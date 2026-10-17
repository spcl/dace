# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""A level map runs as a loop when its axis is the more contiguous one and its planes span every level: a gather by
the level alone counts, a plane range read per level refuses."""

import numpy as np

import dace
from dace.transformation.passes.canonicalize.pipeline import canonicalize
from dace.transformation.passes.cpu_specialization.demote_level_maps import (
    demote_strided_level_maps,
    demote_uncoalesced_level_maps,
)

N = dace.symbol("N")
K = dace.symbol("K")
NAT = dace.symbol("NAT")
G = dace.symbol("G")
M = dace.symbol("M")
B = dace.symbol("B")
L = dace.symbol("L")
P = dace.symbol("P")


@dace.program
def csr_rows(indptr: dace.int64[N + 1], cols: dace.int64[K], vals: dace.float64[K], out: dace.float64[N, N]):
    """spmm's shape: each row writes its own row of ``out`` at the columns its CSR segment names."""
    for i in range(N):
        for e in range(indptr[i], indptr[i + 1]):
            out[i, cols[e]] = out[i, cols[e]] + vals[e]


@dace.program
def level_planes(w: dace.float64[B, L, P], out: dace.float64[B, L, P]):
    """velocity_tendencies' shape on a GPU: a map over the levels whose planes sweep ``(b, p)``, ``p`` contiguous."""
    for jk in dace.map[0:L]:
        out[:, jk, :] = w[:, jk, :] * 2.0


@dace.program
def column_planes(w: dace.float64[B, L, P], out: dace.float64[B, L, P]):
    """The map is over the contiguous axis already: its planes over ``(b, jk)`` are strided."""
    for p in dace.map[0:P]:
        out[:, :, p] = w[:, :, p] * 2.0


def atom_gathers(atom_plane: bool = False) -> dace.SDFG:
    """newdxx_g's shape: a map over atoms whose nested body sweeps planes over every G-vector, gathering from the
    atom's own row of ``table`` (the level alone) into the atom's slice of ``out`` at an offset the body is handed
    (the plane alone, as ``deexx[ijkb0 + ih]``). ``atom_plane`` adds a second plane over the atom's own count,
    ``0:na + 1``, which differs per atom, the way newdxx_g's projector plane does."""
    sdfg = dace.SDFG("atom_gathers")
    for name, shape, dtype in (("table", [NAT, M], dace.float64), ("idx", [G], dace.int64), ("w", [G], dace.float64)):
        sdfg.add_array(name, shape, dtype)
    sdfg.add_array("out", [NAT * G], dace.float64)
    body = dace.SDFG("atom_body")
    for name, shape, dtype in (("table", [NAT, M], dace.float64), ("idx", [G], dace.int64), ("w", [G], dace.float64)):
        body.add_array(name, shape, dtype)
    body.add_array("out", [NAT * G], dace.float64)
    body.add_symbol("na", dace.int64)
    body.add_symbol("base", dace.int64)
    body.add_state().add_mapped_tasklet(
        "gather",
        {"g": "0:G"},
        {"t": dace.Memlet("table[na, 0:M]"), "k": dace.Memlet("idx[g]"), "x": dace.Memlet("w[g]")},
        "o = t[k] * x",
        {"o": dace.Memlet("out[base + g]")},
        external_edges=True,
    )
    if atom_plane:
        body.add_symbol("nproj", dace.int64)
        body.add_array("count", [NAT], dace.float64)
        body.add_state_after(body.start_state).add_mapped_tasklet(
            "project",
            {"p": "0:nproj"},
            {},
            "o = 1.0",
            {"o": dace.Memlet("count[na]", wcr="lambda x, y: x + y")},
            external_edges=True,
        )
        sdfg.add_array("count", [NAT], dace.float64)
    state = sdfg.add_state()
    entry, exit_ = state.add_map("atoms", {"na": "0:NAT"})
    nested = state.add_nested_sdfg(
        body,
        {"table", "idx", "w"},
        {"out", "count"} if atom_plane else {"out"},
        {"na": "na", "base": "na * G", "NAT": "NAT", "M": "M", "G": "G", **({"nproj": "na + 1"} if atom_plane else {})},
    )
    for name, subset in (("table", "0:NAT, 0:M"), ("idx", "0:G"), ("w", "0:G")):
        state.add_memlet_path(
            state.add_read(name), entry, nested, dst_conn=name, memlet=dace.Memlet(f"{name}[{subset}]")
        )
    state.add_memlet_path(
        nested, exit_, state.add_write("out"), src_conn="out", memlet=dace.Memlet("out[na * G:(na + 1) * G]")
    )
    if atom_plane:
        state.add_memlet_path(
            nested, exit_, state.add_write("count"), src_conn="count", memlet=dace.Memlet("count[na]")
        )
    sdfg.validate()
    return sdfg


def canonical(program):
    sdfg = program.to_sdfg(simplify=True)
    canonicalize(sdfg, target="cpu")
    return sdfg


def test_a_row_map_over_csr_segments_stays_a_map():
    sdfg = canonical(csr_rows)
    assert demote_strided_level_maps(sdfg) == 0
    indptr = np.array([0, 2, 3, 5], dtype=np.int64)
    cols = np.array([0, 2, 1, 0, 2], dtype=np.int64)
    vals = np.arange(1.0, 6.0)
    out = np.zeros((3, 3))
    sdfg(indptr=indptr, cols=cols, vals=vals, out=out, N=3, K=5)
    expected = np.zeros((3, 3))
    for i in range(3):
        for e in range(indptr[i], indptr[i + 1]):
            expected[i, cols[e]] += vals[e]
    assert np.allclose(out, expected)


def test_an_atom_map_gathering_its_own_row_is_demoted():
    sdfg = atom_gathers()
    assert demote_strided_level_maps(sdfg) == 1
    rng = np.random.default_rng(0)
    table, idx, w = rng.random((3, 7)), rng.integers(0, 7, 11), rng.random(11)
    out = np.zeros(33)
    sdfg(table=table, idx=idx, w=w, out=out, NAT=3, G=11, M=7)
    assert np.allclose(out, (table[:, idx] * w).ravel())


def test_an_atom_map_with_one_per_atom_plane_is_still_demoted():
    """The G-vector plane spans every atom and moves out; the per-atom plane stays inside the atom loop."""
    sdfg = atom_gathers(atom_plane=True)
    assert demote_strided_level_maps(sdfg) == 1
    rng = np.random.default_rng(0)
    table, idx, w = rng.random((3, 7)), rng.integers(0, 7, 11), rng.random(11)
    out, count = np.zeros(33), np.zeros(3)
    sdfg(table=table, idx=idx, w=w, out=out, count=count, NAT=3, G=11, M=7)
    assert np.allclose(out, (table[:, idx] * w).ravel())
    assert np.allclose(count, [1.0, 2.0, 3.0])


def test_a_level_map_over_contiguous_planes_is_demoted_for_the_gpu():
    """Offloaded as it is, the level map is the kernel and each thread walks whole planes; demoted, the planes are."""
    sdfg = level_planes.to_sdfg(simplify=True)
    assert demote_uncoalesced_level_maps(sdfg) == 1
    w = np.random.default_rng(0).random((2, 3, 5))
    out = np.zeros_like(w)
    sdfg(w=w, out=out, B=2, L=3, P=5)
    assert np.allclose(out, 2.0 * w)


def test_a_map_over_the_contiguous_axis_stays_the_gpu_kernel():
    sdfg = column_planes.to_sdfg(simplify=True)
    assert demote_uncoalesced_level_maps(sdfg) == 0


if __name__ == "__main__":
    test_a_row_map_over_csr_segments_stays_a_map()
    test_an_atom_map_gathering_its_own_row_is_demoted()
    test_an_atom_map_with_one_per_atom_plane_is_still_demoted()
    test_a_level_map_over_contiguous_planes_is_demoted_for_the_gpu()
    test_a_map_over_the_contiguous_axis_stays_the_gpu_kernel()
