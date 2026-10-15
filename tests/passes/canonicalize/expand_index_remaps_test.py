# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Canonicalization expands ``np.roll`` (``CShift``) before fusion, so a stencil written with rolls is one map that
reads its grid at shifted indices -- not one full-size copy of the grid per roll (laplacian_stencil_3d)."""

import numpy as np

import dace
from dace.libraries.standard.nodes.broadcast import Broadcast
from dace.libraries.standard.nodes.cshift import CShift
from dace.sdfg import nodes
from dace.transformation.passes.canonicalize.pipeline import canonicalize

N = dace.symbol("N")


@dace.program
def rolled_stencil(psi: dace.float64[N, N, N], lap: dace.float64[N, N, N]):
    lap[:] = (
        -6.0 * psi
        + np.roll(psi, 1, axis=0)
        + np.roll(psi, -1, axis=0)
        + np.roll(psi, 1, axis=1)
        + np.roll(psi, -1, axis=1)
        + np.roll(psi, 1, axis=2)
        + np.roll(psi, -1, axis=2)
    )


@dace.program
def broadcast_row(row: dace.float64[N], out: dace.float64[N, N]):
    out[:] = np.broadcast_to(row, (N, N)) * 2.0


def full_size_transients(sdfg: dace.SDFG) -> list[str]:
    """The transients of ``sdfg`` (any nesting level) whose size grows with ``N``: a copy of a whole grid."""
    return [
        f"{sd.label}.{name}"
        for sd in sdfg.all_sdfgs_recursive()
        for name, desc in sd.arrays.items()
        if desc.transient
        and "N" in {str(s) for s in dace.symbolic.pystr_to_symbolic(str(desc.total_size)).free_symbols}
    ]


def test_a_rolled_stencil_canonicalizes_to_one_map_and_no_shift_copies():
    sdfg = rolled_stencil.to_sdfg(simplify=True)
    canonicalize(sdfg, validate=True)
    remaps = [n for n, _ in sdfg.all_nodes_recursive() if isinstance(n, (CShift, Broadcast))]
    assert not remaps, f"index remaps survived canonicalization: {remaps}"
    top_maps = [
        n
        for state in sdfg.states()
        for n in state.nodes()
        if isinstance(n, nodes.MapEntry) and state.entry_node(n) is None
    ]
    assert len(top_maps) == 1, f"expected one fused map, got {[str(m) for m in top_maps]}"
    assert not full_size_transients(sdfg), f"full-size transients left: {full_size_transients(sdfg)}"

    size = 7
    psi = np.random.default_rng(0).standard_normal((size, size, size))
    lap = np.zeros_like(psi)
    sdfg(psi=psi, lap=lap, N=size)
    expected = -6.0 * psi
    for axis in (0, 1, 2):
        expected += np.roll(psi, 1, axis=axis) + np.roll(psi, -1, axis=axis)
    assert np.allclose(lap, expected)


def test_a_broadcast_fuses_into_its_consumer():
    sdfg = broadcast_row.to_sdfg(simplify=True)
    canonicalize(sdfg, validate=True)
    assert not [n for n, _ in sdfg.all_nodes_recursive() if isinstance(n, Broadcast)]
    assert not full_size_transients(sdfg), f"full-size transients left: {full_size_transients(sdfg)}"
    row = np.arange(5, dtype=np.float64)
    out = np.zeros((5, 5))
    sdfg(row=row, out=out, N=5)
    assert np.allclose(out, np.broadcast_to(row, (5, 5)) * 2.0)


if __name__ == "__main__":
    test_a_rolled_stencil_canonicalizes_to_one_map_and_no_shift_copies()
    test_a_broadcast_fuses_into_its_consumer()
