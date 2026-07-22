# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""A1 externalize_nest (GLOBAL_LAYOUT_DESIGN.md): each nest of a multi-nest program, cut out into a
standalone SDFG, is bit-exact against its PER-NEST numpy oracle -- inputs produced by earlier nests
are provided (promoted to non-transient by the cutout), outputs start as deterministic noise and
must be fully overwritten. The argument-fill invariants (complex dtypes, and integer fills capped by
the smallest dynamically-indexed extent) are pinned here too."""
import numpy
import pytest

import dace
from dace.transformation.layout.externalize import (externalize_nest, indexed_extent_bound, nest_arguments,
                                                    nest_entries, written_array_names)
from dace.transformation.layout.prepare import prepare_for_layout

from tests.transformations.layout import multinest_programs as fixtures

# Each fixture nest writes exactly one array; its name indexes the per-nest oracle.
NEST_INDEX_BY_OUTPUT = {"B": 0, "C": 1, "D": 2}

N = dace.symbol("N")
M = dace.symbol("M")


@dace.program
def complex_scale(zc: dace.complex64[N], zd: dace.complex128[N], zout: dace.complex128[N]):
    for i in dace.map[0:N]:
        zout[i] = zd[i] + zc[i]


@dace.program
def masked_gather(idx: dace.int64[N], data: dace.float64[M], out: dace.float64[N]):
    """Indirect gather under a mask: the branch makes the read of ``data`` a DYNAMIC memlet, which is
    what marks ``data``'s extent as an in-bounds cap for any integer argument."""
    for i in dace.map[0:N]:
        if idx[i] > 0:
            out[i] = data[idx[i]]


@dace.program
def plain_gather(idx: dace.int64[N], data: dace.float64[M], out: dace.float64[N]):
    """The UNCONDITIONAL gather -- the common source form, and the one the dynamic-memlet signal alone
    misses: the frontend lowers it to a nested SDFG handed all of ``data`` through a STATIC full-range
    memlet, so nothing is marked dynamic and the cap has to recognise the whole-array read instead."""
    for i in dace.map[0:N]:
        out[i] = data[idx[i]]


@dace.program
def int_values(counts: dace.int64[N], out: dace.float64[N]):
    """An integer input used as a VALUE, never as an index -- no dynamic memlet anywhere."""
    for i in dace.map[0:N]:
        out[i] = counts[i] * 1.0


def all_nests(sdfg):
    """Every (state, top-level map entry) pair in the program, any order."""
    return [(state, entry) for state in sdfg.states() for entry in nest_entries(state)]


def externalized_single_nest(program, name):
    """The one externalized nest of a single-nest fixture program."""
    sdfg = program.to_sdfg(simplify=True)
    prepare_for_layout(sdfg)
    nests = all_nests(sdfg)
    assert len(nests) == 1, nests
    state, entry = nests[0]
    return externalize_nest(state, entry, name=name)


@pytest.mark.parametrize("program_name", ["conflict2", "conflict3", "agree2"])
def test_externalized_nests_match_per_nest_oracles(program_name, n=48):
    program, oracle, nest_oracles = fixtures.PROGRAMS[program_name]
    sdfg = program.to_sdfg(simplify=True)
    prepare_for_layout(sdfg)

    inputs = fixtures.make_inputs(n, seed=3)
    chain = {"A": inputs["A"], **oracle(inputs["A"])}  # every array's oracle value, feeds nest inputs

    nests = all_nests(sdfg)
    assert len(nests) == len(nest_oracles)
    covered = set()
    for state, entry in nests:
        ext = externalize_nest(state, entry, name=f"{program_name}_{entry.map.label}_ext")
        written = written_array_names(ext)
        assert len(written) == 1, f"fixture nests write exactly one array, got {written}"
        out_name = written.pop()
        covered.add(out_name)

        provided = {name: chain[name] for name in ext.arrays if name in chain and name != out_name}
        args = nest_arguments(ext, symbols={"N": n}, provided=provided, seed=7)
        assert not numpy.allclose(args[out_name], chain[out_name])  # output starts as noise
        ext(**args, N=n)

        reference = nest_oracles[NEST_INDEX_BY_OUTPUT[out_name]](**chain)
        assert numpy.allclose(args[out_name], reference[out_name]), \
            f"{program_name}: externalized nest writing {out_name} diverges from its oracle"
    assert len(covered) == len(nest_oracles)  # each nest hit a DISTINCT oracle (KeyError above on a foreign one)


def test_externalize_refuses_ambiguous_state_without_map_entry():
    """A state holding several nests must be refused when no map_entry picks one."""
    sdfg = dace.SDFG("two_nests_one_state")
    sdfg.add_array("X", [16], dace.float64)
    sdfg.add_array("Y", [16], dace.float64)
    state = sdfg.add_state("s", is_start_block=True)
    for arr in ("X", "Y"):
        me, mx = state.add_map(f"m_{arr}", {"i": "0:16"})
        t = state.add_tasklet(f"t_{arr}", {"a"}, {"b"}, "b = a + 1.0")
        state.add_memlet_path(state.add_read(arr), me, t, dst_conn="a", memlet=dace.Memlet(f"{arr}[i]"))
        state.add_memlet_path(t, mx, state.add_write(arr), src_conn="b", memlet=dace.Memlet(f"{arr}[i]"))
    with pytest.raises(ValueError, match="top-level"):
        externalize_nest(state)


def test_complex_dtypes_get_a_deterministic_fill(n=6):
    """A complex argument used to hit ``NotImplementedError``, so no complex nest could be
    externalized at all. Both halves must be filled: a real-only fill would let a candidate layout
    that corrupts the imaginary part verify against a reference that never had one."""
    ext = externalized_single_nest(complex_scale, "ext_complex_scale")
    args = nest_arguments(ext, symbols={"N": n}, seed=0)
    assert set(args) == {"zc", "zd", "zout"}

    for name, expected in (("zc", numpy.complex64), ("zd", numpy.complex128), ("zout", numpy.complex128)):
        assert args[name].dtype == expected, (name, args[name].dtype)
        assert args[name].shape == (n, )
        assert numpy.all(args[name].imag != 0.0), name

    again = nest_arguments(ext, symbols={"N": n}, seed=0)
    assert numpy.array_equal(again["zd"], args["zd"])  # same seed -> same buffers


def test_integer_fill_cap_covers_the_unconditional_gather(n=16):
    """The gather form that carries NO dynamic memlet. Keying the cap on ``memlet.dynamic`` alone leaves
    the most ordinary `data[idx[i]]` uncapped -- exactly the case the fix exists for -- because the
    frontend passes the whole of ``data`` into the map body with a static full-range memlet."""
    ext = externalized_single_nest(plain_gather, "ext_plain_gather")
    tight = {"N": n, "M": 4}
    assert indexed_extent_bound(ext, tight) == 4
    idx = nest_arguments(ext, symbols=tight, seed=0)["idx"]
    assert idx.min() >= 0 and idx.max() < 4, idx


def test_integer_fill_is_capped_by_the_indexed_extent(n=16):
    """An integer argument may BE the index of an indirect access, so it is filled in
    ``[0, min(8, smallest indexable extent))``. The fill used to be ``[0, 8)`` regardless, so
    ``data[idx[i]]`` with a data extent below 8 read OUT OF BOUNDS in the REFERENCE run -- invisible to
    verification, because every candidate read the same wrong slot and agreed with it."""
    ext = externalized_single_nest(masked_gather, "ext_masked_gather")

    tight = {"N": n, "M": 4}  # the gathered array is SHORTER than the plain [0, 8) fill
    assert indexed_extent_bound(ext, tight) == 4
    idx = nest_arguments(ext, symbols=tight, seed=0)["idx"]
    assert idx.min() >= 0 and idx.max() < 4, idx

    roomy = {"N": n, "M": 32}  # every dynamically-reached extent is >= 8 -> the cap is inert
    assert indexed_extent_bound(ext, roomy) == n
    assert nest_arguments(ext, symbols=roomy, seed=0)["idx"].max() == 7


def test_no_indirect_access_leaves_the_full_integer_range(n=16):
    """With no dynamic memlet there is nothing to cap against: the bound is ``None`` and integer
    arguments keep the full ``[0, 8)`` fill, so the in-bounds cap never silently narrows the value
    range of an ordinary integer input."""
    ext = externalized_single_nest(int_values, "ext_int_values")
    assert indexed_extent_bound(ext, {"N": n}) is None
    counts = nest_arguments(ext, symbols={"N": n}, seed=0)["counts"]
    assert counts.min() >= 0 and counts.max() == 7


if __name__ == "__main__":
    for name in ["conflict2", "conflict3", "agree2"]:
        test_externalized_nests_match_per_nest_oracles(name)
    test_externalize_refuses_ambiguous_state_without_map_entry()
    test_complex_dtypes_get_a_deterministic_fill()
    test_integer_fill_cap_covers_the_unconditional_gather()
    test_integer_fill_is_capped_by_the_indexed_extent()
    test_no_indirect_access_leaves_the_full_integer_range()
    print("externalize_nest tests PASS")
