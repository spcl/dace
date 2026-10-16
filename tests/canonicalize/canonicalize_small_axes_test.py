# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Parallelization never puts a SMALL axis on a parallel map.

``nblks`` is declared ``SMALL`` (a handful of blocks, much smaller than ``nproma``). Every path that turns a loop into
a map -- the ``ParallelizeLoops`` sweep, a lift a caller has proven independent, the sibling-chain hold that waits for
``WavefrontSkew``, the skew itself and the whole canonicalization pipeline -- leaves a loop over ``nblks`` sequential
and still parallelizes the ``nproma`` loops.
"""

import numpy as np

import dace
from dace import symbolic
from dace.sdfg import nodes
from dace.sdfg.state import LoopRegion, SymbolResolver
from dace.symbolic import Scale
from dace.transformation.passes.canonicalize.finalize import finalize_for_target
from dace.transformation.passes.canonicalize.hold_sibling_chains import ParallelizeLoopsHoldingSiblingChains
from dace.transformation.passes.canonicalize.pipeline import canonicalize
from dace.transformation.passes.canonicalize.wavefront_skew import WavefrontSkew
from dace.transformation.passes.parallelize_loops import ParallelizeLoops

nblks = dace.symbol("nblks", dace.int64, small=True)
nproma = dace.symbol("nproma", dace.int64)

NBLKS, NPROMA = 3, 40


def loops(sdfg: dace.SDFG) -> list[LoopRegion]:
    return [r for r in sdfg.all_control_flow_regions(recursive=True) if isinstance(r, LoopRegion) and r.loop_variable]


def maps(sdfg: dace.SDFG) -> list[tuple[nodes.MapEntry, dace.SDFGState]]:
    return [(n, s) for n, s in sdfg.all_nodes_recursive() if isinstance(n, nodes.MapEntry)]


def small_map_params(sdfg: dace.SDFG) -> list[str]:
    """Every map parameter, anywhere in ``sdfg``, whose range spans SMALL symbols only."""
    resolver = SymbolResolver()
    found = []
    for entry, state in maps(sdfg):
        facts = resolver.facts_at(state, entry)
        for param, (begin, end, _) in zip(entry.map.params, entry.map.range.ranges):
            if symbolic.is_small(end - begin, facts):
                found.append(param)
    return found


@dace.program
def blocked_axpy(A: dace.float64[nblks, nproma], B: dace.float64[nblks, nproma]):
    for b in range(nblks):
        for i in range(nproma):
            B[b, i] = 2.0 * A[b, i] + 1.0


@dace.program
def blocked_columns(A: dace.float64[nproma, nblks], B: dace.float64[nproma, nblks]):
    for i in range(nproma):
        for b in range(nblks):
            B[i, b] = A[i, b] - 3.0 * A[i, 0]


def run_blocked(sdfg: dace.SDFG, shape: tuple[int, int], reference) -> None:
    rng = np.random.default_rng(7)
    A = rng.random(shape)
    B = np.zeros(shape)
    sdfg(A=A, B=B, nblks=NBLKS, nproma=NPROMA)
    assert np.allclose(B, reference(A))


def test_parallelize_loops_keeps_the_block_loop_sequential():
    sdfg = blocked_axpy.to_sdfg(simplify=True)
    assert ParallelizeLoops().apply_pass(sdfg, {})
    sdfg.validate()
    assert small_map_params(sdfg) == []
    assert any("nproma" in str(entry.map.range) for entry, _ in maps(sdfg))
    assert [str(loop.loop_condition.as_string) for loop in loops(sdfg) if "nblks" in loop.loop_condition.as_string]
    run_blocked(sdfg, (NBLKS, NPROMA), lambda A: 2.0 * A + 1.0)


def test_parallelize_loops_keeps_an_inner_block_loop_sequential():
    sdfg = blocked_columns.to_sdfg(simplify=True)
    assert ParallelizeLoops().apply_pass(sdfg, {})
    sdfg.validate()
    assert small_map_params(sdfg) == []
    assert any("nproma" in str(entry.map.range) for entry, _ in maps(sdfg))
    run_blocked(sdfg, (NPROMA, NBLKS), lambda A: A - 3.0 * A[:, :1])


def scaled_loop(extent: str) -> tuple[dace.SDFG, LoopRegion]:
    sdfg = dace.SDFG(f"scaled_loop_{extent}")
    sdfg.add_symbol("nblks", dace.int64, scale=Scale.SMALL)
    sdfg.add_symbol("nproma", dace.int64)
    sdfg.add_array("A", ["nproma"], dace.float64)
    loop = LoopRegion("loop", f"i < {extent}", "i", "i = 0", "i = i + 1")
    sdfg.add_node(loop, is_start_block=True)
    body = loop.add_state("body", is_start_block=True)
    tasklet = body.add_tasklet("one", {}, {"o"}, "o = 1.0")
    body.add_edge(tasklet, "o", body.add_write("A"), None, dace.Memlet("A[i]"))
    return sdfg, loop


def test_a_proven_lift_keeps_a_small_loop_sequential():
    """A caller's independence proof replaces the dependence test, not the scale rule."""
    sdfg, loop = scaled_loop("nblks")
    assert not ParallelizeLoops().parallelize_loop(sdfg, loop, proven=True)
    assert loops(sdfg) == [loop]
    assert maps(sdfg) == []


def test_a_proven_lift_of_a_large_loop_still_lifts():
    sdfg, loop = scaled_loop("nproma")
    assert ParallelizeLoops().parallelize_loop(sdfg, loop, proven=True)
    assert loops(sdfg) == []
    assert [entry.map.params for entry, _ in maps(sdfg)] == [["i"]]


@dace.program
def sibling_chain_with_a_block_prefix(A: dace.float64[nproma, nproma]):
    for i in range(1, nproma):
        for j in range(0, nblks):
            A[i, j] = A[i - 1, j] + 1.0
        for k in range(nblks, nproma):
            A[i, k] = A[i - 1, k] * 0.5


def sibling_chain_reference(A: np.ndarray) -> np.ndarray:
    A = A.copy()
    for i in range(1, NPROMA):
        A[i, :NBLKS] = A[i - 1, :NBLKS] + 1.0
        A[i, NBLKS:] = A[i - 1, NBLKS:] * 0.5
    return A


def test_a_small_sibling_does_not_hold_back_its_large_doall_sibling():
    """Both siblings are DOALL: the chain is no wavefront. The SMALL sibling stays a loop for its scale only, so it
    must not count as the refused sibling that holds the chain back for ``WavefrontSkew``."""
    sdfg = sibling_chain_with_a_block_prefix.to_sdfg(simplify=True)
    ParallelizeLoopsHoldingSiblingChains().apply_pass(sdfg, {})
    sdfg.validate()
    assert small_map_params(sdfg) == []
    ranges = [str(entry.map.range) for entry, _ in maps(sdfg)]
    assert any("nblks" in r and "nproma" in r for r in ranges), ranges
    A = np.random.default_rng(3).random((NPROMA, NPROMA))
    expected = sibling_chain_reference(A)
    sdfg(A=A, nblks=NBLKS, nproma=NPROMA)
    assert np.allclose(A, expected)


@dace.program
def wavefront_over_blocks(aa: dace.float64[nblks, nproma]):
    for i in range(1, nblks):
        for j in range(1, nproma):
            aa[i, j] = (aa[i, j - 1] + aa[i - 1, j]) / 1.9


@dace.program
def wavefront_within_blocks(aa: dace.float64[nproma, nblks]):
    for i in range(1, nproma):
        for j in range(1, nblks):
            aa[i, j] = (aa[i, j - 1] + aa[i - 1, j]) / 1.9


def test_wavefront_skew_refuses_a_nest_with_a_small_axis():
    """The front of a 2-D wavefront is at most as wide as the shorter axis: over a SMALL axis it is a SMALL map."""
    for program in (wavefront_over_blocks, wavefront_within_blocks):
        sdfg = program.to_sdfg(simplify=True)
        assert WavefrontSkew().apply_pass(sdfg, {}) is None, program.name
        assert maps(sdfg) == []


@dace.program
def blocked_two_stage(A: dace.float64[nblks, nproma], B: dace.float64[nblks, nproma]):
    for b in range(nblks):
        for i in range(nproma):
            B[b, i] = A[b, i] * A[b, i]
    for b in range(nblks):
        for i in range(1, nproma):
            B[b, i] = B[b, i] + 0.5 * A[b, i - 1]


def blocked_two_stage_reference(A: np.ndarray) -> np.ndarray:
    B = A * A
    B[:, 1:] += 0.5 * A[:, :-1]
    return B


def test_canonicalize_never_maps_a_block_axis():
    for program, shape, reference in (
        (blocked_axpy, (NBLKS, NPROMA), lambda A: 2.0 * A + 1.0),
        (blocked_columns, (NPROMA, NBLKS), lambda A: A - 3.0 * A[:, :1]),
        (blocked_two_stage, (NBLKS, NPROMA), blocked_two_stage_reference),
    ):
        sdfg = canonicalize(program.to_sdfg(simplify=True), validate=True)
        assert small_map_params(sdfg) == [], program.name
        assert any("nproma" in str(entry.map.range) for entry, _ in maps(sdfg)), program.name
        run_blocked(sdfg, shape, reference)


@dace.program
def blocked_slice(A: dace.float64[nblks, nproma], B: dace.float64[nblks, nproma]):
    B[:] = 2.0 * A + 1.0


def test_the_cpu_target_never_shares_out_a_block_axis():
    """Canonicalization keeps the map a slice makes over ``(nblks, nproma)``; the CPU specialization then shares out
    ``nproma`` and runs ``nblks`` sequentially inside."""
    for program, shape, reference in (
        (blocked_slice, (NBLKS, NPROMA), lambda A: 2.0 * A + 1.0),
        (blocked_axpy, (NBLKS, NPROMA), lambda A: 2.0 * A + 1.0),
        (blocked_columns, (NPROMA, NBLKS), lambda A: A - 3.0 * A[:, :1]),
    ):
        sdfg = finalize_for_target(canonicalize(program.to_sdfg(simplify=True)), "cpu")
        resolver = SymbolResolver()
        for entry, state in maps(sdfg):
            if entry.map.schedule == dace.ScheduleType.CPU_Multicore:
                facts = resolver.facts_at(state, entry)
                ranges = entry.map.range.ranges
                assert not any(symbolic.is_small(end - begin, facts) for begin, end, _ in ranges), program.name
        run_blocked(sdfg, shape, reference)


if __name__ == "__main__":
    test_parallelize_loops_keeps_the_block_loop_sequential()
    test_parallelize_loops_keeps_an_inner_block_loop_sequential()
    test_a_proven_lift_keeps_a_small_loop_sequential()
    test_a_proven_lift_of_a_large_loop_still_lifts()
    test_a_small_sibling_does_not_hold_back_its_large_doall_sibling()
    test_wavefront_skew_refuses_a_nest_with_a_small_axis()
    test_canonicalize_never_maps_a_block_axis()
    test_the_cpu_target_never_shares_out_a_block_axis()
