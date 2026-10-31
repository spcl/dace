# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""A map lowered to a loop stays known-parallel: ``LoopToMap`` lifts it without a dependence test, and loop fusion
keeps the mark only where both loops are parallel and their bodies meet at the same iteration."""

import numpy as np

import dace
from dace.sdfg import nodes
from dace.sdfg.state import LoopRegion
from dace.transformation.dataflow.map_for_loop import MapToForLoop
from dace.transformation.interstate.loop_fusion import LoopFusion
from dace.transformation.interstate.loop_to_map import LoopToMap
from dace.transformation.passes.analysis import loop_analysis
from dace.transformation.passes.canonicalize.normalize_loop_and_map_origin import NormalizeLoopAndMapOrigin

N = dace.symbol("N")
K = dace.symbol("K")


@dace.program
def permute(A: dace.float64[N], idx: dace.int64[N], B: dace.float64[N]):
    for i in dace.map[0:N]:
        B[idx[i]] = A[i]


@dace.program
def same_point(A: dace.float64[N], B: dace.float64[N], C: dace.float64[N]):
    for i in dace.map[0:N]:
        B[i] = A[i] + 1.0
    for i in dace.map[0:N]:
        C[i] = B[i] * 2.0


@dace.program
def read_behind(A: dace.float64[N], B: dace.float64[N], C: dace.float64[N]):
    for i in dace.map[1:N]:
        B[i] = A[i] + 1.0
    for i in dace.map[1:N]:
        C[i] = B[i - 1] * 2.0


@dace.program
def shifted_halves(A: dace.float64[3 * N]):
    for i in dace.map[0:N]:
        A[i + N] = A[i] + 1.0
    for i in dace.map[N : 2 * N]:
        A[i + N] = A[i] + 1.0


@dace.program
def small_scatter(A: dace.float64[K], idx: dace.int64[K], B: dace.float64[K]):
    for i in dace.map[0:K]:
        B[idx[i]] = A[i]


def lowered(program) -> tuple[dace.SDFG, list[LoopRegion]]:
    """``program``'s SDFG with every map lowered to a loop, adjacent loops as direct siblings."""
    sdfg = program.to_sdfg(simplify=True)
    while entries := [n for st in sdfg.states() for n in st.nodes() if isinstance(n, nodes.MapEntry)]:
        MapToForLoop.apply_to(sdfg, map_entry=entries[0])
    sdfg.simplify()
    return sdfg, [r for r in sdfg.all_control_flow_regions() if isinstance(r, LoopRegion)]


def test_a_lowered_map_is_known_parallel_and_lifts_without_a_dependence_test():
    """``B[idx[i]]`` is a scatter no dependence test clears; the map it came from says it is independent."""
    sdfg, loops = lowered(permute)
    assert len(loops) == 1 and loops[0].known_parallel and "B" in loop_analysis.conflict_free_writes(loops[0])
    assert LoopToMap.can_be_applied_to(loops[0].sdfg, loop=loops[0])
    loops[0].known_parallel = False
    loops[0].conflict_free_writes = []
    assert not LoopToMap.can_be_applied_to(loops[0].sdfg, loop=loops[0])


def test_a_known_parallel_loop_over_a_small_symbol_is_still_mapped():
    """The SMALL-scale rule keeps a loop a loop only when its independence is unknown: a lowered map is mapped back,
    and the CPU specialization decides whether it runs single-core."""
    sdfg, loops = lowered(small_scatter)
    sdfg.set_symbol_scale("K", dace.symbolic.Scale.SMALL)
    options = {"keep_small_loops_sequential": True}
    assert len(loops) == 1 and loop_analysis.is_known_parallel(loops[0])
    assert LoopToMap.can_be_applied_to(loops[0].sdfg, options=options, loop=loops[0])
    loops[0].known_parallel = False
    assert not LoopToMap.can_be_applied_to(loops[0].sdfg, options=options, loop=loops[0])


def test_a_rewritten_body_keeps_the_conflict_free_writes_and_a_new_header_drops_both_marks():
    """A body rewrite that keeps what an iteration touches keeps the mark; one that reads another iteration's element
    drops it. Neither can make two iterations' scatters collide, so ``B`` stays conflict-free; a header swap (an
    interchange) means another iteration space, for which nothing was ever known."""
    sdfg, loops = lowered(permute)
    loop = loops[0]
    loop_analysis.mark_known_parallel(loop, {"B"})
    loop.add_state("rewritten")
    assert loop_analysis.is_known_parallel(loop)
    ahead = loop.add_state("read_ahead")
    ahead.add_edge(ahead.add_read("A"), None, ahead.add_write("B"), None, dace.Memlet("A[i + 1]", other_subset="i"))
    assert not loop_analysis.is_known_parallel(loop)
    assert loop_analysis.conflict_free_writes(loop) == {"B"}
    loop.update_statement = dace.properties.CodeBlock(f"{loop.loop_variable} = {loop.loop_variable} + 2")
    assert not loop_analysis.conflict_free_writes(loop)


def fuse(sdfg: dace.SDFG, first: LoopRegion, second: LoopRegion) -> None:
    instance = LoopFusion()
    graph = first.parent_graph
    instance.setup_match(
        sdfg, graph.cfg_id, -1, {LoopFusion.first: graph.node_id(first), LoopFusion.second: graph.node_id(second)}, 0
    )
    instance.defer_doall = False
    assert instance.can_be_applied(graph, 0, sdfg)
    instance.apply(graph, sdfg)


def test_fusing_two_known_parallel_loops_at_the_same_point_stays_known_parallel():
    sdfg, loops = lowered(same_point)
    fuse(sdfg, *sorted(loops, key=lambda loop: loop.parent_graph.node_id(loop)))
    fused = [r for r in sdfg.all_control_flow_regions() if isinstance(r, LoopRegion)]
    assert len(fused) == 1 and fused[0].known_parallel
    n = 16
    A, B, C = np.random.rand(n), np.zeros(n), np.zeros(n)
    sdfg(A=A, B=B, C=C, N=n)
    assert np.allclose(C, (A + 1.0) * 2.0)


def test_fusing_a_read_behind_carries_a_dependence_and_drops_the_mark():
    """``C[i] = B[i - 1]`` after ``B[i] = ...``: legal to fuse, but iteration ``i`` now reads ``i - 1``'s write."""
    sdfg, loops = lowered(read_behind)
    assert all(loop.known_parallel for loop in loops)
    fuse(sdfg, *sorted(loops, key=lambda loop: loop.parent_graph.node_id(loop)))
    fused = [r for r in sdfg.all_control_flow_regions() if isinstance(r, LoopRegion)]
    assert len(fused) == 1 and not fused[0].known_parallel
    n = 16
    A, B, C = np.random.rand(n), np.zeros(n), np.zeros(n)
    sdfg(A=A, B=B, C=C, N=n)
    ref = np.zeros(n)
    ref[2:] = (A[1:-1] + 1.0) * 2.0
    assert np.allclose(C[2:], ref[2:])


def test_a_known_parallel_loop_does_not_fuse_with_a_sequential_one():
    sdfg, loops = lowered(same_point)
    first, second = sorted(loops, key=lambda loop: loop.parent_graph.node_id(loop))
    second.known_parallel = False
    second.pinned_sequential = True
    instance = LoopFusion()
    graph = first.parent_graph
    instance.setup_match(
        sdfg, graph.cfg_id, -1, {LoopFusion.first: graph.node_id(first), LoopFusion.second: graph.node_id(second)}, 0
    )
    instance.defer_doall = False
    assert not instance.can_be_applied(graph, 0, sdfg)


def test_a_known_parallel_loop_fuses_with_a_provably_parallel_one_which_becomes_known_parallel():
    """True with False: the False loop is proven parallel, so it is marked known-parallel and the pair fuses as two
    known-parallel loops."""
    sdfg, loops = lowered(same_point)
    first, second = sorted(loops, key=lambda loop: loop.parent_graph.node_id(loop))
    second.known_parallel = False
    fuse(sdfg, first, second)
    fused = [r for r in sdfg.all_control_flow_regions() if isinstance(r, LoopRegion)]
    assert len(fused) == 1 and fused[0].known_parallel


def test_mixing_is_allowed_when_the_option_is_off():
    sdfg, loops = lowered(same_point)
    first, second = sorted(loops, key=lambda loop: loop.parent_graph.node_id(loop))
    second.known_parallel = False
    second.pinned_sequential = True
    instance = LoopFusion()
    instance.dont_mix_known_parallel = False
    graph = first.parent_graph
    instance.setup_match(
        sdfg, graph.cfg_id, -1, {LoopFusion.first: graph.node_id(first), LoopFusion.second: graph.node_id(second)}, 0
    )
    instance.defer_doall = False
    assert instance.can_be_applied(graph, 0, sdfg)
    instance.apply(graph, sdfg)
    fused = [r for r in sdfg.all_control_flow_regions() if isinstance(r, LoopRegion)]
    assert len(fused) == 1 and not fused[0].known_parallel


def test_adjacent_halves_fuse_side_by_side_once_normalized_and_stay_known_parallel():
    """``[0, N)`` and ``[N, 2N)`` of ``A[i + N] = A[i] + 1``: concatenated, iteration ``i + N`` reads iteration ``i``'s
    write; normalized to ``[0, N)`` and fused side by side, each iteration reads only its own."""
    sdfg, loops = lowered(shifted_halves)
    NormalizeLoopAndMapOrigin().apply_pass(sdfg, {})
    loops = sorted(loops, key=lambda loop: loop.parent_graph.node_id(loop))
    fuse(sdfg, *loops)
    fused = [r for r in sdfg.all_control_flow_regions() if isinstance(r, LoopRegion)]
    assert len(fused) == 1 and fused[0].known_parallel
    n = 8
    A = np.random.rand(3 * n)
    ref = A.copy()
    ref[n : 2 * n] = ref[:n] + 1.0
    ref[2 * n :] = ref[n : 2 * n] + 1.0
    sdfg(A=A, N=n)
    assert np.allclose(A, ref)


if __name__ == "__main__":
    test_a_lowered_map_is_known_parallel_and_lifts_without_a_dependence_test()
    test_a_known_parallel_loop_over_a_small_symbol_is_still_mapped()
    test_a_rewritten_body_keeps_the_conflict_free_writes_and_a_new_header_drops_both_marks()
    test_fusing_two_known_parallel_loops_at_the_same_point_stays_known_parallel()
    test_fusing_a_read_behind_carries_a_dependence_and_drops_the_mark()
    test_a_known_parallel_loop_does_not_fuse_with_a_sequential_one()
    test_a_known_parallel_loop_fuses_with_a_provably_parallel_one_which_becomes_known_parallel()
    test_mixing_is_allowed_when_the_option_is_off()
    test_adjacent_halves_fuse_side_by_side_once_normalized_and_stay_known_parallel()
