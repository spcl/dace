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

N = dace.symbol("N")


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
    assert len(loops) == 1 and loops[0].known_parallel
    assert LoopToMap.can_be_applied_to(loops[0].sdfg, loop=loops[0])
    loops[0].known_parallel = False
    assert not LoopToMap.can_be_applied_to(loops[0].sdfg, loop=loops[0])


def fuse(sdfg: dace.SDFG, first: LoopRegion, second: LoopRegion) -> None:
    instance = LoopFusion()
    graph = first.parent_graph
    instance.setup_match(
        sdfg, graph.cfg_id, -1, {LoopFusion.first: graph.node_id(first), LoopFusion.second: graph.node_id(second)}, 0
    )
    assert instance.can_be_applied(graph, 0, sdfg, allow_doall_fuse=True)
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
    assert not instance.can_be_applied(graph, 0, sdfg, allow_doall_fuse=True)


if __name__ == "__main__":
    test_a_lowered_map_is_known_parallel_and_lifts_without_a_dependence_test()
    test_fusing_two_known_parallel_loops_at_the_same_point_stays_known_parallel()
    test_fusing_a_read_behind_carries_a_dependence_and_drops_the_mark()
    test_a_known_parallel_loop_does_not_fuse_with_a_sequential_one()
