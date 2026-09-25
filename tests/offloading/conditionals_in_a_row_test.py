# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""A loop body holding many conditionals in a row is offloaded in time linear in its blocks.

Every conditional is a diamond whose arms meet again at its close node, and the tail search walked
the section once per ROUTE through it: two routes per conditional, so ``2^k`` walks for ``k`` of them
in a row. ls3df_scf's SCF loop holds 48, and the canon GPU offload of it ran into the 3600 s cap
without finishing, while the CPU column validated in under 5 minutes.
"""
import collections

import numpy as np
import pytest

import dace
from dace.sdfg.state import LoopRegion
from dace.transformation import pass_pipeline as ppl
from dace.transformation.passes.offloading import OffloadToAccelerator
from dace.transformation.passes.offloading.offloading_ir_node import OffloadingIRNode
from dace.ordered import OrderedSet

N = dace.symbol('N')
#: Conditionals in a row: 2^48 routes, which no route-by-route walk finishes.
IN_A_ROW = 48
STEPS = 3


@dace.program
def conditionals_in_a_row(A: dace.float64[N], c: dace.int64):
    for step in range(STEPS):
        for i in dace.map[0:N]:
            A[i] = A[i] * 0.5
        for j in dace.unroll(range(IN_A_ROW)):
            if c > j:
                for i in dace.map[0:N]:
                    A[i] = A[i] + 1.0
            else:
                for i in dace.map[0:N]:
                    A[i] = A[i] - 1.0


def reference(A: np.ndarray, c: int) -> None:
    for step in range(STEPS):
        taken = min(max(c, 0), IN_A_ROW)
        A *= 0.5
        A += taken - (IN_A_ROW - taken)


def ir_diamonds(count: int):
    """``(open, tail)`` for one section of ``count`` two-armed conditionals in a row and a last state.

    The shape the IR pass builds for them: each conditional is ``open_if -> arm -> close_if`` twice,
    and the next conditional hangs off the close node.
    """
    sdfg = dace.SDFG('ir_diamonds')
    section = OffloadingIRNode.new_open_node(sdfg.add_state('section'))
    previous = section
    for k in range(count):
        cond = OffloadingIRNode.new_open_node(sdfg.add_state(f'if_{k}'))
        previous.append_node(cond)
        for arm in ('then', 'else'):
            state = OffloadingIRNode.new_state_node(sdfg.add_state(f'{arm}_{k}'), OrderedSet(), OrderedSet())
            cond.append_node(state)
            state.append_node(cond.close)
        previous = cond.close
    tail = OffloadingIRNode.new_state_node(sdfg.add_state('tail'), OrderedSet(), OrderedSet())
    previous.append_node(tail)
    tail.append_node(section.close)
    return section, tail


def test_a_tail_behind_conditionals_in_a_row_is_found_once():
    """Twenty conditionals are a million routes to the one tail, and the walk listed it once per route."""
    section, tail = ir_diamonds(20)
    assert section.get_all_tails() == [tail]


@pytest.mark.parametrize('count, one_route', [(0, True), (1, False), (IN_A_ROW, False)])
def test_a_conditional_makes_a_section_more_than_one_route(count, one_route):
    """Whether the close node's locations can be read off the section as a straight line: a section
    with a conditional in it is not one, however its arms meet again, and that is answered without
    walking every route."""
    section = ir_diamonds(count)[0]
    assert section.has_one_route() is one_route


def offloaded() -> dace.SDFG:
    sdfg = conditionals_in_a_row.to_sdfg(simplify=True)
    ppl.Pipeline([OffloadToAccelerator()]).apply_pass(sdfg, {})
    sdfg.validate()
    return sdfg


def copies_inside_loops(sdfg: dace.SDFG) -> collections.Counter:
    """``(source, destination)`` of every copy between host and device memory under a loop."""
    found = collections.Counter()
    for node, state in sdfg.all_nodes_recursive():
        if not (isinstance(state, dace.SDFGState) and isinstance(node, dace.nodes.AccessNode)):
            continue
        region = state.parent_graph
        while region is not None and not isinstance(region, (LoopRegion, dace.SDFG)):
            region = region.parent_graph
        if not isinstance(region, LoopRegion):
            continue
        for edge in state.out_edges(node):
            if not isinstance(edge.dst, dace.nodes.AccessNode):
                continue
            on_device = [n.desc(state.sdfg).storage == dace.StorageType.GPU_Global for n in (node, edge.dst)]
            if on_device[0] != on_device[1]:
                found[(node.data, edge.dst.data)] += 1
    return found


def test_conditionals_in_a_row_are_offloaded_and_keep_the_data_on_the_device():
    """The offload finishes, every map is a kernel, and nothing crosses the bus inside the loop."""
    sdfg = offloaded()
    maps = [n for n, parent in sdfg.all_nodes_recursive() if isinstance(n, dace.nodes.MapEntry)]
    assert len(maps) == 2 * IN_A_ROW + 1
    assert all(entry.map.schedule == dace.ScheduleType.GPU_Device for entry in maps)
    assert not copies_inside_loops(sdfg)


@pytest.mark.gpu
@pytest.mark.parametrize('c', [0, 5, IN_A_ROW + 2])
def test_the_offloaded_conditionals_compute_what_numpy_computes(c):
    sdfg = offloaded()
    for node, parent in sdfg.all_nodes_recursive():
        if isinstance(node, dace.nodes.MapEntry) and node.map.schedule == dace.ScheduleType.GPU_Device:
            node.map.gpu_block_size = [128, 1, 1]
    A = np.arange(16, dtype=np.float64)
    want = A.copy()
    reference(want, c)
    sdfg(A=A, c=c, N=16)
    np.testing.assert_array_equal(A, want)
