# Copyright 2019-2023 ETH Zurich and the DaCe authors. All rights reserved.
"""
Tests components in conversion of schedule trees to SDFGs.
"""
import dace
from dace.codegen import control_flow as cf
from dace.properties import CodeBlock
from dace.sdfg import nodes
from dace.sdfg.analysis.schedule_tree import tree_to_sdfg as t2s, treenodes as tn


def test_state_boundaries_none():
    # Manually create a schedule tree
    stree = tn.ScheduleTreeRoot(
        name='tester',
        containers={
            'A': dace.data.Array(dace.float64, [20]),
        },
        children=[
            tn.TaskletNode(nodes.Tasklet('bla', {}, {'out'}, 'out = 1'), {}, {'out': dace.Memlet('A[1]')}),
            tn.TaskletNode(nodes.Tasklet('bla2', {'inp'}, {'out'}, 'out = inp + 1'), {'inp': dace.Memlet('A[1]')},
                           {'out': dace.Memlet('A[1]')}),
        ],
    )

    stree = t2s.insert_state_boundaries_to_tree(stree)
    assert tn.StateBoundaryNode not in [type(n) for n in stree.children]


def test_state_boundaries_waw():
    # Manually create a schedule tree
    stree = tn.ScheduleTreeRoot(
        name='tester',
        containers={
            'A': dace.data.Array(dace.float64, [20]),
        },
        children=[
            tn.TaskletNode(nodes.Tasklet('bla', {}, {'out'}, 'out = 1'), {}, {'out': dace.Memlet('A[1]')}),
            tn.TaskletNode(nodes.Tasklet('bla2', {}, {'out'}, 'out = 2'), {}, {'out': dace.Memlet('A[1]')}),
        ],
    )

    stree = t2s.insert_state_boundaries_to_tree(stree)
    assert [tn.TaskletNode, tn.StateBoundaryNode, tn.TaskletNode] == [type(n) for n in stree.children]


def test_state_boundaries_war():
    # Manually create a schedule tree
    stree = tn.ScheduleTreeRoot(
        name='tester',
        containers={
            'A': dace.data.Array(dace.float64, [20]),
            'B': dace.data.Array(dace.float64, [20]),
        },
        children=[
            tn.TaskletNode(nodes.Tasklet('bla', {'inp'}, {'out'}, 'out = inp + 1'), {'inp': dace.Memlet('A[1]')},
                           {'out': dace.Memlet('B[0]')}),
            tn.TaskletNode(nodes.Tasklet('bla2', {}, {'out'}, 'out = 2'), {}, {'out': dace.Memlet('A[1]')}),
        ],
    )

    stree = t2s.insert_state_boundaries_to_tree(stree)
    assert [tn.TaskletNode, tn.StateBoundaryNode, tn.TaskletNode] == [type(n) for n in stree.children]


def test_state_boundaries_read_write_chain():
    # Manually create a schedule tree
    stree = tn.ScheduleTreeRoot(
        name='tester',
        containers={
            'A': dace.data.Array(dace.float64, [20]),
            'B': dace.data.Array(dace.float64, [20]),
        },
        children=[
            tn.TaskletNode(nodes.Tasklet('bla1', {'inp'}, {'out'}, 'out = inp + 1'), {'inp': dace.Memlet('A[1]')},
                           {'out': dace.Memlet('B[0]')}),
            tn.TaskletNode(nodes.Tasklet('bla2', {'inp'}, {'out'}, 'out = inp + 1'), {'inp': dace.Memlet('B[0]')},
                           {'out': dace.Memlet('A[1]')}),
            tn.TaskletNode(nodes.Tasklet('bla3', {'inp'}, {'out'}, 'out = inp + 1'), {'inp': dace.Memlet('A[1]')},
                           {'out': dace.Memlet('B[0]')}),
        ],
    )

    stree = t2s.insert_state_boundaries_to_tree(stree)
    assert [tn.TaskletNode, tn.TaskletNode, tn.TaskletNode] == [type(n) for n in stree.children]


def test_state_boundaries_data_race():
    # Manually create a schedule tree
    stree = tn.ScheduleTreeRoot(
        name='tester',
        containers={
            'A': dace.data.Array(dace.float64, [20]),
            'B': dace.data.Array(dace.float64, [20]),
        },
        children=[
            tn.TaskletNode(nodes.Tasklet('bla1', {'inp'}, {'out'}, 'out = inp + 1'), {'inp': dace.Memlet('A[1]')},
                           {'out': dace.Memlet('B[0]')}),
            tn.TaskletNode(nodes.Tasklet('bla11', {'inp'}, {'out'}, 'out = inp + 1'), {'inp': dace.Memlet('A[1]')},
                           {'out': dace.Memlet('B[1]')}),
            tn.TaskletNode(nodes.Tasklet('bla2', {'inp'}, {'out'}, 'out = inp + 1'), {'inp': dace.Memlet('B[0]')},
                           {'out': dace.Memlet('A[1]')}),
            tn.TaskletNode(nodes.Tasklet('bla3', {'inp'}, {'out'}, 'out = inp + 1'), {'inp': dace.Memlet('A[1]')},
                           {'out': dace.Memlet('B[0]')}),
        ],
    )

    stree = t2s.insert_state_boundaries_to_tree(stree)
    assert [tn.TaskletNode, tn.TaskletNode, tn.StateBoundaryNode, tn.TaskletNode,
            tn.TaskletNode] == [type(n) for n in stree.children]


def test_state_boundaries_cfg():
    # Manually create a schedule tree
    stree = tn.ScheduleTreeRoot(
        name='tester',
        containers={
            'A': dace.data.Array(dace.float64, [20]),
        },
        children=[
            tn.TaskletNode(nodes.Tasklet('bla1', {}, {'out'}, 'out = 2'), {}, {'out': dace.Memlet('A[1]')}),
            tn.ForScope([
                tn.TaskletNode(nodes.Tasklet('bla2', {}, {'out'}, 'out = i'), {}, {'out': dace.Memlet('A[1]')}),
            ], cf.ForScope(None, None, 'i', None, '0', CodeBlock('i < 20'), 'i + 1', None, [])),
        ],
    )

    stree = t2s.insert_state_boundaries_to_tree(stree)
    assert [tn.TaskletNode, tn.StateBoundaryNode, tn.ForScope] == [type(n) for n in stree.children]


if __name__ == '__main__':
    test_state_boundaries_none()
    test_state_boundaries_waw()
    test_state_boundaries_war()
    test_state_boundaries_read_write_chain()
    test_state_boundaries_data_race()
    test_state_boundaries_cfg()
