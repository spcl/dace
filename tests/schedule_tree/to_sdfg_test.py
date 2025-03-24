# Copyright 2019-2024 ETH Zurich and the DaCe authors. All rights reserved.
"""
Tests components in conversion of schedule trees to SDFGs.
"""
import dace
from dace.codegen import control_flow as cf
from dace.properties import CodeBlock
from dace.sdfg import nodes
from dace.sdfg.analysis.schedule_tree import tree_to_sdfg as t2s, treenodes as tn
import pytest


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


@pytest.mark.parametrize('overlap', (False, True))
def test_state_boundaries_waw_ranges(overlap):
    # Manually create a schedule tree
    N = dace.symbol('N')
    stree = tn.ScheduleTreeRoot(
        name='tester',
        containers={
            'A': dace.data.Array(dace.float64, [20]),
        },
        symbols={'N': N},
        children=[
            tn.TaskletNode(nodes.Tasklet('bla', {}, {'out'}, 'pass'), {}, {'out': dace.Memlet('A[0:N/2]')}),
            tn.TaskletNode(nodes.Tasklet('bla2', {}, {'out'}, 'pass'), {},
                           {'out': dace.Memlet('A[1:N]' if overlap else 'A[N/2+1:N]')}),
        ],
    )

    stree = t2s.insert_state_boundaries_to_tree(stree)
    if overlap:
        assert [tn.TaskletNode, tn.StateBoundaryNode, tn.TaskletNode] == [type(n) for n in stree.children]
    else:
        assert [tn.TaskletNode, tn.TaskletNode] == [type(n) for n in stree.children]


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
            ], cf.ForScope(None, None, True, 'i', None, '0', CodeBlock('i < 20'), 'i + 1', None, [])),
        ],
    )

    stree = t2s.insert_state_boundaries_to_tree(stree)
    assert [tn.TaskletNode, tn.StateBoundaryNode, tn.ForScope] == [type(n) for n in stree.children]


def test_state_boundaries_state_transition():
    # Manually create a schedule tree
    stree = tn.ScheduleTreeRoot(
        name='tester',
        containers={
            'A': dace.data.Array(dace.float64, [20]),
        },
        symbols={
            'N': dace.symbol('N'),
        },
        children=[
            tn.AssignNode('irrelevant', CodeBlock('N + 1'), dace.InterstateEdge(assignments=dict(irrelevant='N + 1'))),
            tn.TaskletNode(nodes.Tasklet('bla', {}, {'out'}, 'out = 2'), {}, {'out': dace.Memlet('A[1]')}),
            tn.AssignNode('relevant', CodeBlock('A[1] + 2'),
                          dace.InterstateEdge(assignments=dict(relevant='A[1] + 2'))),
        ],
    )

    stree = t2s.insert_state_boundaries_to_tree(stree)
    assert [tn.AssignNode, tn.TaskletNode, tn.StateBoundaryNode, tn.AssignNode] == [type(n) for n in stree.children]


@pytest.mark.parametrize('boundary', (False, True))
def test_state_boundaries_propagation(boundary):
    # Manually create a schedule tree
    N = dace.symbol('N')
    stree = tn.ScheduleTreeRoot(
        name='tester',
        containers={
            'A': dace.data.Array(dace.float64, [20]),
        },
        symbols={
            'N': N,
        },
        children=[
            tn.MapScope(node=dace.nodes.MapEntry(dace.nodes.Map('map', ['i'], dace.subsets.Range([(1, N - 1, 1)]))),
                        children=[
                            tn.TaskletNode(nodes.Tasklet('inner', {}, {'out'}, 'out = 2'), {},
                                           {'out': dace.Memlet('A[i]')}),
                        ]),
            tn.TaskletNode(nodes.Tasklet('bla', {}, {'out'}, 'out = 2'), {},
                           {'out': dace.Memlet('A[1]' if boundary else 'A[0]')}),
        ],
    )

    stree = t2s.insert_state_boundaries_to_tree(stree)

    node_types = [type(n) for n in stree.preorder_traversal()]
    if boundary:
        assert [tn.MapScope, tn.TaskletNode, tn.StateBoundaryNode, tn.TaskletNode] == node_types[1:]
    else:
        assert [tn.MapScope, tn.TaskletNode, tn.TaskletNode] == node_types[1:]


@pytest.mark.parametrize("control_flow", (True, False))
def test_create_state_boundary_state_transition(control_flow):
    sdfg = dace.SDFG("tester")
    state = sdfg.add_state("start", is_start_block=True)
    bnode = tn.StateBoundaryNode(control_flow)

    t2s.create_state_boundary(bnode, sdfg, state, t2s.StateBoundaryBehavior.STATE_TRANSITION)
    new_label = "cf_state_boundary" if control_flow else "state_boundary"
    assert ["start", new_label] == [state.label for state in sdfg.states()]


@pytest.mark.xfail(reason="Not yet implemented")
def test_create_state_boundary_empty_memlet(control_flow):
    sdfg = dace.SDFG("tester")
    state = sdfg.add_state("start", is_start_block=True)
    bnode = tn.StateBoundaryNode(control_flow)

    t2s.create_state_boundary(bnode, sdfg, state, t2s.StateBoundaryBehavior.EMPTY_MEMLET)


def test_create_tasklet_raw():
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

    sdfg = stree.as_sdfg()
    assert len(sdfg.states()) == 1
    state = sdfg.states()[0]
    first_tasklet, write_read_node, second_tasklet, write_node = state.nodes()

    assert first_tasklet.label == "bla"
    assert not first_tasklet.in_connectors
    assert first_tasklet.out_connectors.keys() == {"out"}

    assert second_tasklet.label == "bla2"
    assert second_tasklet.in_connectors.keys() == {"inp"}
    assert second_tasklet.out_connectors.keys() == {"out"}

    assert [(first_tasklet, write_read_node), (write_read_node, second_tasklet),
            (second_tasklet, write_node)] == [(edge.src, edge.dst) for edge in state.edges()]


def test_create_tasklet_waw():
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

    sdfg = stree.as_sdfg()
    assert len(sdfg.states()) == 2
    s1, s2 = sdfg.states()

    s1_tasklet, s1_anode = s1.nodes()
    assert [(s1_tasklet, s1_anode)] == [(edge.src, edge.dst) for edge in s1.edges()]

    s2_tasklet, s2_anode = s2.nodes()
    assert [(s2_tasklet, s2_anode)] == [(edge.src, edge.dst) for edge in s2.edges()]


def test_create_for_loop():
    # yapf: disable
    loop=tn.ForScope(
        children=[
            tn.TaskletNode(nodes.Tasklet('bla', {}, {'out'}, 'out = 1'), {}, {'out': dace.Memlet('A[1]')}),
            tn.TaskletNode(nodes.Tasklet('bla', {}, {'out'}, 'out = 2'), {}, {'out': dace.Memlet('A[1]')}),
        ],
        header=cf.ForScope(
            itervar="i", init="0", condition=CodeBlock("i<3"), update="i+1",
            dispatch_state=None, parent=None, last_block=True, guard=None, body=None, init_edges=[]
        )
    )
    # yapf: enable

    # Manually create a schedule tree
    stree = tn.ScheduleTreeRoot(name='tester', containers={'A': dace.data.Array(dace.float64, [20])}, children=[loop])

    sdfg = stree.as_sdfg()
    sdfg.validate()


def test_create_while_loop():
    # yapf: disable
    loop=tn.WhileScope(
        children=[
            tn.TaskletNode(nodes.Tasklet('bla', {}, {'out'}, 'out = 1'), {}, {'out': dace.Memlet('A[1]')}),
            tn.TaskletNode(nodes.Tasklet('bla', {}, {'out'}, 'out = 2'), {}, {'out': dace.Memlet('A[1]')}),
        ],
        header=cf.WhileScope(
            test=CodeBlock("A[1] > 5"),
            dispatch_state=None,
            last_block=True,
            parent=None,
            guard=None,
            body=None
        )
    )
    # yapf: enable

    # Manually create a schedule tree
    stree = tn.ScheduleTreeRoot(name='tester', containers={'A': dace.data.Array(dace.float64, [20])}, children=[loop])

    sdfg = stree.as_sdfg()
    sdfg.validate()


def test_create_if_else():
    # Manually create a schedule tree
    stree = tn.ScheduleTreeRoot(name="tester",
                                containers={'A': dace.data.Array(dace.float64, [20])},
                                children=[
                                    tn.IfScope(condition=CodeBlock("A[0] > 0"),
                                               children=[
                                                   tn.TaskletNode(nodes.Tasklet("bla", {}, {"out"}, "out=1"), {},
                                                                  {"out": dace.Memlet("A[1]")}),
                                               ]),
                                    tn.ElseScope([
                                        tn.TaskletNode(nodes.Tasklet("blub", {}, {"out"}, "out=2"), {},
                                                       {"out": dace.Memlet("A[1]")})
                                    ])
                                ])

    sdfg = stree.as_sdfg()
    sdfg.validate()


def test_create_if_without_else():
    # Manually create a schedule tree
    stree = tn.ScheduleTreeRoot(name="tester",
                                containers={'A': dace.data.Array(dace.float64, [20])},
                                children=[
                                    tn.IfScope(condition=CodeBlock("A[0] > 0"),
                                               children=[
                                                   tn.TaskletNode(nodes.Tasklet("bla", {}, {"out"}, "out=1"), {},
                                                                  {"out": dace.Memlet("A[1]")}),
                                               ]),
                                ])

    sdfg = stree.as_sdfg()
    sdfg.validate()


if __name__ == '__main__':
    test_state_boundaries_none()
    test_state_boundaries_waw()
    test_state_boundaries_waw_ranges(overlap=False)
    test_state_boundaries_waw_ranges(overlap=True)
    test_state_boundaries_war()
    test_state_boundaries_read_write_chain()
    test_state_boundaries_data_race()
    test_state_boundaries_cfg()
    test_state_boundaries_state_transition()
    test_state_boundaries_propagation(boundary=False)
    test_state_boundaries_propagation(boundary=True)
    test_create_state_boundary_state_transition(control_flow=True)
    test_create_state_boundary_state_transition(control_flow=False)
    test_create_state_boundary_empty_memlet()
    test_create_tasklet_raw()
    test_create_tasklet_waw()
    test_create_for_loop()
    test_create_while_loop()
    test_create_if_else()
    test_create_if_without_else()
