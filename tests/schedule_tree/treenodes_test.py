# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.

from dace.sdfg.analysis.schedule_tree import treenodes as tn
from dace import nodes, data, subsets
from dace import Memlet

import dace
import pytest


@pytest.fixture
def tasklet() -> tn.TaskletNode:
    return tn.TaskletNode(nodes.Tasklet("noop", {}, {}, code="pass"), {}, {})


@pytest.mark.parametrize('ScopeClass', (
    tn.ScheduleTreeScope,
    tn.ControlFlowScope,
    tn.GBlock,
    tn.ElseScope,
))
def test_schedule_tree_scope_children(ScopeClass: type[tn.ScheduleTreeScope], tasklet: tn.TaskletNode) -> None:
    scope = ScopeClass(children=[tasklet])

    for child in scope.children:
        assert child.parent == scope

    scope = ScopeClass(children=[])
    scope.add_child(tasklet)

    for child in scope.children:
        assert child.parent == scope

    scope = ScopeClass(children=[])
    scope.add_children([tasklet])

    for child in scope.children:
        assert child.parent == scope


@pytest.mark.parametrize('LoopScope', (
    tn.LoopScope,
    tn.ForScope,
    tn.WhileScope,
    tn.DoWhileScope,
))
def test_loop_scope_children(LoopScope: type[tn.LoopScope], tasklet: tn.TaskletNode) -> None:
    scope = LoopScope(loop=None, children=[tasklet])

    for child in scope.children:
        assert child.parent == scope

    scope = LoopScope(loop=None, children=[])
    scope.add_child(tasklet)

    for child in scope.children:
        assert child.parent == scope

    scope = LoopScope(loop=None, children=[])
    scope.add_children([tasklet])

    for child in scope.children:
        assert child.parent == scope


@pytest.mark.parametrize('IfScope', (
    tn.IfScope,
    tn.StateIfScope,
    tn.ElifScope,
))
def test_if_scope_children(IfScope: type[tn.IfScope | tn.ElifScope], tasklet: tn.TaskletNode) -> None:
    scope = IfScope(condition=None, children=[tasklet])

    for child in scope.children:
        assert child.parent == scope

    scope = IfScope(condition=None, children=[])
    scope.add_child(tasklet)

    for child in scope.children:
        assert child.parent == scope

    scope = IfScope(condition=None, children=[])
    scope.add_children([tasklet])

    for child in scope.children:
        assert child.parent == scope


@pytest.mark.parametrize('DataflowScope', (
    tn.DataflowScope,
    tn.MapScope,
    tn.ConsumeScope,
))
def test_dataflow_scope_children(DataflowScope: type[tn.DataflowScope], tasklet: tn.TaskletNode) -> None:
    scope = DataflowScope(node=None, children=[tasklet])

    for child in scope.children:
        assert child.parent == scope

    scope = DataflowScope(node=None, children=[])
    scope.add_child(tasklet)

    for child in scope.children:
        assert child.parent == scope

    scope = DataflowScope(node=None, children=[])
    scope.add_children([tasklet])

    for child in scope.children:
        assert child.parent == scope


def test_scope_inputs_outputs() -> None:
    write_scalar = tn.TaskletNode(
        nodes.Tasklet('bla', {}, {'out'}, 'out = 1'),
        {},
        {'out': Memlet('scalar[0]')},
    )
    read_scalar = tn.TaskletNode(
        nodes.Tasklet('bla2', {'inp'}, {'out'}, 'out = inp + 1'),
        {'inp': Memlet('scalar[0]')},
        {'out': Memlet('A[1]')},
    )
    map_scope = tn.MapScope(
        node=nodes.MapEntry(nodes.Map('map', ['i'], subsets.Range.from_string("0:20"))),
        children=[write_scalar, read_scalar],
    )

    stree = tn.ScheduleTreeRoot(
        name='tester',
        containers={
            'A': data.Array(dace.float64, [20]),
            'scalar': data.Scalar(dace.float64)
        },
        children=[map_scope],
    )

    assert stree
    assert len(map_scope.input_memlets()) == 0
    assert len(map_scope.output_memlets()) == 2


if __name__ == '__main__':
    test_schedule_tree_scope_children(tn.ScheduleTreeScope, tasklet)
    test_schedule_tree_scope_children(tn.ControlFlowScope, tasklet)
    test_schedule_tree_scope_children(tn.GBlock, tasklet)
    test_schedule_tree_scope_children(tn.ElseScope, tasklet)
    test_loop_scope_children(tn.LoopScope, tasklet)
    test_loop_scope_children(tn.ForScope, tasklet)
    test_loop_scope_children(tn.WhileScope, tasklet)
    test_loop_scope_children(tn.DoWhileScope, tasklet)
    test_if_scope_children(tn.IfScope, tasklet)
    test_if_scope_children(tn.StateIfScope, tasklet)
    test_if_scope_children(tn.ElifScope, tasklet)
    test_dataflow_scope_children(tn.DataflowScope, tasklet)
    test_dataflow_scope_children(tn.MapScope, tasklet)
    test_dataflow_scope_children(tn.ConsumeScope, tasklet)
    test_scope_inputs_outputs()
