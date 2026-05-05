# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.

from dace.sdfg.analysis.schedule_tree import treenodes as tn
from dace import nodes

import pytest


@pytest.fixture
def tasklet() -> nodes.Tasklet:
    return tn.TaskletNode(nodes.Tasklet("noop", {}, {}, code="pass"), {}, {})


@pytest.mark.parametrize('ScopeClass', (
    tn.ScheduleTreeScope,
    tn.ControlFlowScope,
    tn.GBlock,
    tn.ElseScope,
))
def test_schedule_tree_scope_children(ScopeClass: type[tn.ScheduleTreeScope], tasklet: nodes.Tasklet) -> None:
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
def test_loop_scope_children(LoopScope: type[tn.LoopScope], tasklet: nodes.Tasklet) -> None:
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
def test_if_scope_children(IfScope: type[tn.IfScope], tasklet: nodes.Tasklet) -> None:
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
def test_dataflow_scope_children(DataflowScope: type[tn.DataflowScope], tasklet: nodes.Tasklet) -> None:
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
