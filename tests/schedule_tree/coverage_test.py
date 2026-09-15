# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""
Ensures that every schedule tree node type can be converted back into an SDFG.

Every concrete node type in ``treenodes`` has a factory that creates a schedule tree containing it, where possible by
converting a small SDFG such that the SDFG-to-tree conversion is exercised as well. Adding a node type without a
factory fails ``test_all_node_types_have_factory``.
"""
from typing import Callable

import dace
import pytest

from dace import data
from dace.libraries.blas import MatMul
from dace.properties import CodeBlock
from dace.sdfg import nodes
from dace.sdfg.analysis.schedule_tree import treenodes as tn
from dace.sdfg.state import (BreakBlock, ConditionalBlock, ContinueBlock, ControlFlowRegion, LoopRegion, NamedRegion,
                             ReturnBlock)

#: Node types that are only used as base classes and never appear in a schedule tree
ABSTRACT_NODE_TYPES = {tn.ScheduleTreeNode, tn.ScheduleTreeScope, tn.ControlFlowScope, tn.DataflowScope}


def _write_tasklet(state: dace.SDFGState, value: str, memlet: str) -> None:
    """
    Adds a tasklet that writes ``value`` to the data container subset given by ``memlet``.
    """
    tasklet = state.add_tasklet('write', {}, {'out'}, f'out = {value}')
    state.add_edge(tasklet, 'out', state.add_write(memlet.split('[')[0]), None, dace.Memlet(memlet))


def _conditional_sdfg(conditions: list[str | None]) -> dace.SDFG:
    """
    Creates an SDFG with a conditional block that has one branch per given condition (``None`` for else).
    """
    sdfg = dace.SDFG('conditional')
    sdfg.add_symbol('N', dace.int64)
    sdfg.add_array('A', [10], dace.float64)
    block = ConditionalBlock('cond')
    sdfg.add_node(block, is_start_block=True)
    for i, condition in enumerate(conditions):
        body = ControlFlowRegion(f'branch_{i}', sdfg=sdfg)
        block.add_branch(CodeBlock(condition) if condition is not None else None, body)
        _write_tasklet(body.add_state(f'branch_state_{i}', is_start_block=True), str(i), 'A[0]')
    return sdfg


def _loop_sdfg(loop: LoopRegion, increment: bool = False) -> tuple[dace.SDFG, dace.SDFGState]:
    """
    Creates an SDFG with the given loop, whose body writes to ``A[0]``.

    :param loop: The loop region to add.
    :param increment: If True, the loop body increments the symbol ``i`` after writing.
    :return: The SDFG and the body state of the loop.
    """
    sdfg = dace.SDFG('loop')
    sdfg.add_symbol('i', dace.int64)
    sdfg.add_array('A', [10], dace.float64)
    init = sdfg.add_state('init', is_start_block=True)
    sdfg.add_node(loop)
    sdfg.add_edge(init, loop, dace.InterstateEdge(assignments={} if loop.loop_variable else {'i': '0'}))
    body = loop.add_state('body', is_start_block=True)
    _write_tasklet(body, 'i', 'A[0]')
    if increment:
        loop.add_state_after(body, 'increment', assignments={'i': 'i + 1'})
    return sdfg, body


def _map_sdfg() -> dace.SDFG:
    sdfg = dace.SDFG('map')
    sdfg.add_array('A', [10], dace.float64)
    state = sdfg.add_state()
    state.add_mapped_tasklet('write', dict(i='0:10'), {}, 'out = i', {'out': dace.Memlet('A[i]')}, external_edges=True)
    return sdfg


def _root() -> tn.ScheduleTreeRoot:
    sdfg = dace.SDFG('root')
    sdfg.add_state()
    return sdfg.as_schedule_tree()


def _tasklet() -> tn.ScheduleTreeRoot:
    sdfg = dace.SDFG('tasklet')
    sdfg.add_array('A', [10], dace.float64)
    _write_tasklet(sdfg.add_state(), '1', 'A[0]')
    return sdfg.as_schedule_tree()


def _assign() -> tn.ScheduleTreeRoot:
    sdfg = dace.SDFG('assign')
    sdfg.add_symbol('k', dace.int64)
    sdfg.add_array('A', [10], dace.float64)
    first = sdfg.add_state(is_start_block=True)
    second = sdfg.add_state_after(first, assignments={'k': '3'})
    _write_tasklet(second, 'k', 'A[k]')
    return sdfg.as_schedule_tree()


def _unstructured_sdfg() -> dace.SDFG:
    """
    Creates an SDFG with an if/else expressed through conditional inter-state edges, which is unstructured control
    flow in the schedule tree.
    """
    sdfg = dace.SDFG('unstructured')
    sdfg.add_symbol('N', dace.int64)
    sdfg.add_array('A', [10], dace.float64)
    guard = sdfg.add_state('guard', is_start_block=True)
    then_state = sdfg.add_state('then_state')
    else_state = sdfg.add_state('else_state')
    merge = sdfg.add_state('merge')
    _write_tasklet(then_state, '1', 'A[0]')
    _write_tasklet(else_state, '2', 'A[0]')
    _write_tasklet(merge, '3', 'A[1]')
    sdfg.add_edge(guard, then_state, dace.InterstateEdge('N > 0'))
    sdfg.add_edge(guard, else_state, dace.InterstateEdge('N <= 0'))
    sdfg.add_edge(then_state, merge, dace.InterstateEdge())
    sdfg.add_edge(else_state, merge, dace.InterstateEdge())
    return sdfg


def _state_if() -> tn.ScheduleTreeRoot:
    sdfg = dace.SDFG('state_if')
    sdfg.add_symbol('N', dace.int64)
    sdfg.add_array('A', [10], dace.float64)
    first = sdfg.add_state(is_start_block=True)
    second = sdfg.add_state()
    _write_tasklet(second, '1', 'A[0]')
    sdfg.add_edge(first, second, dace.InterstateEdge('N > 0'))
    return sdfg.as_schedule_tree()


def _goto() -> tn.ScheduleTreeRoot:
    sdfg = _conditional_sdfg(['N > 5'])
    branch = sdfg.start_block.branches[0][1]
    branch.remove_node(branch.start_block)
    branch.add_node(ReturnBlock('return'), is_start_block=True)
    _write_tasklet(sdfg.add_state_after(sdfg.start_block), '1', 'A[1]')
    return sdfg.as_schedule_tree()


def _for() -> tn.ScheduleTreeRoot:
    return _loop_sdfg(LoopRegion('loop', 'i < 10', 'i', 'i = 0', 'i = i + 1'))[0].as_schedule_tree()


def _while() -> tn.ScheduleTreeRoot:
    return _loop_sdfg(LoopRegion('loop', 'i < 10'), increment=True)[0].as_schedule_tree()


def _do_while() -> tn.ScheduleTreeRoot:
    return _loop_sdfg(LoopRegion('loop', 'i < 10', inverted=True), increment=True)[0].as_schedule_tree()


def _general_loop() -> tn.ScheduleTreeRoot:
    loop = LoopRegion('loop', 'i < 10', 'i', 'i = 0', 'i = i + 1', inverted=True, update_before_condition=False)
    return _loop_sdfg(loop)[0].as_schedule_tree()


def _loop_control(block_type: type[BreakBlock | ContinueBlock]) -> Callable[[], tn.ScheduleTreeRoot]:

    def factory() -> tn.ScheduleTreeRoot:
        sdfg, body = _loop_sdfg(LoopRegion('loop', 'i < 10', 'i', 'i = 0', 'i = i + 1'))
        loop = body.parent_graph
        control = block_type('control')
        loop.add_node(control)
        loop.add_edge(body, control, dace.InterstateEdge())
        return sdfg.as_schedule_tree()

    return factory


def _consume() -> tn.ScheduleTreeRoot:
    sdfg = dace.SDFG('consume')
    sdfg.add_stream('S', dace.int32, transient=True)
    sdfg.add_array('A', [10], dace.int32)
    state = sdfg.add_state()
    entry, exit_node = state.add_consume('consume', ('p', '4'))
    tasklet = state.add_tasklet('pop', {'inp'}, {'out'}, 'out = inp')
    state.add_edge(state.add_read('S'), None, entry, 'IN_stream', dace.Memlet('S'))
    state.add_edge(entry, 'OUT_stream', tasklet, 'inp', dace.Memlet('S'))
    state.add_memlet_path(tasklet, exit_node, state.add_write('A'), src_conn='out', memlet=dace.Memlet('A[0]'))
    return sdfg.as_schedule_tree()


def _library_call() -> tn.ScheduleTreeRoot:
    sdfg = dace.SDFG('library_call')
    for name, shape in (('A', [5, 4]), ('B', [4, 3]), ('C', [5, 3])):
        sdfg.add_array(name, shape, dace.float64)
    state = sdfg.add_state()
    libnode = MatMul('matmul')
    state.add_node(libnode)
    state.add_edge(state.add_read('A'), None, libnode, '_a', dace.Memlet('A'))
    state.add_edge(state.add_read('B'), None, libnode, '_b', dace.Memlet('B'))
    state.add_edge(libnode, '_c', state.add_write('C'), None, dace.Memlet('C'))
    return sdfg.as_schedule_tree()


def _copy() -> tn.ScheduleTreeRoot:
    sdfg = dace.SDFG('copy')
    sdfg.add_array('A', [10], dace.float64)
    sdfg.add_array('B', [10], dace.float64)
    state = sdfg.add_state()
    state.add_nedge(state.add_read('A'), state.add_write('B'), dace.Memlet('A[0:10]'))
    return sdfg.as_schedule_tree()


def _dynamic_scope_copy() -> tn.ScheduleTreeRoot:
    sdfg = dace.SDFG('dynamic_scope_copy')
    sdfg.add_array('bounds', [2], dace.int64)
    sdfg.add_array('A', [20], dace.float64)
    state = sdfg.add_state()
    entry, exit_node = state.add_map('map', dict(i='begin:end'))
    entry.add_in_connector('begin')
    entry.add_in_connector('end')
    bounds = state.add_read('bounds')
    state.add_edge(bounds, None, entry, 'begin', dace.Memlet('bounds[0]'))
    state.add_edge(bounds, None, entry, 'end', dace.Memlet('bounds[1]'))
    tasklet = state.add_tasklet('write', {}, {'out'}, 'out = i')
    state.add_nedge(entry, tasklet, dace.Memlet())
    state.add_memlet_path(tasklet, exit_node, state.add_write('A'), src_conn='out', memlet=dace.Memlet('A[i]'))
    return sdfg.as_schedule_tree()


def _view() -> tn.ScheduleTreeRoot:
    sdfg = dace.SDFG('view')
    sdfg.add_array('A', [30], dace.float64)
    sdfg.add_view('V', [20], dace.float64)
    state = sdfg.add_state()
    view = state.add_access('V')
    state.add_edge(view, 'views', state.add_write('A'), None, dace.Memlet('A[1:21]'))
    tasklet = state.add_tasklet('write', {}, {'out'}, 'out = 5')
    state.add_edge(tasklet, 'out', view, None, dace.Memlet('V[3]'))
    return sdfg.as_schedule_tree()


def _nview() -> tn.ScheduleTreeRoot:
    inner = dace.SDFG('inner')
    inner.add_array('X', [40], dace.float64)
    _write_tasklet(inner.add_state(), '1', 'X[0]')

    sdfg = dace.SDFG('nview')
    sdfg.add_array('A', [4, 5, 10], dace.float64)
    state = sdfg.add_state()
    nsdfg = state.add_nested_sdfg(inner, {}, {'X'})
    state.add_edge(nsdfg, 'X', state.add_write('A'), None, dace.Memlet('A[0:4, 1, 0:10]'))
    return sdfg.as_schedule_tree()


def _reference_set() -> tn.ScheduleTreeRoot:
    sdfg = dace.SDFG('reference_set')
    sdfg.add_array('A', [20], dace.float64)
    sdfg.add_array('B', [20], dace.float64)
    sdfg.add_reference('ref', [20], dace.float64)
    state = sdfg.add_state()
    ref = state.add_access('ref')
    state.add_edge(state.add_read('A'), None, ref, 'set', dace.Memlet('A[0:20]'))
    state.add_nedge(ref, state.add_write('B'), dace.Memlet('ref[0:20]'))
    return sdfg.as_schedule_tree()


def _named_region() -> tn.ScheduleTreeRoot:
    sdfg = dace.SDFG('named_region')
    sdfg.add_array('A', [10], dace.float64)
    region = NamedRegion('region')
    sdfg.add_node(region, is_start_block=True)
    _write_tasklet(region.add_state('write', is_start_block=True), '1', 'A[0]')
    return sdfg.as_schedule_tree()


def _state_boundary() -> tn.ScheduleTreeRoot:
    # State boundaries are inserted during the conversion to an SDFG, but can also be given explicitly
    return tn.ScheduleTreeRoot(
        name='state_boundary',
        containers={'A': data.Array(dace.float64, [10])},
        children=[
            tn.TaskletNode(nodes.Tasklet('first', {}, {'out'}, 'out = 1'), {}, {'out': dace.Memlet('A[0]')}),
            tn.StateBoundaryNode(),
            tn.TaskletNode(nodes.Tasklet('second', {}, {'out'}, 'out = 2'), {}, {'out': dace.Memlet('A[0]')}),
        ],
    )


FACTORIES: dict[type[tn.ScheduleTreeNode], Callable[[], tn.ScheduleTreeRoot]] = {
    tn.ScheduleTreeRoot: _root,
    tn.GBlock: lambda: _unstructured_sdfg().as_schedule_tree(),
    tn.StateLabel: lambda: _unstructured_sdfg().as_schedule_tree(),
    tn.GotoNode: _goto,
    tn.AssignNode: _assign,
    tn.LoopScope: _general_loop,
    tn.ForScope: _for,
    tn.WhileScope: _while,
    tn.DoWhileScope: _do_while,
    tn.IfScope: lambda: _conditional_sdfg(['N > 0']).as_schedule_tree(),
    tn.StateIfScope: _state_if,
    tn.BreakNode: _loop_control(BreakBlock),
    tn.ContinueNode: _loop_control(ContinueBlock),
    tn.ElifScope: lambda: _conditional_sdfg(['N > 0', 'N < -5']).as_schedule_tree(),
    tn.ElseScope: lambda: _conditional_sdfg(['N > 0', None]).as_schedule_tree(),
    tn.MapScope: lambda: _map_sdfg().as_schedule_tree(),
    tn.ConsumeScope: _consume,
    tn.TaskletNode: _tasklet,
    tn.LibraryCall: _library_call,
    tn.CopyNode: _copy,
    tn.DynScopeCopyNode: _dynamic_scope_copy,
    tn.ViewNode: _view,
    tn.NView: _nview,
    tn.NViewEnd: _nview,
    tn.RefSetNode: _reference_set,
    tn.StateBoundaryNode: _state_boundary,
    tn.NamedRegionScope: _named_region,
}


def _concrete_node_types() -> set[type[tn.ScheduleTreeNode]]:
    """
    Returns all concrete schedule tree node types defined in the ``treenodes`` module.
    """
    return {
        cls
        for cls in vars(tn).values()
        if isinstance(cls, type) and issubclass(cls, tn.ScheduleTreeNode) and cls not in ABSTRACT_NODE_TYPES
    }


def test_all_node_types_have_factory():
    assert set(FACTORIES.keys()) == _concrete_node_types()


@pytest.mark.parametrize('node_type', FACTORIES.keys(), ids=lambda node_type: node_type.__name__)
def test_node_type_to_sdfg(node_type: type[tn.ScheduleTreeNode]):
    stree = FACTORIES[node_type]()
    assert any(type(node) is node_type for node in stree.preorder_traversal())
    stree.as_sdfg(validate=True, simplify=False)


if __name__ == '__main__':
    test_all_node_types_have_factory()
    for node_type in FACTORIES:
        test_node_type_to_sdfg(node_type)
