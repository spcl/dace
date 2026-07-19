# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""
Lowering rules for canonical control flow: ``if``/``elif``/``else``,
``while``, range loops, ``dace.map`` loops, ``break``/``continue``/``pass``.

Loops and maps are emitted with *real* :class:`~dace.sdfg.state.LoopRegion`
and :class:`~dace.sdfg.nodes.MapEntry` objects (not stringly-typed frontend
metadata), so memlet propagation and downstream analysis behave identically
for frontend-produced and SDFG-derived schedule trees.
"""
import ast
from typing import List, Optional, Tuple

from dace import dtypes, subsets, symbolic
from dace.memlet import Memlet
from dace.properties import CodeBlock
from dace.sdfg import nodes
from dace.sdfg.state import LoopRegion
from dace.sdfg.analysis.schedule_tree import treenodes as tn
from dace.frontend.python import astutils
from dace.frontend.python.nextgen.canonical import cpa
from dace.frontend.python.nextgen.common import UnsupportedFeatureError
from dace.frontend.python.nextgen.lowering.access import DataAccess, resolve_access, resolve_symbol_names
from dace.frontend.python.nextgen.lowering.registry import LoweringState, rule
from dace.frontend.python.nextgen.semantics.context import BindingSnapshot
from dace.frontend.python.nextgen.semantics.joins import merge_branches


@rule(ast.If)
def lower_if(statement: ast.If, state: LoweringState) -> None:
    """
    Lower an if/elif/else chain with branch-scoped bindings: each branch is
    lowered from the pre-chain binding state, and the branch-end states are
    merged at the join (see :mod:`~...semantics.joins`). If the join cannot be
    merged soundly, the whole chain is rolled back and re-lowered as a single
    Python callback.
    """
    from dace.frontend.python.nextgen.lowering import dispatch
    mark = state.emitter.checkpoint()
    before = state.context.snapshot()
    try:
        branch_scopes, branch_ends = _lower_if_chain(statement, before, state)
        state.context.restore(before)
        merge_branches(before, branch_ends, branch_scopes, statement, state)
    except UnsupportedFeatureError as reason:
        state.emitter.rollback(mark)
        state.context.restore(before)
        dispatch.fallback_to_callback(statement, state, reason)


def _lower_if_chain(statement: ast.If, before: BindingSnapshot,
                    state: LoweringState) -> Tuple[List[Optional[tn.ScheduleTreeScope]], List[BindingSnapshot]]:
    """
    Emit the scopes of an if/elif/else chain, lowering every branch from the
    ``before`` binding state, and collect (scope, end-state) per path. A chain
    without ``else`` contributes an implicit fall-through path (scope None,
    end state ``before``).
    """
    branch_scopes: List[Optional[tn.ScheduleTreeScope]] = []
    branch_ends: List[BindingSnapshot] = []

    def _lower_branch(scope: tn.ScheduleTreeScope, body: List[ast.stmt]) -> None:
        with state.emitter.scope(scope):
            state.lower_body(body)
        branch_scopes.append(scope)
        branch_ends.append(state.context.snapshot())
        state.context.restore(before)

    condition = CodeBlock(astutils.unparse(resolve_symbol_names(statement.test, state)))
    _lower_branch(tn.IfScope(condition=condition, children=[]), statement.body)

    orelse = statement.orelse
    while len(orelse) == 1 and isinstance(orelse[0], ast.If):
        elif_statement = orelse[0]
        condition = CodeBlock(astutils.unparse(resolve_symbol_names(elif_statement.test, state)))
        _lower_branch(tn.ElifScope(condition=condition, children=[]), elif_statement.body)
        orelse = elif_statement.orelse

    if orelse:
        _lower_branch(tn.ElseScope(children=[]), orelse)
    else:
        branch_scopes.append(None)
        branch_ends.append(before)
    return branch_scopes, branch_ends


@rule(ast.While)
def lower_while(statement: ast.While, state: LoweringState) -> None:

    def _emit(state: LoweringState) -> None:
        condition = astutils.unparse(resolve_symbol_names(statement.test, state))
        loop = LoopRegion(f'while_{statement.lineno}', condition_expr=condition)
        with state.emitter.scope(tn.WhileScope(loop=loop, children=[])):
            state.lower_body(statement.body)

    _lower_loop_with_stability_check(statement, _emit, state)


@rule(ast.For)
def lower_for(statement: ast.For, state: LoweringState) -> None:
    if cpa.is_range_iterator(statement.iter):
        _lower_loop_with_stability_check(statement, lambda s: _lower_range_loop(statement, s), state)
    elif cpa.is_dace_map_iterator(statement.iter):
        _lower_loop_with_stability_check(statement, lambda s: _lower_map_loop(statement, s), state)
    else:
        raise UnsupportedFeatureError(
            f'Non-canonical for-iterator reached lowering: '
            f'{astutils.unparse(statement.iter)}', state.context.filename, statement)


def _lower_loop_with_stability_check(statement: ast.stmt, emit_loop, state: LoweringState) -> None:
    """
    Lower a loop and enforce the loop-entry stability rule: any name bound
    before the loop whose binding the body changed (a different container,
    kind, or static value) would need a φ at the loop head, which the binding
    design intentionally avoids — the loop rolls back and re-lowers as a
    single Python callback instead. In-place rebinding through the same
    container (the common case for scalars) passes.
    """
    from dace.frontend.python.nextgen.lowering import dispatch
    mark = state.emitter.checkpoint()
    before = state.context.snapshot()
    try:
        emit_loop(state)
    except UnsupportedFeatureError as reason:
        state.emitter.rollback(mark)
        state.context.restore(before)
        dispatch.fallback_to_callback(statement, state, reason)
        return
    reason = _loop_instability(before, state)
    if reason is not None:
        state.emitter.rollback(mark)
        state.context.restore(before)
        dispatch.fallback_to_callback(statement, state, reason, category='loop-stability')


def _loop_instability(before: BindingSnapshot, state: LoweringState) -> Optional[str]:
    """The reason a loop body is binding-unstable, or None if it is stable.
    Names first bound inside the body are loop-local and always stable."""
    for name, binding in before.bindings.items():
        current = state.context.bindings.get(name)
        if current is None:
            return f'loop body unbinds "{name}"'
        if (current.kind, current.container) != (binding.kind, binding.container):
            return f'loop-carried rebinding of "{name}" requires a merge at the loop head'
        if binding.kind == 'static' and (state.context.static_values.get(name) is not before.static_values.get(name)):
            return f'loop-carried compile-time value change of "{name}"'
    return None


def _lower_range_loop(statement: ast.For, state: LoweringState) -> None:
    loop_variable = statement.target.id
    start, stop, step = (astutils.unparse(resolve_symbol_names(argument, state)) for argument in statement.iter.args)
    comparator = '<'
    try:
        if (symbolic.pystr_to_symbolic(step) < 0) == True:
            comparator = '>'
    except TypeError:
        pass

    state.context.bind_symbol(loop_variable, _index_dtype(statement.iter.args, state))
    loop = LoopRegion(f'for_{statement.lineno}',
                      condition_expr=f'{loop_variable} {comparator} {stop}',
                      loop_var=loop_variable,
                      initialize_expr=f'{loop_variable} = {start}',
                      update_expr=f'{loop_variable} = {loop_variable} + {step}')
    with state.emitter.scope(tn.ForScope(loop=loop, children=[])):
        state.lower_body(statement.body)


def _lower_map_loop(statement: ast.For, state: LoweringState) -> None:
    targets = statement.target.elts if isinstance(statement.target, ast.Tuple) else [statement.target]
    params = [target.id for target in targets]
    dynamic_inputs: List[tn.DynScopeCopyNode] = []
    ranges = _parse_map_ranges(statement.iter, state, dynamic_inputs)
    if len(params) != len(ranges):
        raise UnsupportedFeatureError('Number of dace.map indices does not match number of ranges',
                                      state.context.filename,
                                      statement,
                                      category='explicit-map')
    for param in params:
        state.context.bind_symbol(param)
    # Dynamic-range inputs (data-dependent bounds) are emitted as siblings
    # immediately preceding the map scope, matching how SDFG-derived schedule
    # trees place them (see sdfg_to_tree.py) rather than as children inside
    # the scope: the scope emitter appends the scope node itself to the
    # *current* (enclosing) scope before entering it, so emitting these first
    # places them right before the map.
    for dynamic_input in dynamic_inputs:
        state.emitter.emit(dynamic_input)
    map_node = nodes.MapEntry(nodes.Map(f'map_{statement.lineno}', params, subsets.Range(ranges)))
    with state.emitter.scope(tn.MapScope(node=map_node, children=[])):
        state.lower_body(statement.body)


def _parse_map_ranges(iterator: ast.Subscript, state: LoweringState,
                      dynamic_inputs: List[tn.DynScopeCopyNode]) -> List[Tuple]:
    """Parse ``dace.map[start:stop:step, ...]`` into inclusive-end symbolic ranges.

    :param dynamic_inputs: Collects a :class:`~dace.sdfg.analysis.schedule_tree.treenodes.DynScopeCopyNode`
                           for every data-dependent bound encountered (see :func:`_dynamic_bound`).
    """
    dimensions = iterator.slice.elts if isinstance(iterator.slice, ast.Tuple) else [iterator.slice]
    ranges = []
    for dimension in dimensions:
        if not isinstance(dimension, ast.Slice):
            raise UnsupportedFeatureError('dace.map dimensions must be slices',
                                          state.context.filename,
                                          iterator,
                                          category='explicit-map')
        start = _bound(dimension.lower, 0, state, dynamic_inputs)
        stop = _bound(dimension.upper, None, state, dynamic_inputs)
        step = _bound(dimension.step, 1, state, dynamic_inputs)
        if stop is None:
            raise UnsupportedFeatureError('dace.map dimensions require an upper bound',
                                          state.context.filename,
                                          iterator,
                                          category='explicit-map')
        ranges.append((start, stop - 1, step))
    return ranges


def _bound(node, default, state: LoweringState, dynamic_inputs: List[tn.DynScopeCopyNode]):
    """
    Resolve a single ``dace.map`` range bound to a symbolic expression.

    A bound that reads a data container — a scalar name, a scalar structure
    member, or a scalar array element like ``A_row[i]`` (index expressions
    are canonical in place, not hoisted) — becomes a fresh dynamic-map-range
    symbol fed by a :class:`DynScopeCopyNode`: see :func:`_dynamic_bound`.
    Purely symbolic expressions (``i + 1``) resolve symbolically. Compound
    expressions that mix data reads with arithmetic (``A_row[i] + 1``) are
    not symbolizable and fall back to a callback.
    """
    if node is None:
        return default
    if isinstance(node, (ast.Name, ast.Attribute, ast.Subscript)):
        # May raise UnsupportedFeatureError (e.g. a data-dependent index
        # inside the bound itself), falling the whole loop back to a callback.
        access = resolve_access(node, state)
        if access is not None:
            return _dynamic_bound(node, access, state, dynamic_inputs)
    expression = astutils.unparse(resolve_symbol_names(node, state))
    try:
        return symbolic.pystr_to_symbolic(expression)
    except Exception:
        # Not symbolizable (e.g. references a value only the interpreter
        # knows); the loop falls back to a callback.
        raise UnsupportedFeatureError(f'Cannot parse loop bound "{expression}" symbolically',
                                      state.context.filename,
                                      node,
                                      category='dynamic-bound')


def _dynamic_bound(node: ast.expr, access: DataAccess, state: LoweringState,
                   dynamic_inputs: List[tn.DynScopeCopyNode]) -> symbolic.symbol:
    """
    Turn a data-dependent ``dace.map`` bound (a scalar integer data access —
    a scalar container or a single array element like ``A_row[i]``) into a
    fresh symbol fed by a dynamic map-range input, recording a
    :class:`~dace.sdfg.analysis.schedule_tree.treenodes.DynScopeCopyNode`
    for the caller to emit right before the map scope.

    :raises UnsupportedFeatureError: If the access is not a single integer
        element (e.g. a whole array, a sub-range, or a floating-point
        scalar) -- those forms are not supported as dynamic map-range inputs.
    """
    if not access.is_scalar_access or access.descriptor.dtype not in dtypes.INTEGER_TYPES:
        raise UnsupportedFeatureError(
            f'Data-dependent dace.map bound "{astutils.unparse(node)}" must be a scalar integer '
            f'element (got subset {access.subset} of {access.descriptor})',
            state.context.filename,
            node,
            category='dynamic-bound')
    symbol_name = state.context.fresh_name('__dyn')
    # A repository-only symbol: registered directly in the symbol table (which
    # *is* the tree root's symbol table), without a source-level name binding.
    state.context.symbols[symbol_name] = symbolic.symbol(symbol_name, access.descriptor.dtype)
    memlet = Memlet(data=access.container, subset=access.subset)
    dynamic_inputs.append(tn.DynScopeCopyNode(target=symbol_name, memlet=memlet))
    return symbolic.symbol(symbol_name, access.descriptor.dtype)


def _index_dtype(bounds: List[ast.expr], state: LoweringState) -> dtypes.typeclass:
    return dtypes.int64


@rule(ast.Break)
def lower_break(statement: ast.Break, state: LoweringState) -> None:
    state.emitter.emit(tn.BreakNode())


@rule(ast.Continue)
def lower_continue(statement: ast.Continue, state: LoweringState) -> None:
    state.emitter.emit(tn.ContinueNode())


@rule(ast.Pass)
def lower_pass(statement: ast.Pass, state: LoweringState) -> None:
    pass
