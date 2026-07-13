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
from typing import List, Tuple

from dace import dtypes, subsets, symbolic
from dace.properties import CodeBlock
from dace.sdfg import nodes
from dace.sdfg.state import LoopRegion
from dace.sdfg.analysis.schedule_tree import treenodes as tn
from dace.frontend.python import astutils
from dace.frontend.python.nextgen.canonical import cpa
from dace.frontend.python.nextgen.common import UnsupportedFeatureError
from dace.frontend.python.nextgen.lowering.access import resolve_symbol_names
from dace.frontend.python.nextgen.lowering.registry import LoweringState, rule


@rule(ast.If)
def lower_if(statement: ast.If, state: LoweringState) -> None:
    condition = CodeBlock(astutils.unparse(resolve_symbol_names(statement.test, state)))
    with state.emitter.scope(tn.IfScope(condition=condition, children=[])):
        state.lower_body(statement.body)
    _lower_orelse(statement.orelse, state)


def _lower_orelse(orelse: List[ast.stmt], state: LoweringState) -> None:
    if not orelse:
        return
    if len(orelse) == 1 and isinstance(orelse[0], ast.If):
        elif_statement = orelse[0]
        condition = CodeBlock(astutils.unparse(resolve_symbol_names(elif_statement.test, state)))
        with state.emitter.scope(tn.ElifScope(condition=condition, children=[])):
            state.lower_body(elif_statement.body)
        _lower_orelse(elif_statement.orelse, state)
        return
    with state.emitter.scope(tn.ElseScope(children=[])):
        state.lower_body(orelse)


@rule(ast.While)
def lower_while(statement: ast.While, state: LoweringState) -> None:
    condition = astutils.unparse(resolve_symbol_names(statement.test, state))
    loop = LoopRegion(f'while_{statement.lineno}', condition_expr=condition)
    with state.emitter.scope(tn.WhileScope(loop=loop, children=[])):
        state.lower_body(statement.body)


@rule(ast.For)
def lower_for(statement: ast.For, state: LoweringState) -> None:
    if cpa.is_range_iterator(statement.iter):
        _lower_range_loop(statement, state)
    elif cpa.is_dace_map_iterator(statement.iter):
        _lower_map_loop(statement, state)
    else:
        raise UnsupportedFeatureError(
            f'Non-canonical for-iterator reached lowering: '
            f'{astutils.unparse(statement.iter)}', state.context.filename, statement)


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
    ranges = _parse_map_ranges(statement.iter, state)
    if len(params) != len(ranges):
        raise UnsupportedFeatureError('Number of dace.map indices does not match number of ranges',
                                      state.context.filename, statement)
    for param in params:
        state.context.bind_symbol(param)
    map_node = nodes.MapEntry(nodes.Map(f'map_{statement.lineno}', params, subsets.Range(ranges)))
    with state.emitter.scope(tn.MapScope(node=map_node, children=[])):
        state.lower_body(statement.body)


def _parse_map_ranges(iterator: ast.Subscript, state: LoweringState) -> List[Tuple]:
    """Parse ``dace.map[start:stop:step, ...]`` into inclusive-end symbolic ranges."""
    dimensions = iterator.slice.elts if isinstance(iterator.slice, ast.Tuple) else [iterator.slice]
    ranges = []
    for dimension in dimensions:
        if not isinstance(dimension, ast.Slice):
            raise UnsupportedFeatureError('dace.map dimensions must be slices', state.context.filename, iterator)
        start = _bound(dimension.lower, 0, state)
        stop = _bound(dimension.upper, None, state)
        step = _bound(dimension.step, 1, state)
        if stop is None:
            raise UnsupportedFeatureError('dace.map dimensions require an upper bound', state.context.filename,
                                          iterator)
        ranges.append((start, stop - 1, step))
    return ranges


def _bound(node, default, state: LoweringState):
    if node is None:
        return default
    return symbolic.pystr_to_symbolic(astutils.unparse(resolve_symbol_names(node, state)))


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
