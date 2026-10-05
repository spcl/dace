# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""
Lowerings of higher-order control-flow operators onto native schedule-tree control flow.

- ``cond``       -> ``IfScope`` / ``ElseScope`` writing into shared join containers
- ``while_loop`` -> ``WhileScope`` over carried containers; the predicate graph is lowered before the loop and again
                    at the end of every iteration into a flag container that the loop condition reads
- ``scan``       -> ``ForScope`` over the leading dimension with carried containers and stacked outputs
- ``map_impl``   -> ``ForScope`` over the leading dimension with stacked outputs (independent iterations)

Subgraph placeholders bind positionally to the HOP operands (verified orders for torch 2.13: ``cond`` branches take
``(operands...)``; ``while_loop`` cond/body take ``(carried..., additional...)``; ``scan`` combine takes
``(carries..., x_slices..., additional...)``; ``map_impl`` body takes ``(x_slices..., args...)``).
"""
from typing import List, Sequence

import torch
from torch.fx import GraphModule

from dace import subsets, symbolic
from dace.properties import CodeBlock
from dace.sdfg.analysis.schedule_tree import treenodes as tn
from dace.sdfg.state import LoopRegion

from ..context import ConstValue, LoweringContext, SymValue, TensorValue, TupleValue, UnsupportedOpError
from . import register_lowering, resolve


def _items(v) -> list:
    if isinstance(v, TupleValue):
        return list(v.items)
    if isinstance(v, (list, tuple)):
        return list(v)
    return [v]


def _graph(v) -> GraphModule:
    v = v.value if isinstance(v, ConstValue) else v
    if not isinstance(v, GraphModule):
        raise UnsupportedOpError('higher-order op', f'expected a subgraph, got {type(v).__name__}')
    return v


def _tensors(ctx: LoweringContext, values: Sequence, what: str) -> List[TensorValue]:
    result = []
    for v in values:
        if not isinstance(v, TensorValue):
            raise UnsupportedOpError(what, f'only tensor operands are supported, got {type(v).__name__}')
        result.append(v)
    return result


def predicate_expr(pred) -> str:
    if isinstance(pred, TensorValue):
        return f'{pred.name}[0] != 0'
    if isinstance(pred, SymValue):
        return f'({pred.expr}) != 0' if not isinstance(pred.expr, int) else ('1' if pred.expr else '0')
    if isinstance(pred, ConstValue):
        return '1' if pred.value else '0'
    raise UnsupportedOpError('predicate', f'unsupported predicate type {type(pred).__name__}')


def _lower_into(ctx: LoweringContext, gm: GraphModule, bindings: Sequence, children: list) -> List:
    with ctx.scope(children):
        return ctx.importer.lower_subgraph(ctx, gm, bindings)


def _store(ctx: LoweringContext, src, dst: TensorValue, dst_subset=None) -> None:
    if not isinstance(src, TensorValue):
        raise UnsupportedOpError('higher-order op', f'subgraph returned a non-tensor value ({type(src).__name__})')
    if src is dst and dst_subset is None:
        return
    ctx.emit_copy(src, dst, dst_subset=dst_subset)


# ---------------------------------------------------------------------- cond
@register_lowering(*resolve('higher_order.cond'))
def lower_cond(ctx: LoweringContext, node, pred, true_graph, false_graph, operands=()):
    vals = node.meta['val']
    vals = list(vals) if isinstance(vals, (list, tuple)) else [vals]
    operands = _items(operands)
    joins = [ctx.add_tensor_like(f't_{node.name}_{k}', v) for k, v in enumerate(vals)]

    then_children: list = []
    outs = _lower_into(ctx, _graph(true_graph), operands, then_children)
    with ctx.scope(then_children):
        for out, join in zip(_items(outs), joins):
            _store(ctx, out, join)
    ctx.emit(tn.IfScope(condition=CodeBlock(predicate_expr(pred)), children=then_children))

    else_children: list = []
    outs = _lower_into(ctx, _graph(false_graph), operands, else_children)
    with ctx.scope(else_children):
        for out, join in zip(_items(outs), joins):
            _store(ctx, out, join)
    ctx.emit(tn.ElseScope(children=else_children))
    return TupleValue(joins) if len(joins) != 1 or isinstance(node.meta['val'], (list, tuple)) else joins[0]


# ---------------------------------------------------------------------- while_loop
@register_lowering(*resolve('higher_order.while_loop'))
def lower_while_loop(ctx: LoweringContext, node, cond_graph, body_graph, carried=(), additional=()):
    carried = _tensors(ctx, _items(carried), 'while_loop carried inputs')
    additional = _items(additional)
    vals = list(node.meta['val'])
    cond_gm, body_gm = _graph(cond_graph), _graph(body_graph)

    # Carried state lives in dedicated containers for the duration of the loop
    state = [ctx.add_tensor_like(f't_{node.name}_c{k}', v) for k, v in enumerate(vals)]
    for src, dst in zip(carried, state):
        ctx.emit_copy(src, dst)

    # Flag container holding the predicate; evaluated before the loop and at the end of every iteration
    flag = ctx.add_array(f't_{node.name}_flag', (), torch.bool, device=carried[0].device if carried else None)

    def evaluate_predicate():
        (pred, ) = _items(ctx.importer.lower_subgraph(ctx, cond_gm, list(state) + additional))
        if isinstance(pred, TensorValue):
            _store(ctx, pred, flag)
        else:
            expr = predicate_expr(pred)
            ctx.emit_tasklet(f'{node.name}_pred', {}, f'__out = {expr}', {'__out': flag.memlet()})

    evaluate_predicate()

    body_children: list = []
    outs = _lower_into(ctx, body_gm, list(state) + additional, body_children)
    with ctx.scope(body_children):
        for out, dst in zip(_items(outs), state):
            _store(ctx, out, dst)
        evaluate_predicate()
    loop = LoopRegion(f'while_{node.name}', condition_expr=f'{flag.name}[0] != 0')
    ctx.emit(tn.WhileScope(loop=loop, children=body_children))
    return TupleValue(state)


# ---------------------------------------------------------------------- scan / map
def _loop_var(ctx: LoweringContext, node) -> symbolic.symbol:
    name = ctx.new_name(f'__{node.name}_i')
    return symbolic.symbol(name, ctx.symtab.dtype)


def _slice_subset(base: TensorValue, i) -> subsets.Range:
    return subsets.Range([(i, i, 1)] + [(0, s - 1, 1) for s in base.tshape[1:]])


def _slice_view(ctx: LoweringContext, prefix: str, base: TensorValue, i) -> TensorValue:
    return ctx.emit_view_raw(prefix, base, base.tshape[1:], base.tstrides[1:], base.torch_dtype, _slice_subset(base, i))


def _for_scope(node, i: symbolic.symbol, length, children: list) -> tn.ForScope:
    loop = LoopRegion(f'for_{node.name}',
                      loop_var=i.name,
                      initialize_expr=f'{i.name} = 0',
                      condition_expr=f'{i.name} < {length}',
                      update_expr=f'{i.name} = {i.name} + 1')
    return tn.ForScope(loop=loop, children=children)


@register_lowering(*resolve('higher_order.scan'))
def lower_scan(ctx: LoweringContext, node, combine_graph, init=(), xs=(), additional=(), *args, **kwargs):
    init = _tensors(ctx, _items(init), 'scan init')
    xs = _tensors(ctx, _items(xs), 'scan xs')
    additional = _items(additional)
    vals = list(node.meta['val'])
    num_carry = len(init)
    if not xs:
        raise UnsupportedOpError(node.target, 'scan without scanned inputs')

    state = [ctx.add_tensor_like(f't_{node.name}_c{k}', v) for k, v in enumerate(vals[:num_carry])]
    for src, dst in zip(init, state):
        ctx.emit_copy(src, dst)
    stacked = [ctx.add_tensor_like(f't_{node.name}_y{j}', v) for j, v in enumerate(vals[num_carry:])]

    i = _loop_var(ctx, node)
    length = xs[0].tshape[0]
    body: list = []
    with ctx.scope(body):
        slices = [_slice_view(ctx, f'v_{node.name}_x{k}', x, i) for k, x in enumerate(xs)]
        outs = _items(ctx.importer.lower_subgraph(ctx, _graph(combine_graph), list(state) + slices + additional))
        for out, dst in zip(outs[:num_carry], state):
            _store(ctx, out, dst)
        for out, dst in zip(outs[num_carry:], stacked):
            _store(ctx, out, dst, _slice_subset(dst, i))
    ctx.emit(_for_scope(node, i, length, body))
    return TupleValue(state + stacked)


@register_lowering(*resolve('higher_order.map_impl', 'higher_order.map'))
def lower_map(ctx: LoweringContext, node, body_graph, xs=(), args=(), *rest, **kwargs):
    xs = _tensors(ctx, _items(xs), 'map xs')
    args = _items(args)
    vals = list(node.meta['val'])
    if not xs:
        raise UnsupportedOpError(node.target, 'map without mapped inputs')
    stacked = [ctx.add_tensor_like(f't_{node.name}_y{j}', v) for j, v in enumerate(vals)]

    i = _loop_var(ctx, node)
    length = xs[0].tshape[0]
    body: list = []
    with ctx.scope(body):
        slices = [_slice_view(ctx, f'v_{node.name}_x{k}', x, i) for k, x in enumerate(xs)]
        outs = _items(ctx.importer.lower_subgraph(ctx, _graph(body_graph), slices + args))
        for out, dst in zip(outs, stacked):
            _store(ctx, out, dst, _slice_subset(dst, i))
    ctx.emit(_for_scope(node, i, length, body))
    return TupleValue(stacked)
