# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Reduction lowerings onto the standard-library ``Reduce`` node."""
from typing import List, Optional, Sequence

import torch

from dace import dtypes
from dace.memlet import Memlet

from ..context import ConstValue, LoweringContext, TensorValue, TupleValue, as_sym
from ..dtypes import to_dace_dtype
from . import register_lowering, resolve

aten = torch.ops.aten

_REDUCTIONS = {
    'sum': ('lambda a, b: a + b', 0),
    'prod': ('lambda a, b: a * b', 1),
    'amax': ('lambda a, b: max(a, b)', None),
    'amin': ('lambda a, b: min(a, b)', None),
}


def _norm_dims(dims, rank: int) -> List[int]:
    if dims is None:
        return list(range(rank))
    if isinstance(dims, (ConstValue, TupleValue)) or not isinstance(dims, (list, tuple)):
        dims = [dims] if not isinstance(dims, TupleValue) else dims.items
    dims = [int(as_sym(d)) for d in dims]
    if len(dims) == 0:
        return list(range(rank))
    return sorted({d + rank if d < 0 else d for d in dims})


def emit_reduce(ctx: LoweringContext, node, tensor: TensorValue, kind: str, dims: List[int],
                keepdim: bool) -> TensorValue:
    """Emits a ``Reduce`` library call; returns a tensor shaped like ``node.meta['val']`` (keepdim via a view)."""
    from dace.libraries.standard import Reduce
    from .pointwise import cast_tensor
    wcr, identity = _REDUCTIONS[kind]
    val = node.meta['val']
    out_dtype = val.dtype
    if tensor.torch_dtype != out_dtype:
        tensor = cast_tensor(ctx, node.name + '_cast', tensor, out_dtype)
    reduced_shape = tuple(s for k, s in enumerate(tensor.tshape) if k not in dims)
    red = ctx.add_array('t_' + node.name + ('_red' if keepdim else ''), reduced_shape, out_dtype, device=val.device)
    axes = None if len(dims) == tensor.rank else tuple(dims)
    if tensor.rank == 0:
        ctx.emit_copy(tensor, red)
    else:
        rnode = Reduce('reduce_' + node.name, wcr, axes, identity)
        rnode.add_in_connector('_in')
        rnode.add_out_connector('_out')
        ctx.emit_library_call(rnode, {'_in': tensor.memlet()}, {'_out': red.memlet()})
    if keepdim and tensor.rank > 0:
        return ctx.emit_view('v_' + node.name, red, val)
    return red


@register_lowering(*resolve('aten.sum.dim_IntList', 'aten.sum.default', 'aten.sum.IntList_out'))
def lower_sum(ctx: LoweringContext, node, tensor: TensorValue, dims=None, keepdim=False, *, dtype=None):
    return emit_reduce(ctx, node, tensor, 'sum', _norm_dims(dims, tensor.rank), bool(as_sym(keepdim)))


@register_lowering(aten.prod.dim_int, aten.prod.default)
def lower_prod(ctx: LoweringContext, node, tensor: TensorValue, dim=None, keepdim=False, *, dtype=None):
    return emit_reduce(ctx, node, tensor, 'prod', _norm_dims(dim, tensor.rank), bool(as_sym(keepdim)))


@register_lowering(aten.amax.default)
def lower_amax(ctx: LoweringContext, node, tensor: TensorValue, dims=None, keepdim=False):
    return emit_reduce(ctx, node, tensor, 'amax', _norm_dims(dims, tensor.rank), bool(as_sym(keepdim)))


@register_lowering(aten.amin.default)
def lower_amin(ctx: LoweringContext, node, tensor: TensorValue, dims=None, keepdim=False):
    return emit_reduce(ctx, node, tensor, 'amin', _norm_dims(dims, tensor.rank), bool(as_sym(keepdim)))


@register_lowering(aten.max.default, aten.min.default)
def lower_max_all(ctx: LoweringContext, node, tensor: TensorValue):
    kind = 'amax' if node.target == aten.max.default else 'amin'
    return emit_reduce(ctx, node, tensor, kind, list(range(tensor.rank)), False)


@register_lowering(aten.mean.dim, aten.mean.default)
def lower_mean(ctx: LoweringContext, node, tensor: TensorValue, dims=None, keepdim=False, *, dtype=None):
    from .pointwise import lower_pointwise_values
    dims = _norm_dims(dims, tensor.rank)
    keepdim = bool(as_sym(keepdim))
    count = 1
    for d in dims:
        count = count * tensor.tshape[d]
    # Sum into a temporary shaped like the output, then divide elementwise
    summed = emit_reduce(ctx, node, tensor, 'sum', dims, keepdim)
    return lower_pointwise_values(ctx, node, [summed], f'{{0}} / ({count})', prefix='t_' + node.name)
