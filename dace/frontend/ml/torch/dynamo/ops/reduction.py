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
    'any': ('lambda a, b: a or b', 0),
    'all': ('lambda a, b: a and b', 1),
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
    """
    Emits a ``Reduce`` library call into a container shaped like ``node.meta['val']``. With ``keepdim`` the node
    writes through a view of the output that drops the kept unit dimensions.
    """
    from .pointwise import cast_tensor
    val = node.meta['val']
    out_dtype = val.dtype
    if tensor.torch_dtype != out_dtype:
        tensor = cast_tensor(ctx, node.name + '_cast', tensor, out_dtype)
    out = ctx.add_tensor_like('t_' + node.name, val)
    if tensor.rank == 0:
        ctx.emit_copy(tensor, out)
        return out
    if keepdim:
        red_shape = tuple(s for k, s in enumerate(out.tshape) if k not in dims)
        red_strides = tuple(s for k, s in enumerate(out.tstrides) if k not in dims)
        target = ctx.emit_view_raw('v_' + node.name + '_red', out, red_shape, red_strides, out.torch_dtype)
    else:
        target = out
    _emit_reduce_into(ctx, node.name, tensor, kind, dims, target)
    return out


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


@register_lowering(*resolve('aten.any.default', 'aten.any.dim', 'aten.any.dims'))
def lower_any(ctx: LoweringContext, node, tensor: TensorValue, dims=None, keepdim=False):
    return emit_reduce(ctx, node, tensor, 'any', _norm_dims(dims, tensor.rank), bool(as_sym(keepdim)))


@register_lowering(*resolve('aten.all.default', 'aten.all.dim', 'aten.all.dims'))
def lower_all(ctx: LoweringContext, node, tensor: TensorValue, dims=None, keepdim=False):
    return emit_reduce(ctx, node, tensor, 'all', _norm_dims(dims, tensor.rank), bool(as_sym(keepdim)))


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


# ---------------------------------------------------------------------- prims-level reductions
def _prims_reduce(kind):

    def lower(ctx: LoweringContext, node, tensor: TensorValue, dims=None, *args, **kwargs):
        return emit_reduce(ctx, node, tensor, kind, _norm_dims(dims, tensor.rank), False)

    return lower


for _name, _kind in (('sum', 'sum'), ('prod', 'prod'), ('amax', 'amax'), ('amin', 'amin')):
    for _op in resolve(f'prims.{_name}'):
        register_lowering(_op)(_prims_reduce(_kind))


@register_lowering(*resolve('prims.var'))
def lower_prims_var(ctx: LoweringContext, node, tensor: TensorValue, dims=None, correction=1, output_dtype=None,
                    **kwargs):
    """``var(x, dims) = sum((x - mean)^2) / (N - correction)`` composed from two reductions and two pointwise maps."""
    from .pointwise import pointwise_into
    dims = _norm_dims(dims, tensor.rank)
    count = 1
    for d in dims:
        count = count * tensor.tshape[d]
    correction = as_sym(correction) if not isinstance(correction, float) else correction
    val = node.meta['val']

    # mean with kept dims (as a view over the reduced container) so that it broadcasts against the input
    red_shape = tuple(s for k, s in enumerate(tensor.tshape) if k not in dims)
    keep_shape = tuple(1 if k in dims else s for k, s in enumerate(tensor.tshape))
    mean = ctx.add_array('t_' + node.name + '_mean', red_shape, tensor.torch_dtype, device=tensor.device)
    summed = ctx.add_array('t_' + node.name + '_sum', red_shape, tensor.torch_dtype, device=tensor.device)
    _emit_reduce_into(ctx, node.name + '_sum', tensor, 'sum', dims, summed)
    pointwise_into(ctx, node.name + '_mean', mean, [summed], f'{{0}} / ({count})')
    from ..context import contiguous_strides
    mean_k = ctx.emit_view_raw('v_' + node.name + '_mean', mean, keep_shape, contiguous_strides(keep_shape),
                               tensor.torch_dtype) if tensor.rank > 0 else mean

    sq = ctx.add_array('t_' + node.name + '_sq', tensor.tshape, tensor.torch_dtype, device=tensor.device)
    pointwise_into(ctx, node.name + '_sq', sq, [tensor, mean_k], '(({0}) - ({1})) * (({0}) - ({1}))')
    out = ctx.add_tensor_like('t_' + node.name, val)
    sqsum = ctx.add_array('t_' + node.name + '_sqsum', red_shape, tensor.torch_dtype, device=tensor.device)
    _emit_reduce_into(ctx, node.name + '_sqsum', sq, 'sum', dims, sqsum)
    pointwise_into(ctx, node.name, out, [sqsum], f'{{0}} / ({count} - ({correction}))')
    return out


def _emit_reduce_into(ctx: LoweringContext, name: str, tensor: TensorValue, kind: str, dims: List[int],
                      out: TensorValue) -> None:
    from dace.libraries.standard import Reduce
    wcr, identity = _REDUCTIONS[kind]
    if tensor.rank == 0:
        ctx.emit_copy(tensor, out)
        return
    axes = None if len(dims) == tensor.rank else tuple(dims)
    rnode = Reduce('reduce_' + name, wcr, axes, identity)
    rnode.add_in_connector('_in')
    rnode.add_out_connector('_out')
    ctx.emit_library_call(rnode, {'_in': tensor.memlet()}, {'_out': out.memlet()})
