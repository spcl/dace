# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""
View and data-movement lowerings: ``view``, ``permute``, ``slice``, ``select``, ``expand``, ``cat``, ``clone``, ...

All reinterpretations use the output FakeTensor's shape and strides (``node.meta['val']``) directly: ATen only emits
these operators when the result is expressible as a strided view of the input storage, so the metadata is exact.
"""
from typing import List, Optional, Sequence

import sympy
import torch

from dace import subsets

from ..context import ConstValue, LoweringContext, SymValue, TensorValue, TupleValue, UnsupportedOpError, as_sym
from . import register_lowering, resolve

aten = torch.ops.aten


def _val(node) -> torch.Tensor:
    return node.meta['val']


def _norm_dim(dim, rank: int) -> int:
    dim = int(as_sym(dim))
    return dim + rank if dim < 0 else dim


def _materialize_if_view(ctx: LoweringContext, t: TensorValue, prefix: str) -> TensorValue:
    """Copies a view into a fresh contiguous array (used when a plain array is required)."""
    if not t.is_view:
        return t
    out = ctx.add_array(prefix, t.tshape, t.torch_dtype, device=t.device)
    ctx.emit_copy(t, out)
    return out


def alias_view(ctx: LoweringContext, node, base: TensorValue, subset: Optional[subsets.Range] = None) -> TensorValue:
    """
    Creates a view of ``base`` with the shape/strides of ``node``'s FakeTensor.

    Views of views are emitted as chains (offsets compose since all strides are relative to the root storage). If
    ``ctx.options.view_chains`` is False, the base is materialized first.
    """
    val = _val(node)
    if base.is_view and not getattr(ctx.options, 'view_chains', True):
        base = _materialize_if_view(ctx, base, node.name + '_mat')
    return ctx.emit_view('v_' + node.name, base, val, subset)


# ---------------------------------------------------------------------- pure reinterpretations
@register_lowering(*resolve('aten.view.default', 'aten._unsafe_view.default', 'aten.reshape.default',
                            'aten.permute.default', 'aten.t.default', 'aten.transpose.int', 'aten.unsqueeze.default',
                            'aten.squeeze.default', 'aten.squeeze.dim', 'aten.squeeze.dims', 'aten.alias.default',
                            'aten.detach.default', 'aten.view.dtype', 'aten.unfold.default', 'aten.as_strided.default',
                            'aten.movedim.int', 'aten.lift_fresh.default', 'aten._reshape_alias.default'))
def lower_reinterpret(ctx: LoweringContext, node, tensor: TensorValue, *args, **kwargs):
    if node.target in (aten.alias.default, aten.detach.default, aten.lift_fresh.default):
        return tensor
    return alias_view(ctx, node, tensor)


# ---------------------------------------------------------------------- select / slice
@register_lowering(aten.select.int)
def lower_select(ctx: LoweringContext, node, tensor: TensorValue, dim, index):
    rank = tensor.rank
    dim = _norm_dim(dim, rank)
    index = as_sym(index)
    if isinstance(index, int) and index < 0:
        index = index + tensor.tshape[dim]
    ranges = [(0, s - 1, 1) for s in tensor.tshape]
    ranges[dim] = (index, index, 1)
    return alias_view(ctx, node, tensor, subsets.Range(ranges))


@register_lowering(aten.slice.Tensor)
def lower_slice(ctx: LoweringContext, node, tensor: TensorValue, dim=0, start=None, end=None, step=1):
    rank = tensor.rank
    dim = _norm_dim(dim, rank)
    step = as_sym(step)
    start = 0 if start is None or (isinstance(start, ConstValue) and start.value is None) else as_sym(start)
    size = tensor.tshape[dim]
    if isinstance(start, int) and start < 0:
        start = start + size
    elif not isinstance(start, int):
        start = sympy.Max(sympy.Min(start, size), 0) if not _nonnegative(start) else start
    # The output extent is given by the FakeTensor; it already accounts for clamping of start/end.
    length = ctx.symtab.to_dace(_val(node).shape[dim])
    ranges = [(0, s - 1, 1) for s in tensor.tshape]
    ranges[dim] = (start, start + (length - 1) * step, step)
    return alias_view(ctx, node, tensor, subsets.Range(ranges))


def _nonnegative(expr) -> bool:
    try:
        return bool(sympy.ask(sympy.Q.nonnegative(expr))) or (expr.is_nonnegative is True)
    except Exception:
        return False


# ---------------------------------------------------------------------- copies
@register_lowering(*resolve('aten.clone.default', 'aten.contiguous.default'))
def lower_clone(ctx: LoweringContext, node, tensor: TensorValue, *, memory_format=None):
    out = ctx.add_tensor_like('t_' + node.name, _val(node), contiguous=True)
    ctx.emit_copy(tensor, out)
    return out


@register_lowering(aten.cat.default)
def lower_cat(ctx: LoweringContext, node, tensors: Sequence[TensorValue], dim=0):
    tensors = [t for t in tensors if not (t.rank == 1 and t.tshape[0] == 0)]
    val = _val(node)
    out = ctx.add_tensor_like('t_' + node.name, val, contiguous=True)
    dim = _norm_dim(dim, out.rank)
    offset = 0
    for t in tensors:
        extent = t.tshape[dim]
        dst_ranges = [(0, s - 1, 1) for s in out.tshape]
        dst_ranges[dim] = (offset, offset + extent - 1, 1)
        ctx.emit_copy(t, out, dst_subset=subsets.Range(dst_ranges))
        offset = offset + extent
    return out


@register_lowering(aten.flip.default)
def lower_flip(ctx: LoweringContext, node, tensor: TensorValue, dims):
    from .pointwise import emit_elementwise
    dims = {_norm_dim(d, tensor.rank) for d in dims}
    out = ctx.add_tensor_like('t_' + node.name, _val(node))
    params = [f'__i{k}' for k in range(out.rank)]
    idx = [f'{out.tshape[k]} - 1 - {p}' if k in dims else p for k, p in enumerate(params)]
    emit_elementwise(ctx, node.name, out, [(tensor, idx)], '__out = __in0')
    return out


@register_lowering(aten.copy.default)
def lower_copy(ctx: LoweringContext, node, dst: TensorValue, src: TensorValue, non_blocking=False):
    """Functional ``copy``: returns ``src`` broadcast (and cast) to ``dst``'s shape and dtype."""
    from .pointwise import lower_pointwise_values
    return lower_pointwise_values(ctx, node, [src], '{0}')
