# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""
View and data-movement lowerings: ``view``, ``permute``, ``slice``, ``select``, ``expand``, ``cat``, ``clone``, ...

All reinterpretations use the output FakeTensor's shape and strides (``node.meta['val']``) directly: ATen only emits
these operators when the result is expressible as a strided view of the input storage, so the metadata is exact.
"""
from typing import Optional, Sequence

import sympy
import torch

from dace import subsets
from dace import symbolic as dsym

from ..context import ConstValue, LoweringContext, TensorValue, TupleValue, UnsupportedOpError, as_sym
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


def alias_view(ctx: LoweringContext,
               node,
               base: TensorValue,
               subset: Optional[subsets.Range] = None,
               val: Optional[torch.Tensor] = None,
               prefix: Optional[str] = None) -> TensorValue:
    """
    Creates a view of ``base`` with the shape/strides of ``node``'s FakeTensor (or ``val``).

    Identity reinterpretations (same shape, strides, dtype, full range) return ``base`` itself. Views of views are
    emitted as chains (offsets compose since all strides are relative to the root storage). If
    ``ctx.options.view_chains`` is False, the base is materialized first.
    """
    val = _val(node) if val is None else val
    tshape = ctx.symtab.shape(val.shape)
    tstrides = ctx.symtab.shape(val.stride())
    if subset is None and tshape == base.tshape and tstrides == base.tstrides and val.dtype == base.torch_dtype:
        return base
    if base.is_view and not getattr(ctx.options, 'view_chains', True):
        base = _materialize_if_view(ctx, base, node.name + '_mat')
    return ctx.emit_view_raw(prefix or ('v_' + node.name), base, tshape, tstrides, val.dtype, subset)


def _slice_subset(tensor: TensorValue, dim: int, start, length, step=1) -> subsets.Range:
    ranges = [(0, s - 1, 1) for s in tensor.tshape]
    ranges[dim] = (start, start + (length - 1) * step, step)
    return subsets.Range(ranges)


# ---------------------------------------------------------------------- pure reinterpretations
@register_lowering(*resolve(
    'aten.view.default', 'aten._unsafe_view.default', 'aten.reshape.default', 'aten.permute.default', 'aten.t.default',
    'aten.transpose.int', 'aten.unsqueeze.default', 'aten.squeeze.default', 'aten.squeeze.dim', 'aten.squeeze.dims',
    'aten.alias.default', 'aten.detach.default', 'aten.view.dtype', 'aten.unfold.default', 'aten.movedim.int',
    'aten.lift_fresh.default', 'aten._reshape_alias.default', 'prims.broadcast_in_dim', 'prims.view_of',
    'prims.transpose', 'prims.squeeze', 'prims.collapse_view', 'prims.split_dim', 'prims.slice_in_dim', 'prims.slice'))
def lower_reinterpret(ctx: LoweringContext, node, tensor: TensorValue, *args, **kwargs):
    if node.target in (aten.alias.default, aten.detach.default, aten.lift_fresh.default):
        return tensor
    return alias_view(ctx, node, tensor)


@register_lowering(*resolve('aten.squeeze_.default', 'aten.squeeze_.dim', 'aten.squeeze_.dims',
                            'aten.unsqueeze_.default', 'aten.t_.default', 'aten.transpose_.default'))
def lower_reinterpret_in_place(ctx: LoweringContext, node, tensor: TensorValue, *args, **kwargs):
    """
    In-place metadata changes (``matmul`` squeezes its result in place) are views, if only the result of the operator
    is used afterwards (as in graphs that ``functionalize`` traces).
    """
    base = node.args[0]
    later = [user for user in base.users if user is not node and _comes_after(user, node)]
    if later:
        raise UnsupportedOpError(node.target, f'{base.name} is used after its shape changes in place')
    return alias_view(ctx, node, tensor)


def _comes_after(node: torch.fx.Node, other: torch.fx.Node) -> bool:
    current = other.next
    while current.op != 'root':
        if current is node:
            return True
        current = current.next
    return False


@register_lowering(*resolve('aten.as_strided.default', 'aten.as_strided_copy.default'))
def lower_as_strided(ctx: LoweringContext, node, tensor: TensorValue, size, stride, storage_offset=None):
    """
    ``as_strided`` relative to ``tensor``'s storage. The offset is expressed as a memlet subset start inside the base:
    it is decomposed greedily over the base's dimensions (largest stride first) when that is exact.
    """
    val = _val(node)
    offset = 0 if storage_offset is None else as_sym(storage_offset)
    base_val = _input_val(node, 0)
    if base_val is not None:
        offset = offset - ctx.symtab.to_dace(base_val.storage_offset())
    if offset == 0:
        return alias_view(ctx, node, tensor)
    indices = _offset_to_indices(offset, tensor.tshape, tensor.tstrides)
    if indices is None:
        raise UnsupportedOpError(node.target,
                                 f'cannot express storage offset {offset} within strides {tensor.tstrides}')
    # The view spans from the offset element; extents along base dims are irrelevant for pointer computation but the
    # subset must stay well-formed, so use a single element per base dimension.
    subset = subsets.Range([(i, i, 1) for i in indices])
    out = alias_view(ctx, node, tensor, subset)
    return out


def _input_val(node, position: int):
    arg = node.args[position] if len(node.args) > position else None
    if isinstance(arg, torch.fx.Node):
        return arg.meta.get('val', None)
    return None


def _offset_to_indices(offset, tshape, tstrides):
    """Greedy decomposition of a linear element offset into per-dimension indices of a strided layout."""
    if len(tshape) == 0:
        return None
    order = list(range(len(tshape)))
    if all(isinstance(s, int) for s in tstrides):
        order.sort(key=lambda k: -tstrides[k])
    indices = [0] * len(tshape)
    remaining = offset
    for k in order:
        stride = tstrides[k]
        if isinstance(stride, int) and stride == 0:
            continue
        if isinstance(remaining, int) and isinstance(stride, int):
            q, remaining = divmod(remaining, stride)
        else:
            q = sympy.simplify(remaining / stride)
            if q.is_integer is not True and not (isinstance(q, sympy.Basic) and q.is_Integer):
                continue
            remaining = 0
        indices[k] = q
    if (isinstance(remaining, int) and remaining == 0) or (isinstance(remaining, sympy.Basic) and remaining == 0):
        return indices
    return None


def _norm_dims_list(dims, rank):
    return [_norm_dim(d, rank) for d in (dims.items if isinstance(dims, TupleValue) else dims)]


@register_lowering(*resolve('aten.split.Tensor', 'aten.split.sizes', 'aten.split_with_sizes.default',
                            'aten.unsafe_split.Tensor', 'aten.unsafe_split_with_sizes.default'))
def lower_split(ctx: LoweringContext, node, tensor: TensorValue, split_size_or_sizes, dim=0):
    vals = list(_val(node))
    dim = _norm_dim(dim, tensor.rank)
    sizes = [ctx.symtab.to_dace(v.shape[dim]) for v in vals]
    views = []
    start = 0
    for k, (v, length) in enumerate(zip(vals, sizes)):
        views.append(
            alias_view(ctx, node, tensor, _slice_subset(tensor, dim, start, length), val=v,
                       prefix=f'v_{node.name}_{k}'))
        start = start + length
    return TupleValue(views)


@register_lowering(*resolve('aten.unbind.int'))
def lower_unbind(ctx: LoweringContext, node, tensor: TensorValue, dim=0):
    vals = list(_val(node))
    dim = _norm_dim(dim, tensor.rank)
    views = []
    for k, v in enumerate(vals):
        ranges = [(0, s - 1, 1) for s in tensor.tshape]
        ranges[dim] = (k, k, 1)
        views.append(alias_view(ctx, node, tensor, subsets.Range(ranges), val=v, prefix=f'v_{node.name}_{k}'))
    return TupleValue(views)


@register_lowering(*resolve('aten.narrow.default'))
def lower_narrow(ctx: LoweringContext, node, tensor: TensorValue, dim, start, length):
    dim = _norm_dim(dim, tensor.rank)
    start = as_sym(start)
    if isinstance(start, int) and start < 0:
        start = start + tensor.tshape[dim]
    return alias_view(ctx, node, tensor, _slice_subset(tensor, dim, start, as_sym(length)))


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
    # The output extent is given by the FakeTensor; it already accounts for clamping of start/end. A data-dependent
    # bound (``x[:t.item()]``) makes it a new unbacked symbol, which stands for the clamped extent computed here.
    extent = _val(node).shape[dim]
    unbacked = ctx.unassigned_unbacked(extent)
    if unbacked is not None:
        length = ctx.alias_unbacked(unbacked, _slice_extent(start, end, step, size))
    else:
        length = ctx.symtab.to_dace(extent)
    if start == 0 and step == 1 and length == size:
        return alias_view(ctx, node, tensor)  # full-range slice: identity (no view emitted)
    return alias_view(ctx, node, tensor, _slice_subset(tensor, dim, start, length, step))


def _slice_extent(start, end, step, size):
    """The number of elements of ``[start:end:step]`` along a dimension of ``size`` elements (``start`` clamped)."""
    if end is None or (isinstance(end, ConstValue) and end.value is None):
        stop = size
    else:
        end = as_sym(end)
        if isinstance(end, int) and end < 0:
            end = end + size
        if isinstance(end, int) or _nonnegative(end):
            stop = sympy.Min(end, size)
        else:  # A negative end counts from the end of the dimension
            stop = dsym.IfExpr(sympy.Ge(end, 0), sympy.Min(end, size), sympy.Max(end + size, 0))
    return sympy.Max(dsym.int_ceil(stop - start, step), 0)


def _nonnegative(expr) -> bool:
    try:
        return bool(sympy.ask(sympy.Q.nonnegative(expr))) or (expr.is_nonnegative is True)
    except Exception:
        return False


# ---------------------------------------------------------------------- copies
@register_lowering(*resolve('aten.clone.default', 'aten.contiguous.default'))
def lower_clone(ctx: LoweringContext, node, tensor: TensorValue, *, memory_format=None):
    out = ctx.add_tensor_like('t_' + node.name, _val(node))
    ctx.emit_copy(tensor, out)
    return out


@register_lowering(aten.cat.default)
def lower_cat(ctx: LoweringContext, node, tensors: Sequence[TensorValue], dim=0):
    tensors = [t for t in tensors if not (t.rank == 1 and t.tshape[0] == 0)]
    val = _val(node)
    out = ctx.add_tensor_like('t_' + node.name, val)
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
