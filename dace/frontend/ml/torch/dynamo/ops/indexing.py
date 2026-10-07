# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Indirect-access lowerings: ``index.Tensor`` (embedding lookups, index_select), ``gather``, ``argmax``/``argmin``."""

from typing import List

import torch

from dace.memlet import Memlet

from ..context import ConstValue, LoweringContext, TensorValue, UnsupportedOpError, as_sym, index_memlet
from .pointwise import cast_expr
from .reduction import emit_reduce_into
from . import register_lowering, resolve

aten = torch.ops.aten


def _range_memlet(name: str, indices: List[str], tshape, dim: int) -> Memlet:
    """Memlet selecting single elements on all dims except ``dim``, which spans the full extent."""
    parts = []
    for k, i in enumerate(indices):
        parts.append(f"0:{tshape[dim]}" if k == dim else str(i))
    return Memlet.simple(name, ", ".join(parts))


def _is_none(v) -> bool:
    return v is None or (isinstance(v, ConstValue) and v.value is None)


@register_lowering(*resolve("aten.index.Tensor", "aten._unsafe_index.Tensor"))
def lower_index(ctx: LoweringContext, node, x: TensorValue, indices):
    """Advanced indexing with exactly one index tensor (other positions ``None``): a gather along that dimension."""
    indices = list(indices.items) if hasattr(indices, "items") and not isinstance(indices, dict) else list(indices)
    active = [(d, idx) for d, idx in enumerate(indices) if not _is_none(idx)]
    if len(active) != 1 or not isinstance(active[0][1], TensorValue):
        raise UnsupportedOpError(node.target, "only indexing with a single index tensor is supported")
    dim, idx = active[0]
    val = node.meta["val"]
    out = ctx.add_tensor_like("t_" + node.name, val)
    # Output dims: x dims before `dim`, then idx dims, then x dims after `dim`
    params = [f"__i{k}" for k in range(out.rank)]
    idx_params = params[dim : dim + idx.rank]
    x_idx = params[:dim] + ["__idx"] + params[dim + idx.rank :]
    size = x.tshape[dim]
    code = f"__p = __idx + ({size} if __idx < 0 else 0)\n__out = {cast_expr('__x[__p]', out.dtype)}"
    inputs = {
        "__idx": index_memlet(idx.name, idx_params),
        "__x": _range_memlet(x.name, [p if p != "__idx" else "0" for p in x_idx], x.tshape, dim),
    }
    ranges = [(0, s - 1, 1) for s in out.tshape]
    ctx.emit_mapped_tasklet(node.name, params, ranges, inputs, code, {"__out": index_memlet(out.name, params)})
    return out


@register_lowering(*resolve("aten.gather.default"))
def lower_gather(ctx: LoweringContext, node, x: TensorValue, dim, index: TensorValue, *, sparse_grad=False):
    dim = int(as_sym(dim))
    dim = dim + x.rank if dim < 0 else dim
    val = node.meta["val"]
    out = ctx.add_tensor_like("t_" + node.name, val)
    params = [f"__i{k}" for k in range(out.rank)]
    size = x.tshape[dim]
    code = f"__p = __idx + ({size} if __idx < 0 else 0)\n__out = {cast_expr('__x[__p]', out.dtype)}"
    inputs = {
        "__idx": index_memlet(index.name, params),
        "__x": _range_memlet(x.name, params, x.tshape, dim),
    }
    ranges = [(0, s - 1, 1) for s in out.tshape]
    ctx.emit_mapped_tasklet(node.name, params, ranges, inputs, code, {"__out": index_memlet(out.name, params)})
    return out


def _arg_reduce(ctx: LoweringContext, node, x: TensorValue, dim, keepdim: bool, kind: str) -> TensorValue:
    """
    ``argmax``/``argmin`` as two DaCe-native steps: the extremum via ``Reduce``, then a map over all elements that
    emits ``index if x == extremum else INT_MAX`` with a write-conflict-resolution ``min`` into the output (so the
    first occurrence wins, matching torch).
    """
    val = node.meta["val"]
    out = ctx.add_tensor_like("t_" + node.name, val)
    sentinel = str(torch.iinfo(torch.int64).max)
    if dim is None or _is_none(dim):
        dims = list(range(x.rank))
        ext = ctx.add_array("t_" + node.name + "_ext", (), x.torch_dtype, device=x.device)
        emit_reduce_into(ctx, node.name + "_ext", x, kind, dims, ext)
        ctx.emit_tasklet(node.name + "_init", {}, f"__out = {cast_expr(sentinel, out.dtype)}", {"__out": out.memlet()})
        params = [f"__j{k}" for k in range(x.rank)]
        flat = " + ".join(f"__j{k}" + "".join(f" * ({s})" for s in x.tshape[k + 1 :]) for k in range(x.rank)) or "0"
        out_memlet = out.memlet()
        out_memlet.wcr = "lambda a, b: min(a, b)"
        code = f"__out = {cast_expr(f'({flat}) if __x == __m else {sentinel}', out.dtype)}"
        ctx.emit_mapped_tasklet(
            node.name,
            params,
            [(0, s - 1, 1) for s in x.tshape],
            {"__x": index_memlet(x.name, params), "__m": ext.memlet()},
            code,
            {"__out": out_memlet},
        )
        return out

    dim = int(as_sym(dim))
    dim = dim + x.rank if dim < 0 else dim
    red_shape = tuple(s for k, s in enumerate(x.tshape) if k != dim)
    ext = ctx.add_array("t_" + node.name + "_ext", red_shape, x.torch_dtype, device=x.device)
    emit_reduce_into(ctx, node.name + "_ext", x, kind, [dim], ext)
    out_params = [f"__i{k}" for k in range(out.rank)]
    emit_init = f"__out = {cast_expr(sentinel, out.dtype)}"
    ctx.emit_mapped_tasklet(
        node.name + "_init",
        out_params,
        [(0, s - 1, 1) for s in out.tshape],
        {},
        emit_init,
        {"__out": index_memlet(out.name, out_params)},
    )
    # Map over all input elements; the output index drops (or keeps as 0) the reduced dimension
    x_params = [f"__j{k}" for k in range(x.rank)]
    red_params = [p for k, p in enumerate(x_params) if k != dim]
    o_params = (x_params[:dim] + ["0"] + x_params[dim + 1 :]) if keepdim else red_params
    out_memlet = index_memlet(out.name, o_params)
    out_memlet.wcr = "lambda a, b: min(a, b)"
    code = f"__out = {cast_expr(f'__j{dim} if __x == __m else {sentinel}', out.dtype)}"
    ctx.emit_mapped_tasklet(
        node.name,
        x_params,
        [(0, s - 1, 1) for s in x.tshape],
        {"__x": index_memlet(x.name, x_params), "__m": index_memlet(ext.name, red_params)},
        code,
        {"__out": out_memlet},
    )
    return out


@register_lowering(*resolve("aten.argmax.default"))
def lower_argmax(ctx: LoweringContext, node, x: TensorValue, dim=None, keepdim=False):
    return _arg_reduce(ctx, node, x, dim, bool(as_sym(keepdim)), "amax")


@register_lowering(*resolve("aten.argmin.default"))
def lower_argmin(ctx: LoweringContext, node, x: TensorValue, dim=None, keepdim=False):
    return _arg_reduce(ctx, node, x, dim, bool(as_sym(keepdim)), "amin")


@register_lowering(*resolve("aten.index_put.default", "aten._unsafe_index_put.default"))
def lower_index_put(ctx: LoweringContext, node, x: TensorValue, indices, values, accumulate=False):
    """
    ``index_put`` with exactly one index tensor (other positions ``None``): a copy of ``x`` into which ``values`` are
    scattered along that dimension, summing on duplicate indices if ``accumulate`` (e.g., embedding gradients).
    """
    indices = list(indices.items) if hasattr(indices, "items") and not isinstance(indices, dict) else list(indices)
    active = [(d, idx) for d, idx in enumerate(indices) if not _is_none(idx)]
    if len(active) != 1 or not isinstance(active[0][1], TensorValue):
        raise UnsupportedOpError(node.target, "only one index tensor is supported")
    dim, idx = active[0]
    out = ctx.add_tensor_like("t_" + node.name, node.meta["val"])
    ctx.emit_copy(x, out)

    # Iterate over the indexed space: x dims before `dim`, then idx dims, then x dims after `dim`
    space = tuple(x.tshape[:dim]) + tuple(idx.tshape) + tuple(x.tshape[dim + 1 :])
    params = [f"__i{k}" for k in range(len(space))]
    idx_params = params[dim : dim + idx.rank]
    out_idx = params[:dim] + ["0"] + params[dim + idx.rank :]
    size = x.tshape[dim]
    inputs = {"__idx": index_memlet(idx.name, idx_params)}
    if isinstance(values, TensorValue):
        offset = len(space) - values.rank
        inputs["__v"] = index_memlet(
            values.name,
            ["0" if (isinstance(s, int) and s == 1) else params[offset + k] for k, s in enumerate(values.tshape)],
        )
        value = cast_expr("__v", out.dtype)
    else:
        value = cast_expr(str(as_sym(values)), out.dtype)
    out_memlet = _range_memlet(out.name, out_idx, out.tshape, dim)
    out_memlet.dynamic = True
    if bool(as_sym(accumulate)):
        out_memlet.wcr = "lambda a, b: a + b"
    code = f"__p = __idx + ({size} if __idx < 0 else 0)\n__out[__p] = {value}"
    ctx.emit_mapped_tasklet(node.name, params, [(0, s - 1, 1) for s in space], inputs, code, {"__out": out_memlet})
    return out
