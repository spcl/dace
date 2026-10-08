# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""
Neural-network operators lowered to DaCe maps: convolution and pooling.

Both are expressed over a zero/-inf padded copy of the input so that the inner loops never touch out-of-bounds
elements. Convolution accumulates with a write-conflict-resolution sum in a nested map; pooling evaluates the
(statically sized) window inside one tasklet per output element.
"""

from typing import List, Sequence

import sympy
import torch

from dace import subsets
from dace.memlet import Memlet

from ..context import LoweringContext, TensorValue, TupleValue, UnsupportedOpError, as_sym, index_memlet
from ..dtypes import is_floating
from .pointwise import cast_expr, emit_elementwise, literal
from .reduction import emit_reduce_into
from . import register_lowering, resolve

aten = torch.ops.aten


def _ints(v, n: int) -> List:
    """Normalizes an int-or-list argument to a list of ``n`` symbolic ints."""
    v = v.items if hasattr(v, "items") and not isinstance(v, dict) else v
    if isinstance(v, (list, tuple)):
        vals = [as_sym(x) for x in v]
        if len(vals) == 1 and n > 1:
            vals = vals * n
        return vals
    return [as_sym(v)] * n


def _pad_input(
    ctx: LoweringContext, name: str, x: TensorValue, lead: int, pads: Sequence, extents: Sequence, fill
) -> TensorValue:
    """
    Returns a copy of ``x`` whose spatial dims (after ``lead`` leading dims) are enlarged to ``extents`` with ``pads``
    leading padding elements filled with ``fill``. If no padding/extension is needed, returns ``x`` itself.
    """
    spatial = x.tshape[lead:]
    if all((isinstance(p, int) and p == 0) for p in pads) and all(e == s for e, s in zip(extents, spatial)):
        return x
    shape = tuple(x.tshape[:lead]) + tuple(extents)
    xp = ctx.add_array(f"t_{name}_padded", shape, x.torch_dtype, device=x.device)
    code = f"__out = {cast_expr(literal(fill, None), xp.dtype)}"
    emit_elementwise(ctx, f"{name}_padfill", xp, [], code)
    dst_ranges = [(0, s - 1, 1) for s in x.tshape[:lead]] + [(p, p + s - 1, 1) for p, s in zip(pads, spatial)]
    ctx.emit_copy(x, xp, dst_subset=subsets.Range(dst_ranges))
    return xp


# ---------------------------------------------------------------------- convolution
@register_lowering(*resolve("aten.convolution.default", "aten._convolution.default"))
def lower_convolution(
    ctx: LoweringContext,
    node,
    x: TensorValue,
    w: TensorValue,
    bias,
    stride,
    padding,
    dilation,
    transposed,
    output_padding,
    groups,
    *rest,
):
    if bool(as_sym(transposed)):
        raise UnsupportedOpError(node.target, "transposed convolution is not supported yet")
    nd = w.rank - 2
    stride, padding, dilation = _ints(stride, nd), _ints(padding, nd), _ints(dilation, nd)
    groups = as_sym(groups)
    val = node.meta["val"]
    out = ctx.add_tensor_like("t_" + node.name, val)
    N, Cin = x.tshape[0], x.tshape[1]
    Cout, Cg = w.tshape[0], w.tshape[1]  # Cg = Cin / groups
    kernel = list(w.tshape[2:])
    ospatial = list(out.tshape[2:])

    # Padded input: spatial extent covers every window position
    extents = [
        sympy.Max(s + 2 * p, (o - 1) * st + (k - 1) * d + 1)
        if not all(isinstance(v, int) for v in (s, p, o, st, k, d))
        else max(s + 2 * p, (o - 1) * st + (k - 1) * d + 1)
        for s, p, o, st, k, d in zip(x.tshape[2:], padding, ospatial, stride, kernel, dilation)
    ]
    xp = _pad_input(ctx, node.name, x, 2, padding, extents, 0)

    # Initialize output with bias (or zero)
    if isinstance(bias, TensorValue):
        emit_elementwise(ctx, node.name + "_init", out, [(bias, ["__i1"])], f"__out = {cast_expr('__in0', out.dtype)}")
    else:
        emit_elementwise(ctx, node.name + "_init", out, [], f"__out = {cast_expr('0', out.dtype)}")

    # Accumulate: outer map over output elements, inner map over the reduction window
    outer_params = ["__n", "__co"] + [f"__o{k}" for k in range(nd)]
    outer_ranges = [(0, N - 1, 1), (0, Cout - 1, 1)] + [(0, o - 1, 1) for o in ospatial]
    inner_params = ["__ci"] + [f"__k{k}" for k in range(nd)]
    inner_ranges = [(0, Cg - 1, 1)] + [(0, k - 1, 1) for k in kernel]
    cout_per_group = Cout / groups if not (isinstance(Cout, int) and isinstance(groups, int)) else Cout // groups
    group = "0" if (isinstance(groups, int) and groups == 1) else f"int_floor(__co, {cout_per_group})"
    in_idx = ["__n", f"({group}) * ({Cg}) + __ci"] + [
        f"__o{k} * {st} + __k{k} * {d}" for k, (st, d) in enumerate(zip(stride, dilation))
    ]
    w_idx = ["__co", "__ci"] + [f"__k{k}" for k in range(nd)]
    out_idx = ["__n", "__co"] + [f"__o{k}" for k in range(nd)]
    out_memlet = index_memlet(out.name, out_idx)
    out_memlet.wcr = "lambda a, b: a + b"
    code = f"__out = {cast_expr('__x', out.dtype)} * {cast_expr('__w', out.dtype)}"
    ctx.emit_nested_mapped_tasklet(
        node.name + "_acc",
        outer_params,
        outer_ranges,
        inner_params,
        inner_ranges,
        {"__x": index_memlet(xp.name, in_idx), "__w": index_memlet(w.name, w_idx)},
        code,
        {"__out": out_memlet},
    )
    return out


# ---------------------------------------------------------------------- pooling
def _window_tasklet(
    ctx: LoweringContext,
    node,
    x: TensorValue,
    kernel,
    stride,
    padding,
    dilation,
    pad_value,
    reduce_code,
    outputs: List[TensorValue],
    extra_code_vars: dict,
) -> None:
    """Emits one map over the output elements; the tasklet reads the whole (static) window via one memlet each."""
    nd = len(kernel)
    out = outputs[0]
    N, C = x.tshape[0], x.tshape[1]
    ospatial = list(out.tshape[2:])
    extents = [
        sympy.Max(s + 2 * p, (o - 1) * st + (k - 1) * d + 1)
        if not all(isinstance(v, int) for v in (s, p, o, st, k, d))
        else max(s + 2 * p, (o - 1) * st + (k - 1) * d + 1)
        for s, p, o, st, k, d in zip(x.tshape[2:], padding, ospatial, stride, kernel, dilation)
    ]
    xp = _pad_input(ctx, node.name, x, 2, padding, extents, pad_value)
    params = ["__n", "__c"] + [f"__o{k}" for k in range(nd)]
    ranges = [(0, N - 1, 1), (0, C - 1, 1)] + [(0, o - 1, 1) for o in ospatial]
    import itertools

    inputs = {}
    positions = list(itertools.product(*[range(int(k)) for k in kernel]))
    for pos in positions:
        name = "__w_" + "_".join(str(p) for p in pos)
        idx = ["__n", "__c"] + [f"__o{k} * {st} + {p * d}" for k, (p, st, d) in enumerate(zip(pos, stride, dilation))]
        inputs[name] = index_memlet(xp.name, idx)
    out_memlets = {f"__out{j}": index_memlet(o.name, params) for j, o in enumerate(outputs)}
    code = reduce_code(positions, extra_code_vars)
    ctx.emit_mapped_tasklet(node.name, params, ranges, inputs, code, out_memlets)


@register_lowering(
    *resolve(
        "aten.max_pool2d_with_indices.default",
        "aten.max_pool3d_with_indices.default",
        "aten.max_pool1d_with_indices.default",
    )
)
def lower_max_pool(
    ctx: LoweringContext, node, x: TensorValue, kernel_size, stride=None, padding=0, dilation=1, ceil_mode=False
):
    nd = x.rank - 2
    kernel = _ints(kernel_size, nd)
    stride = kernel if stride is None or (isinstance(stride, (list, tuple)) and len(stride) == 0) else _ints(stride, nd)
    padding, dilation = _ints(padding, nd), _ints(dilation, nd)
    if not all(isinstance(k, int) for k in kernel):
        raise UnsupportedOpError(node.target, "kernel sizes must be static")
    vals = list(node.meta["val"])
    out = ctx.add_tensor_like("t_" + node.name, vals[0])
    idx = ctx.add_tensor_like("t_" + node.name + "_idx", vals[1])
    spatial = list(x.tshape[2:])
    pad_value = float("-inf") if is_floating(out.dtype) else torch.iinfo(x.torch_dtype).min

    def code(positions, _):
        lines = []
        first = positions[0]
        lines.append(f"best = __w_{'_'.join(map(str, first))}")
        lines.append(f"besti = {_flat_index(first, stride, dilation, padding, spatial)}")
        for pos in positions[1:]:
            name = "__w_" + "_".join(map(str, pos))
            lines.append(f"if {name} > best:")
            lines.append(f"    best = {name}")
            lines.append(f"    besti = {_flat_index(pos, stride, dilation, padding, spatial)}")
        lines.append(f"__out0 = best")
        lines.append(f"__out1 = {cast_expr('besti', idx.dtype)}")
        return "\n".join(lines)

    _window_tasklet(ctx, node, x, kernel, stride, padding, dilation, pad_value, code, [out, idx], {})
    return TupleValue([out, idx])


def _flat_index(pos, stride, dilation, padding, spatial) -> str:
    """Flattened (row-major over the unpadded spatial dims) input index of window position ``pos``."""
    terms = []
    for k, (p, st, d, pad) in enumerate(zip(pos, stride, dilation, padding)):
        coord = f"(__o{k} * {st} + {p * d} - {pad})"
        inner = " * ".join(f"({s})" for s in spatial[k + 1 :])
        terms.append(f"{coord} * {inner}" if inner else coord)
    return " + ".join(terms)


@register_lowering(*resolve("aten.avg_pool2d.default", "aten.avg_pool1d.default", "aten.avg_pool3d.default"))
def lower_avg_pool(
    ctx: LoweringContext,
    node,
    x: TensorValue,
    kernel_size,
    stride=None,
    padding=0,
    ceil_mode=False,
    count_include_pad=True,
    divisor_override=None,
):
    nd = x.rank - 2
    kernel = _ints(kernel_size, nd)
    stride = kernel if stride is None or (isinstance(stride, (list, tuple)) and len(stride) == 0) else _ints(stride, nd)
    padding = _ints(padding, nd)
    if not all(isinstance(k, int) for k in kernel):
        raise UnsupportedOpError(node.target, "kernel sizes must be static")
    if not bool(as_sym(count_include_pad)) and any(not (isinstance(p, int) and p == 0) for p in padding):
        raise UnsupportedOpError(node.target, "count_include_pad=False with padding is not supported yet")
    if bool(as_sym(ceil_mode)):
        raise UnsupportedOpError(node.target, "ceil_mode=True is not supported yet")
    val = node.meta["val"]
    out = ctx.add_tensor_like("t_" + node.name, val)
    divisor = divisor_override if divisor_override is not None and not hasattr(divisor_override, "value") else None
    if divisor is None:
        divisor = 1
        for k in kernel:
            divisor *= int(k)

    def code(positions, _):
        total = " + ".join("__w_" + "_".join(map(str, pos)) for pos in positions)
        return f"__out0 = ({total}) / {cast_expr(str(divisor), out.dtype)}"

    _window_tasklet(ctx, node, x, kernel, stride, padding, [1] * nd, 0, code, [out], {})
    return out


# ---------------------------------------------------------------------- backward operators
def _window_extents(spatial, padding, ospatial, stride, kernel, dilation) -> List:
    """Spatial extents of a padded input that covers every window position (see ``_pad_input``)."""
    return [
        sympy.Max(s + 2 * p, (o - 1) * st + (k - 1) * d + 1)
        for s, p, o, st, k, d in zip(spatial, padding, ospatial, stride, kernel, dilation)
    ]


def _zeros(ctx: LoweringContext, name: str, out: TensorValue) -> None:
    emit_elementwise(ctx, name, out, [], f"__out = {cast_expr('0', out.dtype)}")


@register_lowering(
    *resolve("aten.max_pool2d_with_indices_backward.default", "aten.max_pool3d_with_indices_backward.default")
)
def lower_max_pool_backward(
    ctx: LoweringContext,
    node,
    grad_output: TensorValue,
    x: TensorValue,
    kernel_size,
    stride,
    padding,
    dilation,
    ceil_mode,
    indices: TensorValue,
):
    """Scatters each output gradient to the input element its window selected (``indices`` are flat per plane)."""
    grad_input = ctx.add_tensor_like("t_" + node.name, node.meta["val"])
    _zeros(ctx, node.name + "_init", grad_input)
    lead = grad_input.tshape[:2]
    plane = 1
    for s in grad_input.tshape[2:]:
        plane = plane * s
    # Flat view over the spatial dimensions (indices are positions within one plane)
    flat = ctx.emit_view_raw(
        "v_" + node.name + "_flat",
        grad_input,
        tuple(lead) + (plane,),
        (lead[1] * plane, plane, 1),
        grad_input.torch_dtype,
    )
    params = [f"__i{k}" for k in range(grad_output.rank)]
    out_memlet = Memlet.simple(flat.name, f"{params[0]}, {params[1]}, 0:{plane}")
    out_memlet.dynamic = True
    out_memlet.wcr = "lambda a, b: a + b"
    ctx.emit_mapped_tasklet(
        node.name,
        params,
        [(0, s - 1, 1) for s in grad_output.tshape],
        {"__g": index_memlet(grad_output.name, params), "__ix": index_memlet(indices.name, params)},
        f"__out[__ix] = {cast_expr('__g', grad_input.dtype)}",
        {"__out": out_memlet},
    )
    return grad_input


@register_lowering(*resolve("aten.convolution_backward.default"))
def lower_convolution_backward(
    ctx: LoweringContext,
    node,
    grad_output: TensorValue,
    x: TensorValue,
    w: TensorValue,
    bias_sizes,
    stride,
    padding,
    dilation,
    transposed,
    output_padding,
    groups,
    output_mask,
):
    """
    Gradients of a (non-transposed) convolution with respect to its input, weight, and bias, each as maps with
    write-conflict resolution (the input gradient scatters into a padded buffer).
    """
    if bool(as_sym(transposed)):
        raise UnsupportedOpError(node.target, "transposed convolution is not supported yet")
    nd = w.rank - 2
    stride, padding, dilation = _ints(stride, nd), _ints(padding, nd), _ints(dilation, nd)
    groups = as_sym(groups)
    mask = [bool(as_sym(m)) for m in (output_mask.items if hasattr(output_mask, "items") else output_mask)]
    vals = list(node.meta["val"])
    N = x.tshape[0]
    Cout, Cg = w.tshape[0], w.tshape[1]
    kernel = list(w.tshape[2:])
    ospatial = list(grad_output.tshape[2:])
    cout_per_group = Cout // groups if isinstance(Cout, int) and isinstance(groups, int) else Cout / groups
    group = "0" if (isinstance(groups, int) and groups == 1) else f"int_floor(__co, {cout_per_group})"
    o_params = [f"__o{k}" for k in range(nd)]
    k_params = [f"__k{k}" for k in range(nd)]
    o_ranges = [(0, o - 1, 1) for o in ospatial]
    k_ranges = [(0, k - 1, 1) for k in kernel]
    window = [f"{o} * {st} + {k} * {d}" for o, k, st, d in zip(o_params, k_params, stride, dilation)]
    extents = _window_extents(x.tshape[2:], padding, ospatial, stride, kernel, dilation)
    results = [None, None, None]

    if mask[0]:  # grad_input: scatter grad_output * weight into a padded buffer
        grad_input = ctx.add_tensor_like("t_" + node.name + "_input", vals[0])
        padded = ctx.add_array(
            f"t_{node.name}_input_padded",
            tuple(x.tshape[:2]) + tuple(extents),
            grad_input.torch_dtype,
            device=grad_input.device,
        )
        _zeros(ctx, node.name + "_input_init", padded)
        gi_memlet = index_memlet(padded.name, ["__n", f"({group}) * ({Cg}) + __ci"] + window)
        gi_memlet.wcr = "lambda a, b: a + b"
        code = f"__out = {cast_expr('__g', padded.dtype)} * {cast_expr('__w', padded.dtype)}"
        ctx.emit_nested_mapped_tasklet(
            node.name + "_input",
            ["__n", "__co"] + o_params,
            [(0, N - 1, 1), (0, Cout - 1, 1)] + o_ranges,
            ["__ci"] + k_params,
            [(0, Cg - 1, 1)] + k_ranges,
            {
                "__g": index_memlet(grad_output.name, ["__n", "__co"] + o_params),
                "__w": index_memlet(w.name, ["__co", "__ci"] + k_params),
            },
            code,
            {"__out": gi_memlet},
        )
        src = subsets.Range(
            [(0, s - 1, 1) for s in x.tshape[:2]] + [(p, p + s - 1, 1) for p, s in zip(padding, x.tshape[2:])]
        )
        ctx.emit_copy(padded, grad_input, src_subset=src)
        results[0] = grad_input

    if mask[1]:  # grad_weight: correlate the padded input with grad_output
        grad_weight = ctx.add_tensor_like("t_" + node.name + "_weight", vals[1])
        _zeros(ctx, node.name + "_weight_init", grad_weight)
        xp = _pad_input(ctx, node.name + "_bw", x, 2, padding, extents, 0)
        gw_memlet = index_memlet(grad_weight.name, ["__co", "__ci"] + k_params)
        gw_memlet.wcr = "lambda a, b: a + b"
        code = f"__out = {cast_expr('__g', grad_weight.dtype)} * {cast_expr('__x', grad_weight.dtype)}"
        ctx.emit_nested_mapped_tasklet(
            node.name + "_weight",
            ["__co", "__ci"] + k_params,
            [(0, Cout - 1, 1), (0, Cg - 1, 1)] + k_ranges,
            ["__n"] + o_params,
            [(0, N - 1, 1)] + o_ranges,
            {
                "__g": index_memlet(grad_output.name, ["__n", "__co"] + o_params),
                "__x": index_memlet(xp.name, ["__n", f"({group}) * ({Cg}) + __ci"] + window),
            },
            code,
            {"__out": gw_memlet},
        )
        results[1] = grad_weight

    if mask[2]:  # grad_bias: sum of grad_output over all but the channel dimension
        grad_bias = ctx.add_tensor_like("t_" + node.name + "_bias", vals[2])
        emit_reduce_into(
            ctx, node.name + "_bias", grad_output, "sum", [0] + list(range(2, grad_output.rank)), grad_bias
        )
        results[2] = grad_bias
    return TupleValue(results)
