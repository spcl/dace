# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""
Elementwise lowerings: N-ary pointwise operators with broadcasting, casts, fills, and ranges.

Every pointwise operator becomes one map over the output shape (taken from ``node.meta['val']``) with a single
tasklet. Broadcast inputs use index ``0`` along size-1 dimensions. Type promotion was already performed by ATen, so
inputs are cast to the compute dtype where they differ and the result is cast to the output dtype.
"""

import math
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy
import sympy
import torch

import dace
from dace import dtypes

from ..context import ConstValue, LoweringContext, SymValue, TensorValue, UnsupportedOpError, index_memlet
from ..dtypes import is_boolean, is_floating, to_dace_dtype
from . import register_lowering, resolve

aten = torch.ops.aten
prims = torch.ops.prims

Operand = Union[TensorValue, SymValue, ConstValue, int, float, bool, sympy.Basic]


# ---------------------------------------------------------------------- helpers
def cast_name(dtype: dtypes.typeclass) -> str:
    """Name usable as a cast in Python tasklets (``dace.<name>(x)`` unparses to ``dace::<name>(x)``)."""
    if dtype == dace.bool_:
        return "bool_"
    return dtype.type.__name__


def cast_expr(expr: str, dtype: dtypes.typeclass) -> str:
    return f"dace.{cast_name(dtype)}({expr})"


def _is_one(d) -> bool:
    return (isinstance(d, int) and d == 1) or (isinstance(d, sympy.Basic) and d == 1)


def broadcast_indices(tensor: TensorValue, out_rank: int, params: Sequence[str]) -> List[str]:
    """Right-aligned broadcast indices of ``tensor`` into an output iterated by ``params``."""
    offset = out_rank - tensor.rank
    return ["0" if _is_one(d) else params[offset + k] for k, d in enumerate(tensor.tshape)]


def literal(value: Any, dtype: Optional[dtypes.typeclass]) -> str:
    """Formats a Python/sympy scalar as tasklet code, typed to ``dtype`` where that matters."""
    if isinstance(value, ConstValue):
        value = value.value
    if isinstance(value, SymValue):
        value = value.expr
    if isinstance(value, numpy.generic):  # numpy scalars (e.g. np.False_ from torch's fill values)
        value = value.item()
    if isinstance(value, bool):
        return "True" if value else "False"
    if isinstance(value, (int, sympy.Integer)):
        text = str(int(value))
    elif isinstance(value, float):
        if math.isnan(value):
            text = "nan"
        elif math.isinf(value):
            text = "inf" if value > 0 else "(-inf)"
        else:
            text = repr(value)
    elif isinstance(value, sympy.Basic):
        text = f"({value})"
    else:
        raise UnsupportedOpError("literal", f"cannot embed {type(value).__name__} {value!r} in tasklet code")
    if dtype is not None and not is_boolean(dtype) and not isinstance(value, sympy.Basic):
        return cast_expr(text, dtype)
    return text


def emit_elementwise(
    ctx: LoweringContext,
    name: str,
    out: TensorValue,
    inputs: Sequence[Tuple[TensorValue, Sequence[str]]],
    code: str,
    params: Optional[Sequence[str]] = None,
) -> None:
    """Emits ``map over out: __out = code(__in0, __in1, ...)`` with explicit per-input index expressions."""
    params = list(params) if params is not None else [f"__i{k}" for k in range(out.rank)]
    ranges = [(0, s - 1, 1) for s in out.tshape]
    in_memlets = {f"__in{k}": index_memlet(t.name, idx) for k, (t, idx) in enumerate(inputs)}
    out_memlets = {"__out": index_memlet(out.name, params)}
    ctx.emit_mapped_tasklet(name, params, ranges, in_memlets, code, out_memlets)


def _promote(dtypes_: Sequence[torch.dtype]) -> torch.dtype:
    result = dtypes_[0]
    for d in dtypes_[1:]:
        result = torch.promote_types(result, d)
    return result


def lower_pointwise_values(
    ctx: LoweringContext,
    node,
    operands: Sequence[Operand],
    template: Union[str, Callable[[dtypes.typeclass], str]],
    *,
    nocast: Sequence[int] = (),
    kwargs: Optional[Dict[str, Any]] = None,
    prefix: Optional[str] = None,
    compute_dtype: Optional[torch.dtype] = None,
) -> TensorValue:
    """
    Generic pointwise lowering into a fresh output shaped like ``node.meta['val']``.

    :param operands: Tensor or scalar operands referenced as ``{0}``, ``{1}``, ... in ``template``.
    :param template: Format string (or callable receiving the compute typeclass and returning one).
    :param nocast: Operand positions that must not be cast to the compute dtype (e.g. ``where`` predicates).
    :param kwargs: Extra named scalars available to the template (e.g. ``alpha``).
    :param compute_dtype: Overrides the dtype in which the expression is evaluated.
    """
    val = node.meta["val"]
    out = ctx.add_tensor_like(prefix or ("t_" + node.name), val)
    pointwise_into(ctx, node.name, out, operands, template, nocast=nocast, kwargs=kwargs, compute_dtype=compute_dtype)
    return out


def pointwise_into(
    ctx: LoweringContext,
    name: str,
    out: TensorValue,
    operands: Sequence[Operand],
    template: Union[str, Callable[[dtypes.typeclass], str]],
    *,
    nocast: Sequence[int] = (),
    kwargs: Optional[Dict[str, Any]] = None,
    compute_dtype: Optional[torch.dtype] = None,
) -> None:
    """Emits a broadcasting pointwise expression over ``operands`` into the existing container ``out``."""
    out_dtype = out.dtype

    tensors = [(k, op) for k, op in enumerate(operands) if isinstance(op, TensorValue)]
    if compute_dtype is None:
        if is_boolean(out_dtype) and tensors:
            compute_dtype = _promote([t.torch_dtype for _, t in tensors])
        else:
            compute_dtype = out.torch_dtype
    compute = to_dace_dtype(compute_dtype)

    params = [f"__i{k}" for k in range(out.rank)]
    inputs: List[Tuple[TensorValue, Sequence[str]]] = []
    operand_strs: List[str] = []
    for k, op in enumerate(operands):
        if isinstance(op, TensorValue):
            conn = f"__in{len(inputs)}"
            inputs.append((op, broadcast_indices(op, out.rank, params)))
            if k not in nocast and op.dtype != compute and not is_boolean(compute):
                conn = cast_expr(conn, compute)
            operand_strs.append(conn)
        else:
            operand_strs.append(literal(op, None if k in nocast or is_boolean(compute) else compute))

    named = {k: literal(v, compute) for k, v in (kwargs or {}).items() if _is_scalar(v)}
    if callable(template):
        template = template(compute)
    expr = template.format(*operand_strs, **named)
    code = f"__out = {cast_expr(expr, out_dtype)}"
    emit_elementwise(ctx, name, out, inputs, code, params)


def _is_scalar(v) -> bool:
    if isinstance(v, (ConstValue, SymValue)):
        v = v.value if isinstance(v, ConstValue) else v.expr
    return isinstance(v, (bool, int, float, sympy.Basic))


def cast_tensor(ctx: LoweringContext, name: str, tensor: TensorValue, dtype: torch.dtype) -> TensorValue:
    """Emits an elementwise cast of ``tensor`` to ``dtype`` into a fresh contiguous array."""
    out = ctx.add_array("t_" + name, tensor.tshape, dtype, device=tensor.device)
    params = [f"__i{k}" for k in range(out.rank)]
    code = f"__out = {cast_expr('__in0', out.dtype)}"
    emit_elementwise(ctx, name, out, [(tensor, params)], code, params)
    return out


# ---------------------------------------------------------------------- operator tables
def nan_max(a: str, b: str) -> str:
    """Maximum that propagates NaN from either operand, as ``torch.maximum`` does (``max`` returns ``b`` if ``a`` is NaN)."""
    return f"(({a}) if (({a}) != ({a})) or (({a}) > ({b})) else ({b}))"


def nan_min(a: str, b: str) -> str:
    """Minimum that propagates NaN from either operand, as ``torch.minimum`` does."""
    return f"(({a}) if (({a}) != ({a})) or (({a}) < ({b})) else ({b}))"


def _maximum(dt: dtypes.typeclass) -> str:
    return nan_max("{0}", "{1}") if is_floating(dt) else "max({0}, {1})"


def _minimum(dt: dtypes.typeclass) -> str:
    return nan_min("{0}", "{1}") if is_floating(dt) else "min({0}, {1})"


def _simple(template, nocast=()):

    def lower(ctx, node, *args, **kwargs):
        return lower_pointwise_values(ctx, node, list(args), template, nocast=nocast, kwargs=kwargs)

    return lower


_UNARY = {
    aten.neg: "-({0})",
    aten.abs: "abs({0})",
    aten.sign: "sign({0})",
    aten.exp: "exp({0})",
    aten.exp2: "exp2({0})",
    aten.expm1: "expm1({0})",
    aten.log: "log({0})",
    aten.log2: "log2({0})",
    aten.log10: "log10({0})",
    aten.log1p: "log1p({0})",
    aten.sin: "sin({0})",
    aten.cos: "cos({0})",
    aten.tan: "tan({0})",
    aten.asin: "asin({0})",
    aten.acos: "acos({0})",
    aten.atan: "atan({0})",
    aten.sinh: "sinh({0})",
    aten.cosh: "cosh({0})",
    aten.tanh: "tanh({0})",
    aten.asinh: "asinh({0})",
    aten.acosh: "acosh({0})",
    aten.atanh: "atanh({0})",
    aten.sqrt: "sqrt({0})",
    aten.rsqrt: "1 / sqrt({0})",
    aten.reciprocal: "1 / ({0})",
    aten.square: "({0}) * ({0})",
    aten.erf: "erf({0})",
    aten.erfc: "erfc({0})",
    aten.sigmoid: "1 / (1 + exp(-({0})))",
    aten.relu: lambda dt: nan_max("{0}", "0") if is_floating(dt) else "max({0}, 0)",
    aten.floor: "floor({0})",
    aten.ceil: "ceil({0})",
    aten.round: lambda dt: "rint({0})" if is_floating(dt) else "{0}",  # rint: round half to even, like torch
    aten.trunc: "trunc({0})",
    aten.logical_not: "not ({0})",
    aten.bitwise_not: lambda dt: "not ({0})" if is_boolean(dt) else "~({0})",
    aten.isnan: "({0}) != ({0})",
    aten.isinf: "isinf({0})",
    aten.isfinite: "isfinite({0})",
    aten.positive: "{0}",
}

_BINARY = {
    aten.mul: "({0}) * ({1})",
    aten.div: "({0}) / ({1})",
    aten.true_divide: "({0}) / ({1})",
    aten.maximum: _maximum,
    aten.minimum: _minimum,
    aten.fmax: lambda dt: "fmax({0}, {1})" if is_floating(dt) else "max({0}, {1})",  # fmax/fmin ignore NaN, like torch
    aten.fmin: lambda dt: "fmin({0}, {1})" if is_floating(dt) else "min({0}, {1})",
    aten.atan2: "atan2({0}, {1})",
    aten.fmod: lambda dt: "fmod({0}, {1})" if is_floating(dt) else "({0}) % ({1})",
    aten.remainder: lambda dt: (
        "(({0}) - floor(({0}) / ({1})) * ({1}))" if is_floating(dt) else "((({0}) % ({1})) + ({1})) % ({1})"
    ),
    aten.eq: "({0}) == ({1})",
    aten.ne: "({0}) != ({1})",
    aten.lt: "({0}) < ({1})",
    aten.le: "({0}) <= ({1})",
    aten.gt: "({0}) > ({1})",
    aten.ge: "({0}) >= ({1})",
    aten.logical_and: "({0}) and ({1})",
    aten.logical_or: "({0}) or ({1})",
    aten.logical_xor: "(({0}) != 0) != (({1}) != 0)",
    aten.bitwise_and: lambda dt: "({0}) and ({1})" if is_boolean(dt) else "({0}) & ({1})",
    aten.bitwise_or: lambda dt: "({0}) or ({1})" if is_boolean(dt) else "({0}) | ({1})",
    aten.bitwise_xor: lambda dt: "(({0}) != ({1}))" if is_boolean(dt) else "({0}) ^ ({1})",
    aten.copysign: "copysign({0}, {1})",
    aten.hypot: "hypot({0}, {1})",
    aten.clamp_min: _maximum,
    aten.clamp_max: _minimum,
}

for _op, _template in _UNARY.items():
    register_lowering(_op)(_simple(_template))
for _op, _template in _BINARY.items():
    register_lowering(_op)(_simple(_template))

# prims-level aliases (appear in some decompositions)
_PRIMS = {
    "add": "({0}) + ({1})",
    "sub": "({0}) - ({1})",
    "mul": "({0}) * ({1})",
    "div": "({0}) / ({1})",
    "maximum": _maximum,
    "minimum": _minimum,
    "pow": "pow({0}, {1})",
    "atan2": "atan2({0}, {1})",
    "eq": "({0}) == ({1})",
    "ne": "({0}) != ({1})",
    "lt": "({0}) < ({1})",
    "le": "({0}) <= ({1})",
    "gt": "({0}) > ({1})",
    "ge": "({0}) >= ({1})",
    "neg": "-({0})",
    "abs": "abs({0})",
    "exp": "exp({0})",
    "log": "log({0})",
    "sqrt": "sqrt({0})",
    "rsqrt": "1 / sqrt({0})",
    "sin": "sin({0})",
    "cos": "cos({0})",
    "tanh": "tanh({0})",
    "erf": "erf({0})",
    "floor": "floor({0})",
    "ceil": "ceil({0})",
    "round": lambda dt: "rint({0})" if is_floating(dt) else "{0}",
    "trunc": "trunc({0})",
    "sign": "sign({0})",
    "reciprocal": "1 / ({0})",
    "exp2": "exp2({0})",
    "expm1": "expm1({0})",
    "log1p": "log1p({0})",
    "log2": "log2({0})",
    "log10": "log10({0})",
    "isnan": "({0}) != ({0})",
    "isinf": "isinf({0})",
    "isfinite": "isfinite({0})",
}
for _name, _template in _PRIMS.items():
    for _op in resolve(f"prims.{_name}"):
        register_lowering(_op)(_simple(_template))


@register_lowering(*resolve("prims.where"))
def lower_prims_where(ctx, node, cond, a, b):
    return lower_pointwise_values(ctx, node, [cond, a, b], "(({1}) if ({0}) else ({2}))", nocast=(0,))


@register_lowering(*resolve("prims.convert_element_type"))
def lower_convert_element_type(ctx, node, x, dtype=None):
    return lower_pointwise_values(ctx, node, [x], "{0}")


@register_lowering(aten.add.Tensor, aten.add.Scalar)
def lower_add(ctx, node, a, b, *, alpha=1):
    if _is_unit(alpha):
        return lower_pointwise_values(ctx, node, [a, b], "({0}) + ({1})")
    return lower_pointwise_values(ctx, node, [a, b], "({0}) + {alpha} * ({1})", kwargs={"alpha": alpha})


@register_lowering(aten.sub.Tensor, aten.sub.Scalar)
def lower_sub(ctx, node, a, b, *, alpha=1):
    if _is_unit(alpha):
        return lower_pointwise_values(ctx, node, [a, b], "({0}) - ({1})")
    return lower_pointwise_values(ctx, node, [a, b], "({0}) - {alpha} * ({1})", kwargs={"alpha": alpha})


@register_lowering(aten.rsub.Tensor, aten.rsub.Scalar)
def lower_rsub(ctx, node, a, b, *, alpha=1):
    if _is_unit(alpha):
        return lower_pointwise_values(ctx, node, [a, b], "({1}) - ({0})")
    return lower_pointwise_values(ctx, node, [a, b], "({1}) - {alpha} * ({0})", kwargs={"alpha": alpha})


def _is_unit(v) -> bool:
    v = v.value if isinstance(v, ConstValue) else v
    return isinstance(v, (int, float)) and v == 1


@register_lowering(aten.div.Tensor_mode, aten.div.Scalar_mode)
def lower_div_mode(ctx, node, a, b, *, rounding_mode=None):
    rounding_mode = rounding_mode.value if isinstance(rounding_mode, ConstValue) else rounding_mode
    if rounding_mode is None:
        return lower_pointwise_values(ctx, node, [a, b], "({0}) / ({1})")
    if rounding_mode == "floor":
        return lower_pointwise_values(
            ctx, node, [a, b], lambda dt: "floor(({0}) / ({1}))" if is_floating(dt) else "py_floor({0}, {1})"
        )
    if rounding_mode == "trunc":
        return lower_pointwise_values(
            ctx, node, [a, b], lambda dt: "trunc(({0}) / ({1}))" if is_floating(dt) else "({0}) / ({1})"
        )
    raise UnsupportedOpError(node.target, f"rounding_mode={rounding_mode}")


@register_lowering(aten.pow.Tensor_Scalar, aten.pow.Tensor_Tensor, aten.pow.Scalar)
def lower_pow(ctx, node, a, b):
    exponent = b.value if isinstance(b, ConstValue) else b
    if isinstance(exponent, (int, float)) and not isinstance(exponent, bool):
        if exponent == 2:
            return lower_pointwise_values(ctx, node, [a], "({0}) * ({0})")
        if exponent == 3:
            return lower_pointwise_values(ctx, node, [a], "({0}) * ({0}) * ({0})")
        if exponent == 0.5:
            return lower_pointwise_values(ctx, node, [a], "sqrt({0})")
        if exponent == -0.5:
            return lower_pointwise_values(ctx, node, [a], "1 / sqrt({0})")
        if exponent == -1:
            return lower_pointwise_values(ctx, node, [a], "1 / ({0})")
        if exponent == 1:
            return lower_pointwise_values(ctx, node, [a], "{0}")
    return lower_pointwise_values(ctx, node, [a, b], "pow({0}, {1})")


@register_lowering(*resolve("aten.where.self", "aten.where.ScalarSelf", "aten.where.ScalarOther", "aten.where.Scalar"))
def lower_where(ctx, node, cond, a, b):
    return lower_pointwise_values(ctx, node, [cond, a, b], "(({1}) if ({0}) else ({2}))", nocast=(0,))


@register_lowering(aten.clamp.default, aten.clamp.Tensor)
def lower_clamp(ctx, node, x, min=None, max=None):
    lo = None if min is None or (isinstance(min, ConstValue) and min.value is None) else min
    hi = None if max is None or (isinstance(max, ConstValue) and max.value is None) else max
    operands = [x] + [v for v in (lo, hi) if v is not None]

    def template(dt: dtypes.typeclass) -> str:
        fmax, fmin = (
            (nan_max, nan_min) if is_floating(dt) else (lambda a, b: f"max({a}, {b})", lambda a, b: f"min({a}, {b})")
        )
        expr, k = "{0}", 1
        if lo is not None:
            expr, k = fmax(expr, f"{{{k}}}"), k + 1
        if hi is not None:
            expr = fmin(expr, f"{{{k}}}")
        return expr

    return lower_pointwise_values(ctx, node, operands, template)


@register_lowering(aten.addcmul.default)
def lower_addcmul(ctx, node, a, t1, t2, *, value=1):
    return lower_pointwise_values(ctx, node, [a, t1, t2], "({0}) + {value} * ({1}) * ({2})", kwargs={"value": value})


@register_lowering(aten.addcdiv.default)
def lower_addcdiv(ctx, node, a, t1, t2, *, value=1):
    return lower_pointwise_values(ctx, node, [a, t1, t2], "({0}) + {value} * (({1}) / ({2}))", kwargs={"value": value})


@register_lowering(aten.lerp.Scalar, aten.lerp.Tensor)
def lower_lerp(ctx, node, a, b, w):
    return lower_pointwise_values(ctx, node, [a, b, w], "({0}) + ({2}) * (({1}) - ({0}))")


# ---------------------------------------------------------------------- casts and copies
@register_lowering(
    *resolve(
        "aten._to_copy.default",
        "aten.to.dtype",
        "aten.to.dtype_layout",
        "aten.to.device",
        "aten.lift_fresh_copy.default",
    )
)
def lower_to_copy(ctx, node, x, *args, **kwargs):
    return lower_pointwise_values(ctx, node, [x], "{0}")


# ---------------------------------------------------------------------- fills, ranges
def _fill(ctx, node, value):
    val = node.meta["val"]
    out = ctx.add_tensor_like("t_" + node.name, val)
    code = f"__out = {cast_expr(literal(value, None), out.dtype)}"
    emit_elementwise(ctx, node.name, out, [], code)
    return out


@register_lowering(*resolve("aten.full.default", "aten.full.names"))
def lower_full(ctx, node, size, fill_value, **kwargs):
    return _fill(ctx, node, fill_value)


@register_lowering(aten.full_like.default)
def lower_full_like(ctx, node, x, fill_value, **kwargs):
    return _fill(ctx, node, fill_value)


@register_lowering(aten.new_full.default)
def lower_new_full(ctx, node, x, size, fill_value, **kwargs):
    return _fill(ctx, node, fill_value)


@register_lowering(aten.zeros.default, aten.zeros_like.default, aten.new_zeros.default)
def lower_zeros(ctx, node, *args, **kwargs):
    return _fill(ctx, node, 0)


@register_lowering(aten.ones.default, aten.ones_like.default, aten.new_ones.default)
def lower_ones(ctx, node, *args, **kwargs):
    return _fill(ctx, node, 1)


@register_lowering(
    *resolve(
        "aten.empty.memory_format",
        "aten.empty.default",
        "aten.empty_like.default",
        "aten.new_empty.default",
        "aten.empty_strided.default",
        "aten.empty_permuted.default",
    )
)
def lower_empty(ctx, node, *args, **kwargs):
    # Uninitialized storage: allocate the container, emit nothing.
    return ctx.add_tensor_like("t_" + node.name, node.meta["val"])


@register_lowering(aten.scalar_tensor.default)
def lower_scalar_tensor(ctx, node, value, **kwargs):
    return _fill(ctx, node, value)


@register_lowering(aten.arange.default, aten.arange.start, aten.arange.start_step)
def lower_arange(ctx, node, *args, **kwargs):
    if len(args) == 1:
        start, step = 0, 1
    elif len(args) == 2:
        (start, _), step = args, 1
    else:
        start, _, step = args[:3]
    val = node.meta["val"]
    out = ctx.add_tensor_like("t_" + node.name, val)
    expr = f"{literal(start, out.dtype)} + {cast_expr('__i0', out.dtype)} * {literal(step, out.dtype)}"
    code = f"__out = {cast_expr(expr, out.dtype)}"
    emit_elementwise(ctx, node.name, out, [], code)
    return out


@register_lowering(aten.expand.default)
def lower_expand(ctx, node, x, size, *, implicit=False):
    # Expanded tensors are stride-0 views of the source; emitting a view keeps the broadcast free of data movement.
    from .view import alias_view

    return alias_view(ctx, node, x)
