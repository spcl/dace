# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Lowerings for symbolic-integer (SymInt) arithmetic, tuple indexing, and size queries."""

import builtins
import math
import operator

import sympy
import torch

from dace import dtypes
from dace import symbolic as dsym

from ..context import ConstValue, SymValue, TensorValue, TupleValue, UnsupportedOpError, as_sym
from . import register_lowering, resolve

aten = torch.ops.aten
_SYM_TYPES = (torch.SymInt, torch.SymBool, torch.SymFloat)


def value_from_meta(ctx, node):
    """Returns a SymValue/ConstValue from ``node.meta['val']`` when it is a symbolic or constant scalar."""
    val = node.meta.get("val", None) if node is not None else None
    if isinstance(val, _SYM_TYPES):
        return SymValue(ctx.symtab.to_dace(val))
    if isinstance(val, (bool, int, float)):
        return ConstValue(val)
    return None


def _unwrap_tensor_shape(v: TensorValue):
    return v.tshape


@register_lowering(*resolve("aten.sym_size.int", "aten.sym_size.default", "aten.sym_size"))
def lower_sym_size(ctx, node, tensor, dim=None):
    v = value_from_meta(ctx, node)
    if v is not None:
        return v
    if dim is None:
        return TupleValue([SymValue(s) for s in tensor.tshape])
    return SymValue(tensor.tshape[as_sym(dim)])


@register_lowering(*resolve("aten.sym_stride.int", "aten.sym_stride.default", "aten.sym_stride"))
def lower_sym_stride(ctx, node, tensor, dim=None):
    v = value_from_meta(ctx, node)
    if v is not None:
        return v
    if dim is None:
        return TupleValue([SymValue(s) for s in tensor.tstrides])
    return SymValue(tensor.tstrides[as_sym(dim)])


@register_lowering(*resolve("aten.sym_numel.default", "aten.sym_numel"))
def lower_sym_numel(ctx, node, tensor):
    v = value_from_meta(ctx, node)
    if v is not None:
        return v
    numel = 1
    for s in tensor.tshape:
        numel = numel * s
    return SymValue(numel)


@register_lowering(*resolve("aten.sym_storage_offset.default", "aten.sym_storage_offset"))
def lower_sym_storage_offset(ctx, node, tensor):
    # The torch-level offset (from the FakeTensor), not 0: containers start at the tensor's first element, and
    # ``as_strided`` subtracts the base's torch-level offset to get an offset relative to the container
    v = value_from_meta(ctx, node)
    if v is None:
        raise UnsupportedOpError(node.target, "storage offset without FakeTensor metadata")
    return v


@register_lowering(*resolve("aten._local_scalar_dense.default"))
def lower_local_scalar_dense(ctx, node, tensor):
    """
    ``.item()``: Dynamo's unbacked symbol (e.g., ``u0``) becomes an SDFG symbol assigned from the tensor's element on
    an interstate edge, so that later shapes, indices, and control flow can use the data-dependent value.
    """
    val = node.meta.get("val", None)
    unbacked = ctx.unassigned_unbacked(val)
    if unbacked is None:
        constant = value_from_meta(ctx, node)
        if constant is None:
            raise UnsupportedOpError(node.target, f"unexpected value {val!r}")
        return constant
    if isinstance(val, torch.SymFloat):
        dtype = dtypes.float64
    elif isinstance(val, torch.SymBool):
        dtype = dtypes.bool_
    else:
        dtype = dtypes.int64
    value_range = val.node.shape_env.var_to_range.get(unbacked)
    nonnegative = value_range is not None and bool(value_range.lower >= 0)
    if tensor.is_view:  # Interstate edges read containers, not views
        element = ctx.add_array("item", (), tensor.torch_dtype, transient=True, device=tensor.device)
        ctx.emit_copy(tensor, element)
        tensor = element
    read = f"{tensor.name}[{', '.join(['0'] * len(tensor.desc.shape))}]"
    return SymValue(ctx.assign_symbol(unbacked, read, dtype, nonnegative))


@register_lowering(
    *resolve(
        "aten._assert_scalar.default", "aten.sym_constrain_range.default", "aten.sym_constrain_range_for_size.default"
    )
)
def lower_runtime_assertion(ctx, node, *args, **kwargs):
    """Runtime checks of the value ranges Dynamo assumed for unbacked symbols; not checked in the SDFG."""
    return ConstValue(None)


_BINARY = {
    operator.add: lambda a, b: a + b,
    operator.sub: lambda a, b: a - b,
    operator.mul: lambda a, b: a * b,
    operator.floordiv: lambda a, b: dsym.int_floor(a, b),
    operator.truediv: lambda a, b: a / b,
    operator.mod: lambda a, b: sympy.Mod(a, b),
    operator.pow: lambda a, b: a**b,
    operator.lt: lambda a, b: sympy.Lt(a, b),
    operator.le: lambda a, b: sympy.Le(a, b),
    operator.gt: lambda a, b: sympy.Gt(a, b),
    operator.ge: lambda a, b: sympy.Ge(a, b),
    operator.eq: lambda a, b: sympy.Eq(a, b),
    operator.ne: lambda a, b: sympy.Ne(a, b),
    operator.and_: lambda a, b: sympy.And(a, b),
    operator.or_: lambda a, b: sympy.Or(a, b),
    builtins.max: lambda a, b: sympy.Max(a, b),
    builtins.min: lambda a, b: sympy.Min(a, b),
    torch.sym_max: lambda a, b: sympy.Max(a, b),
    torch.sym_min: lambda a, b: sympy.Min(a, b),
}
_UNARY = {
    operator.neg: lambda a: -a,
    operator.not_: lambda a: sympy.Not(a),
    torch.sym_not: lambda a: sympy.Not(a),
    torch.sym_float: lambda a: sympy.Float(1.0) * a,
    torch.sym_int: lambda a: sympy.floor(a),
    math.floor: lambda a: sympy.floor(a),
    math.ceil: lambda a: sympy.ceiling(a),
    math.trunc: lambda a: sympy.floor(a),
    builtins.abs: lambda a: sympy.Abs(a),
    builtins.float: lambda a: sympy.Float(1.0) * a,
    builtins.int: lambda a: sympy.floor(a),
}
for _name, _fn in (("sym_sqrt", sympy.sqrt), ("sym_log2", lambda a: sympy.log(a, 2))):
    if hasattr(torch, _name):
        _UNARY[getattr(torch, _name)] = _fn


def _is_seq(v):
    return isinstance(v, (list, tuple, TupleValue))


def _items(v):
    return list(v.items) if isinstance(v, TupleValue) else list(v)


@register_lowering(*_BINARY.keys(), *_UNARY.keys())
def lower_symbolic_arith(ctx, node, *args):
    v = value_from_meta(ctx, node)
    if v is not None:
        return v
    target = node.target
    # Sequence concatenation / repetition (e.g. ``shape + [1]``)
    if target is operator.add and len(args) == 2 and _is_seq(args[0]) and _is_seq(args[1]):
        return TupleValue(_items(args[0]) + _items(args[1]))
    if target is operator.mul and len(args) == 2 and (_is_seq(args[0]) or _is_seq(args[1])):
        seq, n = (args[0], args[1]) if _is_seq(args[0]) else (args[1], args[0])
        return TupleValue(_items(seq) * int(as_sym(n)))
    if len(args) == 1 and target in _UNARY:
        return SymValue(_UNARY[target](as_sym(args[0])))
    if len(args) == 2 and target in _BINARY:
        return SymValue(_BINARY[target](as_sym(args[0]), as_sym(args[1])))
    if target in (builtins.max, builtins.min) and len(args) == 1 and _is_seq(args[0]):
        fn = sympy.Max if target is builtins.max else sympy.Min
        return SymValue(fn(*[as_sym(a) for a in _items(args[0])]))
    raise UnsupportedOpError(target, f"symbolic arithmetic with arguments {args}")


@register_lowering(operator.getitem)
def lower_getitem(ctx, node, seq, idx):
    idx = as_sym(idx) if not isinstance(idx, slice) else idx
    if isinstance(seq, TupleValue):
        items = seq.items
    elif isinstance(seq, (list, tuple)):
        items = list(seq)
    elif isinstance(seq, TensorValue):
        raise UnsupportedOpError(operator.getitem, "tensor indexing with getitem should have been decomposed")
    else:
        raise UnsupportedOpError(operator.getitem, f"indexing into {type(seq).__name__}")
    if isinstance(idx, slice):
        return TupleValue(items[idx])
    return items[int(idx)]


@register_lowering(*resolve("torch.sym_ite"))
def lower_sym_ite(ctx, node, cond, a, b):
    v = value_from_meta(ctx, node)
    if v is not None:
        return v
    return SymValue(sympy.Piecewise((as_sym(a), as_sym(cond)), (as_sym(b), True)))
