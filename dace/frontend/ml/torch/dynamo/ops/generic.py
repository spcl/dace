# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""
Generic lowering of ATen operators to library nodes (:mod:`dace.libraries.torch.nodes.aten`).

Every ATen operator without a dedicated lowering becomes a library call: tensor operands are the node's input
connectors (named after the operator schema), the results are output containers shaped like ``node.meta['val']``,
and the remaining arguments are stored on the node. Whether DaCe can implement the operator is decided when the SDFG
is expanded, not while importing the graph.
"""

from collections.abc import Sequence
from typing import Any

import sympy
import torch

from dace import dtypes
from dace.frontend.ml.torch.dynamo.context import (
    RANDOM_SEED_SYMBOL,
    ConstValue,
    LoweringContext,
    SymValue,
    TensorValue,
    TupleValue,
    UnsupportedOpError,
    Value,
    sanitize_name,
)
from dace.libraries.torch import nodes as aten_nodes
from dace.memlet import Memlet

#: Distance (in elements) between the random streams of two random operators of one program
RANDOM_STREAM_STRIDE = 1 << 40


def lower_aten_op(ctx: LoweringContext, node: torch.fx.Node, *args, **kwargs) -> Value:
    """Lowers an ATen operator call to the library node registered for it (``ATenOp`` by default)."""
    target = node.target
    bound = _bind_arguments(target._schema, args, kwargs)
    in_memlets: dict[str, Memlet] = {}
    arguments: dict[str, Any] = {}
    for name, value in bound.items():
        conn = sanitize_name(name)
        if isinstance(value, TensorValue):
            in_memlets[conn] = value.memlet()
        elif _is_tensor_sequence(value):
            for i, item in enumerate(_items(value)):
                if isinstance(item, TensorValue):
                    in_memlets[f"{conn}_{i}"] = item.memlet()
        else:
            arguments[name] = _to_json(ctx, value)

    out_memlets, result = _outputs(ctx, node)
    cls = aten_nodes.node_class(str(target))
    libnode = cls(sanitize_name(node.name), target=str(target), arguments=arguments)
    for conn in in_memlets:
        libnode.add_in_connector(conn)
    for conn in out_memlets:
        libnode.add_out_connector(conn)
    _configure(ctx, libnode)
    ctx.emit_library_call(libnode, in_memlets, out_memlets)
    return result


def _configure(ctx: LoweringContext, libnode: aten_nodes.ATenOp) -> None:
    """Node-specific settings that are not operator arguments."""
    if isinstance(libnode, aten_nodes.NativeDropout):
        libnode.seed = ctx.symtab.define(RANDOM_SEED_SYMBOL, dtypes.int64)
        libnode.offset = ctx.random_streams * RANDOM_STREAM_STRIDE
        ctx.random_streams += 1


def _bind_arguments(schema, args: Sequence[Any], kwargs: dict[str, Any]) -> dict[str, Any]:
    """The operator arguments by schema name, with defaults applied."""
    bound: dict[str, Any] = {}
    for i, arg in enumerate(schema.arguments):
        if arg.name in kwargs:
            bound[arg.name] = kwargs[arg.name]
        elif not arg.kwarg_only and i < len(args):
            bound[arg.name] = args[i]
        elif arg.has_default_value():
            bound[arg.name] = arg.default_value
        else:
            bound[arg.name] = None
    return bound


def _outputs(ctx: LoweringContext, node: torch.fx.Node) -> tuple[dict[str, Memlet], Value]:
    """Output containers shaped like the FakeTensor result(s) of ``node``, by connector name."""
    val = node.meta.get("val")
    if isinstance(val, torch.Tensor):
        out = ctx.add_tensor_like("t_" + node.name, val)
        return {"output": out.memlet()}, out
    if isinstance(val, (tuple, list)):
        memlets: dict[str, Memlet] = {}
        items: list[Value] = []
        for i, v in enumerate(val):
            if isinstance(v, torch.Tensor):
                out = ctx.add_tensor_like(f"t_{node.name}_{i}", v)
                memlets[f"output_{i}"] = out.memlet()
                items.append(out)
            elif v is None:
                items.append(None)
            else:
                raise UnsupportedOpError(node.target, f"result {i} is a {type(v).__name__}, not a tensor")
        return memlets, TupleValue(items)
    raise UnsupportedOpError(node.target, f"the result is a {type(val).__name__}, not a tensor")


def _items(value: Any) -> list[Any]:
    if isinstance(value, TupleValue):
        return list(value.items)
    return list(value)


def _is_tensor_sequence(value: Any) -> bool:
    if isinstance(value, TupleValue):
        value = value.items
    return isinstance(value, (list, tuple)) and any(isinstance(v, TensorValue) for v in value)


def _to_json(ctx: LoweringContext, value: Any) -> Any:
    """A JSON-compatible form of a non-tensor argument (symbolic values become strings)."""
    if isinstance(value, ConstValue):
        value = value.value
    if isinstance(value, SymValue):
        return _symbol_string(value.expr)
    if isinstance(value, TupleValue):
        return [_to_json(ctx, v) for v in value.items]
    if isinstance(value, (list, tuple)):
        return [_to_json(ctx, v) for v in value]
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, (torch.SymInt, torch.SymFloat, torch.SymBool)):
        return _symbol_string(ctx.symtab.to_dace(value))
    if isinstance(value, sympy.Basic):
        return _symbol_string(value)
    return str(value)  # dtypes, devices, layouts, memory formats


def _symbol_string(expr: Any) -> Any:
    if isinstance(expr, (bool, int, float)):
        return expr
    if isinstance(expr, sympy.Basic) and expr.is_Integer:
        return int(expr)
    if isinstance(expr, sympy.Basic) and expr.is_Float:
        return float(expr)
    return str(expr)
