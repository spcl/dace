# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Placement of array elements onto distributed memory nodes (block-cyclic per dimension)."""

import inspect
from collections.abc import Callable, Mapping, Sequence
from typing import Any, NamedTuple

import sympy

from dace import serialize, symbolic


def index_symbol(name: str) -> sympy.Symbol:
    """The symbol a :class:`SymbolicLambda` parameter is traced with: a nonnegative integer index."""
    return sympy.Symbol(name, integer=True, nonnegative=True)


def check_block_cyclic(expr: sympy.Basic, params: Mapping[str, sympy.Symbol]) -> None:
    """Raises ``ValueError`` unless ``expr`` uses only ``+ * // %`` over ``params``, integers and DaCe symbols."""
    if isinstance(expr, sympy.Integer):
        return
    if isinstance(expr, sympy.Symbol):
        if params.get(expr.name) is expr or (isinstance(expr, symbolic.symbol) and expr.name not in params):
            return
        raise ValueError(f"Symbol {expr} is neither a parameter nor a DaCe symbol")
    if isinstance(expr, (sympy.Add, sympy.Mul, sympy.Mod)):
        for arg in expr.args:
            check_block_cyclic(arg, params)
        return
    if isinstance(expr, sympy.Pow) and isinstance(expr.exp, sympy.Integer) and expr.exp > 0:
        check_block_cyclic(expr.base, params)
        return
    if isinstance(expr, sympy.floor):
        # floor(x / y) is how Python's ``//`` arrives; the quotient inside it is the only allowed division
        for part in sympy.fraction(sympy.together(expr.args[0])):
            check_block_cyclic(part, params)
        return
    raise ValueError(f"'{expr}' is outside the block-cyclic family (only + * // % of indices, integers, DaCe symbols)")


@serialize.serializable
class SymbolicLambda:
    """A Python lambda traced once on symbols; keeps the traced expressions, not the function."""

    def __init__(self, fn: Callable[..., Any]):
        names = tuple(inspect.signature(fn).parameters)
        try:
            result = fn(*(index_symbol(name) for name in names))
        except TypeError as ex:  # min/max/comparisons/calls on symbols
            raise ValueError(f"Cannot trace lambda symbolically: {ex}") from ex
        outputs = result if isinstance(result, tuple) else (result,)
        self.set_traced(names, [sympy.Integer(out) if type(out) is int else out for out in outputs])

    def set_traced(self, names: Sequence[str], exprs: Sequence[Any]) -> None:
        self.params = tuple(index_symbol(name) for name in names)
        by_name = {p.name: p for p in self.params}
        self.exprs = []
        for expr in exprs:
            if not isinstance(expr, sympy.Expr):
                raise ValueError(f"Lambda returned {expr!r}, not an integer expression")
            # A DaCe symbol divisor makes ``//`` an int_floor; keep one spelling
            expr = symbolic.relax_int_floor(expr)
            try:
                check_block_cyclic(expr, by_name)
            except ValueError as ex:
                raise ValueError(f"Lambda body {expr}: {ex}") from ex
            self.exprs.append(expr)
        self.exprs = tuple(self.exprs)

    def __call__(self, *args: Any) -> Any:
        """The traced expression(s) with the parameters replaced by ``args``."""
        repl = dict(zip(self.params, args, strict=True))
        outs = tuple(expr.subs(repl, simultaneous=True) for expr in self.exprs)
        return outs[0] if len(outs) == 1 else outs

    def to_json(self) -> dict[str, Any]:
        return {
            "type": "SymbolicLambda",
            "params": [p.name for p in self.params],
            "exprs": [symbolic.serialize_symbolic(e) for e in self.exprs],
        }

    @classmethod
    def from_json(cls, json_obj: dict[str, Any], context: Any = None) -> "SymbolicLambda":
        ret = cls.__new__(cls)
        names = json_obj["params"]
        exprs = []
        for text in json_obj["exprs"]:
            expr = symbolic.deserialize_symbolic(text)
            # Parameters come back as DaCe symbols of the same name
            exprs.append(expr.xreplace({s: index_symbol(s.name) for s in expr.free_symbols if s.name in names}))
        ret.set_traced(names, exprs)
        return ret

    def __repr__(self) -> str:
        return f"SymbolicLambda(({', '.join(p.name for p in self.params)}) -> {', '.join(map(str, self.exprs))})"


class DimPlacement(NamedTuple):
    """One distributed dimension: node part ``(i // block) % count`` (``count`` None: no wrap), times ``stride``."""

    block: sympy.Expr
    count: sympy.Expr | None
    stride: sympy.Expr


def dim_term(term: sympy.Expr, params: Sequence[sympy.Symbol]) -> tuple[int, DimPlacement]:
    stride, part = term.as_independent(*params, as_Add=False)
    count = None
    if isinstance(part, sympy.Mod):
        part, count = part.args
    block = sympy.Integer(1)
    if isinstance(part, sympy.floor):
        part, block = sympy.fraction(sympy.together(part.args[0]))
    if part not in params or any(e.has(*params) for e in (block, count) if e is not None):
        raise ValueError(f"Term '{term}' is not stride * ((index // block) % count) of one index")
    return params.index(part), DimPlacement(block, count, stride)


@serialize.serializable
class Placement:
    """
    Distributes an array over ``nodes`` memory nodes: ``index_map`` maps element indices (one per dimension) to a
    node id. Each dimension is either not distributed or block-cyclic, ``(i // B) % P``, and the node id is a sum of
    such parts times strides. An element's offset within its node is the row-major position of its local indices
    in :meth:`local_shape`, where a distributed dimension's local index is ``(i // (B * P)) * B + i % B``.
    """

    def __init__(self, nodes: Any, index_map: SymbolicLambda):
        if len(index_map.exprs) != 1:
            raise ValueError("A placement maps indices to exactly one node id")
        self.nodes = symbolic.pystr_to_symbolic(nodes)
        self.index_map = index_map
        dims: list[DimPlacement | None] = [None] * len(index_map.params)
        for term in sympy.Add.make_args(index_map.exprs[0]):
            if term == 0:
                continue
            dim, placed = dim_term(term, index_map.params)
            if dims[dim] is not None:
                raise ValueError(f"Index {index_map.params[dim]} appears in more than one term of the node map")
            dims[dim] = placed
        #: Per dimension, its block-cyclic distribution, or None if the dimension is not distributed.
        self.dims: tuple[DimPlacement | None, ...] = tuple(dims)

    def node_of(self, indices: Sequence[Any]) -> sympy.Expr:
        return self.index_map(*indices)

    def local_index(self, dim: int, index: Any) -> Any:
        placed = self.dims[dim]
        if placed is None:
            return index
        if placed.count is None:
            return sympy.Mod(index, placed.block)
        return sympy.floor(index / (placed.block * placed.count)) * placed.block + sympy.Mod(index, placed.block)

    def local_shape(self, shape: Sequence[Any]) -> tuple[Any, ...]:
        """Per dimension, the local extent that holds every node's share (the largest one)."""
        result = []
        for size, placed in zip(shape, self.dims, strict=True):
            if placed is None:
                result.append(size)
            elif placed.count is None:
                result.append(placed.block)
            else:
                result.append(sympy.ceiling(size / (placed.block * placed.count)) * placed.block)
        return tuple(result)

    def offset_of(self, indices: Sequence[Any], shape: Sequence[Any]) -> Any:
        """Element offset of ``indices`` within its node."""
        offset = sympy.Integer(0)
        for dim, (index, extent) in enumerate(zip(indices, self.local_shape(shape), strict=True)):
            offset = offset * extent + self.local_index(dim, index)
        return offset

    def check(self, shape: Sequence[Any], capacity: Any, symbols: Mapping[str, Any] | None = None) -> None:
        """
        Raises ``ValueError`` unless every element maps to a distinct (node, offset) pair, the node ids are exactly
        ``0 .. nodes-1``, and each node's share fits in ``capacity`` elements. Called by backends at lowering time.

        :param shape: The array shape.
        :param capacity: Elements one node can hold.
        :param symbols: Values for symbols, applied before checking.
        """
        values = dict(symbols or {})

        def value(expr: Any) -> sympy.Expr:
            expr = symbolic.pystr_to_symbolic(expr)
            return expr.subs({s: values[s.name] for s in expr.free_symbols if s.name in values})

        # Node ids are injective onto 0 .. nodes-1 iff the strides are a mixed radix over the per-dimension counts
        parts = []
        for size, placed in zip(shape, self.dims, strict=True):
            if placed is not None:
                count = placed.count if placed.count is not None else sympy.ceiling(size / placed.block)
                parts.append((value(placed.stride), value(count)))
        radix = sympy.Integer(1)
        while parts:
            match = next((part for part in parts if symbolic.equal(part[0], radix) is True), None)
            if match is None:
                raise ValueError(f"Placement is not a bijection: no node stride equals {radix} among {parts}")
            parts.remove(match)
            radix *= match[1]
        if symbolic.equal(radix, value(self.nodes)) is not True:
            raise ValueError(f"Placement uses {radix} node ids, not exactly {value(self.nodes)}")
        share = sympy.Mul(*(value(e) for e in self.local_shape(shape)))
        if sympy.Le(share, value(capacity)) is not sympy.true:
            raise ValueError(f"Node share of {share} elements does not fit capacity {value(capacity)}")

    def to_json(self) -> dict[str, Any]:
        return {
            "type": "Placement",
            "nodes": symbolic.serialize_symbolic(self.nodes),
            "index_map": self.index_map.to_json(),
        }

    @classmethod
    def from_json(cls, json_obj: dict[str, Any], context: Any = None) -> "Placement":
        nodes = symbolic.deserialize_symbolic(json_obj["nodes"])
        return cls(nodes, SymbolicLambda.from_json(json_obj["index_map"], context))

    def __repr__(self) -> str:
        return f"Placement(nodes={self.nodes}, {self.index_map})"
