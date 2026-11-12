# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""
Translation of TorchDynamo/ShapeEnv symbolic expressions into DaCe symbolic expressions.

Dynamo represents dynamic sizes and strides as sympy expressions over backed symbols (``s0``, ``s1``, ...). DaCe also
uses sympy, so the translation mostly re-roots the expression on DaCe ``symbol`` objects (minted exactly once per
name) and rewrites the handful of torch-specific sympy functions (``FloorDiv``, ``CeilDiv``, ``PythonMod``, ...).
"""

from collections.abc import Iterable
from typing import Any

import sympy
import torch

import dace
from dace import dtypes, symbolic

SymExpr = int | sympy.Basic


class UnsupportedSymbolicExpression(NotImplementedError):
    pass


_SYM_TYPES = (torch.SymInt, torch.SymBool, torch.SymFloat)

# sympy node types that are rebuilt verbatim over DaCe symbols
_PASSTHROUGH = (
    sympy.Add,
    sympy.Mul,
    sympy.Pow,
    sympy.Max,
    sympy.Min,
    sympy.Abs,
    sympy.Eq,
    sympy.Ne,
    sympy.Lt,
    sympy.Le,
    sympy.Gt,
    sympy.Ge,
    sympy.And,
    sympy.Or,
    sympy.Not,
    sympy.Piecewise,
    sympy.floor,
    sympy.ceiling,
)


def shape_env_of(values: Iterable[Any]) -> Any:
    """The ``ShapeEnv`` of the current compilation, or of fake tensors among ``values`` (``None`` if there is none)."""
    fake_mode = torch._guards.detect_fake_mode(list(values))
    return fake_mode.shape_env if fake_mode is not None else None


class SymbolTable:
    """
    Mints one DaCe symbol per Dynamo shape symbol and translates sympy expressions.

    The table is shared by every FX (sub)graph that originates from the same ShapeEnv, so a symbol that appears in a
    HOP body is bound to the same DaCe symbol as in the enclosing graph.
    """

    def __init__(
        self, dtype: dtypes.typeclass = dace.int64, names: dict[str, str] | None = None, shape_env: Any = None
    ):
        """
        :param dtype: Type of the minted symbols.
        :param names: Renaming of Dynamo symbols to DaCe symbol names.
        :param shape_env: The ``ShapeEnv`` of the symbols, whose value ranges tell which symbols may be negative (a
                          symbolic Python integer, as opposed to a size). Without it, every symbol is assumed
                          nonnegative.
        """
        self.dtype = dtype
        self.symbols: dict[str, symbolic.symbol] = {}
        self.names: dict[str, str] = dict(names or {})  #: Dynamo symbol name -> DaCe symbol name
        self.shape_env = shape_env
        #: Dynamo symbols that stand for an expression over other symbols (e.g., the clamped extent of a slice with a
        #: data-dependent bound); they are substituted instead of becoming DaCe symbols
        self.aliases: dict[str, SymExpr] = {}

    def get(self, sym: sympy.Symbol | str) -> symbolic.symbol:
        name = sym.name if isinstance(sym, sympy.Symbol) else str(sym)
        if name not in self.symbols:
            # ``nonnegative=False`` would assume a negative value: leave the sign unknown instead
            assumptions = {"nonnegative": True} if self._nonnegative(name) else {}
            self.symbols[name] = symbolic.symbol(self.names.get(name, name), self.dtype, **assumptions)
        return self.symbols[name]

    def _nonnegative(self, name: str) -> bool:
        if self.shape_env is None:
            return True
        for sym, value_range in self.shape_env.var_to_range.items():
            if sym.name == name:
                return bool(value_range.lower >= 0)
        return True

    def define(self, name: str, dtype: dtypes.typeclass, nonnegative: bool = False) -> symbolic.symbol:
        """
        Registers a symbol with an explicit type, e.g., an unbacked symbol (``u0``) that the program assigns from
        data, which may be a float, a boolean, or negative.
        """
        if name not in self.symbols:
            # ``nonnegative=False`` would assume a negative value: leave the sign unknown instead
            assumptions = {"nonnegative": True} if nonnegative else {}
            self.symbols[name] = symbolic.symbol(self.names.get(name, name), dtype, **assumptions)
        return self.symbols[name]

    def to_dace(self, expr) -> SymExpr:
        """Translates a Python number, ``torch.SymInt`` or sympy expression into a DaCe symbolic expression."""
        if isinstance(expr, bool):
            return int(expr)
        if isinstance(expr, int):
            return expr
        if isinstance(expr, float):
            return sympy.Float(expr)
        if isinstance(expr, _SYM_TYPES):
            if isinstance(expr, torch.SymInt) and expr.node.constant is not None:
                return int(expr.node.constant)
            expr = expr.node.expr
        if isinstance(expr, sympy.Integer):
            return int(expr)
        if isinstance(expr, (sympy.Rational, sympy.Float)):
            return expr
        if isinstance(expr, (sympy.logic.boolalg.BooleanTrue, sympy.logic.boolalg.BooleanFalse)):
            return 1 if bool(expr) else 0
        if isinstance(expr, sympy.Symbol) and expr.name in self.aliases:
            return self.aliases[expr.name]
        if isinstance(expr, symbolic.symbol):
            return self.get(expr.name)
        if isinstance(expr, sympy.Symbol):
            return self.get(expr)
        if isinstance(expr, sympy.Piecewise):
            return self._conditional(expr)
        if isinstance(expr, _PASSTHROUGH):
            return expr.func(*[self._as_sympy(self.to_dace(a)) for a in expr.args])
        if isinstance(expr, sympy.Basic):
            name = type(expr).__name__
            args = [self._as_sympy(self.to_dace(a)) for a in expr.args]
            if name == "Max":  # torch.utils._sympy.functions.Max (distinct from sympy.Max)
                return sympy.Max(*args)
            if name == "Min":
                return sympy.Min(*args)
            if name == "FloorDiv":
                return symbolic.int_floor(*args)
            if name == "CeilDiv":
                return symbolic.int_ceil(*args)
            if name in ("Mod", "PythonMod"):
                return sympy.Mod(*args)
            if name == "ModularIndexing":
                x, d, m = args
                return sympy.Mod(symbolic.int_floor(x, d), m)
            if name == "Identity":
                return args[0]
            if name in ("IntTrueDiv", "FloatTrueDiv"):
                # True division yielding a float: keep a float factor so generated code does not use integer division
                return sympy.Float(1.0) * args[0] / args[1]
            if name in ("ToFloat", "TruncToFloat"):
                return sympy.Float(1.0) * args[0]
            if name in ("TruncToInt", "FloorToInt"):
                return sympy.floor(args[0])
            if name == "CeilToInt":
                return sympy.ceiling(args[0])
            if name == "RoundToInt":
                return sympy.floor(args[0] + sympy.Rational(1, 2))
            if name in ("PowByNatural", "FloatPow"):
                return args[0] ** args[1]
            if name.startswith("OpaqueUnaryFn_"):
                fn = getattr(sympy, name[len("OpaqueUnaryFn_") :], None)
                if fn is not None:
                    return fn(args[0])
            if name == "IsNonOverlappingAndDenseIndicator":
                raise UnsupportedSymbolicExpression(f"Unsupported layout predicate in shape expression: {expr}")
            raise UnsupportedSymbolicExpression(f"Unsupported symbolic function {name} in expression {expr}")
        raise UnsupportedSymbolicExpression(f"Cannot translate {type(expr).__name__}: {expr!r}")

    @staticmethod
    def _as_sympy(v: SymExpr) -> sympy.Basic:
        return sympy.Integer(v) if isinstance(v, int) else v

    def _conditional(self, expr: sympy.Piecewise) -> SymExpr:
        """
        A ``Piecewise`` (``torch.sym_ite``, a boolean used in arithmetic) as nested DaCe conditional expressions: the
        code generator prints ``IfExpr`` as a ternary, ``Piecewise`` it does not know.
        """
        result = None
        for value, cond in reversed(expr.args):
            value = self._as_sympy(self.to_dace(value))
            if result is None or cond is sympy.true:
                result = value
            else:
                result = symbolic.IfExpr(self._as_sympy(self.to_dace(cond)), value, result)
        return result

    def shape(self, sizes: Iterable) -> tuple[SymExpr, ...]:
        return tuple(self.to_dace(s) for s in sizes)

    def strides(self, strides: Iterable) -> tuple[SymExpr, ...]:
        """
        Translates the strides of a tensor. Torch writes contiguous strides as ``Max(1, n)``, where ``n`` is the
        product of the sizes of the following dimensions; they become ``n``, which equals it wherever the tensor has
        elements, so that contiguous layouts compare equal to :func:`~.context.contiguous_strides`.
        """
        return tuple(_drop_unit_max(self.to_dace(s)) for s in strides)

    @property
    def symbol_types(self) -> dict[str, dtypes.typeclass]:
        return {sym.name: sym.dtype for sym in self.symbols.values()}


def _drop_unit_max(expr: SymExpr) -> SymExpr:
    if not isinstance(expr, sympy.Basic):
        return expr
    return expr.replace(
        lambda e: isinstance(e, sympy.Max) and len(e.args) == 2 and sympy.Integer(1) in e.args,
        lambda e: next(a for a in e.args if a != 1),
    )


def free_symbol_names(expr: SymExpr) -> set:
    if isinstance(expr, int):
        return set()
    return {s.name for s in expr.free_symbols}
