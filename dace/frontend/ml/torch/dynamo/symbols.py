# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""
Translation of TorchDynamo/ShapeEnv symbolic expressions into DaCe symbolic expressions.

Dynamo represents dynamic sizes and strides as sympy expressions over backed symbols (``s0``, ``s1``, ...). DaCe also
uses sympy, so the translation mostly re-roots the expression on DaCe ``symbol`` objects (minted exactly once per
name) and rewrites the handful of torch-specific sympy functions (``FloorDiv``, ``CeilDiv``, ``PythonMod``, ...).
"""
from typing import Dict, Iterable, Optional, Sequence, Tuple, Union

import sympy
import torch

import dace
from dace import dtypes, symbolic

SymExpr = Union[int, sympy.Basic]


class UnsupportedSymbolicExpression(NotImplementedError):
    pass


_SYM_TYPES = (torch.SymInt, torch.SymBool, torch.SymFloat)

# sympy node types that are rebuilt verbatim over DaCe symbols
_PASSTHROUGH = (sympy.Add, sympy.Mul, sympy.Pow, sympy.Max, sympy.Min, sympy.Abs, sympy.Eq, sympy.Ne, sympy.Lt,
                sympy.Le, sympy.Gt, sympy.Ge, sympy.And, sympy.Or, sympy.Not, sympy.Piecewise, sympy.floor,
                sympy.ceiling)


class SymbolTable:
    """
    Mints one DaCe symbol per Dynamo shape symbol and translates sympy expressions.

    The table is shared by every FX (sub)graph that originates from the same ShapeEnv, so a symbol that appears in a
    HOP body is bound to the same DaCe symbol as in the enclosing graph.
    """

    def __init__(self, dtype: dtypes.typeclass = dace.int64):
        self.dtype = dtype
        self.symbols: Dict[str, symbolic.symbol] = {}

    def get(self, sym: Union[sympy.Symbol, str]) -> symbolic.symbol:
        name = sym.name if isinstance(sym, sympy.Symbol) else str(sym)
        if name not in self.symbols:
            self.symbols[name] = symbolic.symbol(name, self.dtype, nonnegative=True)
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
        if isinstance(expr, symbolic.symbol):
            return self.get(expr.name)
        if isinstance(expr, sympy.Symbol):
            return self.get(expr)
        if isinstance(expr, _PASSTHROUGH):
            return expr.func(*[self._as_sympy(self.to_dace(a)) for a in expr.args])
        if isinstance(expr, sympy.Basic):
            name = type(expr).__name__
            args = [self._as_sympy(self.to_dace(a)) for a in expr.args]
            if name == 'FloorDiv':
                return symbolic.int_floor(*args)
            if name == 'CeilDiv':
                return symbolic.int_ceil(*args)
            if name in ('Mod', 'PythonMod'):
                return sympy.Mod(*args)
            if name == 'ModularIndexing':
                x, d, m = args
                return sympy.Mod(symbolic.int_floor(x, d), m)
            if name == 'Identity':
                return args[0]
            if name in ('IntTrueDiv', 'FloatTrueDiv'):
                # True division yielding a float: keep a float factor so generated code does not use integer division
                return sympy.Float(1.0) * args[0] / args[1]
            if name in ('ToFloat', 'TruncToFloat'):
                return sympy.Float(1.0) * args[0]
            if name in ('TruncToInt', 'FloorToInt'):
                return sympy.floor(args[0])
            if name == 'CeilToInt':
                return sympy.ceiling(args[0])
            if name == 'RoundToInt':
                return sympy.floor(args[0] + sympy.Rational(1, 2))
            if name in ('PowByNatural', 'FloatPow'):
                return args[0]**args[1]
            if name.startswith('OpaqueUnaryFn_'):
                fn = getattr(sympy, name[len('OpaqueUnaryFn_'):], None)
                if fn is not None:
                    return fn(args[0])
            if name == 'IsNonOverlappingAndDenseIndicator':
                raise UnsupportedSymbolicExpression(f'Unsupported layout predicate in shape expression: {expr}')
            raise UnsupportedSymbolicExpression(f'Unsupported symbolic function {name} in expression {expr}')
        raise UnsupportedSymbolicExpression(f'Cannot translate {type(expr).__name__}: {expr!r}')

    @staticmethod
    def _as_sympy(v: SymExpr) -> sympy.Basic:
        return sympy.Integer(v) if isinstance(v, int) else v

    def shape(self, sizes: Iterable) -> Tuple[SymExpr, ...]:
        return tuple(self.to_dace(s) for s in sizes)

    @property
    def symbol_types(self) -> Dict[str, dtypes.typeclass]:
        return {name: sym.dtype for name, sym in self.symbols.items()}


def free_symbol_names(expr: SymExpr) -> set:
    if isinstance(expr, int):
        return set()
    return {s.name for s in expr.free_symbols}
