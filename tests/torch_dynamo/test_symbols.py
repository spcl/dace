# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Unit tests for the Dynamo -> DaCe symbolic expression translation (no compilation involved)."""

import pytest

pytest.importorskip("torch", reason="PyTorch not installed. Please install with: pip install dace[ml]")

import sympy  # noqa: E402
import torch  # noqa: E402

import dace  # noqa: E402
from dace import symbolic  # noqa: E402
from dace.frontend.ml.torch.dynamo.symbols import SymbolTable, UnsupportedSymbolicExpression  # noqa: E402


@pytest.mark.torch
def test_symbols_are_minted_once():
    tab = SymbolTable()
    s0 = sympy.Symbol("s0", integer=True)
    a = tab.get(s0)
    b = tab.get(sympy.Symbol("s0"))
    assert a is b
    assert isinstance(a, symbolic.symbol)
    assert a.dtype == dace.int64
    assert tab.symbol_types == {"s0": dace.int64}


@pytest.mark.torch
def test_passthrough_expressions():
    tab = SymbolTable()
    s0, s1 = sympy.Symbol("s0", integer=True), sympy.Symbol("s1", integer=True)
    expr = tab.to_dace(2 * s0 * s1 + 3)
    d0, d1 = tab.get(s0), tab.get(s1)
    assert expr == 2 * d0 * d1 + 3
    assert tab.to_dace(sympy.Integer(5)) == 5 and isinstance(tab.to_dace(sympy.Integer(5)), int)
    assert tab.to_dace(sympy.Max(s0, s1)) == sympy.Max(d0, d1)


@pytest.mark.torch
def test_torch_sympy_functions():
    from torch.utils._sympy.functions import CeilDiv, FloorDiv, ModularIndexing, PythonMod

    tab = SymbolTable()
    s0 = sympy.Symbol("s0", integer=True)
    d0 = tab.get(s0)
    assert tab.to_dace(FloorDiv(s0, 2)) == symbolic.int_floor(d0, 2)
    # torch canonicalizes CeilDiv(a, b) to FloorDiv(a + b - 1, b) on construction
    assert tab.to_dace(CeilDiv(s0, 2)) == symbolic.int_floor(d0 + 1, 2)
    assert tab.to_dace(PythonMod(s0, 3)) == sympy.Mod(d0, 3)
    assert tab.to_dace(ModularIndexing(s0, 2, 3)) == sympy.Mod(symbolic.int_floor(d0, 2), 3)


@pytest.mark.torch
def test_symint_from_shape_env():
    from torch.fx.experimental.symbolic_shapes import ShapeEnv
    from torch._subclasses.fake_tensor import FakeTensorMode

    shape_env = ShapeEnv()
    with FakeTensorMode(shape_env=shape_env) as mode:
        t = mode.from_tensor(torch.empty(4, 6), symbolic_context=None)
        torch._dynamo.mark_dynamic(t, 0)
    # Create a symbolic size directly through the shape env
    sym = shape_env.create_unspecified_symbol(7, source=torch._dynamo.source.LocalSource("x"))
    symint = shape_env.create_symintnode(sym, hint=7)
    tab = SymbolTable()
    expr = tab.to_dace(symint * 2 + 1)
    (name,) = tab.symbols.keys()
    assert expr == 2 * tab.symbols[name] + 1


@pytest.mark.torch
def test_unsupported_function_raises():
    tab = SymbolTable()
    s0 = sympy.Symbol("s0", integer=True)
    with pytest.raises(UnsupportedSymbolicExpression):
        tab.to_dace(sympy.Function("Weird")(s0))


if __name__ == "__main__":
    test_symbols_are_minted_once()
    test_passthrough_expressions()
    test_torch_sympy_functions()
    test_symint_from_shape_env()
    test_unsupported_function_raises()
