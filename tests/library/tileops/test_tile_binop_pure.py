# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""End-to-end correctness for the ``pure`` expansion of :class:`TileBinop`.

Constructs a single-state SDFG that wires two tile inputs through a
``TileBinop`` node into a tile output, runs ``sdfg.expand_library_nodes()``
to lower to the nested ``pure`` SDFG, compiles, and compares against
numpy. K=1 and K=2 shapes are covered with and without ``has_mask``.
"""
import numpy as np
import pytest

import dace
from dace.libraries.tileops import TileBinop


_OP_TAG = {"+": "add", "-": "sub", "*": "mul", "/": "div",
           "min": "min", "max": "max",
           "<": "lt", "<=": "le", ">": "gt", ">=": "ge", "==": "eq", "!=": "ne",
           "&&": "land", "||": "lor"}


def _build_binop_sdfg(widths, op, has_mask, dtype):
    """Build a minimal SDFG: two input tiles -> TileBinop -> output tile."""
    sdfg = dace.SDFG(f"tile_binop_pure_{_OP_TAG[op]}_{'x'.join(str(w) for w in widths)}_{'m' if has_mask else 'nm'}")
    sdfg.add_array("A", widths, dtype, transient=False)
    sdfg.add_array("B", widths, dtype, transient=False)
    sdfg.add_array("C", widths, dtype, transient=False)
    if has_mask:
        sdfg.add_array("M", widths, dace.bool_, transient=False)

    state = sdfg.add_state("main")
    a = state.add_access("A")
    b = state.add_access("B")
    c = state.add_access("C")

    node = TileBinop(name="tb", widths=widths, op=op, has_mask=has_mask)
    state.add_node(node)
    full = ",".join(f"0:{w}" for w in widths)
    state.add_edge(a, None, node, "_a", dace.Memlet(f"A[{full}]"))
    state.add_edge(b, None, node, "_b", dace.Memlet(f"B[{full}]"))
    state.add_edge(node, "_c", c, None, dace.Memlet(f"C[{full}]"))
    if has_mask:
        m = state.add_access("M")
        state.add_edge(m, None, node, "_mask", dace.Memlet(f"M[{full}]"))

    sdfg.expand_library_nodes()
    sdfg.validate()
    return sdfg


@pytest.mark.parametrize("widths", [(8,), (4, 8)])
@pytest.mark.parametrize("op", ["+", "*", "max"])
def test_tile_binop_pure_unmasked(widths, op):
    """Unmasked tile binop matches the numpy reference for K = 1, 2."""
    sdfg = _build_binop_sdfg(widths, op, has_mask=False, dtype=dace.float64)
    rng = np.random.default_rng(seed=42)
    A = rng.random(widths)
    B = rng.random(widths)
    C = np.zeros(widths)
    sdfg(A=A, B=B, C=C)
    if op == "+":
        ref = A + B
    elif op == "*":
        ref = A * B
    else:
        ref = np.maximum(A, B)
    np.testing.assert_allclose(C, ref, rtol=0, atol=0)


@pytest.mark.parametrize("widths", [(8,), (4, 8)])
def test_tile_binop_pure_masked_holds_destination(widths):
    """Masked write must leave inactive lanes untouched (here: zero-init)."""
    sdfg = _build_binop_sdfg(widths, "+", has_mask=True, dtype=dace.float64)
    rng = np.random.default_rng(seed=7)
    A = rng.random(widths)
    B = rng.random(widths)
    C = np.zeros(widths)
    M = np.zeros(widths, dtype=bool)
    M.flat[: M.size // 2] = True
    sdfg(A=A, B=B, C=C, M=M)
    ref = np.where(M, A + B, 0.0)
    np.testing.assert_allclose(C, ref, rtol=0, atol=0)


def test_tile_binop_rejects_mixed_dtype():
    """E2 lock — ``validate`` raises NotImplementedError on cross-dtype operands."""
    widths = (8,)
    sdfg = dace.SDFG("tile_binop_mixed_dtype")
    sdfg.add_array("A", widths, dace.float64)
    sdfg.add_array("B", widths, dace.float32)
    sdfg.add_array("C", widths, dace.float64)
    state = sdfg.add_state("main")
    a, b, c = state.add_access("A"), state.add_access("B"), state.add_access("C")
    node = TileBinop(name="tb_mix", widths=widths, op="+")
    state.add_node(node)
    state.add_edge(a, None, node, "_a", dace.Memlet("A[0:8]"))
    state.add_edge(b, None, node, "_b", dace.Memlet("B[0:8]"))
    state.add_edge(node, "_c", c, None, dace.Memlet("C[0:8]"))
    with pytest.raises(NotImplementedError, match="uniform dtype"):
        sdfg.expand_library_nodes()


def test_tile_binop_rejects_unknown_op():
    """Constructor refuses ops that are not in ``_PY_OP_RHS``."""
    with pytest.raises(ValueError, match="unknown op"):
        TileBinop(name="bad", widths=(8,), op="not-an-op")


def test_tile_binop_rejects_invalid_K():
    """Constructor refuses K outside ``{1, 2, 3}``."""
    with pytest.raises(ValueError, match="length in"):
        TileBinop(name="bad_K", widths=(2, 2, 2, 2), op="+")
    with pytest.raises(ValueError, match="length in"):
        TileBinop(name="bad_K0", widths=(), op="+")


def _build_binop_symbol_rhs_sdfg(widths, expr_b, dtype=dace.float64, free_symbols=()):
    """Build a minimal SDFG: tile input -> TileBinop(kind_b=Symbol) -> tile output."""
    sdfg = dace.SDFG(f"tile_binop_sym_rhs_{'x'.join(str(w) for w in widths)}")
    for sym in free_symbols:
        sdfg.add_symbol(sym, dtype)
    sdfg.add_array("A", widths, dtype, transient=False)
    sdfg.add_array("C", widths, dtype, transient=False)

    state = sdfg.add_state("main")
    a = state.add_access("A")
    c = state.add_access("C")
    node = TileBinop(name="tb_sym", widths=widths, op="+",
                     kind_a="Tile", kind_b="Symbol", expr_b=expr_b)
    state.add_node(node)
    full = ",".join(f"0:{w}" for w in widths)
    state.add_edge(a, None, node, "_a", dace.Memlet(f"A[{full}]"))
    state.add_edge(node, "_c", c, None, dace.Memlet(f"C[{full}]"))

    sdfg.expand_library_nodes()
    sdfg.validate()
    return sdfg


@pytest.mark.parametrize("widths", [(8,), (4, 8)])
def test_tile_binop_kind_b_symbol_with_literal(widths):
    """Symbol-kind RHS with a numeric literal — every lane is shifted by the constant."""
    sdfg = _build_binop_symbol_rhs_sdfg(widths, expr_b="2.5")
    rng = np.random.default_rng(seed=51)
    A = rng.random(widths)
    C = np.zeros(widths)
    sdfg(A=A, C=C)
    np.testing.assert_allclose(C, A + 2.5, rtol=0, atol=0)


def test_tile_binop_kind_b_symbol_with_free_symbol():
    """Symbol-kind RHS resolves a free symbol at runtime."""
    widths = (4, 8)
    sdfg = _build_binop_symbol_rhs_sdfg(widths, expr_b="alpha", free_symbols=("alpha",))
    rng = np.random.default_rng(seed=52)
    A = rng.random(widths)
    C = np.zeros(widths)
    sdfg(A=A, C=C, alpha=0.75)
    np.testing.assert_allclose(C, A + 0.75, rtol=0, atol=0)


def test_tile_binop_rejects_symbol_symbol_pair():
    """Both-sides Symbol is rejected — outside the tile path."""
    with pytest.raises(ValueError, match="at least one operand must be 'Tile'"):
        TileBinop(name="ss", widths=(8,), op="+",
                  kind_a="Symbol", kind_b="Symbol", expr_a="a", expr_b="b")


def test_tile_binop_rejects_symbol_without_expr():
    """Symbol-kind operand requires the corresponding ``expr_*``."""
    with pytest.raises(ValueError, match="kind_b='Symbol' requires expr_b"):
        TileBinop(name="bad", widths=(8,), op="+", kind_b="Symbol")


def test_tile_binop_rejects_unknown_kind():
    """Constructor refuses kinds outside the allowed set."""
    with pytest.raises(ValueError, match="kind_a must be one of"):
        TileBinop(name="bad", widths=(8,), op="+", kind_a="NotAKind")


def _build_binop_scalar_rhs_sdfg(widths, dtype=dace.float64):
    """Build a minimal SDFG: tile input + scalar -> TileBinop(kind_b=Scalar) -> tile."""
    sdfg = dace.SDFG(f"tile_binop_scalar_{'x'.join(str(w) for w in widths)}")
    sdfg.add_array("A", widths, dtype, transient=False)
    sdfg.add_scalar("s", dtype, transient=False)
    sdfg.add_array("C", widths, dtype, transient=False)
    state = sdfg.add_state("main")
    a, s, c = state.add_access("A"), state.add_access("s"), state.add_access("C")
    node = TileBinop(name="tb_scalar", widths=widths, op="*", kind_a="Tile", kind_b="Scalar")
    state.add_node(node)
    full = ",".join(f"0:{w}" for w in widths)
    state.add_edge(a, None, node, "_a", dace.Memlet(f"A[{full}]"))
    state.add_edge(s, None, node, "_b", dace.Memlet("s"))
    state.add_edge(node, "_c", c, None, dace.Memlet(f"C[{full}]"))
    sdfg.expand_library_nodes()
    sdfg.validate()
    return sdfg


@pytest.mark.parametrize("widths", [(8,), (4, 8)])
def test_tile_binop_kind_b_scalar_broadcasts(widths):
    """Scalar-kind RHS (a ``dace.data.Scalar``) broadcasts to every lane."""
    sdfg = _build_binop_scalar_rhs_sdfg(widths)
    rng = np.random.default_rng(seed=61)
    A = rng.random(widths)
    C = np.zeros(widths)
    sdfg(A=A, s=2.5, C=C)
    np.testing.assert_allclose(C, A * 2.5, rtol=0, atol=0)


def test_tile_binop_rejects_no_tile_operand():
    """At least one operand must be a Tile (Scalar/Symbol pair refused)."""
    with pytest.raises(ValueError, match="at least one operand must be 'Tile'"):
        TileBinop(name="ss", widths=(8,), op="+",
                  kind_a="Scalar", kind_b="Symbol", expr_b="b")
