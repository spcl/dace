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


@pytest.mark.parametrize("widths", [(8,), (4, 8), (2, 4, 8)])
@pytest.mark.parametrize("op", ["+", "*", "max"])
def test_tile_binop_pure_unmasked(widths, op):
    """Unmasked tile binop matches the numpy reference for K = 1, 2, 3."""
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
