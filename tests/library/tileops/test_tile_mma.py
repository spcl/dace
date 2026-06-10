# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Unit tests for :class:`TileMMA` -- the K-dim register-tile MMA lib node.

Mirrors cuTile's ``ct.mma(a, b, c)`` primitive with GEMM-style ``alpha`` /
``beta`` compile-time scalar prefactors (matching :class:`Gemm` convention).

Tests cover:

* Constructor validation (widths length, positive dims).
* Validate-time descriptor shape and dtype checks.
* Pure expansion produces correct numerical output across alpha / beta combos
  (full GEMM, accumulate, overwrite, identity).
* Connector wiring with and without ``_c`` input (``beta == 0`` skips the in
  connector).
"""
import dace
import numpy as np
import pytest

from dace.libraries.tileops import TileMMA


def _build_tile_mma_sdfg(M, K_inner, N, alpha=1, beta=1, dtype=dace.float64):
    """Build an SDFG that exercises a single ``TileMMA`` node.

    The SDFG reads three tile-shape transient arrays, calls the lib node, and
    writes the accumulator back out. ``_a``, ``_b``, ``_c`` are flattened into
    1-D arrays at the codegen boundary (the pure expansion addresses them with
    the row-major flat index ``i * N + j``); the tile shapes (M, K_inner),
    (K_inner, N), (M, N) match the lib-node validation.
    """
    sdfg = dace.SDFG("tile_mma_fixture")
    sdfg.add_array("A", (M, K_inner), dtype, transient=False)
    sdfg.add_array("B", (K_inner, N), dtype, transient=False)
    sdfg.add_array("C", (M, N), dtype, transient=False)
    state = sdfg.add_state("s", is_start_block=True)
    a_an = state.add_access("A")
    b_an = state.add_access("B")
    c_in = state.add_access("C") if beta != 0 else None
    c_out = state.add_access("C")
    node = TileMMA(name="mma", widths=(M, K_inner, N), alpha=alpha, beta=beta)
    state.add_node(node)
    state.add_edge(a_an, None, node, "_a", dace.Memlet(f"A[0:{M}, 0:{K_inner}]"))
    state.add_edge(b_an, None, node, "_b", dace.Memlet(f"B[0:{K_inner}, 0:{N}]"))
    if beta != 0:
        state.add_edge(c_in, None, node, "_cin", dace.Memlet(f"C[0:{M}, 0:{N}]"))
    state.add_edge(node, "_c", c_out, None, dace.Memlet(f"C[0:{M}, 0:{N}]"))
    return sdfg


def test_constructor_validates_widths_length():
    with pytest.raises(ValueError, match="widths must be a 3-tuple"):
        TileMMA(name="bad", widths=(8, 8))


def test_constructor_validates_positive_dims():
    with pytest.raises(ValueError, match="every dim must be positive"):
        TileMMA(name="bad", widths=(8, 0, 8))


def test_connectors_omit_cin_when_beta_zero():
    """``beta=0`` means no accumulator read; ``_cin`` is absent."""
    node = TileMMA(name="overwrite", widths=(4, 4, 4), alpha=1, beta=0)
    assert "_cin" not in node.in_connectors
    assert "_c" in node.out_connectors


def test_connectors_include_cin_when_beta_nonzero():
    """``beta != 0`` requires accumulator read on ``_cin``."""
    node = TileMMA(name="accum", widths=(4, 4, 4), alpha=1, beta=1)
    assert "_cin" in node.in_connectors
    assert "_c" in node.out_connectors


@pytest.mark.parametrize("alpha, beta", [
    (1, 0),
    (1, 1),
    (2, 0),
    (2, 3),
])
def test_pure_expansion_matches_numpy(alpha, beta):
    """Pure expansion produces bit-equivalent output to ``alpha * A @ B + beta * C``."""
    M, K_inner, N = 4, 8, 4
    sdfg = _build_tile_mma_sdfg(M, K_inner, N, alpha=alpha, beta=beta)
    sdfg.expand_library_nodes(recursive=True)
    rng = np.random.default_rng(seed=42)
    A = rng.random((M, K_inner))
    B = rng.random((K_inner, N))
    C = rng.random((M, N))
    expected = alpha * (A @ B) + beta * C
    actual = C.copy()
    sdfg.compile()(A=A.copy(), B=B.copy(), C=actual)
    np.testing.assert_allclose(actual, expected, rtol=1e-12)


def test_validate_rejects_shape_mismatch():
    """A descriptor shape mismatching ``(M, K_inner)`` raises at validate time."""
    sdfg = dace.SDFG("bad_shape")
    sdfg.add_array("A", (8, 4), dace.float64, transient=False)
    sdfg.add_array("B", (8, 4), dace.float64, transient=False)
    sdfg.add_array("C", (4, 4), dace.float64, transient=False)
    state = sdfg.add_state("s", is_start_block=True)
    node = TileMMA(name="bad", widths=(4, 4, 4), alpha=1, beta=0)
    state.add_node(node)
    state.add_edge(state.add_access("A"), None, node, "_a", dace.Memlet("A[0:8, 0:4]"))
    state.add_edge(state.add_access("B"), None, node, "_b", dace.Memlet("B[0:8, 0:4]"))
    state.add_edge(node, "_c", state.add_access("C"), None, dace.Memlet("C[0:4, 0:4]"))
    with pytest.raises(ValueError, match="_a' descriptor shape"):
        node.validate(sdfg, state)


def test_validate_rejects_mixed_dtype():
    """Uniform dtype across ``_a``, ``_b``, ``_c`` is required."""
    sdfg = dace.SDFG("mixed_dtype")
    sdfg.add_array("A", (4, 4), dace.float32, transient=False)
    sdfg.add_array("B", (4, 4), dace.float64, transient=False)
    sdfg.add_array("C", (4, 4), dace.float64, transient=False)
    state = sdfg.add_state("s", is_start_block=True)
    node = TileMMA(name="bad", widths=(4, 4, 4), alpha=1, beta=0)
    state.add_node(node)
    state.add_edge(state.add_access("A"), None, node, "_a", dace.Memlet("A[0:4, 0:4]"))
    state.add_edge(state.add_access("B"), None, node, "_b", dace.Memlet("B[0:4, 0:4]"))
    state.add_edge(node, "_c", state.add_access("C"), None, dace.Memlet("C[0:4, 0:4]"))
    with pytest.raises(NotImplementedError, match="uniform dtype"):
        node.validate(sdfg, state)


def test_pure_expansion_8x8():
    """Standard 8x8 / 8x8 / 8x8 tile dimensions exercise the cuTile-friendly
    power-of-2 sizes."""
    M, K_inner, N = 8, 8, 8
    sdfg = _build_tile_mma_sdfg(M, K_inner, N, alpha=1, beta=1)
    sdfg.expand_library_nodes(recursive=True)
    rng = np.random.default_rng(seed=8)
    A = rng.random((M, K_inner))
    B = rng.random((K_inner, N))
    C = rng.random((M, N))
    expected = A @ B + C
    actual = C.copy()
    sdfg.compile()(A=A.copy(), B=B.copy(), C=actual)
    np.testing.assert_allclose(actual, expected, rtol=1e-12)
