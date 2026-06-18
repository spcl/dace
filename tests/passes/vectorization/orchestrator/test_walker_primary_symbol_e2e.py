# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""End-to-end numerical + structural tests for the Symbol operand path.

Two flavors per design 6.2 / 6.5:

* **Data-independent** symbol -- a runtime symbol like ``N`` or a numeric literal that
  does NOT reference any tile iter_var. Broadcast at expansion time;
  ``kind=Symbol`` + inline ``expr_*`` on the lib node.

* **Lane-id-dependent** symbol -- an expression that references an iter_var (e.g.
  ``ii``, ``2 * ii + jj``). Materialised to a per-lane tile by the
  ConvertTaskletsToTileOps lane-id path; the lib node sees ``kind=Tile`` and reads
  from the materialised tile.

Both paths should produce numerically correct output bit-equal to the unvectorized
reference.
"""
import numpy as np
import pytest

import dace
from dace.libraries.tileops import TileBinop
from dace.transformation.passes.vectorization.vectorize_cpu_multi_dim import (VectorizeCPUMultiDim)


def _build_add_symbol_kernel(N):
    """``B[i] = A[i] + SHIFT`` where ``SHIFT`` is a runtime symbol (data-independent)."""
    sdfg = dace.SDFG("add_sym_data_indep")
    sdfg.add_symbol("SHIFT", dace.float64)
    sdfg.add_array("A", (N, ), dace.float64, transient=False)
    sdfg.add_array("B", (N, ), dace.float64, transient=False)
    state = sdfg.add_state("s")
    me, mx = state.add_map("k", {"ii": f"0:{N}"})
    a = state.add_access("A")
    b = state.add_access("B")
    t = state.add_tasklet("body", {"_a"}, {"_b"}, "_b = _a + SHIFT")
    state.add_memlet_path(a, me, t, dst_conn="_a", memlet=dace.Memlet("A[ii]"))
    state.add_memlet_path(t, mx, b, src_conn="_b", memlet=dace.Memlet("B[ii]"))
    return sdfg


def _build_lane_id_kernel(N):
    """``B[i] = A[i] + ii`` -- the lane-id ``ii`` is materialised per lane.

    Expected: for each tile starting at base, ``B[base + l] = A[base + l] + (base + l)``.
    """
    sdfg = dace.SDFG("add_lane_id")
    sdfg.add_array("A", (N, ), dace.float64, transient=False)
    sdfg.add_array("B", (N, ), dace.float64, transient=False)
    state = sdfg.add_state("s")
    me, mx = state.add_map("k", {"ii": f"0:{N}"})
    a = state.add_access("A")
    b = state.add_access("B")
    t = state.add_tasklet("body", {"_a"}, {"_b"}, "_b = _a + ii")
    state.add_memlet_path(a, me, t, dst_conn="_a", memlet=dace.Memlet("A[ii]"))
    state.add_memlet_path(t, mx, b, src_conn="_b", memlet=dace.Memlet("B[ii]"))
    return sdfg


def _build_two_symbol_kernel(N):
    """``B[i] = SHIFT * SCALE`` -- both operands are data-independent symbols.

    Per design 6.2 the output is a Scalar broadcast across the tile (the lib node
    expansion picks the Scalar-output path).
    """
    sdfg = dace.SDFG("two_sym_data_indep")
    sdfg.add_symbol("SHIFT", dace.float64)
    sdfg.add_symbol("SCALE", dace.float64)
    sdfg.add_array("B", (N, ), dace.float64, transient=False)
    state = sdfg.add_state("s")
    me, mx = state.add_map("k", {"ii": f"0:{N}"})
    b = state.add_access("B")
    t = state.add_tasklet("body", set(), {"_b"}, "_b = SHIFT * SCALE")
    state.add_memlet_path(me, t, memlet=dace.Memlet())
    state.add_memlet_path(t, mx, b, src_conn="_b", memlet=dace.Memlet("B[ii]"))
    return sdfg


@pytest.mark.parametrize("N", [8, 16])
def test_symbol_invariant_addition_matches_reference(N):
    """K=1 ``B[i] = A[i] + SHIFT`` with SHIFT runtime symbol."""
    rng = np.random.default_rng(seed=N)
    a = rng.random(N)
    SHIFT = 3.14
    b_ref = np.zeros(N)
    b_vec = np.zeros(N)
    ref = _build_add_symbol_kernel(N)
    ref.name = f"sym_ref_{N}"
    vec = _build_add_symbol_kernel(N)
    vec.name = f"sym_vec_{N}"
    VectorizeCPUMultiDim(widths=(8, ), target_isa="SCALAR").apply_pass(vec, {})
    ref.compile()(A=a.copy(), B=b_ref, SHIFT=SHIFT)
    vec.compile()(A=a.copy(), B=b_vec, SHIFT=SHIFT)
    np.testing.assert_allclose(b_vec, b_ref, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("N", [8, 16])
def test_symbol_lane_id_addition_matches_reference(N):
    """K=1 ``B[i] = A[i] + ii`` -- lane-id materialised."""
    rng = np.random.default_rng(seed=N + 1)
    a = rng.random(N)
    b_ref = np.zeros(N)
    b_vec = np.zeros(N)
    ref = _build_lane_id_kernel(N)
    ref.name = f"lid_ref_{N}"
    vec = _build_lane_id_kernel(N)
    vec.name = f"lid_vec_{N}"
    VectorizeCPUMultiDim(widths=(8, ), target_isa="SCALAR").apply_pass(vec, {})
    ref.compile()(A=a.copy(), B=b_ref)
    vec.compile()(A=a.copy(), B=b_vec)
    np.testing.assert_allclose(b_vec, b_ref, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("N", [8, 16])
def test_two_symbol_broadcast_matches_reference(N):
    """K=1 ``B[i] = SHIFT * SCALE`` -- both data-independent, output is Scalar broadcast."""
    SHIFT, SCALE = 2.5, 0.7
    b_ref = np.zeros(N)
    b_vec = np.zeros(N)
    ref = _build_two_symbol_kernel(N)
    ref.name = f"two_sym_ref_{N}"
    vec = _build_two_symbol_kernel(N)
    vec.name = f"two_sym_vec_{N}"
    VectorizeCPUMultiDim(widths=(8, ), target_isa="SCALAR").apply_pass(vec, {})
    ref.compile()(B=b_ref, SHIFT=SHIFT, SCALE=SCALE)
    vec.compile()(B=b_vec, SHIFT=SHIFT, SCALE=SCALE)
    np.testing.assert_allclose(b_vec, b_ref, rtol=1e-12, atol=1e-12)


# ---- structural tests -----------------------------------------------------


def test_symbol_invariant_does_not_materialise_a_tile():
    """The data-independent symbol path emits a TileBinop with kind=Symbol on the
    Symbol operand AND no per-lane materialised tile transient appears in the body."""
    sdfg = _build_add_symbol_kernel(8)
    VectorizeCPUMultiDim(widths=(8, ), target_isa="SCALAR", expand_tile_nodes=False).apply_pass(sdfg, {})
    # Walk the body NSDFG and check no `_sym_tile`-prefixed transient appears.
    body_nsdfgs = [n for s in sdfg.states() for n in s.nodes() if isinstance(n, dace.nodes.NestedSDFG)]
    assert len(body_nsdfgs) >= 1
    inner = body_nsdfgs[0].sdfg
    sym_tile_arrays = [n for n in inner.arrays if n.startswith("_sym_tile")]
    assert sym_tile_arrays == [], \
        f"data-independent symbol should NOT materialise a per-lane tile, found: {sym_tile_arrays}"
    # And the TileBinop should have kind_b=Symbol.
    body_state = list(inner.states())[0]
    binop = next(n for n in body_state.nodes() if isinstance(n, TileBinop))
    assert binop.kind_b == "Symbol", f"expected kind_b=Symbol for data-independent path, got {binop.kind_b}"


def test_symbol_lane_id_materialises_a_tile():
    """The lane-id-dependent path materialises a per-lane tile and the lib node
    reads from it as a Tile operand."""
    sdfg = _build_lane_id_kernel(8)
    VectorizeCPUMultiDim(widths=(8, ), target_isa="SCALAR", expand_tile_nodes=False).apply_pass(sdfg, {})
    body_nsdfgs = [n for s in sdfg.states() for n in s.nodes() if isinstance(n, dace.nodes.NestedSDFG)]
    inner = body_nsdfgs[0].sdfg
    sym_tile_arrays = [n for n in inner.arrays if n.startswith("_sym_tile") or n.startswith("_idx_")]
    assert len(sym_tile_arrays) >= 1, \
        f"lane-id symbol should materialise a per-lane tile; arrays={list(inner.arrays.keys())}"
    body_state = list(inner.states())[0]
    binop = next(n for n in body_state.nodes() if isinstance(n, TileBinop))
    assert binop.kind_b == "Tile", f"expected kind_b=Tile (materialised lane-id), got {binop.kind_b}"
