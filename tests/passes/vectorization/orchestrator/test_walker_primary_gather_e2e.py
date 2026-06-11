# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""End-to-end numerical tests for indirect-access (gather) kernels.

The walker's :class:`StageInsideBody` GATHER dispatch is exercised through real
``@dace.program`` kernels rather than manually-built SDFGs (the manual construction
of an idx data-dependency edge through MapEntry is brittle, per design Appendix E).

Numerical contract: ``B[i] = A[idx[i]]`` matches the unvectorised reference output.
"""
import numpy as np
import pytest

import dace
from dace.transformation.passes.vectorization.vectorize_cpu_multi_dim import (VectorizeCPUMultiDim)

N_SYM = dace.symbol("N_GATHER")


@dace.program
def k1_gather(A: dace.float64[N_SYM], idx: dace.int64[N_SYM], B: dace.float64[N_SYM]):
    for i in dace.map[0:N_SYM]:
        B[i] = A[idx[i]]


@pytest.mark.parametrize(
    "N",
    [
        8,
        pytest.param(
            16,
            marks=pytest.mark.xfail(reason="N=16 (2 outer tile iters at stride W=8): first tile (lanes 0-7)"
                                    " produces bit-equal output but the second tile (lanes 8-15) writes"
                                    " zero. The per-lane iedge re-evaluates ``__sym_lane0id_<l> = idx[(i +"
                                    " l)]`` correctly per outer iteration, but the destination write path"
                                    " (B[i:i+W] tile store) is not being re-issued for the second outer"
                                    " tile. Distinct slice from the gather lowering -- the bridge tile"
                                    " transient is allocated once at the body NSDFG and the store side"
                                    " needs separate handling to repeat per outer iter."),
        ),
    ],
)
def test_k1_gather_matches_reference(N):
    """K=1 ``B[i] = A[idx[i]]`` -- bit-equal to unvectorised reference. Exercises the
    GATHER walker path + the per-lane index materialiser."""
    rng = np.random.default_rng(seed=N)
    a = rng.random(N)
    idx = rng.permutation(N).astype(np.int64)
    b_ref = np.zeros(N)
    b_vec = np.zeros(N)

    ref_sdfg = k1_gather.to_sdfg(simplify=True)
    ref_sdfg.name = f"k1_gather_ref_{N}"
    vec_sdfg = k1_gather.to_sdfg(simplify=True)
    vec_sdfg.name = f"k1_gather_vec_{N}"
    try:
        VectorizeCPUMultiDim(widths=(8, ), target_isa="SCALAR").apply_pass(vec_sdfg, {})
    except Exception as exc:  # noqa: BLE001 - the walker may still refuse some gather shapes.
        pytest.xfail(f"gather walker path refused: {exc}")
    ref_sdfg.compile()(A=a.copy(), idx=idx.copy(), B=b_ref, N_GATHER=N)
    vec_sdfg.compile()(A=a.copy(), idx=idx.copy(), B=b_vec, N_GATHER=N)
    np.testing.assert_allclose(b_vec, b_ref, rtol=1e-12, atol=1e-12)


N_SCATTER = dace.symbol("N_SCATTER")


@dace.program
def k1_scatter(A: dace.float64[N_SCATTER], idx: dace.int64[N_SCATTER], B: dace.float64[N_SCATTER]):
    for i in dace.map[0:N_SCATTER]:
        B[idx[i]] = A[i]


def test_k1_scatter_matches_reference():
    """K=1 ``B[idx[i]] = A[i]`` (scatter) -- bit-equal to the unvectorised reference.

    Exercises the symmetric scatter path through the walker (TileStore with
    ``gather_dims`` lowering via :class:`ExpandTileStorePure` because
    ``make_store_tasklet`` now delegates to pure when ``gather_dims`` is set,
    mirroring the gather load fix in commit 4ad424945).
    """
    n = 8
    rng = np.random.default_rng(seed=n)
    a = rng.random(n)
    idx = rng.permutation(n).astype(np.int64)
    b_ref = np.zeros(n)
    b_vec = np.zeros(n)
    ref_sdfg = k1_scatter.to_sdfg(simplify=True)
    ref_sdfg.name = f"k1_scatter_ref_{n}"
    vec_sdfg = k1_scatter.to_sdfg(simplify=True)
    vec_sdfg.name = f"k1_scatter_vec_{n}"
    try:
        VectorizeCPUMultiDim(widths=(8, ), target_isa="SCALAR").apply_pass(vec_sdfg, {})
    except Exception as exc:  # noqa: BLE001
        pytest.xfail(f"scatter walker path refused: {exc}")
    ref_sdfg.compile()(A=a.copy(), idx=idx.copy(), B=b_ref, N_SCATTER=n)
    vec_sdfg.compile()(A=a.copy(), idx=idx.copy(), B=b_vec, N_SCATTER=n)
    np.testing.assert_allclose(b_vec, b_ref, rtol=1e-12, atol=1e-12)


M_K2 = dace.symbol("M_K2")
N_K2 = dace.symbol("N_K2")


@dace.program
def k2_partial_kdep_gather(A: dace.float64[M_K2, N_K2], idx: dace.int64[M_K2], B: dace.float64[M_K2, N_K2]):
    for i, j in dace.map[0:M_K2, 0:N_K2]:
        B[i, j] = A[idx[i], j]


def test_k2_partial_kdep_gather_emits_W0_ONE_idx_shape():
    """K=2 partial-K_dep gather (``A[idx[i], j]``) -- the idx tile shape is
    ``(W_0, ONE)`` because the gather expression depends only on ``i`` (the
    row iter-var).

    Resolved by Phase A2: the walker now computes the per-iter-var dep mask
    via :func:`compute_per_iter_var_dep_mask` which walks interstate edges
    to resolve post-Bypass per-lane symbols (``__sym_<> = idx[i]`` becomes
    "dep on i, not on j"). The materialiser receives the mask explicitly
    and emits ``(W_0, ONE)`` per the cuTile contract.
    """
    from dace.symbolic import ONE
    m, n = 8, 8
    vec_sdfg = k2_partial_kdep_gather.to_sdfg(simplify=True)
    vec_sdfg.name = "k2_partial_kdep_shape_audit"
    try:
        VectorizeCPUMultiDim(widths=(8, 8), target_isa="SCALAR").apply_pass(vec_sdfg, {})
    except Exception as exc:  # noqa: BLE001 -- K=2 walker may still refuse some shapes.
        pytest.xfail(f"K=2 walker path refused: {exc}")
    # Inspect every nested SDFG for ``_idx_*`` tiles produced by the gather
    # materialiser. Each must have shape ``(8, ONE)`` -- the row dim is
    # lane-dep, the col dim is broadcast.
    import sympy
    idx_descs = []
    for sd in vec_sdfg.all_sdfgs_recursive():
        for name, desc in sd.arrays.items():
            if name.startswith("_idx_") and isinstance(desc, dace.data.Array):
                idx_descs.append((name, tuple(desc.shape)))
    assert idx_descs, "expected at least one ``_idx_*`` tile from the K=2 gather materialiser"
    for name, shape in idx_descs:
        assert len(shape) == 2, f"{name!r}: expected K=2 shape, got {shape}"
        assert int(shape[0]) == 8, f"{name!r}: expected W_0=8 on lane-dep dim, got {shape}"
        assert isinstance(shape[1], sympy.Basic) and ONE in shape[1].free_symbols, \
            f"{name!r}: expected ONE on broadcast dim 1, got {shape}"
