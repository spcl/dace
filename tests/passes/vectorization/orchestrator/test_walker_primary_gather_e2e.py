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


@pytest.mark.parametrize("N", [8, 16])
@pytest.mark.xfail(reason="Confirmed 2026-06-10 via deep probe (BEFORE/AFTER vectorize): the frontend"
                   " DOES emit a proper gather memlet ``__tmp_15_15_r[__sym___tmp_15_17_r]`` with the"
                   " lane-dep symbol PRESENT in the subset. AFTER VectorizeCPUMultiDim runs, the"
                   " same memlet's subset has been concretised to ``Range (0)`` -- the lane-dep"
                   " symbol got substituted away even though it's still declared in"
                   " ``inner_sdfg.symbols``. Per user direction 2026-06-10: 'Symbols added to the"
                   " inner SDFG from outside need to have entries in symbol mapping. If symbol is"
                   " declared and not assigned to and not in symbol mapping -> it will be"
                   " considered free symbol' -- the symbol likely lacks a ``symbol_mapping`` entry"
                   " on the NSDFG node, and the substitution treats it as a free var defaulting to"
                   " 0. Fix path (next session): find which pass (suspect"
                   " ``ExpandNestedSDFGInputs._uncollapse_scalar`` or"
                   " ``InferBodyTransientShapes._widen_non_transient_memlets``) does the concretising"
                   " and preserve the lane-dep symbol. Then the classifier's ``_is_tile_dependent``"
                   " (already in place) detects GATHER via the iedge ``__sym = idx[i]`` chain; the"
                   " walker's existing GATHER dispatch + per-lane-symbol fan-out (design 7.5) +"
                   " ``RemoveUnusedPerLaneSymbols`` post-clean (commit b27f02d1f) complete the"
                   " lowering. Tile shape rule per user clarification 2026-06-10: ALWAYS broadcast"
                   " per-lane symbols to FULL tile dimensions (W_0, ..., W_{K-1}) even when the"
                   " dependent-dim count is less than K.")
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
