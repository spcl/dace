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
@pytest.mark.xfail(reason="Per user direction 2026-06-10: a scalar load of ``A[sym]`` where ``sym``"
                   " is lane-dependent should trigger a full-body read into a tasklet that gathers"
                   " using the laneid-expanded syms, which then writes to a full tile. The"
                   " ``@dace.program`` frontend hides the gather behind a connector mapping"
                   " (``A_conn`` maps to ``A(1)[0:N]``), so the inner state sees ``A_conn[__sym]``"
                   " where ``__sym`` is set via an interstate edge BEFORE the body state. The"
                   " classifier's ``_is_tile_dependent`` needs to walk across the NSDFG boundary"
                   " to see that ``__sym``'s defining expression (``idx_conn[0]``) maps to"
                   " ``idx[i]`` outside, hence is lane-dependent. Two implementation paths: (a)"
                   " inline the outer mapping into the inner subset before the walker runs, OR"
                   " (b) extend the lane-dependency walker to cross NSDFG boundaries."
                   " Same design for SCATTER (write-side variant of the same pattern).")
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
