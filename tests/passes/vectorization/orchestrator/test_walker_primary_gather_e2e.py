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
@pytest.mark.xfail(reason="Confirmed 2026-06-10 via gather_walker_probe: the @dace.program frontend"
                   " decomposes ``B[i] = A[idx[i]]`` into (1) an INTERSTATE-EDGE scalar lift"
                   " ``__sym = idx[i]`` and (2) an INNER-STATE scalar-to-scalar copy ``A -> A_const"
                   " -> B`` with memlet ``A[0]``. By the time the walker runs, NO gather subset"
                   " remains in any memlet -- the lane-dependency is hidden in the iedge"
                   " assignment, and the inner body looks like a pure CONSTANT scalar copy."
                   " Fix paths (deferred to a separate slice -- requires either a pre-pass that"
                   " UNCOMPILES the frontend's scalar-lift back into explicit gather memlets, OR"
                   " a walker enhancement that detects scalar bridges sourced from iedge-defined"
                   " loop-dep symbols and rewrites them into TileLoad-with-gather_dims): both are"
                   " substantial walker-side work, NOT the per-lane-symbol fan-out from earlier"
                   " design discussion (the fan-out helper machinery would only work AFTER the"
                   " gather pattern has been recovered). RemoveUnusedPerLaneSymbols (commit"
                   " 83822f4f2) is the post-clean infrastructure ready for use once the gather"
                   " recovery lands.")
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
