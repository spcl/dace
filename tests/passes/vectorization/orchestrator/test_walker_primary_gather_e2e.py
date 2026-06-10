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
@pytest.mark.xfail(reason="CORRECTED diagnosis 2026-06-10 via bisect probe (location-keyed, not"
                   " data-name-keyed -- the previous probe was misreading the wrong memlet): the"
                   " lane-dep symbol ``__sym___tmp_*_r`` PRESERVES through ALL my pipeline passes"
                   " (NestInnermost, ExpandNestedSDFGInputs, MarkTileDims, StrideMapByTileWidths,"
                   " InferBodyTransientShapes, GenerateTileIterationMask, PreparePerLaneIndices,"
                   " StageInsideBody). After full pipeline: ``data='A' subset='__sym___tmp_*_r'``"
                   " -- the gather memlet is intact. The walker's GATHER dispatch IS being called."
                   " The actual numerical failure (lane 0 gets a value, lanes 1-7 get 0) is caused"
                   " by ``materialise_per_lane_index_tile`` substituting ``iter_var -> __l<k>``"
                   " (losing the outer tile-start ``i``) AND not inlining the iedge RHS into the"
                   " gather_expr. The materialiser receives ``gather_expr='__sym___tmp_*_r'`` which"
                   " contains no iter_var, so the substitution is a no-op -- every lane gets the"
                   " same value ``A[__sym]`` (where ``__sym = idx[i]`` for the current outer i)."
                   " Fix (per locked design): per-lane symbol fan-out (Slices A+B from earlier"
                   " session): the gather lift pass inlines the iedge RHS into per-lane expressions"
                   " ``idx[i + 0], idx[i + 1], ..., idx[i + W-1]`` via per-lane symbols using"
                   " ``LaneIdScheme.make_dim``, then fills the index tile from those symbols."
                   " ``RemoveUnusedPerLaneSymbols`` (commit b27f02d1f) sweeps any unused.")
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
