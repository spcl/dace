# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""
End-to-end coverage for K-elements-per-iteration strided patterns.

Generalises beyond the s127 contiguous case (K=2, stride=2 → bbox 2*W
contiguous). The handler accepts ``bbox = (W-1)*stride + K`` for any
``K >= 1`` and ``stride >= K``:

- ``K=1, stride>1``         (TSVC s1111 single-element scatter — covered elsewhere)
- ``K>1, stride==K``        (TSVC s127 contiguous bbox — covered in tsvc_additional)
- ``K>1, stride>K``         (multi-element scatter/gather with gaps — THIS FILE)

Two patterns tested here:
- **Scatter**: K=2 contiguous writes per iter at stride 4 → bbox = 7*4+2 = 30.
- **Gather**:  K=2 contiguous reads per iter at stride 4 → bbox = 7*4+2 = 30.
"""
import copy
import pytest
import numpy as np

import dace
from dace.transformation.passes.vectorization.vectorize_cpu import VectorizeCPU
from dace.transformation.interstate import LoopToMap


N = dace.symbol("N")


# --------------------------------------------------------------------------
# Scatter: writes K=2 contiguous elements per iter, stride S=4 between iters.
# --------------------------------------------------------------------------

@dace.program
def _scatter_2_at_stride_4(a: dace.float64[N], b: dace.float64[N]):
    """Each iter writes a[4*i] and a[4*i+1] (2 contiguous, gap of 2 to next)."""
    for i in dace.map[0:N // 4]:
        a[4 * i]     = b[i] * 2.0
        a[4 * i + 1] = b[i] * 3.0


@pytest.mark.parametrize("remainder_strategy", ["divides_evenly", "scalar", "masked"])
def test_multi_elem_scatter_K2_stride4(remainder_strategy):
    """K=2 elements per iter, stride=4 → bbox = (W-1)*4 + 2 = 30.

    Emits 2 ``strided_store`` calls (phase 0 writes a[4*k+0] for
    k=0..W-1, phase 1 writes a[4*k+1]). Elements ``a[4*k+2]`` and
    ``a[4*k+3]`` are NOT touched.
    """
    N_val = 64
    a_ref = np.random.RandomState(0).rand(N_val).astype(np.float64)
    b = np.random.RandomState(1).rand(N_val).astype(np.float64)
    a_vec = a_ref.copy()

    sdfg = _scatter_2_at_stride_4.to_sdfg(simplify=False)
    sdfg.simplify(validate=True, validate_all=True)
    sdfg.apply_transformations_repeated(LoopToMap())
    sdfg.simplify()
    sdfg.name = f"scatter2_s4_ref_{remainder_strategy}"
    sdfg.compile()(a=a_ref, b=b, N=N_val)

    vsdfg = copy.deepcopy(sdfg)
    vsdfg.name = f"scatter2_s4_vec_{remainder_strategy}"
    VectorizeCPU(vector_width=8, fail_on_unvectorizable=True,
                 remainder_strategy=remainder_strategy,
                 use_fp_factor=False,
                 branch_normalization=True if remainder_strategy != "masked" else True).apply_pass(vsdfg, {})
    vsdfg.compile()(a=a_vec, b=b, N=N_val)

    diff = np.max(np.abs(a_ref - a_vec))
    assert diff < 1e-12, f"scatter K=2 stride=4 diff = {diff}"


# --------------------------------------------------------------------------
# Gather: reads K=2 contiguous elements per iter, stride S=4 between iters.
# --------------------------------------------------------------------------

@dace.program
def _gather_2_at_stride_4(a: dace.float64[N], b: dace.float64[N]):
    """Each iter reads b[4*i] and b[4*i+1] (2 contiguous from b)."""
    for i in dace.map[0:N // 4]:
        a[i] = b[4 * i] + b[4 * i + 1]


# ``divides_evenly`` for the gather pattern routes through the bare-
# tasklet ``_generate_strided_loads_to_packed_storage`` path, which
# collides the K=2 read tasklets onto a single ``b_packed`` buffer
# (separate pre-existing bug in the bare-tasklet path, NOT in the
# NSDFG-wrapped multi-element handler this file is testing). Restrict
# to the scalar / masked variants where the P1 wrap routes through
# ``_setup_multi_element_strided_inside_nsdfg``.
@pytest.mark.parametrize("remainder_strategy", ["scalar", "masked"])
def test_multi_elem_gather_K2_stride4(remainder_strategy):
    """K=2 reads per iter, stride=4 → bbox = (W-1)*4 + 2 = 30.

    Emits 2 ``strided_load`` calls (phase 0 reads b[4*k+0] for k=0..W-1
    into the W-wide phase-0 buffer, phase 1 reads b[4*k+1]). The body
    then adds the two W-wide buffers and writes to a[i].
    """
    N_val = 64
    a_ref = np.zeros(N_val, dtype=np.float64)
    b = np.random.RandomState(2).rand(N_val).astype(np.float64)
    a_vec = a_ref.copy()

    sdfg = _gather_2_at_stride_4.to_sdfg(simplify=False)
    sdfg.simplify(validate=True, validate_all=True)
    sdfg.apply_transformations_repeated(LoopToMap())
    sdfg.simplify()
    sdfg.name = f"gather2_s4_ref_{remainder_strategy}"
    sdfg.compile()(a=a_ref, b=b, N=N_val)

    vsdfg = copy.deepcopy(sdfg)
    vsdfg.name = f"gather2_s4_vec_{remainder_strategy}"
    VectorizeCPU(vector_width=8, fail_on_unvectorizable=True,
                 remainder_strategy=remainder_strategy,
                 use_fp_factor=False,
                 branch_normalization=True).apply_pass(vsdfg, {})
    vsdfg.compile()(a=a_vec, b=b, N=N_val)

    diff = np.max(np.abs(a_ref - a_vec))
    assert diff < 1e-12, f"gather K=2 stride=4 diff = {diff}"
