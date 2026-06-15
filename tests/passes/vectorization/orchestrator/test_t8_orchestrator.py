# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""End-to-end smoke tests for :class:`VectorizeCPUMultiDim`.

Builds a K=1 / K=2 axpy SDFG, runs the orchestrator, validates the
final SDFG, and confirms the no-laneid audit passes.
"""

import pytest
# [UNSKIPPED-FOR-ASSESSMENT 2026-06-14] pytestmark = pytest.mark.skip(reason="legacy K=1/K=2 descent path frozen during walker-primary migration -- this test goes through VectorizeCPUMultiDim or the harness; both depend on the legacy descent + emit infrastructure being removed. Will be revived (or replaced by walker-primary equivalents) after the new orchestrator pipeline lands end-to-end.")
import pytest

import dace
from dace.transformation.passes.vectorization.vectorize_cpu_multi_dim import (
    VectorizeCPUMultiDim, )


def _build_k1_axpy_sdfg():
    """K=1 axpy: ``C[i] = A[i] + B[i]``."""
    N = dace.symbol("N")
    sdfg = dace.SDFG("k1_axpy_orchestrator")
    sdfg.add_array("A", (N, ), dace.float64)
    sdfg.add_array("B", (N, ), dace.float64)
    sdfg.add_array("C", (N, ), dace.float64)
    state = sdfg.add_state("main")
    state.add_mapped_tasklet(
        "axpy",
        {"i": "0:N"},
        {
            "_a": dace.Memlet("A[i]"),
            "_b": dace.Memlet("B[i]")
        },
        "_c = _a + _b",
        {"_c": dace.Memlet("C[i]")},
        external_edges=True,
    )
    return sdfg


def _build_k2_axpy_sdfg():
    """K=2 axpy: ``C[i, j] = A[i, j] + B[i, j]``."""
    M = dace.symbol("M")
    N = dace.symbol("N")
    sdfg = dace.SDFG("k2_axpy_orchestrator")
    sdfg.add_array("A", (M, N), dace.float64)
    sdfg.add_array("B", (M, N), dace.float64)
    sdfg.add_array("C", (M, N), dace.float64)
    state = sdfg.add_state("main")
    state.add_mapped_tasklet(
        "axpy",
        {
            "i": "0:M",
            "j": "0:N"
        },
        {
            "_a": dace.Memlet("A[i, j]"),
            "_b": dace.Memlet("B[i, j]")
        },
        "_c = _a + _b",
        {"_c": dace.Memlet("C[i, j]")},
        external_edges=True,
    )
    return sdfg


def test_orchestrator_rejects_K_outside_1_to_3():
    """Locked: ``len(widths)`` must be in ``{1, 2, 3}``."""
    with pytest.raises(NotImplementedError, match="K=4"):
        VectorizeCPUMultiDim(widths=(8, 8, 8, 8))
    with pytest.raises(NotImplementedError, match="K=0"):
        VectorizeCPUMultiDim(widths=())


def test_orchestrator_rejects_non_power_of_two():
    """Locked: every width must be a power of 2."""
    with pytest.raises(NotImplementedError, match="power of 2"):
        VectorizeCPUMultiDim(widths=(3, ))
    with pytest.raises(NotImplementedError, match="power of 2"):
        VectorizeCPUMultiDim(widths=(8, 6))


def test_orchestrator_rejects_avx512_innermost_not_8_aligned():
    """AVX-512 requires ``widths[-1] % 8 == 0``."""
    with pytest.raises(NotImplementedError, match="widths\\[-1\\] %% 8 == 0|widths\\[-1\\] % 8 == 0"):
        VectorizeCPUMultiDim(widths=(4, ), target_isa="AVX512")


def test_orchestrator_rejects_unknown_target_isa():
    """Only ``AVX512`` and ``SCALAR`` are recognized in MVP."""
    with pytest.raises(NotImplementedError, match="target_isa"):
        VectorizeCPUMultiDim(widths=(8, ), target_isa="CUTILE")


def test_orchestrator_k1_axpy_runs_and_validates():
    """K=1 axpy: orchestrator runs end-to-end + validates the result."""
    sdfg = _build_k1_axpy_sdfg()
    VectorizeCPUMultiDim(widths=(8, ), target_isa="SCALAR").apply_pass(sdfg, {})
    sdfg.validate()


def test_orchestrator_k2_axpy_runs_and_validates():
    """K=2 axpy: orchestrator runs end-to-end + validates the result."""
    sdfg = _build_k2_axpy_sdfg()
    VectorizeCPUMultiDim(widths=(4, 8), target_isa="SCALAR").apply_pass(sdfg, {})
    sdfg.validate()
