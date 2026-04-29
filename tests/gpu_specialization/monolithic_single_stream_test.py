# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Tests for :class:`MonolithicSingleStreamGPUScheduler`.

The monolithic strategy is the right fit for all-on-GPU stencils: every
kernel runs on stream 0, and the only places that need an explicit
``cudaStreamSynchronize`` are the host transfer boundaries (one state-end
sync per state with a ``CopyLibraryNode``, plus a trailing sync at any
program-sink state without one).

For a typical "copy-in / iterate / copy-out" pattern the result is **two**
sync tasklets total: one after the H2D copy state and one at the program
exit (which is also the D2H state).
"""
import dace
import numpy as np
import pytest

from dace.codegen import common
from dace.transformation.auto.auto_optimize import auto_optimize
from dace.transformation.passes.gpu_specialization.gpu_specialization_pipeline import GPUStreamPipeline
from dace.transformation.passes.gpu_specialization.gpu_stream_scheduling import MonolithicSingleStreamGPUScheduler

N = dace.symbol('N')


@dace.program
def jacobi_2d(TSTEPS: dace.int32, A: dace.float32[N, N], B: dace.float32[N, N]):
    for _ in range(1, TSTEPS):
        B[1:-1, 1:-1] = 0.2 * (A[1:-1, 1:-1] + A[1:-1, :-2] + A[1:-1, 2:] + A[2:, 1:-1] + A[:-2, 1:-1])
        A[1:-1, 1:-1] = 0.2 * (B[1:-1, 1:-1] + B[1:-1, :-2] + B[1:-1, 2:] + B[2:, 1:-1] + B[:-2, 1:-1])


@dace.program
def heat_3d(TSTEPS: dace.int64, A: dace.float64[N, N, N], B: dace.float64[N, N, N]):
    for _ in range(1, TSTEPS):
        B[1:-1, 1:-1,
          1:-1] = (0.125 * (A[2:, 1:-1, 1:-1] - 2.0 * A[1:-1, 1:-1, 1:-1] + A[:-2, 1:-1, 1:-1]) + 0.125 *
                   (A[1:-1, 2:, 1:-1] - 2.0 * A[1:-1, 1:-1, 1:-1] + A[1:-1, :-2, 1:-1]) + 0.125 *
                   (A[1:-1, 1:-1, 2:] - 2.0 * A[1:-1, 1:-1, 1:-1] + A[1:-1, 1:-1, 0:-2]) + A[1:-1, 1:-1, 1:-1])
        A[1:-1, 1:-1,
          1:-1] = (0.125 * (B[2:, 1:-1, 1:-1] - 2.0 * B[1:-1, 1:-1, 1:-1] + B[:-2, 1:-1, 1:-1]) + 0.125 *
                   (B[1:-1, 2:, 1:-1] - 2.0 * B[1:-1, 1:-1, 1:-1] + B[1:-1, :-2, 1:-1]) + 0.125 *
                   (B[1:-1, 1:-1, 2:] - 2.0 * B[1:-1, 1:-1, 1:-1] + B[1:-1, 1:-1, 0:-2]) + B[1:-1, 1:-1, 1:-1])


def _count_sync_tasklets(sdfg):
    """Count sync tasklets across the whole SDFG hierarchy."""
    backend = common.get_gpu_backend()
    needle = f"{backend}StreamSynchronize("
    count = 0
    for nsdfg in sdfg.all_sdfgs_recursive():
        for state in nsdfg.states():
            for node in state.nodes():
                if isinstance(node, dace.nodes.Tasklet) and needle in node.code.as_string:
                    count += 1
    return count


def _build_gpu_sdfg(program, *, monolithic: bool):
    """to_sdfg → auto_optimize for GPU → run the requested stream pipeline."""
    sdfg = program.to_sdfg()
    sdfg = auto_optimize(sdfg, dace.dtypes.DeviceType.GPU)
    strategy = MonolithicSingleStreamGPUScheduler() if monolithic else None
    GPUStreamPipeline(scheduling_strategy=strategy).apply_pass(sdfg, {})
    return sdfg


@pytest.mark.gpu
@pytest.mark.new_gpu_codegen_only
def test_monolithic_jacobi_2d_two_syncs_and_correctness():
    TSTEPS, n_val = 20, 30
    rng = np.random.default_rng(0)
    A = rng.standard_normal((n_val, n_val), dtype=np.float32)
    B = rng.standard_normal((n_val, n_val), dtype=np.float32)
    A_ref, B_ref = A.copy(), B.copy()

    sdfg = _build_gpu_sdfg(jacobi_2d, monolithic=True)
    sync_count = _count_sync_tasklets(sdfg)
    assert sync_count == 2, (f"Monolithic jacobi_2d should produce exactly 2 sync tasklets "
                             f"(one after the H2D copy state, one at program exit); got {sync_count}.")

    A_gpu, B_gpu = A.copy(), B.copy()
    sdfg(A=A_gpu, B=B_gpu, TSTEPS=TSTEPS, N=n_val)

    jacobi_2d.f(TSTEPS, A_ref, B_ref)
    assert np.allclose(A_gpu, A_ref, rtol=1e-5, atol=1e-6)
    assert np.allclose(B_gpu, B_ref, rtol=1e-5, atol=1e-6)


@pytest.mark.gpu
@pytest.mark.new_gpu_codegen_only
def test_monolithic_heat_3d_two_syncs_and_correctness():
    TSTEPS, n_val = 20, 10
    rng = np.random.default_rng(0)
    A = rng.standard_normal((n_val, n_val, n_val), dtype=np.float64)
    B = A.copy()
    A_ref, B_ref = A.copy(), B.copy()

    sdfg = _build_gpu_sdfg(heat_3d, monolithic=True)
    sync_count = _count_sync_tasklets(sdfg)
    assert sync_count == 2, (f"Monolithic heat_3d should produce exactly 2 sync tasklets "
                             f"(one after the H2D copy state, one at program exit); got {sync_count}.")

    A_gpu, B_gpu = A.copy(), B.copy()
    sdfg(A=A_gpu, B=B_gpu, TSTEPS=TSTEPS, N=n_val)

    heat_3d.f(TSTEPS, A_ref, B_ref)
    assert np.allclose(A_gpu, A_ref, rtol=1e-10, atol=1e-12)
    assert np.allclose(B_gpu, B_ref, rtol=1e-10, atol=1e-12)


def test_monolithic_strategy_rejects_cpu_only_program():
    """The strategy must crash on a CPU-only SDFG — it's opted into explicitly."""

    @dace.program
    def add_cpu(A: dace.float32[16], B: dace.float32[16], C: dace.float32[16]):
        for i in dace.map[0:16]:
            C[i] = A[i] + B[i]

    sdfg = add_cpu.to_sdfg()  # CPU only, no GPU transformations.
    with pytest.raises(ValueError, match="MonolithicSingleStreamGPUScheduler requires every"):
        GPUStreamPipeline(scheduling_strategy=MonolithicSingleStreamGPUScheduler()).apply_pass(sdfg, {})
