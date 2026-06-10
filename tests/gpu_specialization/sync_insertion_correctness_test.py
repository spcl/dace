# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""End-to-end correctness + sync-count tests for :class:`AutoSingleStreamGPUScheduler`.

Each test defines a small ``@dace.program`` using only ``range(...)`` for host-side
sequential loops and ``dace.map[...]`` (no explicit ``@ScheduleType``) for
data-parallel loops that the GPU transformation is free to offload. The SDFG is built
via the standard path ``to_sdfg -> auto_optimize(DeviceType.GPU) -> GPUStreamPipeline``,
then two contracts are asserted:

1. The pipeline emits the expected number of ``cudaStreamSynchronize`` / ``hipStream
   Synchronize`` tasklets (so a missing or extra sync surfaces immediately, not as
   silent wrong output).
2. Running the SDFG and comparing against the in-Python numpy reference returns
   element-wise-matching outputs.

The patterns target the rules ``AutoSingleStreamGPUScheduler`` claims to handle:

* **Three-state CPU -> GPU -> CPU** -- canonical
  :class:`SplitStateByGPUClass` shape; the host suffix reads the kernel output.
* **Mixed-class single source state** -- a host reduction packed in the same source
  state as the kernel; ``SplitStateByGPUClass`` must lift the prefix out.
* **Two independent parallel components feeding one host reader** -- two parallelisable
  maps writing different transients followed by a host accumulator.
* **Parallel output consumed by a host-side loop** -- the host loop reads a kernel-
  written array.
* **Loop body alternating host init and parallel compute** -- per-iteration host
  init -> parallelisable map -> host accumulate.

Tests require a working GPU runtime -- marked ``gpu`` + ``new_gpu_codegen_only``.
"""
import numpy as np
import pytest

import dace
from dace.codegen import common
from dace.transformation.auto.auto_optimize import auto_optimize
from dace.transformation.passes.gpu_specialization.gpu_specialization_pipeline import GPUStreamPipeline

pytestmark = [pytest.mark.gpu, pytest.mark.new_gpu_codegen_only]

_N = dace.symbol('_N', dtype=dace.int64)


def _count_sync_tasklets(sdfg: dace.SDFG) -> int:
    """Count ``cudaStreamSynchronize`` / ``hipStreamSynchronize`` tasklets across the
    SDFG hierarchy. Mirrors the helper in ``monolithic_single_stream_test.py`` so the
    GPU-backend prefix is resolved through the canonical
    :func:`dace.codegen.common.get_gpu_backend`."""
    backend = common.get_gpu_backend()
    needle = f"{backend}StreamSynchronize("
    count = 0
    for nsdfg in sdfg.all_sdfgs_recursive():
        for state in nsdfg.states():
            for node in state.nodes():
                if isinstance(node, dace.nodes.Tasklet) and needle in node.code.as_string:
                    count += 1
    return count


def _build_gpu_sdfg(program) -> dace.SDFG:
    """``to_sdfg -> auto_optimize(GPU) -> GPUStreamPipeline``.

    Mirrors ``tests/gpu_specialization/monolithic_single_stream_test.py:_build_gpu_sdfg``
    so the build path is identical to what other sync-count tests use.
    """
    sdfg = program.to_sdfg(simplify=True)
    sdfg = auto_optimize(sdfg, dace.dtypes.DeviceType.GPU)
    GPUStreamPipeline().apply_pass(sdfg, {})
    return sdfg


def _run_gpu_and_check(program,
                       *,
                       args: dict,
                       symbols: dict,
                       expected: dict,
                       expected_sync_count: int,
                       rtol: float = 1e-10,
                       atol: float = 1e-12):
    """Build the GPU SDFG, assert the expected sync-tasklet count, run it, and compare
    its output arrays element-wise to a numpy reference.

    :param program: A ``@dace.program``-decorated function.
    :param args: kwargs dict for the call (may be mutated by the run).
    :param symbols: Symbol values for the call.
    :param expected: Mapping of output-arg-name -> expected numpy array after the run.
    :param expected_sync_count: Number of sync tasklets the pipeline must emit.
    :param rtol: relative tolerance for ``np.testing.assert_allclose``.
    :param atol: absolute tolerance.
    """
    sdfg = _build_gpu_sdfg(program)
    got_sync_count = _count_sync_tasklets(sdfg)
    assert got_sync_count == expected_sync_count, (f"Expected {expected_sync_count} sync tasklet(s); "
                                                   f"got {got_sync_count}.")
    sdfg(**args, **symbols)
    for name, want in expected.items():
        np.testing.assert_allclose(args[name], want, rtol=rtol, atol=atol, err_msg=f'arg "{name}" mismatch')


# ---------------------------------------------------------------------------
# Case 1: three-state CPU -> GPU -> CPU
# ---------------------------------------------------------------------------
@dace.program
def _three_state_cpu_gpu_cpu(A: dace.float64[_N], B: dace.float64[_N], C: dace.float64[_N]):
    scale = np.float64(0.0)
    for i in range(_N):
        scale = scale + A[i] * 0.5
    for i in dace.map[0:_N]:
        B[i] = A[i] * scale
    for i in range(_N):
        C[i] = B[i] + 1.0


def test_three_state_cpu_gpu_cpu_chain_has_one_sync_and_matches_numpy():
    """CPU prefix -> GPU kernel -> CPU suffix reading the kernel output. Expect one
    sync tasklet between the GPU state and the CPU suffix (the suffix reads B)."""
    N = 64
    A = np.array(np.linspace(0.0, 1.0, N, dtype=np.float64))
    B = np.zeros(N, dtype=np.float64)
    C = np.zeros(N, dtype=np.float64)

    expected_scale = (A * 0.5).sum()
    expected_B = A * expected_scale
    expected_C = expected_B + 1.0

    _run_gpu_and_check(_three_state_cpu_gpu_cpu,
                       args=dict(A=A, B=B, C=C),
                       symbols=dict(_N=N),
                       expected=dict(B=expected_B, C=expected_C),
                       expected_sync_count=1,
                       rtol=1e-12)


# ---------------------------------------------------------------------------
# Case 2: mixed-class single source state
# ---------------------------------------------------------------------------
@dace.program
def _mixed_class_single_state(A: dace.float64[_N], B: dace.float64[_N]):
    scratch = np.zeros((_N, ), dtype=np.float64)
    for i in range(_N):
        scratch[i] = A[i] - 1.0
    for i in dace.map[0:_N]:
        B[i] = scratch[i] * 2.0


def test_mixed_class_single_state_has_one_sync_and_matches_numpy():
    """Host-side scratch init feeds a parallel-map consumer. ``SplitStateByGPUClass``
    must lift the prefix out; expect one program-end sync (the residual sink is the
    GPU kernel state)."""
    N = 48
    A = np.array(np.linspace(2.0, 4.0, N, dtype=np.float64))
    B = np.zeros(N, dtype=np.float64)
    expected_B = (A - 1.0) * 2.0
    _run_gpu_and_check(_mixed_class_single_state,
                       args=dict(A=A, B=B),
                       symbols=dict(_N=N),
                       expected=dict(B=expected_B),
                       expected_sync_count=1)


# ---------------------------------------------------------------------------
# Case 3: two independent parallel components feeding one host reader
# ---------------------------------------------------------------------------
@dace.program
def _two_parallel_writers_one_host_reader(A: dace.float64[_N], B: dace.float64[_N], C: dace.float64[_N]):
    X = np.empty((_N, ), dtype=np.float64)
    for i in dace.map[0:_N]:
        X[i] = A[i] * 2.0
    Y = np.empty((_N, ), dtype=np.float64)
    for i in dace.map[0:_N]:
        Y[i] = B[i] * 0.5
    for i in range(_N):
        C[i] = X[i] + Y[i]


def test_two_independent_parallel_writers_one_host_reader_has_one_sync_and_matches_numpy():
    """Two parallel kernels feed a single host-side reader. Expect one sync tasklet
    before the host reader (the scheduler folds both kernels' flushes into one)."""
    N = 32
    A = np.array(np.linspace(0.1, 1.0, N, dtype=np.float64))
    B = np.array(np.linspace(0.5, 1.5, N, dtype=np.float64))
    C = np.zeros(N, dtype=np.float64)
    expected_C = A * 2.0 + B * 0.5
    _run_gpu_and_check(_two_parallel_writers_one_host_reader,
                       args=dict(A=A, B=B, C=C),
                       symbols=dict(_N=N),
                       expected=dict(C=expected_C),
                       expected_sync_count=1)


# ---------------------------------------------------------------------------
# Case 4: parallel output consumed by host loop
# ---------------------------------------------------------------------------
@dace.program
def _parallel_output_consumed_by_host_loop(A: dace.float64[_N], B: dace.float64[_N], C: dace.float64[_N]):
    for i in dace.map[0:_N]:
        B[i] = A[i] + 10.0
    for i in range(_N):
        C[i] = B[i] * 0.25


def test_parallel_output_consumed_by_host_loop_has_one_sync_and_matches_numpy():
    """A parallel kernel writes B; a subsequent host loop reads B. Expect one sync
    tasklet between the kernel state and the host loop."""
    N = 24
    A = np.array(np.linspace(0.0, 5.0, N, dtype=np.float64))
    B = np.zeros(N, dtype=np.float64)
    C = np.zeros(N, dtype=np.float64)
    expected_B = A + 10.0
    expected_C = expected_B * 0.25
    _run_gpu_and_check(_parallel_output_consumed_by_host_loop,
                       args=dict(A=A, B=B, C=C),
                       symbols=dict(_N=N),
                       expected=dict(B=expected_B, C=expected_C),
                       expected_sync_count=1)


# ---------------------------------------------------------------------------
# Case 5: loop body alternating host init and parallel compute per iteration
# ---------------------------------------------------------------------------
@dace.program
def _per_iter_host_then_parallel_then_host(A: dace.float64[_N], B: dace.float64[_N]):
    for k in range(4):
        scratch = np.empty((_N, ), dtype=np.float64)
        for i in range(_N):
            scratch[i] = A[i] + np.float64(k)
        gscratch = np.empty((_N, ), dtype=np.float64)
        for i in dace.map[0:_N]:
            gscratch[i] = scratch[i] * np.float64(k + 1)
        for i in range(_N):
            B[i] = B[i] + gscratch[i]


def test_loop_body_alternating_host_parallel_has_one_sync_and_matches_numpy():
    """Loop body does host init -> parallel kernel -> host accumulate. The sync lives
    statically *inside* the loop body (executed every iteration), so expect one sync
    tasklet -- not four."""
    N = 16
    A = np.array(np.linspace(0.0, 2.0, N, dtype=np.float64))
    B = np.zeros(N, dtype=np.float64)
    expected_B = sum((A + np.float64(k)) * np.float64(k + 1) for k in range(4))
    _run_gpu_and_check(_per_iter_host_then_parallel_then_host,
                       args=dict(A=A, B=B),
                       symbols=dict(_N=N),
                       expected=dict(B=expected_B),
                       expected_sync_count=1)


if __name__ == '__main__':
    import sys
    sys.exit(pytest.main([__file__, '-v']))
