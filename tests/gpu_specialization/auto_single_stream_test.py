# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Tests for :class:`AutoSingleStreamGPUScheduler`.

The strategy is the new default for the GPU stream pipeline. It classifies each top-level node
in the SDFG hierarchy as CPU / GPU / MIXED, pins every GPU consumer to stream 0 when no MIXED
node exists, and splices a one-tasklet sync state between any GPU state and a CPU successor
(or any successor reached via an iedge whose condition / assignment reads a GPU-written
array). If any top-level node classifies as MIXED, the strategy falls back to
:class:`NaiveGPUStreamScheduler` and emits a ``UserWarning``.

Tests build SDFGs through the Python frontend, run ``apply_gpu_transformations``, then push them
through the pipeline so we can inspect the *real* shape the strategy sees in production.
"""
import warnings

import dace
import numpy as np
import pytest

from dace.codegen import common
from dace.transformation.auto.auto_optimize import auto_optimize
from dace.transformation.passes.gpu_specialization.gpu_specialization_pipeline import GPUStreamPipeline
from dace.transformation.passes.gpu_specialization.gpu_stream_scheduling import (AutoSingleStreamGPUScheduler)
from dace.transformation.passes.gpu_specialization.helpers.gpu_helpers import get_gpu_stream_array_name

N = dace.symbol('N')

_STREAM_ARRAY = get_gpu_stream_array_name()


def _sync_tasklets(state):
    backend = common.get_gpu_backend()
    needle = f"{backend}StreamSynchronize("
    return [n for n in state.nodes() if isinstance(n, dace.nodes.Tasklet) and needle in n.code.as_string]


def _all_sync_states(sdfg):
    out = []
    for nsdfg in sdfg.all_sdfgs_recursive():
        for state in nsdfg.states():
            if _sync_tasklets(state):
                out.append((nsdfg, state))
    return out


def _build_gpu_sdfg(program, *, strategy=None):
    sdfg = program.to_sdfg()
    sdfg = auto_optimize(sdfg, dace.dtypes.DeviceType.GPU)
    GPUStreamPipeline(scheduling_strategy=strategy).apply_pass(sdfg, {})
    return sdfg


# Default-strategy contract.


def test_default_strategy_is_auto_single_stream():
    """``GPUStreamPipeline()`` selects :class:`AutoSingleStreamGPUScheduler`."""
    pipe = GPUStreamPipeline()
    assert isinstance(pipe._scheduling_strategy, AutoSingleStreamGPUScheduler)


# Pure GPU programs -- single stream, sync state at the program exit only.


@dace.program
def jacobi_2d(TSTEPS: dace.int32, A: dace.float32[N, N], B: dace.float32[N, N]):
    for _ in range(1, TSTEPS):
        B[1:-1, 1:-1] = 0.2 * (A[1:-1, 1:-1] + A[1:-1, :-2] + A[1:-1, 2:] + A[2:, 1:-1] + A[:-2, 1:-1])
        A[1:-1, 1:-1] = 0.2 * (B[1:-1, 1:-1] + B[1:-1, :-2] + B[1:-1, 2:] + B[2:, 1:-1] + B[:-2, 1:-1])


@pytest.mark.gpu
@pytest.mark.new_gpu_codegen_only
def test_pure_gpu_jacobi_2d_one_stream_sync_at_exit():
    """Pure GPU SDFG: every kernel pinned to stream 0; one sync state appended at the
    region-level sink; numerical match with the CPU reference."""
    TSTEPS, n_val = 8, 16
    rng = np.random.default_rng(0)
    A = rng.standard_normal((n_val, n_val), dtype=np.float32)
    B = rng.standard_normal((n_val, n_val), dtype=np.float32)
    A_ref, B_ref = A.copy(), B.copy()

    sdfg = _build_gpu_sdfg(jacobi_2d)

    # Every map-entry / libnode / runtime tasklet consumer carries stream 0.
    streams_seen = {
        n.gpu_stream_id
        for nsdfg in sdfg.all_sdfgs_recursive()
        for state in nsdfg.states()
        for n in state.nodes() if n.gpu_stream_id is not None
    }
    assert streams_seen == {0}, f"Expected single stream {{0}}, got {streams_seen}"

    # At least one sync state inserted (program-end sink).
    sync_locations = _all_sync_states(sdfg)
    assert sync_locations, "AutoSingleStreamGPUScheduler should append at least one sync state"

    # Numerical correctness.
    A_gpu, B_gpu = A.copy(), B.copy()
    sdfg(A=A_gpu, B=B_gpu, TSTEPS=TSTEPS, N=n_val)
    jacobi_2d.f(TSTEPS, A_ref, B_ref)
    assert np.allclose(A_gpu, A_ref, rtol=1e-5, atol=1e-6)
    assert np.allclose(B_gpu, B_ref, rtol=1e-5, atol=1e-6)


# Pure CPU program -- no streams allocated, no sync states.


def test_pure_cpu_program_no_streams():
    """A CPU-only SDFG must not allocate the ``gpu_streams`` array and must not insert sync
    states. The classifier sees only CPU work, so the assignment dict is empty and wiring
    short-circuits."""

    @dace.program
    def cpu_add(A: dace.float32[16], B: dace.float32[16], C: dace.float32[16]):
        for i in dace.map[0:16] @ dace.dtypes.ScheduleType.Sequential:
            C[i] = A[i] + B[i]

    sdfg = cpu_add.to_sdfg()
    # No apply_gpu_transformations -- the program stays on host.
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        GPUStreamPipeline().apply_pass(sdfg, {})

    # The strategy must not warn about its own fallback (no MIXED nodes here).
    fallback = [
        w for w in caught if 'AutoSingleStreamGPUScheduler' in str(w.message) and 'falling back' in str(w.message)
    ]
    assert not fallback, f"Strategy must not fall back on pure-CPU input. Got: {[str(w.message) for w in caught]}"

    # No sync states, no stream consumers (no GPU work to wire).
    assert not _all_sync_states(sdfg), "no sync states must be emitted for pure-CPU input"
    streams_seen = {
        n.gpu_stream_id
        for nsdfg in sdfg.all_sdfgs_recursive()
        for state in nsdfg.states()
        for n in state.nodes() if n.gpu_stream_id is not None
    }
    assert not streams_seen, f"no GPU consumers expected, got assignments: {streams_seen}"


# CPU -> GPU transition needs no sync state; GPU -> CPU needs one.


def test_mixed_program_fallback_to_naive_emits_warning():
    """A NestedSDFG that interleaves a free host tasklet with a GPU kernel inside *the same*
    NestedSDFG (no inner GPU_Device map wrapping the tasklet) classifies as MIXED. The strategy
    must warn and delegate to :class:`NaiveGPUStreamScheduler`."""

    # Construct a NestedSDFG with one host-side tasklet and one GPU_Device map. Built by hand
    # because the frontend won't produce a free host tasklet next to a GPU kernel in one body.
    outer = dace.SDFG('outer_mixed')
    outer.add_array('A', [16], dace.float32, storage=dace.dtypes.StorageType.GPU_Global)
    outer.add_array('B', [16], dace.float32, storage=dace.dtypes.StorageType.GPU_Global)
    outer.add_array('C', [1], dace.float32, storage=dace.dtypes.StorageType.GPU_Global)
    state = outer.add_state('main')

    inner = dace.SDFG('inner_mixed')
    inner.add_array('a', [16], dace.float32, storage=dace.dtypes.StorageType.GPU_Global)
    inner.add_array('b', [16], dace.float32, storage=dace.dtypes.StorageType.GPU_Global)
    inner.add_array('c', [1], dace.float32, storage=dace.dtypes.StorageType.GPU_Global)
    istate = inner.add_state('mixed')
    # Host-side bare tasklet (no GPU map ancestor inside the inner SDFG).
    t = istate.add_tasklet('host_tasklet', {}, {'__out': dace.pointer(dace.float32)},
                           '__out = 0.0',
                           language=dace.Language.CPP)
    cw = istate.add_write('c')
    istate.add_edge(t, '__out', cw, None, dace.Memlet('c[0]'))
    # GPU_Device map alongside it.
    me, mx = istate.add_map('gpu_map', dict(i='0:16'), schedule=dace.dtypes.ScheduleType.GPU_Device)
    kt = istate.add_tasklet('kernel_body', {'_a': dace.float32}, {'_b': dace.float32}, '_b = _a + 1.0')
    ar = istate.add_read('a')
    bw = istate.add_write('b')
    istate.add_memlet_path(ar, me, kt, dst_conn='_a', memlet=dace.Memlet('a[i]'))
    istate.add_memlet_path(kt, mx, bw, src_conn='_b', memlet=dace.Memlet('b[i]'))

    nsdfg_node = state.add_nested_sdfg(inner, {'a', 'b'}, {'c', 'b'}, symbol_mapping={})
    state.add_edge(state.add_read('A'), None, nsdfg_node, 'a', dace.Memlet('A[0:16]'))
    state.add_edge(state.add_read('B'), None, nsdfg_node, 'b', dace.Memlet('B[0:16]'))
    state.add_edge(nsdfg_node, 'b', state.add_write('B'), None, dace.Memlet('B[0:16]'))
    state.add_edge(nsdfg_node, 'c', state.add_write('C'), None, dace.Memlet('C[0:1]'))

    strategy = AutoSingleStreamGPUScheduler()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        GPUStreamPipeline(scheduling_strategy=strategy).apply_pass(outer, {})

    fallback_msgs = [
        w for w in caught if 'AutoSingleStreamGPUScheduler' in str(w.message) and 'falling back' in str(w.message)
    ]
    assert fallback_msgs, ("Expected an AutoSingleStreamGPUScheduler fallback warning when a "
                           f"NestedSDFG mixes CPU + GPU work. Got: {[str(w.message) for w in caught]}")
    # Naive-style wiring landed: gpu_streams allocated.
    assert _STREAM_ARRAY in outer.arrays


# GPU -> GPU with no host iedge work: no sync state spliced between consecutive GPU states.


@dace.program
def two_gpu_kernels_chain(A: dace.float32[N], B: dace.float32[N], C: dace.float32[N]):
    for i in dace.map[0:N]:
        B[i] = A[i] * 2.0
    for i in dace.map[0:N]:
        C[i] = B[i] + 1.0


@pytest.mark.gpu
@pytest.mark.new_gpu_codegen_only
def test_chain_of_gpu_kernels_one_sync_at_exit():
    """Two GPU kernels in sequence with no host-side iedge work: stream 0 ordering is
    enough; expect exactly one sync state (the program-end sync), not one per kernel."""
    n_val = 64
    rng = np.random.default_rng(1)
    A = rng.standard_normal(n_val).astype(np.float32)
    B = np.zeros(n_val, dtype=np.float32)
    C = np.zeros(n_val, dtype=np.float32)

    sdfg = _build_gpu_sdfg(two_gpu_kernels_chain)

    # No more than one sync state per region-level sink.
    sync_locations = _all_sync_states(sdfg)
    assert 1 <= len(sync_locations) <= 2, (f"Expected at most one sync state per GPU sink "
                                           f"in a linear-chain program, got {len(sync_locations)}")

    # Numerical correctness (manual reference, no .f() since N is a symbol the Python
    # frontend doesn't strip from the body).
    sdfg(A=A.copy(), B=B, C=C, N=n_val)
    B_ref = A * 2.0
    C_ref = B_ref + 1.0
    assert np.allclose(C, C_ref, rtol=1e-5, atol=1e-6)


# End-to-end: split + auto strategy run through the full codegen pipeline; compile + run.


@pytest.mark.gpu
@pytest.mark.new_gpu_codegen_only
def test_e2e_cpu_init_feeds_gpu_kernel_via_split():
    """CPU computes a scalar (``s = x + 1``), GPU map uses it. After the codegen pipeline runs
    its preprocess + scheduler chain, the resulting SDFG must compile and produce the right
    numerical result. The split pass is responsible for moving the CPU init into its own state
    so the scheduler doesn't fall back to Naive.
    """

    @dace.program
    def init_then_kernel(A: dace.float32[N], B: dace.float32[N], x: dace.float32):
        s = x + dace.float32(1.0)
        for i in dace.map[0:N]:
            B[i] = A[i] * s

    n_val = 32
    rng = np.random.default_rng(7)
    A = rng.standard_normal(n_val).astype(np.float32)
    x = np.float32(0.5)
    expected = A * (x + np.float32(1.0))

    sdfg = _build_gpu_sdfg(init_then_kernel)
    B = np.zeros(n_val, dtype=np.float32)
    sdfg(A=A.copy(), B=B, x=x, N=n_val)
    assert np.allclose(B, expected, rtol=1e-5, atol=1e-6), (f"B mismatch -- got {B[:4]}, expected {expected[:4]}")


@pytest.mark.gpu
@pytest.mark.new_gpu_codegen_only
def test_e2e_gpu_kernel_writes_scalar_consumed_by_cpu():
    """GPU map writes a scalar via a single-iteration kernel; CPU tasklet then reads the scalar
    and applies a finalize. Tests the suffix lift: the CPU work must land in a trailing state,
    not before the kernel."""

    @dace.program
    def kernel_then_finalize(A: dace.float32[N], out: dace.float32[1]):
        s = np.zeros(1, dtype=np.float32)
        for i in dace.map[0:1]:
            s[0] = A[i] + dace.float32(1.0)
        out[0] = s[0] * dace.float32(2.0)

    n_val = 32
    rng = np.random.default_rng(11)
    A = rng.standard_normal(n_val).astype(np.float32)
    expected = np.float32((A[0] + 1.0) * 2.0)

    sdfg = _build_gpu_sdfg(kernel_then_finalize)
    out = np.zeros(1, dtype=np.float32)
    sdfg(A=A.copy(), out=out, N=n_val)
    assert np.isclose(out[0], expected, rtol=1e-5, atol=1e-5), (f"out[0]={out[0]} expected {expected}")


if __name__ == '__main__':
    import sys
    sys.exit(pytest.main([__file__, '-v']))
