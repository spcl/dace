# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Tests for :class:`AutoGPUStreamScheduler`.

The strategy is the new default for the GPU stream pipeline. It classifies each top-level node
in the SDFG hierarchy as CPU / GPU / MIXED, pins every GPU consumer to stream 0 when no MIXED
node exists, and splices a one-tasklet sync state between any GPU state and a CPU successor
(or any successor reached via an iedge whose condition / assignment reads a GPU-written
array). If any top-level node classifies as MIXED, the strategy falls back to
:class:`PerComponentGPUStreamScheduler` and emits a ``UserWarning``.

Tests build SDFGs through the Python frontend, run ``apply_gpu_transformations``, then push them
through the pipeline so we can inspect the *real* shape the strategy sees in production.
"""

import warnings

import numpy as np
import pytest

import dace
from dace.codegen import common
from dace.transformation.auto.auto_optimize import auto_optimize
from dace.transformation.passes.gpu_specialization.gpu_specialization_pipeline import GPUStreamPipeline
from dace.transformation.passes.gpu_specialization.gpu_stream_scheduling import (
    AutoGPUStreamScheduler,
    SingleStreamGPUScheduler,
)
from dace.transformation.passes.gpu_specialization.helpers.gpu_helpers import get_gpu_stream_array_name

N = dace.symbol("N")

_STREAM_ARRAY = get_gpu_stream_array_name()


def _sync_tasklets(state):
    backend = common.get_gpu_backend()
    needle = f"{backend}StreamSynchronize("
    return [n for n in state.nodes() if isinstance(n, dace.nodes.Tasklet) and needle in n.code.as_string]


def _all_sync_states(sdfg):
    return [(nsdfg, state) for nsdfg in sdfg.all_sdfgs_recursive() for state in nsdfg.states() if _sync_tasklets(state)]


def _build_gpu_sdfg(program, *, strategy=None):
    sdfg = program.to_sdfg()
    sdfg = auto_optimize(sdfg, dace.dtypes.DeviceType.GPU)
    GPUStreamPipeline(scheduling_strategy=strategy).apply_pass(sdfg, {})
    return sdfg


# Default-strategy contract.


def test_default_strategy_is_auto_single_stream():
    """``GPUStreamPipeline()`` selects :class:`AutoGPUStreamScheduler`."""
    pipe = GPUStreamPipeline()
    assert isinstance(pipe._scheduling_strategy, AutoGPUStreamScheduler)


# Pure GPU programs -- single stream, sync state at the program exit only.


# Pure CPU program -- no streams allocated, no sync states.


# CPU -> GPU transition needs no sync state; GPU -> CPU needs one.


def mixed_host_gpu_sdfg() -> dace.SDFG:
    """A NestedSDFG whose one state holds a free host tasklet next to a GPU kernel, so it classifies as MIXED."""
    # Built by hand: the frontend won't produce a free host tasklet next to a GPU kernel in one body.
    outer = dace.SDFG("outer_mixed")
    outer.add_array("A", [16], dace.float32, storage=dace.dtypes.StorageType.GPU_Global)
    outer.add_array("B", [16], dace.float32, storage=dace.dtypes.StorageType.GPU_Global)
    outer.add_array("C", [1], dace.float32, storage=dace.dtypes.StorageType.GPU_Global)
    state = outer.add_state("main")

    inner = dace.SDFG("inner_mixed")
    inner.add_array("a", [16], dace.float32, storage=dace.dtypes.StorageType.GPU_Global)
    inner.add_array("b", [16], dace.float32, storage=dace.dtypes.StorageType.GPU_Global)
    inner.add_array("c", [1], dace.float32, storage=dace.dtypes.StorageType.GPU_Global)
    istate = inner.add_state("mixed")
    # Host-side bare tasklet (no GPU map ancestor inside the inner SDFG).
    t = istate.add_tasklet(
        "host_tasklet", {}, {"__out": dace.pointer(dace.float32)}, "__out = 0.0", language=dace.Language.CPP
    )
    cw = istate.add_write("c")
    istate.add_edge(t, "__out", cw, None, dace.Memlet("c[0]"))
    # GPU_Device map alongside it.
    me, mx = istate.add_map("gpu_map", dict(i="0:16"), schedule=dace.dtypes.ScheduleType.GPU_Device)
    kt = istate.add_tasklet("kernel_body", {"_a": dace.float32}, {"_b": dace.float32}, "_b = _a + 1.0")
    ar = istate.add_read("a")
    bw = istate.add_write("b")
    istate.add_memlet_path(ar, me, kt, dst_conn="_a", memlet=dace.Memlet("a[i]"))
    istate.add_memlet_path(kt, mx, bw, src_conn="_b", memlet=dace.Memlet("b[i]"))

    nsdfg_node = state.add_nested_sdfg(inner, {"a": None, "b": None}, {"c": None, "b": None}, symbol_mapping={})
    state.add_edge(state.add_read("A"), None, nsdfg_node, "a", dace.Memlet("A[0:16]"))
    state.add_edge(state.add_read("B"), None, nsdfg_node, "b", dace.Memlet("B[0:16]"))
    state.add_edge(nsdfg_node, "b", state.add_write("B"), None, dace.Memlet("B[0:16]"))
    state.add_edge(nsdfg_node, "c", state.add_write("C"), None, dace.Memlet("C[0:1]"))
    return outer


def test_mixed_program_fallback_to_per_component_emits_warning():
    """A MIXED NestedSDFG makes the auto strategy warn and delegate to :class:`PerComponentGPUStreamScheduler`."""
    outer = mixed_host_gpu_sdfg()
    strategy = AutoGPUStreamScheduler()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        GPUStreamPipeline(scheduling_strategy=strategy).apply_pass(outer, {})

    fallback_msgs = [
        w for w in caught if "AutoGPUStreamScheduler" in str(w.message) and "falling back" in str(w.message)
    ]
    assert fallback_msgs, (
        "Expected an AutoGPUStreamScheduler fallback warning when a "
        f"NestedSDFG mixes CPU + GPU work. Got: {[str(w.message) for w in caught]}"
    )
    # Per-component wiring landed: gpu_streams allocated.
    assert _STREAM_ARRAY in outer.arrays


def test_single_stream_strategy_rejects_a_mixed_program():
    with pytest.raises(ValueError, match="mix host and GPU work"):
        GPUStreamPipeline(scheduling_strategy=SingleStreamGPUScheduler()).apply_pass(mixed_host_gpu_sdfg(), {})


# GPU -> GPU with no host iedge work: no sync state spliced between consecutive GPU states.


# End-to-end: split + auto strategy run through the full codegen pipeline; compile + run.


@pytest.mark.gpu
@pytest.mark.new_gpu_codegen_only
def test_e2e_cpu_init_feeds_gpu_kernel_via_split():
    """CPU computes a scalar (``s = x + 1``), GPU map uses it. After the codegen pipeline runs
    its preprocess + scheduler chain, the resulting SDFG must compile and produce the right
    numerical result. The split pass is responsible for moving the CPU init into its own state
    so the scheduler doesn't fall back to per-component streams.
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
    assert np.allclose(B, expected, rtol=1e-5, atol=1e-6), f"B mismatch -- got {B[:4]}, expected {expected[:4]}"


# ICON-stencil patterns (from the compute_rho_theta legacy-vs-experimental investigation).
# A real ICON stencil compiles to a single SDFG with many GPU maps writing GPU transients
# (the ``gtir_tmp`` pool) consumed downstream. The scheduler pins everything to stream 0 and
# adds the host-blocking ``cudaStreamSynchronize`` exactly once (the region sink) -- regardless
# of kernel/transient count, since stream-0 ordering covers every GPU->GPU hand-off. That single
# per-SDFG sink sync is correct but conservative: it serializes the host across many back-to-back
# stencil calls.


@dace.program
def stencil_chain_with_transients(A: dace.float64[N], OUT: dace.float64[N]):
    # Neighbour-offset reads keep the stages as separate kernels (resist map fusion).
    t1 = np.zeros_like(A)
    t2 = np.zeros_like(A)
    for i in dace.map[1 : N - 1]:
        t1[i] = A[i - 1] + 2.0 * A[i] + A[i + 1]
    for i in dace.map[1 : N - 1]:
        t2[i] = t1[i - 1] - t1[i + 1]
    for i in dace.map[1 : N - 1]:
        OUT[i] = 0.5 * t2[i] + A[i]


@dace.program
def deep_stencil_pipeline(A: dace.float64[N], OUT: dace.float64[N]):
    t1 = np.zeros_like(A)
    t2 = np.zeros_like(A)
    t3 = np.zeros_like(A)
    t4 = np.zeros_like(A)
    t5 = np.zeros_like(A)
    for i in dace.map[1 : N - 1]:
        t1[i] = A[i - 1] + A[i + 1]
    for i in dace.map[1 : N - 1]:
        t2[i] = t1[i - 1] + t1[i + 1]
    for i in dace.map[1 : N - 1]:
        t3[i] = t2[i - 1] + t2[i + 1]
    for i in dace.map[1 : N - 1]:
        t4[i] = t3[i - 1] + t3[i + 1]
    for i in dace.map[1 : N - 1]:
        t5[i] = t4[i - 1] + t4[i + 1]
    for i in dace.map[1 : N - 1]:
        OUT[i] = t5[i - 1] + t5[i + 1]


def test_sync_count_independent_of_kernel_count():
    """Pass-structure only (no GPU needed): a deeper GPU pipeline must still get exactly one
    program-end sync. Codifies that the per-SDFG sink sync count does not grow with the number
    of kernels -- the over-synchronization is per-SDFG, not per-kernel."""
    sdfg = _build_gpu_sdfg(deep_stencil_pipeline)
    gpu_device_maps = [
        n
        for nsdfg in sdfg.all_sdfgs_recursive()
        for s in nsdfg.states()
        for n in s.nodes()
        if isinstance(n, dace.nodes.MapEntry) and n.map.schedule == dace.dtypes.ScheduleType.GPU_Device
    ]
    assert len(gpu_device_maps) >= 3, f"expected a deep multi-kernel pipeline, got {len(gpu_device_maps)}"
    syncs = _all_sync_states(sdfg)
    assert len(syncs) == 1, f"sync count must be independent of kernel count (expected 1, got {len(syncs)})"


# compiler.cuda.synchronize_on_exit -- drop the per-SDFG exit sync for GPU-resident pipelines.
# The exit sync is what serializes the host across back-to-back stencil calls; for a host app that
# shares one GPU stream and syncs at its own boundaries (e.g. icon4py) it is pure overhead. Opting
# out removes it for sinks whose outputs stay GPU-resident, while sinks that write host-visible
# (numpy / copy-out) outputs are always synchronized so correctness is preserved.


def _build_gpu_resident_pipeline(strategy=None):
    """Hand-built pure-GPU pipeline with GPU_Global, non-transient I/O (no copy-out) -- the
    icon4py shape where outputs stay on the device."""
    GPU = dace.dtypes.StorageType.GPU_Global
    sdfg = dace.SDFG("gpu_resident_pipeline")
    for nm in ("A", "B", "C"):
        sdfg.add_array(nm, [16], dace.float32, storage=GPU)
    s0 = sdfg.add_state("k0")
    me0, mx0 = s0.add_map("m0", dict(i="0:16"), schedule=dace.dtypes.ScheduleType.GPU_Device)
    t0 = s0.add_tasklet("b0", {"a": None}, {"o": None}, "o = a * 2.0")
    s0.add_memlet_path(s0.add_read("A"), me0, t0, dst_conn="a", memlet=dace.Memlet("A[i]"))
    s0.add_memlet_path(t0, mx0, s0.add_write("B"), src_conn="o", memlet=dace.Memlet("B[i]"))
    s1 = sdfg.add_state_after(s0, "k1")
    me1, mx1 = s1.add_map("m1", dict(i="0:16"), schedule=dace.dtypes.ScheduleType.GPU_Device)
    t1 = s1.add_tasklet("b1", {"a": None}, {"o": None}, "o = a + 1.0")
    s1.add_memlet_path(s1.add_read("B"), me1, t1, dst_conn="a", memlet=dace.Memlet("B[i]"))
    s1.add_memlet_path(t1, mx1, s1.add_write("C"), src_conn="o", memlet=dace.Memlet("C[i]"))
    GPUStreamPipeline(scheduling_strategy=strategy or AutoGPUStreamScheduler()).apply_pass(sdfg, {})
    return sdfg


def test_gpu_resident_sink_synced_by_default():
    """Default (synchronize_on_exit=True): a GPU-resident sink still gets its exit sync."""
    sdfg = _build_gpu_resident_pipeline()
    assert len(_all_sync_states(sdfg)) == 1, "default must keep the exit sync"


def test_gpu_resident_sink_exit_sync_dropped_when_opted_out():
    """synchronize_on_exit=False: a GPU-resident sink (no host-visible output) gets NO exit sync;
    stream-0 ordering still covers the GPU->GPU hand-off. This removes the per-stencil host stall."""
    with dace.config.set_temporary("compiler", "cuda", "synchronize_on_exit", value=False):
        sdfg = _build_gpu_resident_pipeline()
    assert len(_all_sync_states(sdfg)) == 0, "opted-out GPU-resident sink must get no exit sync"
    streams = {
        n.gpu_stream_id
        for nsdfg in sdfg.all_sdfgs_recursive()
        for s in nsdfg.states()
        for n in s.nodes()
        if n.gpu_stream_id is not None
    }
    assert streams == {0}, f"streams must still be wired for ordering, got {streams}"


@pytest.mark.gpu
@pytest.mark.new_gpu_codegen_only
def test_host_visible_output_always_synced_even_when_opted_out():
    """Safety: even with synchronize_on_exit=False, a sink that writes a host (numpy) output keeps
    its sync, so copy-out SDFGs stay correct."""
    n_val = 128
    with dace.config.set_temporary("compiler", "cuda", "synchronize_on_exit", value=False):
        sdfg = _build_gpu_sdfg(stencil_chain_with_transients)
        assert _all_sync_states(sdfg), "host-visible (numpy) output must still be synchronized"
        rng = np.random.default_rng(5)
        A = rng.standard_normal(n_val)
        OUT = np.zeros(n_val)
        sdfg(A=A.copy(), OUT=OUT, N=n_val)
    t1 = np.zeros(n_val)
    t2 = np.zeros(n_val)
    ref = np.zeros(n_val)
    t1[1:-1] = A[:-2] + 2.0 * A[1:-1] + A[2:]
    t2[1:-1] = t1[:-2] - t1[2:]
    ref[1:-1] = 0.5 * t2[1:-1] + A[1:-1]
    assert np.allclose(OUT, ref, rtol=1e-9, atol=1e-12), f"numerical mismatch: {OUT[:4]} vs {ref[:4]}"


if __name__ == "__main__":
    test_default_strategy_is_auto_single_stream()
    test_mixed_program_fallback_to_per_component_emits_warning()
    test_single_stream_strategy_rejects_a_mixed_program()
    test_sync_count_independent_of_kernel_count()
    test_gpu_resident_sink_synced_by_default()
    test_gpu_resident_sink_exit_sync_dropped_when_opted_out()
    test_e2e_cpu_init_feeds_gpu_kernel_via_split()
    test_host_visible_output_always_synced_even_when_opted_out()
