# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""End-to-end test: a ``__field_operator_testee`` SDFG round-trips through
serialise/deserialise while still compiling and producing the right result.

Captures the structural skeleton of the original SDFG:

  * top-level state with Tasklets writing Register-storage Scalars, a
    NestedSDFG (the scan body's stand-in), GPU_Global Array transients, and
    copy Tasklets feeding output Arrays;
  * two ConditionalBlocks guarded by a host symbol (``metrics_level >= 10``),
    mirroring the metrics-entry / metrics-exit pattern.

The test compiles the SDFG once (driving the GPU pipeline), saves it to disk
*after* the pipeline has run, reloads it from disk, compiles again, and
finally runs the binary to check the numerical result. The mid-flight
serialise/deserialise is the load-bearing assertion -- per-node
``gpu_stream_id`` and all post-pipeline wiring must survive the round-trip.
"""
import os
import tempfile

import numpy as np
import pytest

import dace
from dace import dtypes
from dace.sdfg.state import ConditionalBlock, ControlFlowRegion
from dace.memlet import Memlet
from dace.properties import CodeBlock

N = dace.symbol('N')
M = dace.symbol('M')
metrics_level = dace.symbol('metrics_level', dace.int64)


@dace.program
def _scan_body(inp_local: dace.float64[M], out_local: dace.float64[M]):
    for i in dace.map[0:M]:
        out_local[i] = inp_local[i] * 2.0 + 1.0


@dace.program
def _field_operator_core(inp: dace.float64[N], out_0: dace.float64[M], out_1: dace.float64[M]):
    """Captures the original SDFG's Map -> NSDFG -> Array data path.

    Two NSDFG-style scans, each writing a GPU_Global transient Array that's
    copied to one of the output Arrays. The frontend produces the Tasklet ->
    Register Scalar pattern naturally via the loop-carried ``tmp_0``/``tmp_1``
    Python locals.
    """
    tmp_0: dace.float64 = 1.0
    tmp_1: dace.float64 = 2.0
    tmp_6 = np.ndarray(M, dtype=np.float64)
    tmp_7 = np.ndarray(M, dtype=np.float64)
    _scan_body(inp[:M], tmp_6)
    _scan_body(inp[:M], tmp_7)
    for i in dace.map[0:M]:
        out_0[i] = tmp_6[i] + tmp_0
        out_1[i] = tmp_7[i] + tmp_1


def _attach_metrics_blocks(sdfg: dace.SDFG) -> None:
    """Add ConditionalBlock metrics_entry / metrics_exit around the body.

    Each block contains one state and is guarded by ``metrics_level >= 10``.
    Mirrors the ``__field_operator_testee`` instrumentation gating pattern.
    """
    sdfg.add_symbol('metrics_level', dace.int64)
    sdfg.add_array('gt_compute_time',
                   shape=(1, ),
                   dtype=dace.float64,
                   storage=dtypes.StorageType.CPU_Heap,
                   transient=False)
    sdfg.add_scalar('gt_start_time', dace.int64, storage=dtypes.StorageType.CPU_Heap, transient=True)

    # The pre-existing body state(s) live under ``sdfg``. Insert two
    # ConditionalBlocks: one as the new start, one after the original sink.
    original_start = sdfg.start_block
    sink_blocks = [b for b in sdfg.nodes() if sdfg.out_degree(b) == 0]
    sink_block = sink_blocks[0] if sink_blocks else original_start

    # Entry conditional becomes the new start of the SDFG.
    entry_region = ConditionalBlock('metrics_entry')
    sdfg.add_node(entry_region, is_start_block=True)
    entry_body = ControlFlowRegion('metrics_entry_body', sdfg=sdfg)
    entry_state = entry_body.add_state('metrics_entry_collect', is_start_block=True)
    entry_region.add_branch(CodeBlock('metrics_level >= 10'), entry_body)
    t_in = entry_state.add_tasklet('gt_start_timer', set(), {'out'}, 'out = 0')
    t_in_an = entry_state.add_access('gt_start_time')
    entry_state.add_edge(t_in, 'out', t_in_an, None, Memlet(data='gt_start_time', subset='0'))

    # Exit conditional.
    exit_region = ConditionalBlock('metrics_exit')
    sdfg.add_node(exit_region)
    exit_body = ControlFlowRegion('metrics_exit_body', sdfg=sdfg)
    exit_state = exit_body.add_state('metrics_exit_collect', is_start_block=True)
    exit_region.add_branch(CodeBlock('metrics_level >= 10'), exit_body)
    t_out = exit_state.add_tasklet('gt_stop_timer', {'start'}, {'out'}, 'out = start - start')
    t_out_an = exit_state.add_access('gt_start_time')
    t_out_dst = exit_state.add_access('gt_compute_time')
    exit_state.add_edge(t_out_an, None, t_out, 'start', Memlet(data='gt_start_time', subset='0'))
    exit_state.add_edge(t_out, 'out', t_out_dst, None, Memlet(data='gt_compute_time', subset='0'))

    sdfg.add_edge(entry_region, original_start, dace.InterstateEdge())
    sdfg.add_edge(sink_block, exit_region, dace.InterstateEdge())


def _build_field_operator_sdfg() -> dace.SDFG:
    sdfg = _field_operator_core.to_sdfg(simplify=False)
    _attach_metrics_blocks(sdfg)
    sdfg.apply_gpu_transformations()
    sdfg.validate()
    return sdfg


def _run(sdfg: dace.SDFG, n: int, m: int, metrics: int):
    """Compile + execute, return ``out_0`` and ``out_1`` for assertion."""
    inp = np.arange(n, dtype=np.float64)
    out_0 = np.zeros(m, dtype=np.float64)
    out_1 = np.zeros(m, dtype=np.float64)
    gt_compute_time = np.zeros(1, dtype=np.float64)
    sdfg(inp=inp, out_0=out_0, out_1=out_1, gt_compute_time=gt_compute_time, N=n, M=m, metrics_level=metrics)
    return out_0, out_1


def _expected(n: int, m: int):
    base = np.arange(n, dtype=np.float64)[:m] * 2.0 + 1.0
    return base + 1.0, base + 2.0


@pytest.mark.gpu
def test_field_operator_compile():
    """Compile-only smoke for the reconstructed field_operator topology."""
    sdfg = _build_field_operator_sdfg()
    sdfg.compile()


@pytest.mark.gpu
def test_field_operator_roundtrip_and_run():
    """Mid-flight save/load: build -> compile (warms pipeline) -> save -> reload -> compile -> run."""
    sdfg = _build_field_operator_sdfg()

    # First compile drives the GPU pipeline -> writes gpu_stream_id properties
    # and lays down the gpu_streams wiring. Both must survive the round-trip.
    sdfg.compile()

    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, 'field_operator_postpipeline.sdfg')
        sdfg.save(path)
        reloaded = dace.SDFG.from_file(path)

    # Re-compile from the loaded SDFG -- the bug we guarded against
    # (No GPU stream assigned to node X) would re-surface here if assignments
    # weren't persisted.
    reloaded.compile()

    n, m = 16, 8
    out_0, out_1 = _run(reloaded, n, m, metrics=0)
    exp_0, exp_1 = _expected(n, m)
    np.testing.assert_allclose(out_0, exp_0)
    np.testing.assert_allclose(out_1, exp_1)
