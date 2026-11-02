# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""GPU_Events on nodes inside GPU kernels: an event per thread block, on the GPU's global timer."""

import numpy as np
import pytest

import dace
from dace.sdfg import nodes

N = 8  # Thread blocks
W = 64  # Threads per block

GPU_Global = dace.StorageType.GPU_Global
TB = dace.ScheduleType.GPU_ThreadBlock


@dace.program
def twice(a: dace.float32[W], b: dace.float32[W]):
    for j in dace.map[0:W] @ TB:
        b[j] = 2 * a[j]


@dace.program
def rows(a: dace.float32[N, W] @ GPU_Global, b: dace.float32[N, W] @ GPU_Global):
    for i in dace.map[0:N] @ dace.ScheduleType.GPU_Device:
        twice(a[i], b[i])


def instrumented_rows() -> tuple[dace.SDFG, nodes.NestedSDFG]:
    """``rows`` with the nested program of every thread block instrumented (before simplification, which keeps it)."""
    sdfg = rows.to_sdfg(simplify=False)
    task = next(n for n, _ in sdfg.all_nodes_recursive() if isinstance(n, nodes.NestedSDFG) and "twice" in n.label)
    task.instrument = dace.InstrumentationType.GPU_Events
    sdfg.simplify()
    return sdfg, task


def test_inlining_keeps_instrumented_nested_sdfg():
    sdfg, task = instrumented_rows()
    assert any(n is task for n, _ in sdfg.all_nodes_recursive())


@pytest.mark.gpu
def test_event_per_thread_block():
    import cupy

    sdfg, task = instrumented_rows()
    a = cupy.random.rand(N, W).astype(np.float32)
    b = cupy.zeros((N, W), dtype=np.float32)
    sdfg(a=a, b=b)
    assert cupy.allclose(b, 2 * a)

    report = sdfg.get_latest_report()
    events = [e for e in report.events if e.additional_info and "block" in e.additional_info]
    assert len(events) == N
    assert sorted(e.additional_info["block"] for e in events) == list(range(N))
    assert sorted(e.tid for e in events) == list(range(N))  # A thread of the trace per thread block
    for e in events:
        assert e.name == task.label
        assert e.duration >= 0
        assert e.additional_info["cycles"] > 0
        assert e.additional_info["sm"] >= 0
    # The device events are on the host's time axis: within a minute of the host's events
    starts = [e.timestamp for e in events]
    assert max(starts) - min(starts) < 60e6
    # Summarized over all thread blocks
    (times,) = report.durations[events[0].uuid].values()
    assert list(times.keys()) == [-1] and len(times[-1]) == N


@pytest.mark.gpu
def test_per_thread_node_is_rejected():
    sdfg = rows.to_sdfg(simplify=True)
    tasklet = next(n for n, _ in sdfg.all_nodes_recursive() if isinstance(n, nodes.Tasklet))
    tasklet.instrument = dace.InstrumentationType.GPU_Events  # Inside the thread-block map: runs per thread
    with pytest.raises(TypeError, match="threads of a block run together"):
        sdfg.generate_code()


if __name__ == "__main__":
    test_inlining_keeps_instrumented_nested_sdfg()
    test_event_per_thread_block()
    test_per_thread_node_is_rejected()
