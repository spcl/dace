# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""GPU_Events on nodes and control flow regions inside GPU kernels: an event per thread block, on the GPU's global
timer."""

import numpy as np
import pytest

import dace
from dace.sdfg import nodes
from dace.sdfg.state import NamedRegion

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


def instrumented(label: str) -> tuple[dace.SDFG, nodes.NestedSDFG]:
    """``rows`` with a nested SDFG instrumented before simplification: the body of the map (``rows``), which stays a
    nested SDFG in the scope of the map, or the call of ``twice`` at the top level of that body, which is inlined into
    it as a region."""
    sdfg = rows.to_sdfg(simplify=False)
    node = next(n for n, _ in sdfg.all_nodes_recursive() if isinstance(n, nodes.NestedSDFG) and n.label.endswith(label))
    node.instrument = dace.InstrumentationType.GPU_Events
    sdfg.simplify()
    return sdfg, node


def test_inlining_keeps_instrumented_nested_sdfg_in_a_scope():
    sdfg, body = instrumented("test_rows")
    assert any(n is body for n, _ in sdfg.all_nodes_recursive())
    assert body.instrument == dace.InstrumentationType.GPU_Events


def test_inlining_turns_an_instrumented_call_into_a_region():
    sdfg, call = instrumented("test_twice")
    assert not any(n is call for n, _ in sdfg.all_nodes_recursive())
    # (Its label ends with that of the call: inlining the region of the call statement prefixes it)
    (region,) = [r for r in sdfg.all_control_flow_regions(recursive=True) if isinstance(r, NamedRegion)]
    assert region.label.endswith(call.label) and region.instrument == dace.InstrumentationType.GPU_Events


def check_event_per_thread_block(sdfg: dace.SDFG, name: str):
    import cupy

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
        assert name.startswith(e.name)  # (Names in reports are truncated)
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
def test_event_per_thread_block():
    sdfg, body = instrumented("test_rows")
    check_event_per_thread_block(sdfg, body.label)


@pytest.mark.gpu
def test_event_per_thread_block_of_a_region():
    """The inlined call, a region in the kernel, measured as the nested SDFG would be."""
    sdfg, _ = instrumented("test_twice")
    (region,) = [r for r in sdfg.all_control_flow_regions(recursive=True) if isinstance(r, NamedRegion)]
    check_event_per_thread_block(sdfg, region.label)


@pytest.mark.gpu
def test_per_thread_node_is_rejected():
    sdfg = rows.to_sdfg(simplify=True)
    tasklet = next(n for n, _ in sdfg.all_nodes_recursive() if isinstance(n, nodes.Tasklet))
    tasklet.instrument = dace.InstrumentationType.GPU_Events  # Inside the thread-block map: runs per thread
    with pytest.raises(TypeError, match="threads of a block run together"):
        sdfg.generate_code()


if __name__ == "__main__":
    test_inlining_keeps_instrumented_nested_sdfg_in_a_scope()
    test_inlining_turns_an_instrumented_call_into_a_region()
    test_event_per_thread_block()
    test_event_per_thread_block_of_a_region()
    test_per_thread_node_is_rejected()
