# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Instrumentation of control flow regions, and of nested SDFGs that inlining turns into regions (see also
``gpu_kernel_instrumentation_test.py`` for regions in GPU kernels)."""

import numpy as np
import pytest

import dace
from dace.sdfg import nodes
from dace.sdfg.state import LoopRegion, NamedRegion

N = 16
rng = np.random.default_rng(0)


@dace.program
def scale(a: dace.float64[N], b: dace.float64[N]):
    for i in range(N):
        b[i] = 2 * a[i]
    b[0] += 1


@dace.program
def calls_scale(a: dace.float64[N], b: dace.float64[N]):
    scale(a, b)


def instrumented_call(kind: dace.InstrumentationType) -> tuple[dace.SDFG, str]:
    """``calls_scale`` with the call of ``scale`` instrumented, simplified (which inlines it)."""
    sdfg = calls_scale.to_sdfg(simplify=False)
    call = next(n for n, _ in sdfg.all_nodes_recursive() if isinstance(n, nodes.NestedSDFG))
    call.instrument = kind
    sdfg.simplify()
    return sdfg, call.label


def run(sdfg: dace.SDFG) -> dace.codegen.instrumentation.InstrumentationReport:
    a, b = rng.random(N), np.zeros(N)
    sdfg(a=a, b=b)
    expected = 2 * a
    expected[0] += 1
    assert np.allclose(b, expected)
    return sdfg.get_latest_report()


def test_instrumented_call_is_inlined_as_a_region():
    sdfg, label = instrumented_call(dace.InstrumentationType.Timer)
    assert not any(isinstance(n, nodes.NestedSDFG) for n, _ in sdfg.all_nodes_recursive())
    # (Its label ends with that of the call: inlining the region of the call statement prefixes it)
    (region,) = [r for r in sdfg.all_control_flow_regions() if isinstance(r, NamedRegion)]
    assert region.label.endswith(label) and region.instrument == dace.InstrumentationType.Timer
    assert any(isinstance(r, LoopRegion) for r in region.all_control_flow_regions())

    (event,) = run(sdfg).events
    assert f"NamedRegion {region.label}".startswith(event.name)  # (Names in reports are truncated)


def test_instrumentation_of_a_region_is_serialized():
    sdfg, label = instrumented_call(dace.InstrumentationType.Timer)
    loaded = dace.SDFG.from_json(sdfg.to_json())
    (region,) = [r for r in loaded.all_control_flow_regions() if isinstance(r, NamedRegion)]
    assert region.instrument == dace.InstrumentationType.Timer


def test_timer_on_a_loop():
    sdfg = scale.to_sdfg()
    (loop,) = [r for r in sdfg.all_control_flow_regions() if isinstance(r, LoopRegion)]
    loop.instrument = dace.InstrumentationType.Timer
    report = run(sdfg)
    assert [e.name for e in report.events] == [f"LoopRegion {loop.label}"]


@pytest.mark.gpu
def test_gpu_events_on_a_region_in_host_code():
    """Events on the GPU stream around the kernels of the region, as for a state."""
    sdfg, _ = instrumented_call(dace.InstrumentationType.GPU_Events)
    sdfg.apply_gpu_transformations()
    (region,) = [r for r in sdfg.all_control_flow_regions() if isinstance(r, NamedRegion)]
    (event,) = run(sdfg).events
    assert f"NamedRegion {region.label}".startswith(event.name)  # (Names in reports are truncated)


if __name__ == "__main__":
    test_instrumented_call_is_inlined_as_a_region()
    test_instrumentation_of_a_region_is_serialized()
    test_timer_on_a_loop()
    test_gpu_events_on_a_region_in_host_code()
