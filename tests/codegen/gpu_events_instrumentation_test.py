# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""``GPU_Events`` instrumentation times kernels only, on the stream each kernel is launched on."""

import re
import warnings

import pytest

import dace

N = dace.symbol("N")


@dace.program
def axpy(a: dace.float64, x: dace.float64[N], y: dace.float64[N]):
    y[:] = a * x + y


def instrumented_code() -> str:
    """axpy on the GPU with every kernel instrumented."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sdfg = axpy.to_sdfg(simplify=True)
        sdfg.specialize({"N": 1024})
        sdfg.apply_gpu_transformations()
        for node, _ in sdfg.all_nodes_recursive():
            if isinstance(node, dace.nodes.MapEntry) and node.map.schedule == dace.ScheduleType.GPU_Device:
                node.map.instrument = dace.InstrumentationType.GPU_Events
        return "\n".join(code.clean_code for code in sdfg.generate_code())


@pytest.mark.new_gpu_codegen_only
def test_gpu_events_instrumentation_records_on_the_kernels_stream():
    """A kernel's events record on the stream the kernel is launched on, not on the default stream."""
    # With the default of -1 streams every event would record on nullptr, hiding a stream mismatch.
    with dace.config.set_temporary("compiler", "cuda", "max_concurrent_streams", value=4):
        code = instrumented_code()

    lines = [line.strip() for line in code.splitlines()]
    checked = 0
    for k, line in enumerate(lines):
        begin = re.fullmatch(r"\w+EventRecord\(__dace_ev_b\d+_\d+_\d+, (\S+)\);", line)
        if begin:
            # The launch block that follows binds the kernel's stream.
            bindings = (re.fullmatch(r"gpuStream_t __dace_current_stream = (\S+);", n) for n in lines[k + 1 : k + 4])
            bound = next((b for b in bindings if b), None)
            assert bound and bound.group(1) == begin.group(1), lines[k : k + 4]
            checked += 1
    assert checked == 2, code


@pytest.mark.parametrize("where", ["sdfg", "state", "tasklet", "host_map"])
def test_gpu_events_outside_a_kernel_are_invalid(where):
    """GPU events time kernels only: on an SDFG, a state, a tasklet or a non-kernel map they fail validation."""
    sdfg = axpy.to_sdfg(simplify=True)
    if where == "sdfg":
        sdfg.instrument = dace.InstrumentationType.GPU_Events
    elif where == "state":
        sdfg.start_state.instrument = dace.InstrumentationType.GPU_Events
    else:
        kind = dace.nodes.Tasklet if where == "tasklet" else dace.nodes.MapEntry
        node = next(n for n, _ in sdfg.all_nodes_recursive() if isinstance(n, kind))
        (node.map if where == "host_map" else node).instrument = dace.InstrumentationType.GPU_Events
    with pytest.raises(dace.sdfg.InvalidSDFGError, match="GPU_Events instrumentation times kernels only"):
        sdfg.validate()


def test_gpu_events_on_a_kernel_are_valid():
    sdfg = axpy.to_sdfg(simplify=True)
    sdfg.apply_gpu_transformations()
    for node, _ in sdfg.all_nodes_recursive():
        if isinstance(node, dace.nodes.MapEntry) and node.map.schedule == dace.ScheduleType.GPU_Device:
            node.map.instrument = dace.InstrumentationType.GPU_Events
    sdfg.validate()


def test_the_instrument_hook_refuses_gpu_events_on_states():
    with pytest.raises(ValueError, match="kernels only"):
        with dace.instrument(dace.InstrumentationType.GPU_Events, filter="*", annotate_states=True):
            axpy(a=1.0, x=dace.ndarray([4], dace.float64), y=dace.ndarray([4], dace.float64))


if __name__ == "__main__":
    for where in ("sdfg", "state", "tasklet", "host_map"):
        test_gpu_events_outside_a_kernel_are_invalid(where)
    test_gpu_events_on_a_kernel_are_valid()
    test_gpu_events_instrumentation_records_on_the_kernels_stream()
