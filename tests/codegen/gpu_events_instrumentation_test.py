# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""``GPU_Events`` instrumentation reads a node's stream from ``Node.gpu_stream_id`` (a map exit takes its entry's)."""

import re
import warnings

import pytest

import dace

N = dace.symbol("N")


@dace.program
def axpy(a: dace.float64, x: dace.float64[N], y: dace.float64[N]):
    y[:] = a * x + y


def instrumented_code() -> str:
    """axpy on the GPU with every state instrumented."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sdfg = axpy.to_sdfg(simplify=True)
        sdfg.specialize({"N": 1024})
        sdfg.apply_gpu_transformations()
        for state in sdfg.all_states():
            state.instrument = dace.InstrumentationType.GPU_Events
        return "\n".join(code.clean_code for code in sdfg.generate_code())


@pytest.mark.new_gpu_codegen_only
def test_gpu_events_instrumentation_resolves_a_real_stream_index():
    """Per-node events (three-part ids) must record on the stream their ``gpu_streams[i]`` edge names.

    State-level events hardcode stream 0, so only the per-node ids isolate the stream lookup.
    """
    # With the default of -1 streams every event records on nullptr, hiding a lookup that resolves to -1.
    with dace.config.set_temporary("compiler", "cuda", "max_concurrent_streams", value=4):
        code = instrumented_code()

    node_streams = {m.group(1) for m in re.finditer(r"EventRecord\(__dace_ev_[be]\d+_\d+_\d+, (\S+)\);", code)}
    assert node_streams, "no per-node EventRecord call was emitted"
    assert all(re.fullmatch(r"__state->gpu_context->streams\[\d+\]", s) for s in node_streams), node_streams


if __name__ == "__main__":
    test_gpu_events_instrumentation_resolves_a_real_stream_index()
