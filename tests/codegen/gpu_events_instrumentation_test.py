# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""GPU_Events instrumentation on the experimental GPU codegen path.

``GPUEventProvider._get_gpu_stream`` reads a node's assigned stream off the
``gpu_streams[i]`` in-edge it was wired to by ``stream_lowering_helpers``.
That memlet is built from a bare integer index
(``dace.Memlet(f"gpu_streams[{i}]")``), which the frontend parses into a
single-point ``Range``, not a plain scalar -- handing the ``Range`` itself to
``int()`` raises.

Only ``generate_code()`` is exercised, so this needs no GPU for compilation
nor a run.
"""
import re
import warnings

import dace

N = dace.symbol('N')


@dace.program
def axpy(a: dace.float64, x: dace.float64[N], y: dace.float64[N]):
    y[:] = a * x + y


def instrumented_code() -> str:
    """axpy, GPU-transformed and specialized, every state instrumented with GPU_Events."""
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        sdfg = axpy.to_sdfg(simplify=True)
        sdfg.specialize({'N': 1024})
        sdfg.apply_gpu_transformations()
        for state in sdfg.all_states():
            state.instrument = dace.InstrumentationType.GPU_Events
        return '\n'.join(code.clean_code for code in sdfg.generate_code())


def test_gpu_events_instrumentation_resolves_a_real_stream_index():
    """generate_code() must succeed, and each per-node event -- the ones read off a
    ``gpu_streams[i]`` in-edge via ``_get_gpu_stream`` -- must carry the real index that
    edge names, not a blind ``nullptr`` fallback.

    ``_idstr`` gives a state-level event a 2-part id (``<cfg>_<state>``) and a per-node
    event a 3-part id (``<cfg>_<state>_<node>``); on_state_begin/on_state_end hardcode
    stream 0 for the state-level id regardless of this bug, so matching *any* EventRecord
    would pass even with ``_get_gpu_stream`` broken. Anchoring on the 3-part ids isolates
    the code path this test is actually about.
    """
    # max_concurrent_streams=-1 (the default) always emits nullptr regardless of the
    # resolved index, which would hide a stream lookup that quietly resolves to -1.
    with dace.config.set_temporary('compiler', 'cuda', 'max_concurrent_streams', value=4):
        code = instrumented_code()

    node_records = list(re.finditer(r'EventRecord\(__dace_ev_[be]\d+_\d+_\d+, (\S+)\);', code))
    assert node_records, 'no per-node EventRecord call was emitted, so this test is anchored on nothing'
    node_streams = {m.group(1) for m in node_records}
    assert all(re.fullmatch(r'__state->gpu_context->streams\[\d+\]', s) for s in node_streams), (
        f'a per-node EventRecord fell back to the default stream instead of resolving the real '
        f'gpu_streams[] index its in-edge names: {node_streams}')


if __name__ == '__main__':
    test_gpu_events_instrumentation_resolves_a_real_stream_index()
