# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""State-boundary synchronization of the GPU streams a state actually used.

A stream-unaware GPU callback is pinned to the legacy default stream. DaCe's own streams are
``StreamNonBlocking``, which opts them out of that stream's implicit serialization, so it needs
synchronizing at the end of a state like any other -- otherwise the host reads the callback's output
as soon as ``__program_*`` returns, whatever the device is still doing.

These assert on emitted code, so they need a GPU for neither compilation nor a run.
"""
import re
import warnings

import numpy as np

import dace


def dace_inhibitor(f):
    """Keeps the frontend from inlining the function, so it becomes a callback tasklet."""
    return f


@dace_inhibitor
def scale_on_its_own_stream(arr):
    arr *= 2


def generated_code_for(program, *args) -> str:
    """Every generated file for ``program`` after GPU transformation, joined."""
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        sdfg = program.to_sdfg(*args)
        sdfg.apply_gpu_transformations()
        return '\n'.join(code.clean_code for code in sdfg.generate_code())


def callback_code() -> str:
    """A GPU array handed to a callback that never names ``__dace_current_stream``."""

    @dace.program
    def gpucallback(A: dace.float64[20]):
        tmp = dace.ndarray([20], dace.float64, storage=dace.StorageType.GPU_Global)
        tmp[:] = A
        scale_on_its_own_stream(tmp)
        A[:] = tmp

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        sdfg = gpucallback.to_sdfg()
        return '\n'.join(code.clean_code for code in sdfg.generate_code())


def test_the_default_stream_is_synchronized_at_the_end_of_its_state():
    """Nothing else orders it: DaCe's streams are non-blocking, and the host returns right after."""
    code = callback_code()
    assert re.search(
        r'StreamSynchronize\(nullptr\)',
        code), ('the state pinned to the default stream ends without synchronizing it, so the host reads the '
                'callback output while the device may still be writing it')


def test_the_synchronization_follows_the_data_movement_it_guards():
    """A sync emitted before the copy it is meant to cover orders nothing."""
    code = callback_code()
    last_copy = max(m.start() for m in re.finditer(r'MemcpyAsync\([^;]*nullptr\)', code))
    sync = re.search(r'StreamSynchronize\(nullptr\)', code)
    assert sync and sync.start() > last_copy, 'the default-stream synchronization precedes the last copy on it'


def test_the_default_stream_is_never_rendered_as_an_array_index():
    """``nullptr`` names the stream directly; the context's array holds only the created streams."""
    code = callback_code()
    bad = re.findall(r'\w+\[nullptr\]', code)
    assert not bad, f'the default stream is used to index an array, which does not compile: {bad}'


def test_a_state_on_a_created_stream_still_synchronizes_that_stream():
    """The default stream is an addition, not a replacement -- the indexed streams still sync."""

    @dace.program
    def doubler(a: dace.float64[64], b: dace.float64[64]):
        b[:] = a * 2

    code = generated_code_for(doubler, np.zeros(64), np.zeros(64))
    assert re.search(r'StreamSynchronize\(__state->gpu_context->streams\[\d+\]\)',
                     code), ('a state whose work runs on a created stream no longer synchronizes it')


if __name__ == '__main__':
    test_the_default_stream_is_synchronized_at_the_end_of_its_state()
    test_the_synchronization_follows_the_data_movement_it_guards()
    test_the_default_stream_is_never_rendered_as_an_array_index()
    test_a_state_on_a_created_stream_still_synchronizes_that_stream()
