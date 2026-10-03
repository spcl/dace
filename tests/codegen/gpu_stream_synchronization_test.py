# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""State-boundary synchronization of the GPU streams a state actually used.

The stream a stream-unaware GPU callback's surrounding work runs on -- the legacy target pins it to
the default stream, the experimental one issues it on a created stream -- has to be synchronized at
the end of its state. DaCe's created streams are ``StreamNonBlocking``, which opts them out of the
default stream's implicit serialization, so nothing else orders them: otherwise the host reads the
callback's output as soon as ``__program_*`` returns, whatever the device is still doing.

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
    enqueues = list(re.finditer(r'Async\(', code))
    assert enqueues, 'no asynchronous work was enqueued, so this test is anchored on nothing'
    syncs = list(re.finditer(r'StreamSynchronize\(', code))
    assert syncs and syncs[-1].start() > enqueues[-1].start(), (
        'the state ends with asynchronous work still in flight, so the host reads the '
        'callback output while the device may still be writing it')


def test_the_synchronization_fences_the_callback_rather_than_preceding_it():
    """A sync emitted before the work it guards orders nothing.

    Anchored on the callback, the one thing both simplification settings emit alike: with
    ``optimizer.automatic_simplification`` off the callback gets its own state and the copies around
    it sit on a created stream, so there is no null-stream copy here to anchor on. The callback's
    output leaves the device through an asynchronous copy, so the fence is the LAST synchronization:
    it has to follow the callback, with nothing queued on any stream after it.
    """
    code = callback_code()
    callback = re.search(r'scale_on_its_own_stream\([A-Za-z_]\w*\);', code)
    assert callback, 'the callback tasklet was not emitted, so this test is anchored on nothing'
    syncs = list(re.finditer(r'StreamSynchronize\(', code))
    assert syncs, 'no stream synchronization was emitted'
    assert syncs[-1].start() > callback.start(), 'every stream synchronization precedes the callback it guards'
    assert not re.search(r'Async\(', code[syncs[-1].end():]), \
        'work is queued on a stream after the last synchronization, so nothing fences it'


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
    # The experimental target reads the created streams through a local alias of the context array.
    assert re.search(
        r'StreamSynchronize\((?:__state->gpu_context->streams\[\d+\]|gpu_streams\[gpu_streams_idx\(\d+\)\])\)',
        code), ('a state whose work runs on a created stream no longer synchronizes it')


if __name__ == '__main__':
    test_the_default_stream_is_synchronized_at_the_end_of_its_state()
    test_the_synchronization_fences_the_callback_rather_than_preceding_it()
    test_the_default_stream_is_never_rendered_as_an_array_index()
    test_a_state_on_a_created_stream_still_synchronizes_that_stream()
