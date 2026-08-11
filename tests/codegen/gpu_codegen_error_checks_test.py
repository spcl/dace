# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Generated GPU code has to check what it calls, and has to stop before it uses what failed.

These assert on emitted code, so they need a GPU for neither compilation nor a run.
"""
import re

import dace
from dace import dtypes

BAILOUT = r'if \(__result\)'


def generated_code(sdfg: dace.SDFG) -> str:
    """Every generated file for ``sdfg``, joined."""
    return '\n'.join(code.clean_code for code in sdfg.generate_code())


def init_function(code: str, name: str) -> str:
    """The body of ``__dace_init_<name>``."""
    match = re.search(r'__dace_init_' + re.escape(name) + r'\(.*?\n\}', code, re.S)
    assert match, f'no __dace_init_{name} was emitted, so this test is anchored on nothing'
    return match.group(0)


def persistent_gpu_transient() -> dace.SDFG:
    """A persistent GPU transient, whose allocation is hoisted into the init function."""
    sdfg = dace.SDFG('persistent_gpu_transient')
    sdfg.add_array('A', [20], dace.float64, storage=dtypes.StorageType.GPU_Global)
    sdfg.add_array('B', [20], dace.float64, storage=dtypes.StorageType.GPU_Global)
    sdfg.add_transient('T', [20],
                       dace.float64,
                       storage=dtypes.StorageType.GPU_Global,
                       lifetime=dtypes.AllocationLifetime.Persistent)

    state = sdfg.add_state('main')
    a = state.add_access('A')
    t = state.add_access('T')
    b = state.add_access('B')
    state.add_nedge(a, t, dace.Memlet('A[0:20]'))
    state.add_nedge(t, b, dace.Memlet('T[0:20]'))
    sdfg.validate()
    return sdfg


def test_a_failed_target_initializer_stops_before_the_state_it_left_unset():
    """``__dace_init_cuda`` returns early without a gpu_context when no device is present."""
    init = init_function(generated_code(persistent_gpu_transient()), 'persistent_gpu_transient')
    initializer = re.search(r'__result \|= __dace_init_cuda\(', init)
    assert initializer, 'the CUDA target initializer is not called, so this test is anchored on nothing'
    allocation = re.search(r'DACE_GPU_CHECK\(', init)
    assert allocation, 'the persistent GPU allocation was not hoisted into the init function'
    bailout = re.search(BAILOUT, init[initializer.end():allocation.start()])
    assert bailout, ('the persistent GPU allocation runs even when __dace_init_cuda failed, and every DACE_GPU_CHECK '
                     'in it dereferences the gpu_context that the failed initializer never constructed')


def test_the_init_function_still_checks_what_runs_after_the_allocations():
    """Environment and SDFG-level init code can fail too, so the later guard has to stay."""
    init = init_function(generated_code(persistent_gpu_transient()), 'persistent_gpu_transient')
    allocation = re.search(r'DACE_GPU_CHECK\(', init)
    assert allocation, 'the persistent GPU allocation was not hoisted into the init function'
    assert re.search(BAILOUT, init[allocation.end():]), (
        'nothing checks __result after the allocation and init code, so a failure there returns a live state')


if __name__ == '__main__':
    test_a_failed_target_initializer_stops_before_the_state_it_left_unset()
    test_the_init_function_still_checks_what_runs_after_the_allocations()
