# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Generated GPU code has to check what it calls, and has to stop before it uses what failed.

These assert on emitted code, so they need a GPU for neither compilation nor a run.
"""
import re

import dace
from dace import dtypes
from dace.libraries import blas

BAILOUT = r'if \(__result\)'
# Matches DACE_GPU_CHECK and its _RETURN/_RETURN_VAL variants, either backend target.
GPU_CHECK = r'DACE_GPU_CHECK\w*\('
SYNC_CALL = r'(?:cuda|hip)StreamSynchronize\('
CUBLAS_CALL = r'cublas[A-Z]\w*\('


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


def cross_stream_consumer() -> dace.SDFG:
    """Two independent kernels feeding a third, so one producer is ordered against another stream."""
    sdfg = dace.SDFG('cross_stream_consumer')
    for name in ('A', 'B', 'C'):
        sdfg.add_array(name, [20], dace.float64, storage=dtypes.StorageType.GPU_Global)
    sdfg.add_transient('T1', [20], dace.float64, storage=dtypes.StorageType.GPU_Global)
    sdfg.add_transient('T2', [20], dace.float64, storage=dtypes.StorageType.GPU_Global)

    state = sdfg.add_state('main')
    accesses = {name: state.add_access(name) for name in ('A', 'B', 'C', 'T1', 'T2')}

    def producer(name: str, src: str, dst: str, code: str) -> None:
        entry, exit_ = state.add_map(name, {'i': '0:20'}, schedule=dtypes.ScheduleType.GPU_Device)
        tasklet = state.add_tasklet(name + '_t', {'inp'}, {'out'}, code)
        entry.add_in_connector('IN_x')
        entry.add_out_connector('OUT_x')
        exit_.add_in_connector('IN_y')
        exit_.add_out_connector('OUT_y')
        state.add_edge(accesses[src], None, entry, 'IN_x', dace.Memlet(f'{src}[0:20]'))
        state.add_edge(entry, 'OUT_x', tasklet, 'inp', dace.Memlet(f'{src}[i]'))
        state.add_edge(tasklet, 'out', exit_, 'IN_y', dace.Memlet(f'{dst}[i]'))
        state.add_edge(exit_, 'OUT_y', accesses[dst], None, dace.Memlet(f'{dst}[0:20]'))

    producer('k1', 'A', 'T1', 'out = inp * 2')
    producer('k2', 'B', 'T2', 'out = inp * 3')

    entry, exit_ = state.add_map('k3', {'i': '0:20'}, schedule=dtypes.ScheduleType.GPU_Device)
    tasklet = state.add_tasklet('k3_t', {'p', 'q'}, {'out'}, 'out = p + q')
    for conn in ('IN_1', 'IN_2'):
        entry.add_in_connector(conn)
    for conn in ('OUT_1', 'OUT_2'):
        entry.add_out_connector(conn)
    exit_.add_in_connector('IN_c')
    exit_.add_out_connector('OUT_c')
    state.add_edge(accesses['T1'], None, entry, 'IN_1', dace.Memlet('T1[0:20]'))
    state.add_edge(accesses['T2'], None, entry, 'IN_2', dace.Memlet('T2[0:20]'))
    state.add_edge(entry, 'OUT_1', tasklet, 'p', dace.Memlet('T1[i]'))
    state.add_edge(entry, 'OUT_2', tasklet, 'q', dace.Memlet('T2[i]'))
    state.add_edge(tasklet, 'out', exit_, 'IN_c', dace.Memlet('C[i]'))
    state.add_edge(exit_, 'OUT_c', accesses['C'], None, dace.Memlet('C[0:20]'))
    sdfg.validate()
    return sdfg


def cublas_gemm(alpha: float = 1.0) -> dace.SDFG:
    """A GEMM library node expanded onto cuBLAS. ``alpha`` off 1.0 also brings out the pointer mode."""
    sdfg = dace.SDFG('cublas_gemm')
    for name in ('A', 'B', 'C'):
        sdfg.add_array(name, [20, 20], dace.float64, storage=dtypes.StorageType.GPU_Global)

    state = sdfg.add_state('main')
    node = blas.Gemm('gemm', alpha=alpha)
    node.implementation = 'cuBLAS'
    state.add_node(node)
    state.add_edge(state.add_read('A'), None, node, '_a', dace.Memlet('A[0:20, 0:20]'))
    state.add_edge(state.add_read('B'), None, node, '_b', dace.Memlet('B[0:20, 0:20]'))
    state.add_edge(node, '_c', state.add_write('C'), None, dace.Memlet('C[0:20, 0:20]'))
    sdfg.expand_library_nodes()
    sdfg.validate()
    return sdfg


def test_a_failed_target_initializer_stops_before_the_state_it_left_unset():
    """The CUDA target initializer returns early without a gpu_context when no device is present."""
    init = init_function(generated_code(persistent_gpu_transient()), 'persistent_gpu_transient')
    initializer = re.search(r'__result \|= __dace_init_\w*cuda\w*\(', init)
    assert initializer, 'the CUDA target initializer is not called, so this test is anchored on nothing'
    allocation = re.search(GPU_CHECK, init)
    assert allocation, 'the persistent GPU allocation was not hoisted into the init function'
    bailout = re.search(BAILOUT, init[initializer.end():allocation.start()])
    assert bailout, ('the persistent GPU allocation runs even when the CUDA target initializer failed, and every '
                     'GPU check in it dereferences the gpu_context that the failed initializer never constructed')


def test_the_init_function_still_checks_what_runs_after_the_allocations():
    """Environment and SDFG-level init code can fail too, so the later guard has to stay."""
    init = init_function(generated_code(persistent_gpu_transient()), 'persistent_gpu_transient')
    allocation = re.search(GPU_CHECK, init)
    assert allocation, 'the persistent GPU allocation was not hoisted into the init function'
    assert re.search(BAILOUT, init[allocation.end():]), (
        'nothing checks __result after the allocation and init code, so a failure there returns a live state')


def test_cross_stream_event_synchronization_is_checked():
    """The producers have to be ordered against their consumer, and the ordering has to be checked.

    The experimental CUDA target establishes cross-stream ordering without events: every kernel of
    the state is issued on the same stream in dependency order (stream FIFO replaces
    ``StreamWaitEvent``), and the host is blocked on that stream by a ``StreamSynchronize`` before
    anything may read the results. A silent ``StreamSynchronize`` failure loses that ordering just
    as a silent ``EventRecord`` did, so every emitted one has to be checked.
    """
    code = generated_code(cross_stream_consumer())
    launches = list(
        re.finditer(
            r'gpuStream_t __dace_current_stream = (?P<stream>[^;]+);\s*'
            r'__dace_runkernel_cross_stream_consumer_(?P<kernel>k\d)_', code))
    kernels = [launch.group('kernel') for launch in launches]
    assert sorted(kernels) == ['k1', 'k2', 'k3'], f'expected the three kernel launches, got {kernels}'
    assert kernels[-1] == 'k3', f'the consumer k3 is issued before a producer it reads from: {kernels}'
    streams = {launch.group('kernel'): launch.group('stream').strip() for launch in launches}
    assert streams['k3'] == streams['k1'] and streams['k3'] == streams['k2'], (
        f'the consumer is issued on a different stream than its producers with no event between them, '
        f'so nothing orders them: {streams}')
    syncs = list(re.finditer(SYNC_CALL, code))
    last_launch = launches[-1].end()
    assert any(sync.start() > last_launch for sync in syncs), (
        'no stream synchronization follows the kernels, so nothing orders the GPU work with the host')
    unchecked = [sync.group(0) for sync in syncs if not re.search(GPU_CHECK + r'$', code[:sync.start()])]
    assert not unchecked, f'stream synchronization emitted without an error check: {unchecked}'


def test_cublas_calls_are_checked():
    """cuBLAS reports through its return value only, so a dropped status is a silently wrong result."""
    for alpha in (1.0, 2.0):
        code = generated_code(cublas_gemm(alpha))
        calls = list(re.finditer(CUBLAS_CALL, code))
        assert calls, f'no cuBLAS call was emitted for alpha={alpha}, so this test is anchored on nothing'
        unchecked = [
            call.group(0) for call in calls
            if not code[:call.start()].rstrip().endswith('dace::blas::CheckCublasError(')
        ]
        assert not unchecked, f'cuBLAS called without checking its status: {unchecked}'


def test_the_cublas_gemm_expansion_still_emits_the_pointer_mode_switch():
    """The check has to wrap the pointer mode switch, not replace it."""
    code = generated_code(cublas_gemm(2.0))
    assert 'cublasSetPointerMode(__dace_cublas_handle, CUBLAS_POINTER_MODE_HOST)' in code
    assert 'cublasSetPointerMode(__dace_cublas_handle, CUBLAS_POINTER_MODE_DEVICE)' in code


if __name__ == '__main__':
    test_a_failed_target_initializer_stops_before_the_state_it_left_unset()
    test_the_init_function_still_checks_what_runs_after_the_allocations()
    test_cross_stream_event_synchronization_is_checked()
    test_cublas_calls_are_checked()
    test_the_cublas_gemm_expansion_still_emits_the_pointer_mode_switch()
