# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Codegen coverage for ``ExpandReduceCUDABlockAtomic``.

The block-atomic reduce folds each thread's partial across the thread block with
``gpucub::BlockReduce`` and commits ONE atomic per block into a length-1 global
output. We assert the emitted CUDA (the cub call + the thread-0 ``reduce_atomic``)
without a GPU; when ``nvcc`` is present we also compile the generated TU.
"""
import shutil
import warnings

import numpy as np
import pytest

import dace
from dace.memlet import Memlet

_HAS_NVCC = shutil.which("nvcc") is not None


def _build_block_atomic_sum_sdfg():
    """Sum a 128-vector into a length-1 output: 2 blocks x 64 threads, each block
    reduces its 64 partials and atomically adds the block sum into ``B[0]``."""
    sdfg = dace.SDFG('block_atomic_reduction')
    sdfg.add_array('A', (128, ), dace.float32)
    sdfg.add_array('B', (1, ), dace.float32)
    sdfg.add_transient('tA', (1, ), dace.float32)
    state = sdfg.add_state('a')

    A = state.add_access('A')
    B = state.add_access('B')
    me, mx = state.add_map('grid', dict(bi='0:2'))
    mei, mxi = state.add_map('block', dict(i='0:64'))
    red = state.add_reduce('lambda a, b: a + b', None, 0)
    red.implementation = 'CUDA (block atomic)'
    tA = state.add_access('tA')

    state.add_edge(A, None, me, None, Memlet.simple(A, '0:128'))
    state.add_edge(me, None, mei, None, Memlet.simple(A, '(64*bi):(64*bi+64)'))
    state.add_edge(mei, None, tA, None, Memlet.simple('A', '(64*bi+i)'))
    state.add_edge(tA, None, red, '_in', Memlet.simple(tA, '0'))
    # The atomic lives inside the expansion; the block sum drains to the single
    # global element B[0] (dynamic: exactly one atomic per block, from thread 0).
    e_out = state.add_edge(red, '_out', mxi, None, Memlet.simple('B', '0', num_accesses=-1))
    state.add_edge(mxi, None, mx, None, Memlet.simple('B', '0', num_accesses=-1))
    state.add_edge(mx, None, B, None, Memlet.simple(B, '0'))
    sdfg.fill_scope_connectors()

    # Declared, not derived. What this test is about is the CUDA the expansion emits, and the
    # expansion has requirements of its own: a thread-block map to size ``BlockReduce`` from, and a
    # register input holding the per-thread partial. Asking an offloader for them would assert its
    # placement policy here -- and it has no reason to produce a thread-block map at all, since
    # inserting those is a later pass.
    me.map.schedule = dace.ScheduleType.GPU_Device
    mei.map.schedule = dace.ScheduleType.GPU_ThreadBlock
    sdfg.arrays['tA'].storage = dace.StorageType.Register
    sdfg.validate()
    return sdfg


def _generated_cuda(sdfg):
    code_objects = sdfg.generate_code()
    return "\n".join(co.code for co in code_objects if co.language in ("cu", "cuda")) or \
        "\n".join(co.code for co in code_objects)


def test_block_atomic_emits_cub_and_atomic():
    sdfg = _build_block_atomic_sum_sdfg()
    code = _generated_cuda(sdfg)
    assert "gpucub::BlockReduce<float, 64>" in code, "block reduce not typed to the 64-thread block"
    assert ".Reduce(" in code, "cub block Reduce call missing"
    assert "reduce_atomic" in code, "thread-0 atomic to the global output missing"
    assert "threadIdx.x == 0" in code, "atomic not guarded to a single thread per block"
    assert "__shared__" in code, "block-reduce temp storage not in shared memory"


@pytest.mark.skipif(not _HAS_NVCC, reason="nvcc not available; compile check skipped")
def test_block_atomic_compiles():
    sdfg = _build_block_atomic_sum_sdfg()
    sdfg.compile()


@pytest.mark.gpu
def test_block_atomic_runs():
    """Runs on the device. ``A`` and ``B`` are read and written by the kernel, so they are device
    arrays here. Host buffers under a ``GPU_Device`` map cause a memory-access fault on a discrete GPU."""
    import cupy  # GPU-only dependency; a CPU collection of this file must not need it
    sdfg = _build_block_atomic_sum_sdfg()
    for name in ('A', 'B'):
        sdfg.arrays[name].storage = dace.StorageType.GPU_Global
    A = np.random.rand(128).astype(np.float32)
    arrays = {'A': cupy.asarray(A), 'B': cupy.zeros(1, dtype=np.float32)}
    sdfg(**arrays)
    assert abs(arrays['B'].get()[0] - np.sum(A)) / 128.0 <= 1e-4


def last_negative_index_sdfg() -> dace.SDFG:
    """A device map folding ``max`` of an ``int64`` index into one element: tsvc s331's kernel shape."""
    sdfg = dace.SDFG('last_negative_index_int64')
    sdfg.add_array('A', (1000, ), dace.float64, storage=dace.StorageType.GPU_Global)
    sdfg.add_array('R', (1, ), dace.int64, storage=dace.StorageType.GPU_Global)
    state = sdfg.add_state('a')
    entry, exit_node = state.add_map('scan', dict(i='0:1000'), schedule=dace.ScheduleType.GPU_Device)
    pick = state.add_tasklet('pick', {'a'}, {'out'}, 'out = i if a < 0.0 else -1')
    state.add_memlet_path(state.add_read('A'), entry, pick, dst_conn='a', memlet=Memlet('A[i]'))
    state.add_memlet_path(pick,
                          exit_node,
                          state.add_write('R'),
                          src_conn='out',
                          memlet=Memlet('R[0]', wcr='lambda x, y: max(x, y)'))
    sdfg.validate()
    return sdfg


def test_a_device_max_into_an_int64_uses_the_64_bit_atomic():
    code = _generated_cuda(last_negative_index_sdfg())
    assert '_wcr_fixed<dace::ReductionType::Max, int64_t>' in code, 'the int64 max fold is not emitted'


@pytest.mark.gpu
def test_a_device_max_into_an_int64_finds_the_last_negative_index():
    import cupy  # GPU-only dependency; a CPU collection of this file must not need it
    host_a = np.random.default_rng(11).uniform(-1.0, 1.0, 1000)
    arrays = {'A': cupy.asarray(host_a), 'R': cupy.full(1, np.iinfo(np.int64).min, dtype=np.int64)}
    last_negative_index_sdfg()(**arrays)
    assert int(arrays['R'].get()[0]) == int(np.flatnonzero(host_a < 0.0)[-1])


def device_fold_sdfg(name: str, dtype: dace.typeclass, body: str, wcr: str) -> dace.SDFG:
    """A device map folding ``body`` of each ``A[i]`` into ``R[0]`` with ``wcr``: an atomic per thread."""
    sdfg = dace.SDFG(name)
    sdfg.add_array('A', (1000, ), dace.float64, storage=dace.StorageType.GPU_Global)
    sdfg.add_array('R', (1, ), dtype, storage=dace.StorageType.GPU_Global)
    state = sdfg.add_state('a')
    entry, exit_node = state.add_map('scan', dict(i='0:1000'), schedule=dace.ScheduleType.GPU_Device)
    entry.map.gpu_block_size = [128, 1, 1]
    fold = state.add_tasklet('fold', {'a'}, {'out'}, f'out = {body}')
    state.add_memlet_path(state.add_read('A'), entry, fold, dst_conn='a', memlet=Memlet('A[i]'))
    state.add_memlet_path(fold, exit_node, state.add_write('R'), src_conn='out', memlet=Memlet('R[0]', wcr=wcr))
    sdfg.validate()
    return sdfg


#: The device folds that have no native atomic: a 16-byte complex sum and a one-byte bool or/and.
#: npbench vexx_k carries the complex sum and the bool or on the GPU canonicalize column.
DEVICE_FOLDS = {
    'complex128_sum': (dace.complex128, 'a * (1.0 + 2.0j)', 'lambda x, y: x + y'),
    'complex64_sum': (dace.complex64, 'dace.complex64(a)', 'lambda x, y: x + y'),
    'bool_or': (dace.bool_, 'a > 0.99', 'lambda x, y: x or y'),
    'bool_and': (dace.bool_, 'a > 0.01', 'lambda x, y: x and y'),
}


def device_fold(key: str) -> dace.SDFG:
    dtype, body, wcr = DEVICE_FOLDS[key]
    return device_fold_sdfg(f'device_fold_{key}', dtype, body, wcr)


def test_a_complex_fold_seeds_its_partial_with_both_parts():
    """The partial's identity is written ``complex128(re, im)``; ``float()`` of it dropped the
    imaginary part, and warned, for every complex fold."""
    with warnings.catch_warnings():
        warnings.simplefilter('error', np.exceptions.ComplexWarning)
        code = _generated_cuda(device_fold('complex128_sum'))
    assert 'dace::complex128(0.0, 0.0)' in code, 'the complex identity lost a part'


@pytest.mark.skipif(not (_HAS_NVCC or shutil.which('hipcc')), reason='no GPU compiler; compile check skipped')
@pytest.mark.parametrize('key', list(DEVICE_FOLDS))
def test_a_device_fold_without_a_native_atomic_compiles(key):
    """There is no atomicAdd for a complex operand, no 16-byte atomicCAS for the fallback to
    use, and no atomicOr/atomicAnd for a bool: all of them failed to compile in the kernel."""
    device_fold(key).compile()


@pytest.mark.gpu
@pytest.mark.parametrize('key', list(DEVICE_FOLDS))
def test_a_device_fold_without_a_native_atomic_computes_the_fold(key):
    import cupy  # GPU-only dependency; a CPU collection of this file must not need it
    dtype, _, _ = DEVICE_FOLDS[key]
    host_a = np.random.default_rng(5).uniform(0.0, 1.0, 1000)
    if key.endswith('_sum'):
        scale = 1.0 + 2.0j if dtype is dace.complex128 else 1.0
        expected, start = np.sum(host_a * scale), 0
    elif key == 'bool_or':
        expected, start = bool(np.any(host_a > 0.99)), False
    else:
        expected, start = bool(np.all(host_a > 0.01)), True
    arrays = {'A': cupy.asarray(host_a), 'R': cupy.full(1, start, dtype=dtype.type)}
    device_fold(key)(**arrays)
    got = arrays['R'].get()[0]
    if key.endswith('_sum'):
        assert abs(got - expected) <= 1e-4 * abs(expected), (got, expected)
    else:
        assert bool(got) == expected, (got, expected)


if __name__ == '__main__':
    test_block_atomic_emits_cub_and_atomic()
    print('codegen ok')
