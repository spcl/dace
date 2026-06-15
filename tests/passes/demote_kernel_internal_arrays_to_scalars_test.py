# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Unit tests for :class:`DemoteKernelInternalArraysToScalars`.

The pass is the inverse of ``PromoteGPUScalarsToArrays``: a length-1 ``Array``
that lives entirely inside a ``GPU_Device`` kernel (and is not a kernel output)
is demoted back to a ``Scalar``. Genuine kernel outputs (``GPU_Global`` storage
or written across a ``GPU_Device`` ``MapExit``) and ``GPU_Shared`` arrays are
kept.
"""
import dace
from dace import data, dtypes
from dace.transformation.passes.promote_gpu_scalars_to_arrays import InferDefaultSchedulesAndStorages
from dace.transformation.passes.demote_kernel_internal_arrays_to_scalars import (DemoteKernelInternalArraysToScalars,
                                                                                 written_by_gpu_map_exit)

GPU_GLOBAL = dtypes.StorageType.GPU_Global
GPU_SHARED = dtypes.StorageType.GPU_Shared
REGISTER = dtypes.StorageType.Register
GPU_DEVICE = dtypes.ScheduleType.GPU_Device


def _kernel_with_inner(inner: dace.SDFG, in_conns, out_conns) -> dace.SDFG:
    """Wrap ``inner`` in a ``GPU_Device`` map: ``A[i] -> inner -> B[i]``."""
    sdfg = dace.SDFG('outer_' + inner.name)
    sdfg.add_array('A', (16, ), dace.float64, storage=GPU_GLOBAL)
    sdfg.add_array('B', (16, ), dace.float64, storage=GPU_GLOBAL)
    st = sdfg.add_state('main')
    entry, exit_ = st.add_map('kmap', dict(i='0:16'), schedule=GPU_DEVICE)
    n = st.add_nested_sdfg(inner, set(in_conns), set(out_conns))
    st.add_memlet_path(st.add_access('A'), entry, n, dst_conn='a', memlet=dace.Memlet('A[i]'))
    st.add_memlet_path(n, exit_, st.add_access('B'), src_conn='b', memlet=dace.Memlet('B[i]'))
    return sdfg


def _run(sdfg: dace.SDFG):
    InferDefaultSchedulesAndStorages().apply_pass(sdfg, {})
    return DemoteKernelInternalArraysToScalars().apply_pass(sdfg, {})


def test_kernel_internal_transient_demoted():
    """A length-1 Register transient inside a device function becomes a Scalar."""
    inner = dace.SDFG('roundtrip')
    inner.add_array('a', (1, ), dace.float64)
    inner.add_array('b', (1, ), dace.float64)
    inner.add_array('tmp', (1, ), dace.float64, transient=True, storage=REGISTER)
    s = inner.add_state('s')
    t1 = s.add_tasklet('t1', {'x'}, {'y'}, 'y = x + 1.0')
    t2 = s.add_tasklet('t2', {'x'}, {'y'}, 'y = x * 2.0')
    tmp = s.add_access('tmp')
    s.add_edge(s.add_access('a'), None, t1, 'x', dace.Memlet('a[0]'))
    s.add_edge(t1, 'y', tmp, None, dace.Memlet('tmp[0]'))
    s.add_edge(tmp, None, t2, 'x', dace.Memlet('tmp[0]'))
    s.add_edge(t2, 'y', s.add_access('b'), None, dace.Memlet('b[0]'))

    sdfg = _kernel_with_inner(inner, {'a'}, {'b'})
    assert _run(sdfg) is not None
    assert isinstance(inner.arrays['tmp'], data.Scalar), inner.arrays['tmp']


def test_gpu_global_len1_array_not_demoted():
    """A ``GPU_Global`` length-1 array (a promoted kernel output) is kept as an Array."""
    inner = dace.SDFG('keepglobal')
    inner.add_array('a', (1, ), dace.float64)
    inner.add_array('b', (1, ), dace.float64)
    inner.add_array('acc', (1, ), dace.float64, transient=True, storage=GPU_GLOBAL)
    s = inner.add_state('s')
    t = s.add_tasklet('t', {'x'}, {'y'}, 'y = x')
    acc = s.add_access('acc')
    s.add_edge(s.add_access('a'), None, t, 'x', dace.Memlet('a[0]'))
    s.add_edge(t, 'y', acc, None, dace.Memlet('acc[0]'))
    t2 = s.add_tasklet('t2', {'x'}, {'y'}, 'y = x')
    s.add_edge(acc, None, t2, 'x', dace.Memlet('acc[0]'))
    s.add_edge(t2, 'y', s.add_access('b'), None, dace.Memlet('b[0]'))

    sdfg = _kernel_with_inner(inner, {'a'}, {'b'})
    _run(sdfg)
    assert isinstance(inner.arrays['acc'], data.Array), inner.arrays['acc']


def test_gpu_shared_len1_array_not_demoted():
    """A ``GPU_Shared`` length-1 array (cross-thread) is kept as an Array."""
    inner = dace.SDFG('keepshared')
    inner.add_array('a', (1, ), dace.float64)
    inner.add_array('b', (1, ), dace.float64)
    inner.add_array('sh', (1, ), dace.float64, transient=True, storage=GPU_SHARED)
    s = inner.add_state('s')
    t = s.add_tasklet('t', {'x'}, {'y'}, 'y = x')
    sh = s.add_access('sh')
    s.add_edge(s.add_access('a'), None, t, 'x', dace.Memlet('a[0]'))
    s.add_edge(t, 'y', sh, None, dace.Memlet('sh[0]'))
    t2 = s.add_tasklet('t2', {'x'}, {'y'}, 'y = x')
    s.add_edge(sh, None, t2, 'x', dace.Memlet('sh[0]'))
    s.add_edge(t2, 'y', s.add_access('b'), None, dace.Memlet('b[0]'))

    sdfg = _kernel_with_inner(inner, {'a'}, {'b'})
    _run(sdfg)
    assert isinstance(inner.arrays['sh'], data.Array), inner.arrays['sh']


def test_kernel_output_written_across_gpu_exit_not_demoted():
    """A length-1 array written across a ``GPU_Device`` ``MapExit`` is a kernel output -> kept."""
    sdfg = dace.SDFG('kout')
    sdfg.add_array('A', (16, ), dace.float64, storage=GPU_GLOBAL)
    sdfg.add_array('out', (1, ), dace.float64, transient=True, storage=REGISTER)
    st = sdfg.add_state('main')
    entry, exit_ = st.add_map('kmap', dict(i='0:1'), schedule=GPU_DEVICE)
    t = st.add_tasklet('t', {'x'}, {'y'}, 'y = x')
    st.add_memlet_path(st.add_access('A'), entry, t, dst_conn='x', memlet=dace.Memlet('A[i]'))
    st.add_memlet_path(t, exit_, st.add_access('out'), src_conn='y', memlet=dace.Memlet('out[0]'))

    InferDefaultSchedulesAndStorages().apply_pass(sdfg, {})
    assert written_by_gpu_map_exit(sdfg, 'out') is True
    DemoteKernelInternalArraysToScalars().apply_pass(sdfg, {})
    assert isinstance(sdfg.arrays['out'], data.Array), sdfg.arrays['out']


def test_non_gpu_len1_array_untouched():
    """Outside any GPU kernel, length-1 arrays are left alone (this is a GPU-codegen pass)."""
    sdfg = dace.SDFG('host')
    sdfg.add_array('a', (1, ), dace.float64, transient=True)
    sdfg.add_array('b', (1, ), dace.float64)
    st = sdfg.add_state('s')
    t = st.add_tasklet('t', {'x'}, {'y'}, 'y = x')
    st.add_edge(st.add_access('a'), None, t, 'x', dace.Memlet('a[0]'))
    st.add_edge(t, 'y', st.add_access('b'), None, dace.Memlet('b[0]'))

    InferDefaultSchedulesAndStorages().apply_pass(sdfg, {})
    assert DemoteKernelInternalArraysToScalars().apply_pass(sdfg, {}) is None
    assert isinstance(sdfg.arrays['a'], data.Array)


if __name__ == '__main__':
    test_kernel_internal_transient_demoted()
    test_gpu_global_len1_array_not_demoted()
    test_gpu_shared_len1_array_not_demoted()
    test_kernel_output_written_across_gpu_exit_not_demoted()
    test_non_gpu_len1_array_untouched()
    print('ok')
