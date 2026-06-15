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


def test_library_node_output_not_demoted():
    """A length-1 array written by a ``LibraryNode`` (e.g. a cub block ``Reduce``) is kept: the
    library expansion emits raw ``buf[0]`` indexing that scalar-conversion never rewrites. This is
    the block-reduction regression -- demoting it yields ``float tB; ... tB[0] = ...``."""
    from dace.memlet import Memlet
    from dace.transformation.interstate import GPUTransformSDFG

    sdfg = dace.SDFG('blockreduce')
    sdfg.add_array('A', (128, ), dace.float32)
    sdfg.add_array('B', (2, ), dace.float32)
    sdfg.add_transient('tA', (2, ), dace.float32)
    sdfg.add_transient('tB', (1, ), dace.float32)  # length-1 reduce output
    state = sdfg.add_state('a')
    A, B = state.add_access('A'), state.add_access('B')
    me, mx = state.add_map('mymap', dict(bi='0:2'))
    mei, mxi = state.add_map('mymap2', dict(i='0:32'))
    red = state.add_reduce('lambda a, b: a + b', None, 0)
    red.implementation = 'CUDA (block)'
    tA, tB = state.add_access('tA'), state.add_access('tB')
    wt = state.add_tasklet('writeout', {'inp'}, {'out'}, 'if i == 0: out = inp')
    state.add_edge(A, None, me, None, Memlet.simple(A, '0:128'))
    state.add_edge(me, None, mei, None, Memlet.simple(A, '(64*bi):(64*bi+64)'))
    state.add_edge(mei, None, tA, None, Memlet.simple('A', '(64*bi+2*i):(64*bi+2*i+2)'))
    state.add_edge(tA, None, red, None, Memlet.simple(tA, '0:2'))
    state.add_edge(red, None, tB, None, Memlet.simple(tB, '0'))
    state.add_edge(tB, None, wt, 'inp', Memlet.simple(tB, '0'))
    state.add_edge(wt, 'out', mxi, None, Memlet.simple('B', 'bi', num_accesses=-1))
    state.add_edge(mxi, None, mx, None, Memlet.simple(B, 'bi'))
    state.add_edge(mx, None, B, None, Memlet.simple(B, '0:2'))
    sdfg.fill_scope_connectors()
    sdfg.apply_transformations(GPUTransformSDFG, options={'sequential_innermaps': False})

    InferDefaultSchedulesAndStorages().apply_pass(sdfg, {})
    DemoteKernelInternalArraysToScalars().apply_pass(sdfg, {})
    assert isinstance(sdfg.arrays['tB'], data.Array), sdfg.arrays['tB']


def test_gpu_scheduled_nested_map_boundary_not_demoted():
    """A length-1 array on the boundary of a GPU-scheduled *nested* map (a thread-block sub-kernel)
    is kept -- it is a multi-thread buffer, not a single per-thread value."""
    sdfg = dace.SDFG('nestedgpu')
    sdfg.add_array('A', (16, ), dace.float64, storage=GPU_GLOBAL)
    sdfg.add_array('B', (16, ), dace.float64, storage=GPU_GLOBAL)
    sdfg.add_transient('buf', (1, ), dace.float64, storage=REGISTER)
    st = sdfg.add_state('main')
    dev_e, dev_x = st.add_map('dev', dict(i='0:16'), schedule=GPU_DEVICE)
    # A nested GPU thread-block map writes the length-1 buffer.
    tb_e, tb_x = st.add_map('tb', dict(j='0:1'), schedule=dtypes.ScheduleType.GPU_ThreadBlock)
    t = st.add_tasklet('t', {'x'}, {'y'}, 'y = x')
    buf = st.add_access('buf')
    st.add_memlet_path(st.add_access('A'), dev_e, tb_e, t, dst_conn='x', memlet=dace.Memlet('A[i]'))
    st.add_memlet_path(t, tb_x, buf, src_conn='y', memlet=dace.Memlet('buf[0]'))
    st.add_memlet_path(buf, dev_x, st.add_access('B'), memlet=dace.Memlet('B[i]'))

    InferDefaultSchedulesAndStorages().apply_pass(sdfg, {})
    DemoteKernelInternalArraysToScalars().apply_pass(sdfg, {})
    assert isinstance(sdfg.arrays['buf'], data.Array), sdfg.arrays['buf']


if __name__ == '__main__':
    test_kernel_internal_transient_demoted()
    test_gpu_global_len1_array_not_demoted()
    test_gpu_shared_len1_array_not_demoted()
    test_kernel_output_written_across_gpu_exit_not_demoted()
    test_non_gpu_len1_array_untouched()
    test_library_node_output_not_demoted()
    test_gpu_scheduled_nested_map_boundary_not_demoted()
    print('ok')
