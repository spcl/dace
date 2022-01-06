# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
""" Tests different allocation lifetimes. """
import pytest

import dace
from dace.codegen.targets import framecode
from dace.sdfg import infer_types
import numpy as np

N = dace.symbol('N')


def _test_determine_alloc(lifetime: dace.AllocationLifetime, unused: bool = False) -> dace.SDFG:
    """ Creates an SDFG playground for determining allocation. """
    sdfg = dace.SDFG('lifetimetest')
    sdfg.add_array('A', [N], dace.float64)
    sdfg.add_array('B', [N], dace.float64)
    sdfg.add_transient('unused', [N], dace.float64, lifetime=lifetime)
    state = sdfg.add_state()
    me, mx = state.add_map('m', dict(i='0:N'))

    #########################################################################
    nsdfg = dace.SDFG('nested')
    nsdfg.add_array('A', [N], dace.float64)
    nsdfg.add_array('B', [N], dace.float64)
    nsdfg.add_transient('tmp', [N], dace.float64, dace.StorageType.GPU_Global, lifetime=lifetime)
    nsdfg.add_transient('tmp2', [1], dace.float64, dace.StorageType.Register, lifetime=lifetime)
    nstate = nsdfg.add_state()
    ime, imx = nstate.add_map('m2', dict(i='0:20'), schedule=dace.ScheduleType.GPU_Device)
    t1 = nstate.add_access('tmp')
    t2 = nstate.add_access('tmp2')
    nstate.add_nedge(t1, t2, dace.Memlet('tmp[0]'))
    nstate.add_memlet_path(nstate.add_read('A'), ime, t1, memlet=dace.Memlet('A[i]'))
    nstate.add_memlet_path(t2, imx, nstate.add_write('B'), memlet=dace.Memlet('B[0]', wcr='lambda a,b: a+b'))
    #########################################################################
    nsdfg_node = state.add_nested_sdfg(nsdfg, None, {'A'}, {'B'})
    state.add_memlet_path(state.add_read('A'), me, nsdfg_node, dst_conn='A', memlet=dace.Memlet('A[0:N]'))
    state.add_memlet_path(nsdfg_node, mx, state.add_write('B'), src_conn='B', memlet=dace.Memlet('B[0:N]'))

    # Set default storage/schedule types in SDFG
    infer_types.set_default_schedule_and_storage_types(sdfg, None)

    return sdfg, (sdfg, state, me, nsdfg, nstate, ime)


def _check_alloc(id, name, codegen, scope):
    # for sdfg_id, _, node in codegen.to_allocate[scope]:
    #     if id == sdfg_id and name == node.data:
    #         return True
    for sdfg, _, node, _, _, _ in codegen.to_allocate[scope]:
        if sdfg.sdfg_id == id and name == node.data:
            return True
    return False


def test_determine_alloc_scope():
    sdfg, scopes = _test_determine_alloc(dace.AllocationLifetime.Scope)
    codegen = framecode.DaCeCodeGenerator()
    codegen.determine_allocation_lifetime(sdfg)

    # tmp cannot be allocated within the inner scope because it is GPU_Global
    assert _check_alloc(1, 'tmp', codegen, scopes[-2])
    assert _check_alloc(1, 'tmp2', codegen, scopes[-1])


def test_determine_alloc_state():
    sdfg, scopes = _test_determine_alloc(dace.AllocationLifetime.State, unused=True)
    codegen = framecode.DaCeCodeGenerator()
    codegen.determine_allocation_lifetime(sdfg)

    # Ensure that unused transients are not allocated
    assert not any('__0_unused' in field for field in codegen.statestruct)

    assert _check_alloc(1, 'tmp', codegen, scopes[-2])
    assert _check_alloc(1, 'tmp2', codegen, scopes[-2])


def test_determine_alloc_sdfg():
    sdfg, scopes = _test_determine_alloc(dace.AllocationLifetime.SDFG)
    codegen = framecode.DaCeCodeGenerator()
    codegen.determine_allocation_lifetime(sdfg)

    assert _check_alloc(1, 'tmp', codegen, scopes[-3])
    assert _check_alloc(1, 'tmp2', codegen, scopes[-3])


def test_determine_alloc_global():
    sdfg, scopes = _test_determine_alloc(dace.AllocationLifetime.Global)
    codegen = framecode.DaCeCodeGenerator()
    codegen.determine_allocation_lifetime(sdfg)
    assert any('__1_tmp' in field for field in codegen.statestruct)
    assert any('__1_tmp2' in field for field in codegen.statestruct)
    assert _check_alloc(1, 'tmp', codegen, sdfg)
    assert _check_alloc(1, 'tmp2', codegen, sdfg)


@pytest.mark.gpu
def test_persistent_gpu_copy_regression():

    sdfg = dace.SDFG('copynd')
    state = sdfg.add_state()

    nsdfg = dace.SDFG('copynd_nsdfg')
    nstate = nsdfg.add_state()

    sdfg.add_array("input", [2, 2], dace.float64)
    sdfg.add_array("input_gpu", [2, 2],
                   dace.float64,
                   transient=True,
                   storage=dace.StorageType.GPU_Global,
                   lifetime=dace.AllocationLifetime.Persistent)
    sdfg.add_array("__return", [2, 2], dace.float64)

    nsdfg.add_array("ninput", [2, 2],
                    dace.float64,
                    storage=dace.StorageType.GPU_Global,
                    lifetime=dace.AllocationLifetime.Persistent)
    nsdfg.add_array("transient_heap", [2, 2],
                    dace.float64,
                    transient=True,
                    storage=dace.StorageType.CPU_Heap,
                    lifetime=dace.AllocationLifetime.Persistent)
    nsdfg.add_array("noutput", [2, 2],
                    dace.float64,
                    storage=dace.dtypes.StorageType.CPU_Heap,
                    lifetime=dace.AllocationLifetime.Persistent)

    a_trans = nstate.add_access("transient_heap")
    nstate.add_edge(nstate.add_read("ninput"), None, a_trans, None, nsdfg.make_array_memlet("transient_heap"))
    nstate.add_edge(a_trans, None, nstate.add_write("noutput"), None, nsdfg.make_array_memlet("transient_heap"))

    a_gpu = state.add_read("input_gpu")
    nsdfg_node = state.add_nested_sdfg(nsdfg, None, {"ninput"}, {"noutput"})
    wR = state.add_write("__return")

    state.add_edge(state.add_read("input"), None, a_gpu, None, sdfg.make_array_memlet("input"))
    state.add_edge(a_gpu, None, nsdfg_node, "ninput", sdfg.make_array_memlet("input_gpu"))
    state.add_edge(nsdfg_node, "noutput", wR, None, sdfg.make_array_memlet("__return"))
    result = sdfg(input=np.ones((2, 2), dtype=np.float64))
    assert np.all(result == np.ones((2, 2)))


@pytest.mark.gpu
def test_persistent_gpu_transpose_regression():
    @dace.program
    def test_persistent_transpose(A: dace.float64[5, 3]):
        return np.transpose(A)

    sdfg = test_persistent_transpose.to_sdfg()

    sdfg.expand_library_nodes()
    sdfg.simplify()
    sdfg.apply_gpu_transformations()

    for _, _, arr in sdfg.arrays_recursive():
        if arr.transient and arr.storage == dace.StorageType.GPU_Global:
            arr.lifetime = dace.AllocationLifetime.Persistent
    A = np.random.rand(5, 3)
    result = sdfg(A=A)
    assert np.allclose(np.transpose(A), result)


def test_alloc_persistent_register():
    """ Tries to allocate persistent register array. Should fail. """
    @dace.program
    def lifetimetest(input: dace.float64[N]):
        tmp = dace.ndarray([1], input.dtype)
        return tmp + 1

    sdfg: dace.SDFG = lifetimetest.to_sdfg()
    sdfg.arrays['tmp'].storage = dace.StorageType.Register
    sdfg.arrays['tmp'].lifetime = dace.AllocationLifetime.Persistent

    try:
        sdfg.validate()
        raise AssertionError('SDFG should not be valid')
    except dace.sdfg.InvalidSDFGError:
        print('Exception caught, test passed')


def test_alloc_persistent():
    @dace.program
    def persistentmem(output: dace.int32[1]):
        tmp = dace.ndarray([1], output.dtype, lifetime=dace.AllocationLifetime.Persistent)
        if output[0] == 1.0:
            tmp[0] = 0
        else:
            tmp[0] += 3
            output[0] = tmp[0]

    # Repeatedly invoke program. Since memory is persistent, output is expected
    # to increase with each call
    csdfg = persistentmem.compile()
    value = np.ones([1], dtype=np.int32)
    csdfg(output=value)
    assert value[0] == 1
    value[0] = 2
    csdfg(output=value)
    assert value[0] == 3
    csdfg(output=value)
    assert value[0] == 6

    del csdfg


def test_alloc_persistent_threadlocal():
    @dace.program
    def persistentmem(output: dace.int32[2]):
        tmp = dace.ndarray([2],
                           output.dtype,
                           storage=dace.StorageType.CPU_ThreadLocal,
                           lifetime=dace.AllocationLifetime.Persistent)
        if output[0] == 1.0:
            for i in dace.map[0:2]:
                tmp[i] = i
        else:
            for i in dace.map[0:2]:
                tmp[i] += 3
                output[i] = tmp[i]

    # Repeatedly invoke program. Since memory is persistent, output is expected
    # to increase with each call
    csdfg = persistentmem.compile()
    value = np.ones([2], dtype=np.int32)
    csdfg(output=value)
    assert value[0] == 1
    assert value[1] == 1
    value[0] = 4
    value[1] = 2
    csdfg(output=value)
    assert value[0] == 3
    assert value[1] == 4
    csdfg(output=value)
    assert value[0] == 6
    assert value[1] == 7

    del csdfg


def test_alloc_multistate():
    i = dace.symbol('i')
    sdfg = dace.SDFG('multistate')
    sdfg.add_array('A', [20], dace.float64)
    sdfg.add_array('B', [20], dace.float64)
    sdfg.add_transient('tmp', [i + 1], dace.float64)

    init = sdfg.add_state()
    end = sdfg.add_state()
    s2 = sdfg.add_state()
    sdfg.add_loop(init, s2, end, 'i', '0', 'i < 5', 'i + 1')

    s1 = sdfg.add_state_before(s2)

    ar = s1.add_read('A')
    tw = s1.add_write('tmp')
    s1.add_nedge(ar, tw, dace.Memlet('A[0:i+1]'))

    tr = s2.add_read('tmp')
    bw = s2.add_write('B')
    s2.add_nedge(tr, bw, dace.Memlet('tmp'))

    A = np.random.rand(20)
    B = np.random.rand(20)
    sdfg(A=A, B=B)
    assert np.allclose(A[:5], B[:5])


def test_nested_view_samename():
    @dace.program
    def incall(a, b):
        tmp = a.reshape([10, 2])
        tmp[:] += 1
        return tmp

    @dace.program
    def top(a: dace.float64[20]):
        tmp = dace.ndarray([20], dace.float64, lifetime=dace.AllocationLifetime.Persistent)
        return incall(a, tmp)

    sdfg = top.to_sdfg(simplify=False)

    a = np.random.rand(20)
    ref = a.copy()
    b = sdfg(a)
    assert np.allclose(b, ref.reshape(10, 2) + 1)


def test_nested_persistent():
    @dace.program
    def nestpers(a):
        tmp = np.ndarray([20], np.float64)
        tmp[:] = a + 1
        return tmp

    @dace.program
    def toppers(a: dace.float64[20]):
        return nestpers(a)

    sdfg = toppers.to_sdfg(simplify=False)
    for _, _, arr in sdfg.arrays_recursive():
        if arr.transient:
            arr.lifetime = dace.AllocationLifetime.Persistent

    a = np.random.rand(20)
    b = sdfg(a)
    assert np.allclose(b, a + 1)


def test_persistent_scalar():
    @dace.program
    def perscal(a: dace.float64[20]):
        tmp = dace.define_local_scalar(dace.float64, lifetime=dace.AllocationLifetime.Persistent)
        tmp[:] = a[1] + 1
        return tmp

    a = np.random.rand(20)
    b = perscal(a)
    assert np.allclose(b, a[1] + 1)


if __name__ == '__main__':
    test_determine_alloc_scope()
    test_determine_alloc_state()
    test_determine_alloc_sdfg()
    test_determine_alloc_global()
    test_persistent_gpu_copy_regression()
    test_persistent_gpu_transpose_regression()
    test_alloc_persistent_register()
    test_alloc_persistent()
    test_alloc_persistent_threadlocal()
    test_alloc_multistate()
    test_nested_view_samename()
    test_nested_persistent()
    test_persistent_scalar()
