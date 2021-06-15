# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
""" Tests different allocation lifetimes. """
import pytest

import dace
import numpy as np

N = dace.symbol('N')


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
    nstate.add_edge(nstate.add_read("ninput"), None, a_trans, None,
                    nsdfg.make_array_memlet("transient_heap"))
    nstate.add_edge(a_trans, None, nstate.add_write("noutput"), None,
                    nsdfg.make_array_memlet("transient_heap"))

    a_gpu = state.add_read("input_gpu")
    nsdfg_node = state.add_nested_sdfg(nsdfg, None, {"ninput"}, {"noutput"})
    wR = state.add_write("__return")

    state.add_edge(state.add_read("input"), None, a_gpu, None,
                   sdfg.make_array_memlet("input"))
    state.add_edge(a_gpu, None, nsdfg_node, "ninput",
                   sdfg.make_array_memlet("input_gpu"))
    state.add_edge(nsdfg_node, "noutput", wR, None,
                   sdfg.make_array_memlet("__return"))
    result = sdfg(input=np.ones((2, 2), dtype=np.float64))
    assert np.all(result == np.ones((2, 2)))


@pytest.mark.gpu
def test_persistent_gpu_transpose_regression():
    @dace.program
    def test_persistent_transpose(A: dace.float64[5, 3]):
        return np.transpose(A)

    sdfg = test_persistent_transpose.to_sdfg()

    sdfg.expand_library_nodes()
    sdfg.apply_strict_transformations()
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
        tmp = dace.ndarray([1],
                           output.dtype,
                           lifetime=dace.AllocationLifetime.Persistent)
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


if __name__ == '__main__':
    test_persistent_gpu_copy_regression()
    test_persistent_gpu_transpose_regression()
    test_alloc_persistent_register()
    test_alloc_persistent()
    test_alloc_persistent_threadlocal()
