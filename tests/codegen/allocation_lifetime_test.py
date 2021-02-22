# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
""" Tests different allocation lifetimes. """
import dace
import numpy as np

N = dace.symbol('N')


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
    def persistentmem(output: dace.int32[1]):
        tmp = dace.ndarray([1],
                           output.dtype,
                           storage=dace.StorageType.CPU_ThreadLocal,
                           lifetime=dace.AllocationLifetime.Persistent)
        if output[0] == 1.0:
            for i in dace.map[0:1]:
                tmp[i] = 0
        else:
            for i in dace.map[0:1]:
                tmp[i] += 3
                output[i] = tmp[i]

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


if __name__ == '__main__':
    test_alloc_persistent_register()
    test_alloc_persistent()
    test_alloc_persistent_threadlocal()
