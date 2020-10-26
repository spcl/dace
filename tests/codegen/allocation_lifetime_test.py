# Copyright 2019-2020 ETH Zurich and the DaCe authors. All rights reserved.
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


if __name__ == '__main__':
    test_alloc_persistent_register()
