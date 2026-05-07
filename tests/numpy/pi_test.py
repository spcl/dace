# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
import dace
import numpy as np
import math


def test_pi_tasklet():

    @dace.program
    def returnpi(result: dace.float64[1]):
        with dace.tasklet:
            r = math.pi
            r >> result[0]

    a = np.random.rand(1)
    returnpi(a)
    assert np.allclose(a, np.array(math.pi))


def test_pi_numpy():

    @dace.program
    def returnpi(result: dace.float64[1]):
        result[0] = math.pi

    a = np.random.rand(1)
    returnpi(a)
    assert np.allclose(a, np.array(math.pi))


def test_piarray_numpy():

    @dace.program
    def returnpi(result: dace.float64[20]):
        result[:] = math.pi

    a = np.random.rand(20)
    returnpi(a)
    assert np.allclose(a, np.array([math.pi] * 20))


if __name__ == '__main__':
    test_pi_tasklet()
    test_pi_numpy()
    test_piarray_numpy()
