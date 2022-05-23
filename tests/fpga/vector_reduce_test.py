# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
""" Sums all the element of the vector with a reduce. """

import dace
import numpy as np
import argparse
from dace.fpga_testing import fpga_test
from dace.transformation.interstate import FPGATransformSDFG

N = dace.symbol('N')


@dace.program
def vector_reduce(x: dace.float32[N], s: dace.scalar(dace.float32)):
    #transient
    tmp = dace.define_local([N], dtype=x.dtype)

    @dace.map
    def sum(i: _[0:N]):
        in_x << x[i]
        out >> tmp[i]

        out = in_x

    dace.reduce(lambda a, b: a + b, tmp, s, axis=(0), identity=0)


@fpga_test()
def test_vector_reduce():

    N.set(24)

    # Initialize arrays: X, Y and Z
    X = np.random.rand(N.get()).astype(dace.float32.type)
    s = dace.scalar(dace.float32)

    sdfg = vector_reduce.to_sdfg()
    sdfg.apply_transformations(FPGATransformSDFG)
    sdfg(x=X, s=s, N=N)

    # Compute expected result
    s_exp = 0.0
    for x in X:
        s_exp += x
    diff = np.linalg.norm(s_exp - s) / N.get()
    assert diff <= 1e-5

    return sdfg


if __name__ == "__main__":
    test_vector_reduce(None)
