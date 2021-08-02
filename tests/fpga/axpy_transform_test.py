# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
import argparse
import dace
import numpy as np
from dace.fpga_testing import fpga_test
from dace.transformation.interstate import FPGATransformSDFG

N = dace.symbol('N')


@dace.program(dace.float64, dace.float64[N], dace.float64[N])
def axpy(A, X, Y):
    @dace.map(_[0:N])
    def multiplication(i):
        in_A << A
        in_X << X[i]
        in_Y << Y[i]
        out >> Y[i]

        out = in_A * in_X + in_Y


@fpga_test()
def test_axpy_transformed():

    n = 24

    print(f'Scalar-vector multiplication {n}')

    A = dace.float64(np.random.rand())
    X = np.random.rand(n)
    Y = np.random.rand(n)
    expected = A * X + Y

    # Obtain SDFG from @dace.program
    sdfg = axpy.to_sdfg()

    # Convert SDFG to FPGA using a transformation
    sdfg.apply_transformations(FPGATransformSDFG)

    # Specialize and execute SDFG on FPGA
    sdfg._name = f'axpy_fpga_{n}'
    sdfg.specialize(dict(N=n))
    sdfg(A=A, X=X, Y=Y)

    diff = np.linalg.norm(expected - Y) / n
    assert diff <= 1e-5

    return sdfg


if __name__ == "__main__":
    test_axpy_transformed(None)
