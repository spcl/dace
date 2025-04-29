# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
import argparse
import dace
import numpy as np
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("N", type=int, nargs="?", default=24)
    args = vars(parser.parse_args())

    print('Scalar-vector multiplication %d' % (args['N']))

    A = dace.float64(np.random.rand())
    X = np.random.rand(args['N'])
    Y = np.random.rand(args['N'])
    expected = A * X + Y

    # Obtain SDFG from @dace.program
    sdfg = axpy.to_sdfg()

    # Convert SDFG to FPGA using a transformation
    sdfg.apply_transformations(FPGATransformSDFG)

    # Specialize and execute SDFG on FPGA
    sdfg._name = 'axpy_fpga_%d' % args['N']
    sdfg.specialize(dict(N=args['N']))
    sdfg(A=A, X=X, Y=Y)

    diff = np.linalg.norm(expected - Y) / args['N']
    print("Difference:", diff)
    exit(0 if diff <= 1e-5 else 1)
