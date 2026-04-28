# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
import dace
import numpy as np
import scipy as sp

import tests.codegen.sve.common as common
import pytest

N = dace.symbol('N')


@dace.program
def axpy(A: dace.float64, X: dace.float64[N], Y: dace.float64[N]):

    @dace.map(_[0:N])
    def multiplication(i):
        in_A << A
        in_X << X[i]
        in_Y << Y[i]
        out >> Y[i]

        out = in_A * in_X + in_Y


@pytest.mark.sve
def test_axpy():
    print("==== Program start ====")

    N = 24

    print('Scalar-vector multiplication %d' % (N))

    # Initialize arrays: Randomize A and X, zero Y
    A = dace.float64(np.random.rand())
    X = np.random.rand(N).astype(np.float64)
    Y = np.random.rand(N).astype(np.float64)

    A_regression = np.float64()
    X_regression = np.ndarray([N], dtype=np.float64)
    Y_regression = np.ndarray([N], dtype=np.float64)
    A_regression = A
    X_regression[:] = X[:]
    Y_regression[:] = Y[:]

    sdfg = common.vectorize(axpy)

    sdfg(A=A, X=X, Y=Y, N=N)

    c_axpy = sp.linalg.blas.get_blas_funcs('axpy', arrays=(X_regression, Y_regression))
    if dace.Config.get_bool('profiling'):
        dace.timethis('axpy', 'BLAS', (2 * N), c_axpy, X_regression, Y_regression, N, A_regression)
    else:
        c_axpy(X_regression, Y_regression, N, A_regression)

    diff = np.linalg.norm(Y_regression - Y) / N
    print("Difference:", diff)
    print("==== Program end ====")
    assert diff <= 1e-5
