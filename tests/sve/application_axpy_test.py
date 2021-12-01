# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
import dace
import numpy as np
import scipy as sp

import tests.sve.common as common
import pytest

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


@pytest.mark.sve
def test_axpy():
    print("==== Program start ====")

    N.set(24)

    print('Scalar-vector multiplication %d' % (N.get()))

    # Initialize arrays: Randomize A and X, zero Y
    A = dace.float64(np.random.rand())
    X = np.random.rand(N.get()).astype(np.float64)
    Y = np.random.rand(N.get()).astype(np.float64)

    A_regression = np.float64()
    X_regression = np.ndarray([N.get()], dtype=np.float64)
    Y_regression = np.ndarray([N.get()], dtype=np.float64)
    A_regression = A
    X_regression[:] = X[:]
    Y_regression[:] = Y[:]

    sdfg = common.vectorize(axpy)

    sdfg(A=A, X=X, Y=Y, N=N)

    c_axpy = sp.linalg.blas.get_blas_funcs('axpy',
                                            arrays=(X_regression,
                                                    Y_regression))
    if dace.Config.get_bool('profiling'):
        dace.timethis('axpy', 'BLAS', (2 * N.get()), c_axpy, X_regression,
                        Y_regression, N.get(), A_regression)
    else:
        c_axpy(X_regression, Y_regression, N.get(), A_regression)

    diff = np.linalg.norm(Y_regression - Y) / N.get()
    print("Difference:", diff)
    print("==== Program end ====")
    assert diff <= 1e-5
