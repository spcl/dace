# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
import dace
import numpy as np
from dace.transformation.dataflow import Vectorization


@dace.program
def tovec(A: dace.float64[20]):
    return A + A


N = dace.symbol('N')


@dace.program
def tovec_sym(x: dace.float32[N], y: dace.float32[N], z: dace.float32[N]):
    @dace.map
    def sum(i: _[0:N]):
        xx << x[i]
        yy << y[i]
        zz << z[i]
        out >> z[i]

        out = xx + yy + zz


@dace.program
def tovec_uneven(A: dace.float64[N + 2]):
    for i in dace.map[1:N + 1]:
        with dace.tasklet:
            a << A[i]
            b >> A[i]
            b = a + a


def test_vectorization():
    sdfg: dace.SDFG = tovec.to_sdfg()
    assert sdfg.apply_transformations(Vectorization, options={'vector_len':
                                                              2}) == 1
    assert 'vec<double, 2>' in sdfg.generate_code()[0].code
    A = np.random.rand(20)
    B = sdfg(A=A)
    assert np.allclose(B, A * 2)


def test_vectorization_uneven():
    sdfg: dace.SDFG = tovec_uneven.to_sdfg()

    A = np.ones([22], np.float64)
    result = np.array([1.] + [2.] * 20 + [1.], dtype=np.float64)
    sdfg(A=A, N=20)
    assert np.allclose(A, result)

    sdfg.coarsen_dataflow()
    assert sdfg.apply_transformations(Vectorization, options={'vector_len':
                                                              2}) == 1
    assert 'vec<double, 2>' in sdfg.generate_code()[0].code

    A = np.ones([22], np.float64)
    sdfg(A=A, N=20)
    assert np.allclose(A, result)


def test_vectorization_postamble():
    sdfg: dace.SDFG = tovec_sym.to_sdfg()
    sdfg.coarsen_dataflow()
    assert sdfg.apply_transformations(Vectorization) == 1
    assert 'vec<float, 4>' in sdfg.generate_code()[0].code
    csdfg = sdfg.compile()

    for N in range(24, 29):
        x = np.random.rand(N).astype(np.float32)
        y = np.random.rand(N).astype(np.float32)
        z = np.random.rand(N).astype(np.float32)
        expected = x + y + z

        csdfg(x=x, y=y, z=z, N=N)
        assert np.allclose(z, expected)


def test_propagate_parent():
    sdfg: dace.SDFG = tovec.to_sdfg()
    assert sdfg.apply_transformations(Vectorization,
                                      options={
                                          'vector_len': 2,
                                          'propagate_parent': True
                                      }) == 1
    assert 'vec<double, 2>' in sdfg.generate_code()[0].code
    A = np.random.rand(20)
    B = sdfg(A=A)
    assert np.allclose(B.reshape(20), A * 2)


if __name__ == '__main__':
    test_vectorization()
    test_vectorization_uneven()
    test_vectorization_postamble()
    test_propagate_parent()
