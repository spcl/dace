import dace
import numpy as np
from dace.transformation.dataflow import Vectorization


@dace.program
def tovec(A: dace.float64[20]):
    return A + A


N = dace.symbol('N')


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

    assert sdfg.apply_transformations(Vectorization, options={'vector_len':
                                                              2}) == 1
    assert 'vec<double, 2>' in sdfg.generate_code()[0].code

    A = np.ones([22], np.float64)
    sdfg(A=A, N=20)
    assert np.allclose(A, result)


if __name__ == '__main__':
    test_vectorization()
    test_vectorization_uneven()