import dace
import numpy as np


@dace.program
def oneret(A: dace.float64[20]):
    return A*2


def test_oneret():
    A = np.random.rand(20)
    result = np.random.rand(20)
    oneret(A, __return=result)
    assert np.allclose(result, A*2)


@dace.program
def multiret(A: dace.float64[20]):
    return A*3, A*4, A


def test_multiret():
    A = np.random.rand(20)
    result = [np.random.rand(20) for i in range(3)]
    multiret(A, __return_0=result[0], __return_1=result[1],
             __return_2=result[2])
    assert np.allclose(result[0], A * 3)
    assert np.allclose(result[1], A * 4)
    assert np.allclose(result[2], A)


if __name__ == '__main__':
    test_oneret()
    test_multiret()
