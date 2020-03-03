import dace
import numpy as np
import math


@dace.program
def loop_nesting(A: dace.float32[100]):
    for i in range(0, 100, 10):
        tmp = np.ndarray([10], np.float32)
        tmp << A[i:i + 10]
        for j in dace.map[0:10]:
            with dace.tasklet:
                t << tmp[j]
                t2 >> tmp[j]
                t2 = t * 2
        tmp >> A[i:i + 10]


def test_loop_nesting():
    A = np.random.rand(100).astype(np.float32)
    expected = A.copy()
    for i in range(0, 100, 10):
        expected[i:i + 10] *= 2
    sdfg = loop_nesting.to_sdfg()
    sdfg.apply_strict_transformations()
    sdfg(A=A)
    assert np.allclose(A, expected)


@dace.program
def nested(A: dace.float64[20], i: dace.int32):
    return A + i


@dace.program
def symscope_loop(A: dace.float64[20]):
    for i in range(1, 20):
        A[:] = nested(A, i)


def test_symbol_loop():
    A = np.random.rand(20)
    expected = np.copy(A)
    for i in range(1, 20):
        expected += i

    symscope_loop(A)
    assert np.allclose(A, expected)


@dace.program
def symscope_map(A: dace.float64[19, 20]):
    for i in dace.map[1:20]:
        A[i - 1, :] = nested(A[i - 1, :], i)


def test_symbol_map():
    A = np.random.rand(19, 20)
    expected = np.copy(A)
    for i in range(1, 20):
        expected[i - 1] += i
    symscope_map(A)
    assert np.allclose(A, expected)


N = dace.symbol('N')
p = dace.symbol('p')
v = 4


@dace.program
def nested2(A: dace.float64[N, N], k0: dace.int32):
    for k in range(k0 + p, k0 + v + p):
        for i in parrange(k + 1, k0 + v + p):
            # A[i,k] /= A[k,k]
            with dace.tasklet:
                akk << A[k, k]
                aik << A[i, k]
                aout = aik / akk
                aout >> A[i, k]


@dace.program
def symscope_combined(A: dace.float64[N, N]):
    for k in range(0, N, v):
        for pp in parrange(0, 1):
            nested2(A, k0=k, p=pp, N=N)


def test_symscope_combined():
    symscope_combined.compile()


if __name__ == '__main__':
    test_loop_nesting()
    test_symbol_loop()
    test_symbol_map()
    test_symscope_combined()
