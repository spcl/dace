""" Tests automatic detection and baking of callbacks in the Python frontend. """
import dace
import numpy as np

N = dace.symbol('N')


def almost_gemm(A, alpha, B):
    return alpha * A @ B


def scale(C, beta):
    C *= beta


@dace.program
def autocallback(A: dace.float32[N, N], B: dace.float32[N, N], C: dace.float32[N, N], beta: dace.float32):
    tmp = almost_gemm(A, 0.5, B)
    scale(C, beta)
    C += tmp


def test_automatic_callback():
    N.set(24)
    A = np.random.rand(24, 24).astype(np.float32)
    B = np.random.rand(24, 24).astype(np.float32)
    C = np.random.rand(24, 24).astype(np.float32)
    beta = np.float32(np.random.rand())
    expected = 0.5 * A @ B + beta * C

    autocallback(A, B, C, beta)

    diff = np.linalg.norm(C - expected) / dace.eval(N * N)
    print('Difference:', diff)
    assert diff <= 1e-5


@dace.program
def modcallback(A: dace.float64[N, N], B: dace.float64[N]):
    tmp: dace.float64[N] = np.median(A, axis=1)
    B[:] = tmp


def test_callback_from_module():
    N.set(24)
    A = np.random.rand(24, 24)
    B = np.random.rand(24)
    modcallback(A, B)
    diff = np.linalg.norm(B - np.median(A, axis=1))
    print('Difference:', diff)
    assert diff <= 1e-5


def sq(a):
    return a * a


@dace.program
def tasklet_callback(A: dace.float64[N, N], B: dace.float64[N, N]):
    for i, j in dace.map[0:N, 0:N]:
        with dace.tasklet:
            a << A[i, j]
            b >> B[i, j]
            b = sq(a)


def test_callback_tasklet():
    N.set(24)
    A = np.random.rand(24, 24)
    B = np.random.rand(24, 24)
    tasklet_callback(A, B)
    diff = np.linalg.norm(B - A * A)
    print('Difference:', diff)
    assert diff <= 1e-5


if __name__ == '__main__':
    test_automatic_callback()
    test_callback_from_module()
    test_callback_tasklet()
