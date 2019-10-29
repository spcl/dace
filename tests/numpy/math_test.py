import numpy as np
import dace

M, N = 24, 24


@dace.program
def exponentf(A: dace.float32[M, N], B: dace.float32[M, N]):
    B[:] = exp(A)

@dace.program
def exponentc(A: dace.complex64[M, N], B: dace.complex64[M, N]):
    B[:] = exp(A)


if __name__ == '__main__':

    A = np.random.rand(M, N).astype(np.float32)
    daceB = np.zeros([M, N], dtype=np.float32)
    exponentf(A, daceB)
    numpyB = np.exp(A)
    relerr = np.linalg.norm(numpyB - daceB) / np.linalg.norm(numpyB)
    print('Relative error:', relerr)
    assert relerr < 1e-5

    A = np.random.rand(M, N).astype(np.float32) + 1j*np.random.rand(M, N).astype(np.float32)
    daceB = np.zeros([M, N], dtype=np.complex64)
    exponentc(A, daceB)
    numpyB = np.exp(A)
    relerr = np.linalg.norm(numpyB - daceB) / np.linalg.norm(numpyB)
    print('Relative error:', relerr)
    assert relerr < 1e-5
