import dace
import numpy as np


N = dace.symbol('N')
K = dace.symbol('K')
M = dace.symbol('M')

# method 1
@dace.program
def GEMM1(A: dace.float64[M, K], B: dace.float64[K, N],
             C: dace.float64[M, N]):
        # Transient variable
    tmp = np.ndarray([M, N, K], dtype=A.dtype)
    @dace.map
    def multiplication(i: _[0:M], j: _[0:N], k: _[0:K]):
        in_A << A[i,k]
        in_B << B[k,j]
        out >> tmp[i,j,k]
        out = in_A * in_B

    dace.reduce(lambda a, b: a + b, tmp, C, axis=2)

if __name__ == '__main__':
    GEMM1.compile(strict = False)
    N.set(300)
    K.set(300)
    M.set(300)
    A = numpy.random.rand(N.get(),K.get())
    B = numpy.random.rand(K.get(),M.get())
    R = numpy.ndarray(shape=[N,M])
