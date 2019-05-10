import numpy as np
import dapp

#M,N,K = (dapp.symbol(name) for name in ['M', 'N', 'K'])
M, N, K = 24, 24, 24


@dapp.program
def gemm(A: dapp.float32[M, K], B: dapp.float32[K, N], C: dapp.float32[M, N],
         alpha: dapp.float32, beta: dapp.float32):
    C[:] = alpha * A @ B + beta * C


if __name__ == '__main__':
    A = np.random.rand(M, K).astype(np.float32)
    B = np.random.rand(K, N).astype(np.float32)
    C = np.zeros([M, N], dtype=np.float32)
    gemm(A, B, C, 1.0, 1.0)

    realC = 1.0 * (A @ B) + 1.0 * C
    diff = np.linalg.norm(C - realC) / (M * N)
    print('Difference:', diff)
    exit(1 if diff >= 1e-5 else 0)
