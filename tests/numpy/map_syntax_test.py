import numpy as np
import dace

M, N, K = (dace.symbol(name) for name in ['M', 'N', 'K'])


@dace.program
def copy3d(A: dace.float32[M, N, K], B: dace.float32[M, N, K]):
    for i in parrange(M):
        for j, k in dace.map[0:N, 0:K]:
            with dace.tasklet:
                a << A[i, j, k]
                b >> B[i, j, k]
                b = a


if __name__ == '__main__':
    [sym.set(24) for sym in [M, N, K]]
    A = np.random.rand(M.get(), N.get(), K.get()).astype(np.float32)
    B = np.random.rand(M.get(), N.get(), K.get()).astype(np.float32)
    copy3d(A, B)

    diff = np.linalg.norm(B - A) / (M.get() * N.get())
    print('Difference:', diff)
    exit(1 if diff >= 1e-5 else 0)
