import numpy as np
import dace

M, K = (dace.symbol(name) for name in ['M', 'K'])


@dace.program
def ttest(A: dace.float32[M, K], B: dace.float32[M, K]):
    s = np.ndarray(shape=(K, M), dtype=np.int32)
    t = np.ndarray(A.shape, A.dtype)

    for i, j in dace.map[0:M, 0:K]:
        t[i, j] = 1.0

    t += 5 * A
    B -= t
    # B += 5 * A @ B @ A @ B


if __name__ == '__main__':
    M.set(13)
    K.set(25)
    A = np.random.rand(M.get(), K.get()).astype(np.float32)
    B = np.random.rand(M.get(), K.get()).astype(np.float32)

    realB = B - 5 * A - 1.0
    ttest(A, B)

    diff = np.linalg.norm(B - realB) / (M.get() * K.get())
    print('Difference:', diff)
    exit(1 if diff >= 1e-5 else 0)
