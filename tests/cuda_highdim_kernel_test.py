# Copyright 2019-2020 ETH Zurich and the DaCe authors. All rights reserved.
import dace
import numpy as np

# Symbols
N = dace.symbol('N')
M = dace.symbol('M')
K = dace.symbol('K')
L = dace.symbol('L')

X = dace.symbol('X')
Y = dace.symbol('Y')
Z = dace.symbol('Z')
W = dace.symbol('W')
U = dace.symbol('U')


@dace.program
def highdim(A: dace.uint64[N, M, K, L, X, Y, Z, W, U],
            B: dace.uint64[N, M, K, L]):
    @dace.mapscope
    def kernel(i: _[5:N - 5], j: _[0:M], k: _[7:K - 1], l: _[0:L]):
        @dace.map
        def block(a: _[0:X], b: _[0:Y], c: _[1:Z], d: _[2:W - 2], e: _[0:U]):
            input << A[i, j, k, l, a, b, c, d, e]
            output >> B(1, lambda a, b: a + b)[i, j, k, l]
            output = input


def makendrange(*args):
    result = []
    for i in range(0, len(args), 2):
        result.append((args[i], args[i + 1] - 1, 1))
    return result


if __name__ == '__main__':
    # 4D kernel with 5D block
    N.set(12)
    M.set(3)
    K.set(14)
    L.set(15)
    X.set(1)
    Y.set(2)
    Z.set(3)
    W.set(4)
    U.set(5)
    dims = tuple(s.get() for s in (N, M, K, L, X, Y, Z, W, U))
    outdims = tuple(s.get() for s in (N, M, K, L))
    print('High-dimensional GPU kernel test', dims)

    A = dace.ndarray((N, M, K, L, X, Y, Z, W, U), dtype=dace.uint64)
    B = dace.ndarray((N, M, K, L), dtype=dace.uint64)
    A[:] = np.random.randint(10, size=dims).astype(np.uint64)
    B[:] = np.zeros(outdims, dtype=np.uint64)
    B_regression = np.zeros(outdims, dtype=np.uint64)

    # Equivalent python code
    for i, j, k, l in dace.ndrange(
            makendrange(5,
                        N.get() - 5, 0, M.get(), 7,
                        K.get() - 1, 0, L.get())):
        for a, b, c, d, e in dace.ndrange(
                makendrange(0, X.get(), 0, Y.get(), 1, Z.get(), 2,
                            W.get() - 2, 0, U.get())):
            B_regression[i, j, k, l] += A[i, j, k, l, a, b, c, d, e]

    highdim(A, B)

    diff = np.linalg.norm(B_regression - B) / (N.get() * M.get() * K.get() *
                                               L.get())
    print('Difference:', diff)
    exit(0 if diff <= 1e-5 else 1)
