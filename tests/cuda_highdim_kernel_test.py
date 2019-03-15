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
    @dace.map
    def kernel(i: _[5:N - 5], j: _[0:M], k: _[7:K - 1], l: _[0:L]):
        @dace.map
        def block(a: _[0:X], b: _[0:Y], c: _[1:Z], d: _[2:W - 2], e: _[0:U]):
            input << A[i, j, k, l, a, b, c, d, e]
            output >> B(1, lambda a, b: a + b)[i, j, k, l]
            output = input


def makendrange(*args):
    result = []
    for i in range(0, len(args), 2):
        result.append((dace.eval(args[i]), dace.eval(args[i + 1] - 1), 1))
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

    A = np.random.randint(10, size=dims).astype(np.uint64)
    B = np.zeros(outdims, dtype=np.uint64)
    B_regression = np.zeros(outdims, dtype=np.uint64)

    # Equivalent python code
    for i, j, k, l in dace.ndrange(
            makendrange(5, N - 5, 0, M, 7, K - 1, 0, L)):
        for a, b, c, d, e in dace.ndrange(
                makendrange(0, X, 0, Y, 1, Z, 2, W - 2, 0, U)):
            B_regression[i, j, k, l] += A[i, j, k, l, a, b, c, d, e]

    highdim(A, B)

    diff = np.linalg.norm(B_regression - B) / dace.eval(N * M * K * L)
    print('Difference:', diff)
    exit(0 if diff <= 1e-5 else 1)
