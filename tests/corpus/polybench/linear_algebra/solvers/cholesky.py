# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
import dace
import numpy as np

N = dace.symbol('N')

#datatypes = [dace.float64, dace.int32, dace.float32]
datatype = dace.float64

# Dataset sizes
sizes = [{N: 40}, {N: 120}, {N: 400}, {N: 2000}, {N: 4000}]

args = [([N, N], datatype)]


def init_array(A, n):
    for i in range(0, n, 1):
        for j in range(0, i + 1, 1):
            # Python does modulo, while C does remainder ...
            A[i, j] = datatype(-(j % n)) / n + 1
        for j in range(i + 1, n, 1):
            A[i, j] = datatype(0)
        A[i, i] = datatype(1)

    A[:] = np.dot(A, np.transpose(A))


@dace.program
def cholesky(A: datatype[N, N]):
    # ``numpy.linalg.cholesky`` returns the lower-triangular factor ``L`` (``A = L @ L.T``),
    # which the DaCe frontend lowers to the LAPACK-backed ``Cholesky`` (POTRF) library node
    # instead of a hand-written Crout loop nest. The original in-place kernel wrote ``L`` into
    # ``A``'s lower triangle + diagonal and left the strictly-upper triangle (the symmetric
    # input) untouched, so copy back only the lower triangle to preserve that exact convention.
    L = np.linalg.cholesky(A)
    for i in dace.map[0:N]:
        for j in dace.map[0:i + 1]:
            A[i, j] = L[i, j]


def print_result(filename, *args, n=None, **kwargs):
    with open(filename, 'w') as fp:
        fp.write("==BEGIN DUMP_ARRAYS==\n")
        fp.write("begin dump: %s\n" % 'A')
        for i in range(0, n):
            for j in range(0, i + 1):
                fp.write("{:.7f} ".format(args[0][i, j]))
            fp.write("\n")
        fp.write("\nend   dump: %s\n" % 'A')
        fp.write("==END   DUMP_ARRAYS==\n")


if __name__ == '__main__':
    import polybench  # noqa: E402  (CLI only; corpus loads module without it)
    polybench.main(sizes, args, print_result, init_array, cholesky)
