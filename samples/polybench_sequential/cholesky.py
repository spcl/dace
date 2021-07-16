# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
import math
import dace
import polybench
import numpy as np

N = dace.symbol('N')

#datatypes = [dace.float64, dace.int32, dace.float32]
datatype = dace.float64

# Dataset sizes
sizes = [{N: 40}, {N: 120}, {N: 400}, {N: 2000}, {N: 4000}]

args = [([N, N], datatype)]


def init_array(A):
    n = N.get()

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
    for i in range(N):
        for j in range(i):
            for k in range(j):
                with dace.tasklet:
                    Aij_out >> A[i][j]
                    Aij_in << A[i][j]
                    Aik_in << A[i][k]
                    Ajk_in << A[j][k]
                    Aij_out = Aij_in - (Aik_in * Ajk_in)
            with dace.tasklet:
                Aij_out >> A[i][j]
                Aij_in << A[i][j]
                Ajj_in << A[j][j]
                Aij_out = Aij_in / Ajj_in
        for k in range(i):
            with dace.tasklet:
                Aii_out >> A[i][i]
                Aii_in << A[i][i]
                Aik_in << A[i][k]
                Aii_out = Aii_in - (Aik_in * Aik_in)
        with dace.tasklet:
            Aii_out >> A[i][i]
            Aii_in << A[i][i]
            Aii_out = math.sqrt(Aii_in)

def print_result(filename, *args):
    with open(filename, 'w') as fp:
        fp.write("==BEGIN DUMP_ARRAYS==\n")
        fp.write("begin dump: %s\n" % 'A')
        for i in range(0, N.get()):
            for j in range(0, i + 1):
                fp.write("{:.7f} ".format(args[0][i, j]))
            fp.write("\n")
        fp.write("\nend   dump: %s\n" % 'A')
        fp.write("==END   DUMP_ARRAYS==\n")


if __name__ == '__main__':
    polybench.main(sizes, args, print_result, init_array, cholesky)
