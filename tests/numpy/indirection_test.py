import dace
import numpy as np

# Declaration of symbolic variables
M, N = (dace.symbol(name) for name in ['M', 'N'])


@dace.program
def overlap(A: dace.float64[M], x: dace.int32[N]):

    for i in dace.map[0:M]:
        A[i] = 1.0
    for j in dace.map[1:N]:
        A[x[j]] += A[x[j-1]]


if __name__ == '__main__':
    overlap.compile()