# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
import math
import dace
import polybench

N = dace.symbol('N')

#datatypes = [dace.float64, dace.int32, dace.float32]
datatype = dace.float64

# Dataset sizes
sizes = [{
    N: 40,
}, {
    N: 120,
}, {
    N: 400,
}, {
    N: 2000,
}, {
    N: 4000,
}]

args = [([N], datatype), ([N], datatype), ([N], datatype), ([N], datatype), ([N, N], datatype)]


def init_array(x1, x2, y_1, y_2, A):
    n = N.get()

    for i in range(n):
        x1[i] = datatype(i % n) / n
        x2[i] = datatype((i + 1) % n) / n
        y_1[i] = datatype((i + 3) % n) / n
        y_2[i] = datatype((i + 4) % n) / n
        for j in range(n):
            A[i, j] = datatype(i * j % n) / n


@dace.program(datatype[N], datatype[N], datatype[N], datatype[N], datatype[N, N])
def mvt(x1, x2, y_1, y_2, A):
    @dace.map
    def compute(i: _[0:N], j: _[0:N]):
        in_A1 << A[i, j]
        in_A2 << A[j, i]
        iny1 << y_1[j]
        iny2 << y_2[j]
        out1 >> x1(1, lambda a, b: a + b)[i]
        out2 >> x2(1, lambda a, b: a + b)[i]
        out1 = in_A1 * iny1
        out2 = in_A2 * iny2


if __name__ == '__main__':
    polybench.main(sizes, args, [(0, 'x1'), (1, 'x2')], init_array, mvt)
