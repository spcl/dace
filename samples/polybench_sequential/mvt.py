# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
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

args = [([N], datatype), ([N], datatype), ([N], datatype), ([N], datatype),
        ([N, N], datatype)]


def init_array(x1, x2, y_1, y_2, A):
    n = N.get()

    for i in range(n):
        x1[i] = datatype(i % n) / n
        x2[i] = datatype((i + 1) % n) / n
        y_1[i] = datatype((i + 3) % n) / n
        y_2[i] = datatype((i + 4) % n) / n
        for j in range(n):
            A[i, j] = datatype(i * j % n) / n


@dace.program(datatype[N], datatype[N], datatype[N], datatype[N],
              datatype[N, N])
def mvt(x1, x2, y_1, y_2, A):
    for i in range(N):
        for j in range(N):
            with dace.tasklet:
                in_A1 << A[i, j]
                iny1 << y_1[j]
                x1_in << x1[i]
                x1_out >> x1[i]
                x1_out = x1_in + in_A1 * iny1
            # x1[i] = x1[i] + A[i][j] * y_1[j]
    for i in range(N):
        for j in range(N):
            with dace.tasklet:
                in_A2 << A[j, i]
                iny2 << y_2[j]
                x2_in << x2[i]
                x2_out >> x2[i]
                x2_out = x2_in + in_A2 * iny2
            # x2[i] = x2[i] + A[j][i] * y_2[j]


if __name__ == '__main__':
    polybench.main(sizes, args, [(0, 'x1'), (1, 'x2')], init_array, mvt)
