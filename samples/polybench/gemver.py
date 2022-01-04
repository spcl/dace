# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
import math
import dace
import polybench

N = dace.symbol('N')

#datatypes = [dace.float64, dace.int32, dace.float32]
datatype = dace.float64

# Dataset sizes
sizes = [{N: 40}, {N: 120}, {N: 400}, {N: 2000}, {N: 4000}]

args = [([N, N], datatype), ([N], datatype), ([N], datatype), ([N], datatype), ([N], datatype), ([N], datatype),
        ([N], datatype), ([N], datatype), ([N], datatype), ([1], datatype), ([1], datatype)]

outputs = [(5, 'w')]


def init_array(A, u1, v1, u2, v2, w, x, y, z, alpha, beta):
    n = N.get()

    alpha[0] = datatype(1.5)
    beta[0] = datatype(1.2)

    for i in range(n):
        u1[i] = i
        u2[i] = ((i + 1) / n) / 2.0
        v1[i] = ((i + 1) / n) / 4.0
        v2[i] = ((i + 1) / n) / 6.0
        y[i] = ((i + 1) / n) / 8.0
        z[i] = ((i + 1) / n) / 9.0
        x[i] = 0.0
        w[i] = 0.0
        for j in range(n):
            A[i, j] = datatype(i * j % n) / n


@dace.program(datatype[N, N], datatype[N], datatype[N], datatype[N], datatype[N], datatype[N], datatype[N], datatype[N],
              datatype[N], datatype[1], datatype[1])
def gemver(A, u1, v1, u2, v2, w, x, y, z, alpha, beta):
    @dace.map
    def add_uv(i: _[0:N], j: _[0:N]):
        iu1 << u1[i]
        iv1 << v1[j]
        iu2 << u2[i]
        iv2 << v2[j]
        ia << A[i, j]
        oa >> A[i, j]

        oa = ia + iu1 * iv1 + iu2 * iv2

    @dace.map
    def comp_y(i: _[0:N], j: _[0:N]):
        ib << beta
        ia << A[j, i]
        iy << y[j]
        ox >> x(1, lambda a, b: a + b)[i]

        ox = ib * ia * iy

    @dace.map
    def comp_xz(i: _[0:N]):
        ix << x[i]
        iz << z[i]
        ox >> x[i]
        ox = ix + iz

    @dace.map
    def comp_w(i: _[0:N], j: _[0:N]):
        ialpha << alpha
        ia << A[i, j]
        ix << x[j]
        ow >> w(1, lambda a, b: a + b)[i]
        ow = ialpha * ia * ix


if __name__ == '__main__':
    polybench.main(sizes, args, outputs, init_array, gemver)
