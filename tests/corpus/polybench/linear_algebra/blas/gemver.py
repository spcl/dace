# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
import dace
import numpy as np

N = dace.symbol('N')

#datatypes = [dace.float64, dace.int32, dace.float32]
datatype = dace.float64

# Dataset sizes
sizes = [{N: 40}, {N: 120}, {N: 400}, {N: 2000}, {N: 4000}]

args = [([N, N], datatype), ([N], datatype), ([N], datatype), ([N], datatype), ([N], datatype), ([N], datatype),
        ([N], datatype), ([N], datatype), ([N], datatype), ([1], datatype), ([1], datatype)]

outputs = [(5, 'w')]


def init_array(A, u1, v1, u2, v2, w, x, y, z, alpha, beta, n):
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


@dace.program
def gemver(A: datatype[N, N], u1: datatype[N], v1: datatype[N], u2: datatype[N], v2: datatype[N], w: datatype[N],
           x: datatype[N], y: datatype[N], z: datatype[N], alpha: datatype[1], beta: datatype[1]):

    # npbench formulation: rank-2 update of ``A`` (two outer products), then two Gemv sweeps.
    # ``@`` lowers to Gemv library nodes; ``np.multiply.outer`` to an outer-product map.
    # ``alpha``/``beta`` are 1-element arrays in the corpus signature, so index the scalar out.
    A += np.multiply.outer(u1, v1) + np.multiply.outer(u2, v2)
    x += beta[0] * y @ A + z
    w += alpha[0] * A @ x


if __name__ == '__main__':
    import polybench  # noqa: E402  (CLI only; corpus loads module without it)
    polybench.main(sizes, args, outputs, init_array, gemver)
