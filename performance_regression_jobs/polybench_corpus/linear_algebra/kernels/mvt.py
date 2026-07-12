# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
import dace

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


def init_array(x1, x2, y_1, y_2, A, n):
    for i in range(n):
        x1[i] = datatype(i % n) / n
        x2[i] = datatype((i + 1) % n) / n
        y_1[i] = datatype((i + 3) % n) / n
        y_2[i] = datatype((i + 4) % n) / n
        for j in range(n):
            A[i, j] = datatype(i * j % n) / n


@dace.program
def mvt(x1: datatype[N], x2: datatype[N], y_1: datatype[N], y_2: datatype[N], A: datatype[N, N]):

    # npbench formulation: ``x1 += A @ y_1`` and ``x2 += y_2 @ A`` (two Gemv library nodes).
    x1 += A @ y_1
    x2 += y_2 @ A


if __name__ == '__main__':
    import polybench  # noqa: E402  (CLI only; corpus loads module without it)
    polybench.main(sizes, args, [(0, 'x1'), (1, 'x2')], init_array, mvt)
