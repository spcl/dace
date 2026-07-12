# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
import dace

N = dace.symbol('N')
M = dace.symbol('M')

#datatypes = [dace.float64, dace.int32, dace.float32]
datatype = dace.float64

# Dataset sizes
sizes = [{
    M: 38,
    N: 42,
}, {
    M: 116,
    N: 124,
}, {
    M: 390,
    N: 410,
}, {
    M: 1900,
    N: 2100,
}, {
    M: 1800,
    N: 2200,
}]

args = [([N, M], datatype), ([M], datatype), ([N], datatype), ([M], datatype), ([N], datatype)]


def init_array(A, s, q, p, r, n, m):
    for i in range(m):
        p[i] = datatype(i % m) / m
    for i in range(n):
        r[i] = datatype(i % n) / n
        for j in range(m):
            A[i, j] = datatype(i * (j + 1) % n) / n


@dace.program
def bicg(A: datatype[N, M], s: datatype[M], q: datatype[N], p: datatype[M], r: datatype[N]):

    # npbench formulation: ``s = r @ A`` and ``q = A @ p`` (two Gemv library nodes).
    s[:] = r @ A
    q[:] = A @ p


if __name__ == '__main__':
    import polybench  # noqa: E402  (CLI only; corpus loads module without it)
    polybench.main(sizes, args, [(1, 's'), (2, 'q')], init_array, bicg)
