# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
import dace

N = dace.symbol('N')

#datatypes = [dace.float64, dace.int32, dace.float32]
datatype = dace.float64

# Dataset sizes
sizes = [{N: 40}, {N: 120}, {N: 400}, {N: 2000}, {N: 4000}]

args = [([N, N], datatype), ([N], datatype), ([N], datatype)]


def init_array(L, x, b, n):
    x[:] = datatype(-999)
    for i in range(0, n, 1):
        b[i] = datatype(i)
    for i in range(0, n, 1):
        for j in range(0, i + 1, 1):
            L[i, j] = 2 * datatype(i + n - j + 1) / n
        for j in range(i + 1, n, 1):
            L[i, j] = datatype(0)


@dace.program
def trisolv(L: datatype[N, N], x: datatype[N], b: datatype[N]):
    # npbench formulation: forward substitution with a ``@`` inner product (Dot library node)
    # instead of a scalar ``j`` reduction loop.
    for i in range(N):
        x[i] = (b[i] - L[i, :i] @ x[:i]) / L[i, i]


if __name__ == '__main__':
    import polybench  # noqa: E402  (CLI only; corpus loads module without it)
    polybench.main(sizes, args, [(1, 'x')], init_array, trisolv)
