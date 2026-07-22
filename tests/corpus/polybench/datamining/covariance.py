# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
import dace
import numpy as np

M = dace.symbol('M')
N = dace.symbol('N')

#datatypes = [dace.float64, dace.int32, dace.float32]
datatype = dace.float64

# Dataset sizes
sizes = [{M: 28, N: 32}, {M: 80, N: 100}, {M: 240, N: 260}, {M: 1200, N: 1400}, {M: 2600, N: 3000}]

args = [([N, M], datatype), ([M, M], datatype), ([M], datatype)]


def init_array(data, cov, mean, n, m):
    for i in range(n):
        for j in range(m):
            data[i, j] = datatype(i * j) / m


@dace.program
def covariance(data: datatype[N, M], cov: datatype[M, M], mean: datatype[M]):
    # npbench formulation: column-mean centering (a Reduce library node), then one
    # matrix-vector inner product ``data[:, i] @ data[:, i:M]`` (Gemv) per column, mirrored
    # across the diagonal. ``float_n`` in npbench is the row count ``N``.
    mean[:] = np.mean(data, axis=0)
    np.subtract(data, mean, out=data)
    for i in range(M):
        cov[i, i:M] = data[:, i] @ data[:, i:M] / (datatype(N) - 1.0)
        cov[i:M, i] = cov[i, i:M]


if __name__ == '__main__':
    import polybench  # noqa: E402  (CLI only; corpus loads module without it)
    polybench.main(sizes, args, [(1, 'cov')], init_array, covariance)
