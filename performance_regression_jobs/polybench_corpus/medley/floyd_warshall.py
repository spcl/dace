# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
import dace
import numpy as np

N = dace.symbol('N')

#datatypes = [dace.float64, dace.int32, dace.float32]
datatype = dace.int32

# Dataset sizes
sizes = [{N: 60}, {N: 180}, {N: 500}, {N: 2800}, {N: 5600}]

args = [([N, N], datatype)]


def init_array(path, n):
    for i in range(n):
        for j in range(n):
            path[i, j] = datatype(i * j % 7 + 1)
            if (i + j) % 13 == 0 or (i + j) % 7 == 0 or (i + j) % 11 == 0:
                path[i, j] = datatype(999)


@dace.program
def floyd_warshall(path: datatype[N, N]):
    # npbench formulation: each k-step is a vectorized ``np.minimum`` against the outer-sum
    # ``np.add.outer(path[:, k], path[k, :])`` (elementwise maps, no scalar min-reduction loop).
    for k in range(N):
        path[:] = np.minimum(path[:], np.add.outer(path[:, k], path[k, :]))


if __name__ == '__main__':
    import polybench  # noqa: E402  (CLI only; corpus loads module without it)
    if polybench:
        polybench.main(sizes, args, [(0, 'path')], init_array, floyd_warshall)
    else:
        init_array(*args, **{str(k).lower(): v for k, v in sizes[2].items()})
        floyd_warshall(*args)
