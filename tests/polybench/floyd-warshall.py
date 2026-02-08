# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
import math
import dace
try:
    import polybench
except ImportError:
    polybench = None

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

    @dace.mapscope
    def k_map(k: _[0:N]):

        @dace.map
        def ij_map(i: _[0:N], j: _[0:N]):
            ik_dist << path[i, k]
            kj_dist << path[k, j]
            out >> path(1, lambda x, y: min(x, y))[i, j]
            out = ik_dist + kj_dist


if __name__ == '__main__':
    if polybench:
        polybench.main(sizes, args, [(0, 'path')], init_array, floyd_warshall)
    else:
        init_array(*args, **{str(k).lower(): v for k, v in sizes[2].items()})
        floyd_warshall(*args)
