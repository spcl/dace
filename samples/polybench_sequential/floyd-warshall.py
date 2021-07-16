# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
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


def init_array(path):
    n = N.get()

    for i in range(n):
        for j in range(n):
            path[i, j] = datatype(i * j % 7 + 1)
            if (i + j) % 13 == 0 or (i + j) % 7 == 0 or (i + j) % 11 == 0:
                path[i, j] = datatype(999)


@dace.program(datatype[N, N])
def floyd_warshall(path):
    for k in range(N):
        for i in range(N):
            for j in range(N):
                with dace.tasklet:
                    ik_dist << path[i, k]
                    kj_dist << path[k, j]
                    path_in << path[i, j]
                    path_out >> path[i, j]
                    path_out = min(ik_dist + kj_dist, path_in)


if __name__ == '__main__':
    polybench.main(sizes, args, [(0, 'path')], init_array, floyd_warshall)