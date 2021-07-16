# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
import math
import dace
import polybench

M = dace.symbol('M')
N = dace.symbol('N')

#datatypes = [dace.float64, dace.int32, dace.float32]
datatype = dace.float64

# Dataset sizes
sizes = [{
    M: 28,
    N: 32
}, {
    M: 80,
    N: 100
}, {
    M: 240,
    N: 260
}, {
    M: 1200,
    N: 1400
}, {
    M: 2600,
    N: 3000
}]

args = [([N, M], datatype), ([M, M], datatype), ([M], datatype),
        ([M], datatype), M, N]


def init_array(data, corr, mean, stddev, M, N):
    n = N.get()
    m = M.get()
    for i in range(n):
        for j in range(m):
            data[i, j] = datatype(i * j) / m + i


@dace.program(datatype[N, M], datatype[M, M], datatype[M], datatype[M])
def correlation(data, corr, mean, stddev):
    for j in range(M):
        mean[j] = 0.0
        for i in range(N):
            with dace.tasklet:
                data_in << data[i, j]
                mean_in << mean[j]
                mean_out >> mean[j]
                mean_out = mean_in + data_in
            # mean[j] += data[i, j]
        mean[j] /= N

    for j in range(M):
        stddev[j] = 0.0
        for i in range(N):
            with dace.tasklet:
                mean_in << mean[j]
                data_in << data[i, j]
                stddev_in << stddev[j]
                stddev_out >> stddev[j]
                stddev_out = stddev_in + ((data_in-mean_in) * (data_in-mean_in))
            # stddev[j] += (data[i, j] - mean[j]) * (data[i, j] - mean[j])
        with dace.tasklet:
            inp << stddev[j]
            out >> stddev[j]
            out = math.sqrt(inp / N)
            if out <= 0.1:
                out = 1.0
        # stddev[j] = math.sqrt(stddev[j] / N)

    for i in range(N):
        for j in range(M):
            # data[i, j] -= mean[j]
            # data[i, j] /= math.sqrt(datatype(N)) * stddev[j]
            with dace.tasklet:
                ind << data[i, j]
                m << mean[j]
                sd << stddev[j]
                oud >> data[i, j]
                oud = (ind - m) / (math.sqrt(datatype(N)) * sd)

    for i in range(M-1):
        corr[i, i] = 1.0
        for j in range(i+1, M):
            corr[i, j] = 0.0
            for k in range(N):
                with dace.tasklet:
                    d1_in << data[k, i]
                    d2_in << data[k, j]
                    corr_in << corr[i, j]
                    corr_out >> corr[i, j]
                    corr_out = corr_in + (d1_in * d2_in)
                # corr[i, j] += (data[k, i] * data[k, j])
            corr[j, i] = corr[i, j]

    corr[M - 1, M - 1] = 1.0

if __name__ == '__main__':
    polybench.main(sizes, args, [(1, 'corr')], init_array, correlation)


