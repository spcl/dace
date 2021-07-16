# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
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

args = [([N, M], datatype), ([M, M], datatype), ([M], datatype), M, N]


def init_array(data, cov, mean, M, N):
    n = N.get()
    m = M.get()
    for i in range(n):
        for j in range(m):
            data[i, j] = datatype(i * j) / m


@dace.program(datatype[N, M], datatype[M, M], datatype[M])
def covariance(data, cov, mean):
    for j in range(M):
        with dace.tasklet:
            mean_out >> mean[j]
            mean_out = 0.0
        # mean[j] = 0.0
        for i in range(N):
            with dace.tasklet:
                data_in << data[i, j]
                mean_in << mean[j]
                mean_out >> mean[j]
                mean_out = mean_in + data_in
            # mean[j] += data[i, j]
        with dace.tasklet:
            mean_in << mean[j]
            mean_out >> mean[j]
            mean_out = mean_in / N
        # mean[j] /= N

    for i in range(N):
        for j in range(M):
            with dace.tasklet:
                ind << data[i, j]
                m << mean[j]
                oud >> data[i, j]
                oud = ind - m

    for i in range(M):
        for j in range(M):
            with dace.tasklet:
                cov_ij_out >> cov[i, j]
                cov_ij_out = 0.0
            # cov[i, j] = 0.0
            for k in range(N):
                with dace.tasklet:
                    indi << data[k, i]
                    indj << data[k, j]
                    cov_ij_in << cov[i, j]
                    cov_ij_out >> cov[i, j]
                    cov_ij_out = cov_ij_in + (indi * indj)
                # cov[i, j] += data[k, i] * data[k, j]
            with dace.tasklet:
                cov_ij_in << cov[i, j]
                cov_ij_out >> cov[i, j]
                # cov_ji_out >> cov[j, i]
                cov_ij_out = cov_ij_in / (N - 1)
                # cov_ji_out = cov_ij_out
            with dace.tasklet:
                cov_ij_in << cov[i, j]
                cov_ji_out >> cov[j, i]
                cov_ji_out = cov_ij_in
            # cov[i, j] /= (N - 1)
            # cov[j, i] = cov[i, j]


if __name__ == '__main__':
    polybench.main(sizes, args, [(1, 'cov')], init_array, covariance)
