# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
import dace
import polybench

M = dace.symbol('M')
N = dace.symbol('N')

#datatypes = [dace.float64, dace.int32, dace.float32]
datatype = dace.float64

# Dataset sizes
sizes = [{M: 28, N: 32}, {M: 80, N: 100}, {M: 240, N: 260}, {M: 1200, N: 1400}, {M: 2600, N: 3000}]

args = [([N, M], datatype), ([M, M], datatype), ([M], datatype)]


def init_array(data, cov, mean):
    n = N.get()
    m = M.get()
    for i in range(n):
        for j in range(m):
            data[i, j] = datatype(i * j) / m


@dace.program(datatype[N, M], datatype[M, M], datatype[M])
def covariance(data, cov, mean):
    mean[:] = 0.0

    @dace.map
    def comp_mean(j: _[0:M], i: _[0:N]):
        inp << data[i, j]
        out >> mean(1, lambda x, y: x + y)[j]
        out = inp

    @dace.map
    def comp_mean2(j: _[0:M]):
        inp << mean[j]
        out >> mean[j]
        out = inp / N

    @dace.map
    def sub_mean(i: _[0:N], j: _[0:M]):
        ind << data[i, j]
        m << mean[j]
        oud >> data[i, j]
        oud = ind - m

    @dace.mapscope
    def comp_cov_row(i: _[0:M]):
        @dace.mapscope
        def comp_cov_col(j: _[i:M]):
            with dace.tasklet:
                cov_ij >> cov[i, j]
                cov_ij = 0.0

            @dace.map
            def comp_cov_k(k: _[0:N]):
                indi << data[k, i]
                indj << data[k, j]
                cov_ij >> cov(1, lambda x, y: x + y)[i, j]
                cov_ij = (indi * indj)

            with dace.tasklet:
                cov_ij_in << cov[i, j]
                cov_ij_out >> cov[i, j]
                cov_ji_out >> cov[j, i]
                cov_ij_out = cov_ij_in / (N - 1)
                cov_ji_out = cov_ij_out


if __name__ == '__main__':
    polybench.main(sizes, args, [(1, 'cov')], init_array, covariance)
