# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
import math
import dace
import polybench

M = dace.symbol('M')
N = dace.symbol('N')

#datatypes = [dace.float64, dace.int32, dace.float32]
datatype = dace.float64

# Dataset sizes
sizes = [{M: 28, N: 32}, {M: 80, N: 100}, {M: 240, N: 260}, {M: 1200, N: 1400}, {M: 2600, N: 3000}]

args = [([N, M], datatype), ([M, M], datatype), ([M], datatype), ([M], datatype), M, N]


def init_array(data, corr, mean, stddev, M, N):
    n = N.get()
    m = M.get()
    for i in range(n):
        for j in range(m):
            data[i, j] = datatype(i * j) / m + i


@dace.program(datatype[N, M], datatype[M, M], datatype[M], datatype[M])
def correlation(data, corr, mean, stddev):
    @dace.map
    def comp_mean(j: _[0:M], i: _[0:N]):
        inp << data[i, j]
        out >> mean(1, lambda x, y: x + y, 0)[j]
        out = inp

    @dace.map
    def comp_mean2(j: _[0:M]):
        inp << mean[j]
        out >> mean[j]
        out = inp / N

    @dace.map
    def comp_stddev(j: _[0:M], i: _[0:N]):
        inp << data[i, j]
        inmean << mean[j]
        out >> stddev(1, lambda x, y: x + y, 0)[j]
        out = (inp - inmean) * (inp - inmean)

    @dace.map
    def comp_stddev2(j: _[0:M]):
        inp << stddev[j]
        out >> stddev[j]
        out = math.sqrt(inp / N)
        if out <= 0.1:
            out = 1.0

    @dace.map
    def center_data(i: _[0:N], j: _[0:M]):
        ind << data[i, j]
        m << mean[j]
        sd << stddev[j]
        oud >> data[i, j]
        oud = (ind - m) / (math.sqrt(datatype(N)) * sd)

    @dace.map
    def comp_corr_diag(i: _[0:M]):
        corrout >> corr[i, i]
        corrout = 1.0

    @dace.mapscope
    def comp_corr_row(i: _[0:M - 1]):
        @dace.mapscope
        def comp_corr_col(j: _[i + 1:M]):
            @dace.map
            def comp_cov_k(k: _[0:N]):
                indi << data[k, i]
                indj << data[k, j]
                cov_ij >> corr(1, lambda x, y: x + y, 0)[i, j]
                cov_ij = (indi * indj)

    @dace.mapscope
    def symmetrize(i: _[0:M - 1]):
        @dace.map
        def symmetrize_col(j: _[i + 1:M]):
            corrin << corr[i, j]
            corrout >> corr[j, i]
            corrout = corrin


if __name__ == '__main__':
    polybench.main(sizes, args, [(1, 'corr')], init_array, correlation)
