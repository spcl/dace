# Copyright 2019-2020 ETH Zurich and the DaCe authors. All rights reserved.
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
    def sub_mean(i: _[0:N], j: _[0:M]):
        ind << data[i, j]
        m << mean[j]
        oud >> data[i, j]
        oud = ind - m

    @dace.mapscope
    def comp_cov_row(i: _[0:M]):
        @dace.mapscope
        def comp_cov_col(j: _[i:M]):
            @dace.map
            def comp_cov_k(k: _[0:N]):
                indi << data[k, i]
                indj << data[k, j]
                cov_ij >> cov(1, lambda x, y: x + y, 0)[i, j]
                cov_ij = (indi * indj)

    @dace.mapscope
    def symmetrize(i: _[0:M]):
        @dace.map
        def symmetrize_col(j: _[i:M]):
            cov_ij << cov[i, j]
            covout >> cov(2)[:, :]
            covout[i, j] = cov_ij / (N - 1)
            covout[j, i] = cov_ij / (N - 1)


if __name__ == '__main__':
    polybench.main(sizes, args, [(1, 'cov')], init_array, covariance)
