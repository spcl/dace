# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
import dace
import polybench

N = dace.symbol('N')
M = dace.symbol('M')

#datatypes = [dace.float64, dace.int32, dace.float32]
datatype = dace.float64

# Dataset sizes
sizes = [{
    M: 38,
    N: 42,
}, {
    M: 116,
    N: 124,
}, {
    M: 390,
    N: 410,
}, {
    M: 1900,
    N: 2100,
}, {
    M: 1800,
    N: 2200,
}]

args = [([N, M], datatype), ([M], datatype), ([N], datatype), ([M], datatype),
        ([N], datatype)]


def init_array(A, s, q, p, r):
    n = N.get()
    m = M.get()

    for i in range(m):
        p[i] = datatype(i % m) / m
    for i in range(n):
        r[i] = datatype(i % n) / n
        for j in range(m):
            A[i, j] = datatype(i * (j + 1) % n) / n


@dace.program(datatype[N, M], datatype[M], datatype[N], datatype[M],
              datatype[N])
def bicg(A, s, q, p, r):
    for i in range(M):
        s[i] = 0
    for i in range(N):
        q[i] = 0
        for j in range(M):
            with dace.tasklet:
                r_in <<  r[i]
                a_in << A[i, j]
                s_in << s[j]
                s_out >> s[j]
                s_out = s_in + (r_in * a_in)
            # s[j] = s[j] + r[i] * A[i, j]
            with dace.tasklet:
                a_in << A[i, j]
                p_in << p[j]
                q_in << q[i]
                q_out >> q[i]
                q_out = q_in + (a_in * p_in)
            # q[i] = q[i] + A[i, j] * p[j]


if __name__ == '__main__':
    polybench.main(sizes, args, [(1, 's'), (2, 'q')], init_array, bicg)
