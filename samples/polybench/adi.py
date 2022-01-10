# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
import dace
import polybench

N = dace.symbol('N')
tsteps = dace.symbol('tsteps')

#datatypes = [dace.float64, dace.int32, dace.float32]
datatype = dace.float64

# Dataset sizes
sizes = [{
    tsteps: 20,
    N: 20
}, {
    tsteps: 40,
    N: 60
}, {
    tsteps: 100,
    N: 200
}, {
    tsteps: 500,
    N: 1000
}, {
    tsteps: 1000,
    N: 2000
}]

args = [([N, N], datatype)]


def init_array(u):
    n = N.get()
    for i in range(n):
        for j in range(n):
            u[i, j] = datatype(i + n - j) / n


@dace.program
def adi(u: datatype[N, N]):
    v = dace.define_local([N, N], datatype)
    p = dace.define_local([N, N], datatype)
    q = dace.define_local([N, N], datatype)

    A = dace.define_local([1], datatype)
    B = dace.define_local([1], datatype)
    C = dace.define_local([1], datatype)
    D = dace.define_local([1], datatype)
    E = dace.define_local([1], datatype)
    F = dace.define_local([1], datatype)

    with dace.tasklet:
        out_a >> A
        out_b >> B
        out_c >> C
        out_d >> D
        out_e >> E
        out_f >> F
        out_a = -(datatype(2) * (datatype(1) / tsteps) / (datatype(1) / (N * N))) / datatype(2)
        out_b = datatype(1) + (datatype(2) * (datatype(1) / tsteps) / (datatype(1) / (N * N)))
        out_c = -(datatype(2) * (datatype(1) / tsteps) / (datatype(1) / (N * N))) / datatype(2)
        out_d = -(datatype(1) * (datatype(1) / tsteps) / (datatype(1) / (N * N))) / datatype(2)
        out_e = datatype(1) + (datatype(1) * (datatype(1) / tsteps) / (datatype(1) / (N * N)))
        out_f = -(datatype(1) * (datatype(1) / tsteps) / (datatype(1) / (N * N))) / datatype(2)

    for t in range(tsteps):
        # Column Sweep
        for i in dace.map[1:N - 1]:
            with dace.tasklet:
                v0i >> v[0, i]
                pi0 >> p[i, 0]
                qi0 >> q[i, 0]
                v0i = 1.0
                pi0 = 0.0
                qi0 = 1.0

            for j in range(1, N - 1):
                with dace.tasklet:
                    a << A
                    b << B
                    c << C
                    d << D
                    f << F
                    pjm1 << p[i, j - 1]
                    qjm1 << q[i, j - 1]
                    uim1 << u[j, i - 1]
                    uji << u[j, i]
                    uip1 << u[j, i + 1]
                    pij >> p[i, j]
                    qij >> q[i, j]

                    pij = -c / (a * pjm1 + b)
                    qij = (-d * uim1 + (1.0 + 2.0 * d) * uji - f * uip1 - a * qjm1) / (a * pjm1 + b)
            with dace.tasklet:
                out >> u[i, N - 1]
                out = 1.0
            for j in range(N - 2, 0, -1):
                with dace.tasklet:
                    pij << p[i, j]
                    vjp1 << v[j + 1, i]
                    qij << q[i, j]
                    vji >> v[j, i]
                    vji = pij * vjp1 + qij

        # Row Sweep
        for i in dace.map[1:N - 1]:
            with dace.tasklet:
                ui0 >> u[i, 0]
                pi0 >> p[i, 0]
                qi0 >> q[i, 0]
                ui0 = 1.0
                pi0 = 0.0
                qi0 = 1.0

            for j in range(1, N - 1):
                with dace.tasklet:
                    a << A
                    c << C
                    d << D
                    e << E
                    f << F
                    pjm1 << p[i, j - 1]
                    qjm1 << q[i, j - 1]
                    vim1 << v[i - 1, j]
                    vij << v[i, j]
                    vip1 << v[i + 1, j]
                    pij >> p[i, j]
                    qij >> q[i, j]

                    pij = -f / (d * pjm1 + e)
                    qij = (-a * vim1 + (1.0 + 2.0 * a) * vij - c * vip1 - d * qjm1) / (d * pjm1 + e)
            with dace.tasklet:
                out >> u[i, N - 1]
                out = 1.0
            for j in range(N - 2, 0, -1):
                with dace.tasklet:
                    pij << p[i, j]
                    ujp1 << u[i, j + 1]
                    qij << q[i, j]
                    uij >> u[i, j]
                    uij = pij * ujp1 + qij


if __name__ == '__main__':
    polybench.main(sizes, args, [(0, 'u')], init_array, adi)
