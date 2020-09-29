# Copyright 2019-2020 ETH Zurich and the DaCe authors. All rights reserved.
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

DX = datatype(1.0) / N
DY = datatype(1.0) / N
DT = datatype(1.0) / tsteps
B1 = datatype(2.0)
B2 = datatype(1.0)
mul1 = B1 * DT / (DX * DX)
mul2 = B2 * DT / (DY * DY)
a = -mul1 / datatype(2.0)
b = datatype(1.0) + mul1
c = a
d = -mul2 / datatype(2.0)
e = datatype(1.0) + mul2
f = d
args = [([N, N], datatype)]


def init_array(u):
    n = N.get()
    for i in range(n):
        for j in range(n):
            u[i, j] = datatype(i + n - j) / n


@dace.program(datatype[N, N])
def adi(u):
    v = dace.define_local([N, N], datatype)
    p = dace.define_local([N, N], datatype)
    q = dace.define_local([N, N], datatype)

    a = dace.define_local([1], datatype)
    b = dace.define_local([1], datatype)
    c = dace.define_local([1], datatype)
    d = dace.define_local([1], datatype)
    e = dace.define_local([1], datatype)
    f = dace.define_local([1], datatype)

    @dace.tasklet
    def init():
        out_a >> a
        out_b >> b
        out_c >> c
        out_d >> d
        out_e >> e
        out_f >> f
        out_a = -(datatype(2) * (datatype(1) / tsteps) /
                  (datatype(1) / (N * N))) / datatype(2)
        out_b = datatype(1) + (datatype(2) * (datatype(1) / tsteps) /
                               (datatype(1) / (N * N)))
        out_c = -(datatype(2) * (datatype(1) / tsteps) /
                  (datatype(1) / (N * N))) / datatype(2)
        out_d = -(datatype(1) * (datatype(1) / tsteps) /
                  (datatype(1) / (N * N))) / datatype(2)
        out_e = datatype(1) + (datatype(1) * (datatype(1) / tsteps) /
                               (datatype(1) / (N * N)))
        out_f = -(datatype(1) * (datatype(1) / tsteps) /
                  (datatype(1) / (N * N))) / datatype(2)

    for t in range(tsteps):
        # TODO(later): For more transformability, convert to nested SDFG
        @dace.map
        def colsweep(i: _[1:N - 1]):
            uin_prev << u[:, i - 1]
            uin << u[:, i]
            uin_next << u[:, i + 1]
            pin << p[i, :]
            qin << q[i, :]
            in_a << a
            in_b << b
            in_c << c
            in_d << d
            in_e << e
            in_f << f
            vin << v[:, i]

            # Intermediate outputs that are re-read are marked as
            # dynamic access
            v0i >> v(-1)[0, i]
            pi0 >> p(-1)[i, 0]
            qi0 >> q(-1)[i, 0]
            vNi >> v(-1)[N - 1, i]
            pout >> p[i, :]
            qout >> q[i, :]
            vout >> v[:, i]

            # Init
            v0i = datatype(1.0)
            pi0 = datatype(0.0)
            qi0 = datatype(1.0)

            # Column sweep
            for j in range(1, N - 1):
                pout[j] = -in_c / (in_a * pin[j - 1] + in_b)
                qout[j] = (-in_d * uin_prev[j] +
                           (datatype(1.0) + datatype(2.0) * in_d) * uin[j] -
                           in_f * uin_next[j] -
                           in_a * qin[j - 1]) / (in_a * pin[j - 1] + in_b)

            vNi = datatype(1.0)

            # Column sweep 2
            for j in range(N - 2, 0, -1):
                vout[j] = pin[j] * vin[j + 1] + qin[j]

        # TODO(later): For more transformability, convert to nested SDFG
        @dace.map
        def rowsweep(ir: _[1:N - 1]):
            vin_prev << v[ir - 1, :]
            vin << v[ir, :]
            vin_next << v[ir + 1, :]
            uin << u[ir, :]
            pin << p[ir, :]
            qin << q[ir, :]
            in_a << a
            in_b << b
            in_c << c
            in_d << d
            in_e << e
            in_f << f

            u0i >> u(-1)[ir, 0]
            pi0 >> p(-1)[ir, 0]
            qi0 >> q(-1)[ir, 0]

            pout >> p[ir, :]
            qout >> q[ir, :]
            uout >> u[ir, :]

            uNi >> u(-1)[ir, N - 1]

            u0i = datatype(1.0)
            pi0 = datatype(0.0)
            qi0 = datatype(1.0)

            for j in range(1, N - 1):
                pout[j] = -in_f / (in_d * pin[j - 1] + in_e)
                qout[j] = (-in_a * vin_prev[j] +
                           (datatype(1.0) + datatype(2.0) * in_a) * vin[j] -
                           in_c * vin_next[j] -
                           in_d * qin[j - 1]) / (in_d * pin[j - 1] + in_e)

            uNi = datatype(1.0)

            for j in range(N - 2, 0, -1):
                uout[j] = pin[j] * uin[j + 1] + qin[j]


if __name__ == '__main__':
    polybench.main(sizes, args, [(0, 'u')], init_array, adi)
