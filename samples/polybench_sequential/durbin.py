# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
import dace
import polybench

N = dace.symbol('N')

#datatypes = [dace.float64, dace.int32, dace.float32]
datatype = dace.float64

# Dataset sizes
sizes = [{N: 40}, {N: 120}, {N: 400}, {N: 2000}, {N: 4000}]

args = [([N], datatype), ([N], datatype)]


def init_array(r, y):
    n = N.get()

    for i in range(0, n):
        r[i] = datatype(n + 1 - i)


@dace.program(datatype[N], datatype[N])
def durbin(r, y):
    alpha = dace.define_local([1], datatype)
    beta = dace.define_local([1], datatype)
    _sum = dace.define_local([1], datatype)
    z = dace.define_local([N], datatype)

    beta = 1.0
    with dace.tasklet:
        in_r << r[0]
        out_y >> y[0]
        out_a >> alpha
        out_y = -in_r
        out_a = -in_r
    # y[0] = -r[0]
    # alpha = -r[0]

    for k in range(1, N):
        with dace.tasklet:
            in_a << alpha
            in_b << beta
            out_b >> beta
            out_b = (1 - in_a * in_a) * in_b
        # beta = (1-alpha * alpha) * beta
        _sum = 0.0
        for i in range(k):
            with dace.tasklet:
                in_r << r[k-i-1]
                in_y << y[i]
                in_sum << _sum
                out_sum >> _sum
                out_sum = in_sum + (in_r * in_y)
            # _sum += r[k-i-1] * y[i]
        with dace.tasklet:
            in_r << r[k]
            in_sum << _sum
            in_b << beta
            out_a >> alpha
            out_a = -(in_r + in_sum) / in_b
        # alpha = - (r[k] + _sum) / beta
        for i in range(k):
            with dace.tasklet:
                in_y << y[i]
                kin_y << y[k-i-1]
                in_a << alpha
                out_z >> z[i]
                out_z = in_y + (in_a * kin_y)
            # z[i] = y[i] + alpha * y[k-i-1]
        for i in range(k):
            y[i] = z[i]
        y[k] = alpha


if __name__ == '__main__':
    polybench.main(sizes, args, [(1, 'y')], init_array, durbin)
