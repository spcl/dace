# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
import math
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
    sum = dace.define_local([1], datatype)
    z = dace.define_local([N], datatype)

    @dace.tasklet
    def init():
        in_r << r[0]
        out_y >> y[0]
        out_a >> alpha
        out_b >> beta
        out_y = -in_r
        out_a = -in_r
        out_b = datatype(1)

    for k in range(1, N, 1):

        @dace.tasklet
        def k_init():
            in_a << alpha
            in_b << beta
            out_b >> beta
            out_sum >> sum
            out_b = (datatype(1) - in_a * in_a) * in_b
            out_sum = datatype(0)

        @dace.map
        def set_sum(i: _[0:k]):
            in_r << r[k - i - 1]
            in_y << y[i]
            out_sum >> sum(1, lambda x, y: x + y)
            out_sum = in_r * in_y

        @dace.tasklet
        def set_alpha():
            in_r << r[k]
            in_sum << sum
            in_b << beta
            out_a >> alpha
            out_a = -(in_r + in_sum) / in_b

        @dace.map
        def set_zeta(i: _[0:k]):
            in_y << y[i]
            kin_y << y[k - i - 1]
            in_a << alpha
            out_z >> z[i]
            out_z = in_y + in_a * kin_y

        @dace.map
        def set_y1(i: _[0:k]):
            in_z << z[i]
            out_y >> y[i]
            out_y = in_z

        @dace.tasklet
        def set_y2():
            in_a << alpha
            out_y >> y[k]
            out_y = in_a


if __name__ == '__main__':
    polybench.main(sizes, args, [(1, 'y')], init_array, durbin)
