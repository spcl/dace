# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
import math
import dace
import polybench

W = dace.symbol('W')
H = dace.symbol('H')

#datatypes = [dace.float64, dace.int32, dace.float32]
datatype = dace.float32

# Dataset sizes
sizes = [{
    W: 64,
    H: 64,
}, {
    W: 192,
    H: 128,
}, {
    W: 720,
    H: 480,
}, {
    W: 4096,
    H: 2160,
}, {
    W: 7680,
    H: 4320,
}]

args = [
    ([W, H], datatype),
    ([W, H], datatype),
]

# Constants
alpha = datatype(0.25)
k = (datatype(1.0) - math.exp(-alpha)) * (datatype(1.0) - math.exp(-alpha)) / (
    datatype(1.0) + datatype(2.0) * alpha * math.exp(-alpha) - math.exp(datatype(2.0) * alpha))
a1 = a5 = k
a2 = a6 = k * math.exp(-alpha) * (alpha - datatype(1.0))
a3 = a7 = k * math.exp(-alpha) * (alpha + datatype(1.0))
a4 = a8 = -k * math.exp(datatype(-2.0) * alpha)
b1 = math.pow(datatype(2.0), -alpha)
b2 = -math.exp(datatype(-2.0) * alpha)
c1 = c2 = 1


def init_array(imgIn, imgOut, w, h):
    for i in range(w):
        for j in range(h):
            imgIn[i, j] = datatype((313 * i + 991 * j) % 65536) / 65535.0


@dace.program
def deriche(imgIn: datatype[W, H], imgOut: datatype[W, H]):
    y1 = dace.define_local([W, H], dtype=datatype)
    y2 = dace.define_local([W, H], dtype=datatype)
    ym1 = dace.define_local([1], datatype)
    ym2 = dace.define_local([1], datatype)
    xm1 = dace.define_local([1], datatype)
    tm1 = dace.define_local([1], datatype)
    yp1 = dace.define_local([1], datatype)
    yp2 = dace.define_local([1], datatype)
    xp1 = dace.define_local([1], datatype)
    xp2 = dace.define_local([1], datatype)
    tp1 = dace.define_local([1], datatype)
    tp2 = dace.define_local([1], datatype)

    for i in range(W):

        @dace.tasklet
        def reset():
            in_ym1 >> ym1
            in_ym2 >> ym2
            in_xm1 >> xm1
            in_ym1 = 0
            in_ym2 = 0
            in_xm1 = 0

        for j in range(H):

            @dace.tasklet
            def comp_y1():
                in_img << imgIn[i, j]
                in_xm1 << xm1
                in_ym1 << ym1
                in_ym2 << ym2
                out_y1 >> y1[i, j]
                out_xm1 >> xm1
                out_ym1 >> ym1
                out_ym2 >> ym2
                out_y1 = a1 * in_img + a2 * in_xm1 + b1 * in_ym1 + b2 * in_ym2
                out_xm1 = in_img
                out_ym2 = in_ym1
                out_ym1 = out_y1

    for i in range(W):

        @dace.tasklet
        def reset2():
            in_yp1 >> yp1
            in_yp2 >> yp2
            in_xp1 >> xp1
            in_xp2 >> xp2
            in_yp1 = 0
            in_yp2 = 0
            in_xp1 = 0
            in_xp2 = 0

        for j in range(H - 1, -1, -1):

            @dace.tasklet
            def comp_y2():
                in_img << imgIn[i, j]
                in_xp1 << xp1
                in_xp2 << xp2
                in_yp1 << yp1
                in_yp2 << yp2
                out_y2 >> y2[i, j]
                out_xp1 >> xp1
                out_xp2 >> xp2
                out_yp1 >> yp1
                out_yp2 >> yp2
                out_y2 = a3 * in_xp1 + a4 * in_xp2 + b1 * in_yp1 + b2 * in_yp2
                out_xp2 = in_xp1
                out_xp1 = in_img
                out_yp2 = in_yp1
                out_yp1 = out_y2

    @dace.map
    def comp_iout(i: _[0:W], j: _[0:H]):
        in_y1 << y1[i, j]
        in_y2 << y2[i, j]
        out_img >> imgOut[i, j]
        out_img = c1 * (in_y1 + in_y2)

    for j in range(H):

        @dace.tasklet
        def reset3():
            in_ym1 >> ym1
            in_ym2 >> ym2
            in_tm1 >> tm1
            in_ym1 = 0
            in_ym2 = 0
            in_tm1 = 0

        for i in range(W):

            @dace.tasklet
            def comp_y12():
                in_img << imgOut[i, j]
                in_tm1 << tm1
                in_ym1 << ym1
                in_ym2 << ym2
                out_y1 >> y1[i, j]
                out_tm1 >> tm1
                out_ym1 >> ym1
                out_ym2 >> ym2
                out_y1 = a5 * in_img + a6 * in_tm1 + b1 * in_ym1 + b2 * in_ym2
                out_tm1 = in_img
                out_ym2 = in_ym1
                out_ym1 = out_y1

    for j in range(H):

        @dace.tasklet
        def reset4():
            in_yp1 >> yp1
            in_yp2 >> yp2
            in_tp1 >> tp1
            in_tp2 >> tp2
            in_yp1 = 0
            in_yp2 = 0
            in_tp1 = 0
            in_tp2 = 0

        for i in range(W - 1, -1, -1):

            @dace.tasklet
            def comp_y22():
                in_img << imgOut[i, j]
                in_tp1 << tp1
                in_tp2 << tp2
                in_yp1 << yp1
                in_yp2 << yp2
                out_y2 >> y2[i, j]
                out_tp1 >> tp1
                out_tp2 >> tp2
                out_yp1 >> yp1
                out_yp2 >> yp2
                out_y2 = a7 * in_tp1 + a8 * in_tp2 + b1 * in_yp1 + b2 * in_yp2
                out_tp2 = in_tp1
                out_tp1 = in_img
                out_yp2 = in_yp1
                out_yp1 = out_y2

    @dace.map
    def comp_iout2(i: _[0:W], j: _[0:H]):
        in_y1 << y1[i, j]
        in_y2 << y2[i, j]
        out_img >> imgOut[i, j]
        out_img = c1 * (in_y1 + in_y2)


if __name__ == '__main__':
    polybench.main(sizes, args, [(1, 'imgOut')], init_array, deriche)
