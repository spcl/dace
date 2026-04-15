# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
import dace
import polybench

NX = dace.symbol('NX')
NY = dace.symbol('NY')
TMAX = dace.symbol('TMAX')

#datatypes = [dace.float64, dace.int32, dace.float32]
datatype = dace.float64

# Dataset sizes
sizes = [{
    TMAX: 20,
    NX: 20,
    NY: 30
}, {
    TMAX: 40,
    NX: 60,
    NY: 80
}, {
    TMAX: 100,
    NX: 200,
    NY: 240
}, {
    TMAX: 500,
    NX: 1000,
    NY: 1200
}, {
    TMAX: 1000,
    NX: 2000,
    NY: 2600
}]
args = [
    ([NX, NY], datatype),  # ex
    ([NX, NY], datatype),  # ey
    ([NX, NY], datatype),  # hz
    ([TMAX], datatype),  # _fict_
    # NX,
    # NY,
    # TMAX
]


def init_array(ex, ey, hz, _fict_, nx, ny, tmax):
    for i in range(tmax):
        _fict_[i] = datatype(i)
    for i in range(nx):
        for j in range(ny):
            ex[i, j] = datatype(i * (j + 1)) / nx
            ey[i, j] = datatype(i * (j + 2)) / ny
            hz[i, j] = datatype(i * (j + 3)) / nx


@dace.program
def fdtd2d(ex: datatype[NX, NY], ey: datatype[NX, NY], hz: datatype[NX, NY], _fict_: datatype[TMAX]):
    for t in range(TMAX):

        @dace.map
        def col0(j: _[0:NY]):
            fict << _fict_[t]
            out >> ey[0, j]
            out = fict

        @dace.map
        def update_ey(i: _[1:NX], j: _[0:NY]):
            eyin << ey[i, j]
            hz1 << hz[i, j]
            hz2 << hz[i - 1, j]
            eyout >> ey[i, j]
            eyout = eyin - datatype(0.5) * (hz1 - hz2)

        @dace.map
        def update_ex(i: _[0:NX], j: _[1:NY]):
            exin << ex[i, j]
            hz1 << hz[i, j]
            hz2 << hz[i, j - 1]
            exout >> ex[i, j]
            exout = exin - datatype(0.5) * (hz1 - hz2)

        @dace.map
        def update_hz(i: _[0:NX - 1], j: _[0:NY - 1]):
            hzin << hz[i, j]
            ex1 << ex[i, j + 1]
            ex2 << ex[i, j]
            ey1 << ey[i + 1, j]
            ey2 << ey[i, j]
            hzout >> hz[i, j]
            hzout = hzin - datatype(0.7) * (ex1 - ex2 + ey1 - ey2)


if __name__ == '__main__':
    polybench.main(sizes, args, [(0, 'ex'), (1, 'ey'), (2, 'hz')], init_array, fdtd2d)
