# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
import dace

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
    # npbench formulation: slice-vectorized FDTD field updates.
    for t in range(TMAX):
        ey[0, :] = _fict_[t]
        ey[1:, :] -= 0.5 * (hz[1:, :] - hz[:-1, :])
        ex[:, 1:] -= 0.5 * (hz[:, 1:] - hz[:, :-1])
        hz[:-1, :-1] -= 0.7 * (ex[:-1, 1:] - ex[:-1, :-1] + ey[1:, :-1] - ey[:-1, :-1])


if __name__ == '__main__':
    import polybench  # noqa: E402  (CLI only; corpus loads module without it)
    polybench.main(sizes, args, [(0, 'ex'), (1, 'ey'), (2, 'hz')], init_array, fdtd2d)
