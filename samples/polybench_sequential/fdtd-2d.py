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


def init_array(ex, ey, hz, _fict_):  #, NX, NY, TMAX):
    nx = NX.get()
    ny = NY.get()
    tmax = TMAX.get()
    for i in range(tmax):
        _fict_[i] = datatype(i)
    for i in range(nx):
        for j in range(ny):
            ex[i, j] = datatype(i * (j + 1.0)) / nx
            ey[i, j] = datatype(i * (j + 2.0)) / ny
            hz[i, j] = datatype(i * (j + 3.0)) / nx
    pass


@dace.program(datatype[NX, NY], datatype[NX, NY], datatype[NX, NY],
              datatype[TMAX])
def fdtd2d(ex, ey, hz, _fict_):
    for t in range(TMAX):
        for j in range(NY):
            with dace.tasklet:
                fict_in << _fict_[t]
                ey_out >> ey[0][j]
                ey_out = fict_in
        for i in range(1, NX):
            for j in range(NY):
                with dace.tasklet:
                    hz0_in << hz[i][j]
                    hz1_in << hz[i-1][j]
                    ey_in << ey[i][j]
                    ey_out >> ey[i][j]
                    ey_out = ey_in - (0.5 * (hz0_in-hz1_in))
                # ey[i][j] -= 0.5 * (hz[i][j]-hz[i-1][j])
        for i in range(NX):
            for j in range(1, NY):
                with dace.tasklet:
                    hz0_in << hz[i][j]
                    hz1_in << hz[i][j-1]
                    ex_in << ex[i][j]
                    ex_out >> ex[i][j]
                    ex_out = ex_in - (0.5 * (hz0_in-hz1_in))
                # ex[i][j] -= 0.5 * (hz[i][j]-hz[i][j-1])
        for i in range(NX-1):
            for j in range(NY-1):
                with dace.tasklet:
                    ex0_in << ex[i][j+1]
                    ex1_in << ex[i][j]
                    ey0_in << ey[i+1][j]
                    ey1_in << ey[i][j]
                    hz_in << hz[i][j]
                    hz_out >> hz[i][j]
                    hz_out = hz_in - (0.7 * (ex0_in - ex1_in + ey0_in - ey1_in))
                # hz[i][j] -= 0.7 * (ex[i][j+1] - ex[i][j] + ey[i+1][j] - ey[i][j])


if __name__ == '__main__':
    polybench.main(sizes, args, [(0, 'ex'), (1, 'ey'), (2, 'hz')], init_array,
                   fdtd2d)
