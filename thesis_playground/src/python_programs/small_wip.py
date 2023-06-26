import dace
import numpy as np

KLEV = dace.symbol('KLEV')
NBLOCKS = dace.symbol('NBLOCKS')
KLON = 4


@dace.program
def small_wip(inp: dace.float64[KLEV+1, KLON, NBLOCKS], out: dace.float64[KLEV, KLON, NBLOCKS]):
    for jn in dace.map[0:NBLOCKS]:
        tmp = np.zeros([KLEV+1, KLON])
        for jl in range(KLON):
            for jk in range(KLEV):
                tmp[jk, jl] = inp[jk, jl, jn] + inp[jk+1, jl, jn]

        for jl in range(KLON):
            for jk in range(KLEV):
                out[jk, jl, jn] = tmp[jk, jl] + tmp[jk+1, jl]
