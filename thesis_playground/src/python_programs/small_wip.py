import dace
import numpy as np

KLEV = dace.symbol('KLEV')
NBLOCKS = dace.symbol('NBLOCKS')
# KLON = dace.symbol('KLON')
KLON = 5


@dace.program
def small_wip(inp1: dace.float64[KLEV+1, KLON, NBLOCKS], inp2: dace.float64[KLEV, KLON, NBLOCKS],
              out: dace.float64[KLEV, KLON, NBLOCKS]):
    for jn in dace.map[0:NBLOCKS]:
        tmp = np.zeros([KLEV, KLON])
        for jl in range(KLON):
            for jk in range(KLEV):
                tmp[jk, jl] = inp1[jk, jl, jn] - inp2[jk, jl, jn]

        for jl in range(KLON):
            for jk in range(KLEV):
                out[jk, jl, jn] = inp1[jk, jl, jn] - inp1[jk+1, jl, jn] + tmp[jk, jl]


def small_wip_manual(inp1: dace.float64[KLEV+1, KLON, NBLOCKS], inp2: dace.float64[KLEV, KLON, NBLOCKS],
                     out: dace.float64[KLEV, KLON, NBLOCKS]):

    for jn in dace.map[0:NBLOCKS]:
        for jl in range(KLON):
            for jk in range(KLEV):
                tmp = inp1[jk, jl, jn] - inp2[jk, jl, jn]
                out[jk, jl, jn] = inp1[jk, jl, jn] - inp1[jk+1, jl, jn] + tmp[jk, jl]
