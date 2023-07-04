import dace
import numpy as np

KLEV = dace.symbol('KLEV')
NBLOCKS = dace.symbol('NBLOCKS')
# KLON = dace.symbol('KLON')
KLON = 5


@dace.program
def small_wip_4(inp1: dace.float64[KLEV+1, KLON, NBLOCKS], inp2: dace.float64[KLEV, KLON, NBLOCKS],
                out: dace.float64[KLEV, KLON, NBLOCKS]):
    for jn in dace.map[0:NBLOCKS]:
        tmp = np.zeros([KLEV, KLON])
        for jl in range(KLON):
            for jk in range(KLEV):
                tmp[jk, jl] = inp1[jk, jl, jn] - inp2[jk, jl, jn]

        for jl in range(KLON):
            for jk in range(1, KLEV):
                # if tmp[jk, jl] > 0.5:
                out[jk, jl, jn] = inp1[jk, jl, jn] - tmp[jk-1, jl] + tmp[jk, jl]
                # else:
                    # out[jk, jl, jn] = inp2[jk, jl, jn] - tmp[jk-1, jl] + tmp[jk, jl]
