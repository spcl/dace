import dace
import numpy as np

KLEV = dace.symbol('KLEV')
NBLOCKS = dace.symbol('NBLOCKS')
KLON = 5


@dace.program
def small_wip_3(inp: dace.float64[KLEV+1, KLON, NBLOCKS], out: dace.float64[KLEV, KLON, NBLOCKS]
              ):
    for jn in dace.map[0:NBLOCKS]:
        tmp1 = np.zeros([KLEV+1, KLON])
        tmp2 = np.zeros([KLEV+1, KLON])

        for jl in range(KLON):
            for jk in range(KLEV):
                tmp1[jk, jl] = inp[jk, jl, jn] + inp[jk+1, jl, jn]

        for jl in range(KLON):
            for jk in range(KLEV):
                tmp2[jk, jl] = inp[jk, jl, jn] - inp[jk+1, jl, jn]

        for jl in range(KLON):
            for jk in range(KLEV):
                out[jk, jl, jn] = tmp1[jk, jl] - tmp2[jk, jl]
