import dace

KLEV = dace.symbol('KLEV')
NBLOCKS = dace.symbol('NBLOCKS')
NCLV = dace.symbol('NCLV')


@dace.program
def mwe_memlet_range(
        inp1: dace.float64[KLEV, NBLOCKS],
        inp2: dace.float64[KLEV, NBLOCKS],
        inp3: dace.float64[KLEV, NBLOCKS, NCLV],
        out1: dace.float64[KLEV, NBLOCKS]):

    for jn in dace.map[0:NBLOCKS]:
        for jk in range(KLEV):
            if inp2[jn, jk] > 0.5:
                out1[jn, jk] = inp1[jn, jk] + inp3[jn, jk, 3]
            else:
                out1[jn, jk] = inp1[jn, jk] + inp3[jn, jk, 4]


