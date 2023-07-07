import dace

KLEV = dace.symbol('KLEV')
NBLOCKS = dace.symbol('NBLOCKS')
NCLV = dace.symbol('NCLV')
NCLDQL = dace.symbol('NCLDQL')
NCLDQI = dace.symbol('NCLDQI')


@dace.program
def mwe_map_similar_size(
        inp1: dace.float64[KLEV, NBLOCKS],
        inp2: dace.float64[KLEV, NBLOCKS],
        inp3: dace.float64[KLEV, NBLOCKS, NCLV],
        out1: dace.float64[KLEV, NBLOCKS]):

    for jn in dace.map[0:NBLOCKS]:
        for jk in range(KLEV):
            out1[jn, jk] = inp1[jn, jk] + inp3[jn, jk, NCLDQI]

        for jk in range(1, KLEV):
            out1[jn, jk] += inp2[jn, jk] + inp3[jn, jk, NCLDQL]
