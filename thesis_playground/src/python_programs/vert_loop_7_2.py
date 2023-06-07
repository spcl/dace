"""
Remove the unneccessary ZSOLQA computation
"""
import dace
import numpy as np


NBLOCKS = dace.symbol('NBLOCKS')
KLEV = dace.symbol('KLEV')
NCLV = dace.symbol('NCLV')
KLON = dace.symbol('KLON')
NCLDTOP = dace.symbol('NCLDTOP')
KFDIA = dace.symbol('KFDIA')
KIDIA = dace.symbol('KIDIA')
NCLDQL = dace.symbol('NCLDQL')
NCLDQI = dace.symbol('NCLDQI')
NCLDQS = dace.symbol('NCLDQS')
NCLDTOP = 2


@dace.program
def vert_loop_7_2(
            PTSPHY: dace.float64,
            RLMIN: dace.float64,
            ZEPSEC: dace.float64,
            RG: dace.float64,
            PLU_NF: dace.float64[KLEV, KLON, NBLOCKS],
            LDCUM_NF: dace.int32[KLON, NBLOCKS],
            PAPH_NF: dace.float64[KLEV+1, KLON, NBLOCKS],
            PLUDE_NF: dace.float64[KLEV, KLON, NBLOCKS]
        ):

    for JN in dace.map[0:NBLOCKS:KLON]:
        ZDTGDP = np.zeros([KLON], dtype=np.float64)
        ZDP = np.zeros([KLON], dtype=np.float64)
        ZGDP = np.zeros([KLON], dtype=np.float64)

        # The starting range should be NCLDTOP-1 and end KLEV but using these leads to very strange behaviour
        for JK in range(1, 137):
            # for JL in range(KIDIA-1, KFDIA):
            for JL in range(0, 1):
                ZDP[JL] = PAPH_NF[JK+1, JL, JN]-PAPH_NF[JK, JL, JN]
                ZGDP[JL] = RG/ZDP[JL]
                ZDTGDP[JL] = PTSPHY*ZGDP[JL]

            # for JL in range(KIDIA-1, KFDIA):
            for JL in range(0, 1):
                PLUDE_NF[JK, JL, JN] = PLUDE_NF[JK, JL, JN]*ZDTGDP[JL]
                if not (LDCUM_NF[JL, JN] and PLUDE_NF[JK, JL, JN] > RLMIN and PLU_NF[JK+1, JL, JN] > ZEPSEC):
                    PLUDE_NF[JK, JL, JN] = 0.0
