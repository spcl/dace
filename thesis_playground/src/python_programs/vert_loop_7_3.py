"""
ZSOLQA is passed in and out as well
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
def vert_loop_7_3(
            PTSPHY: dace.float64,
            RLMIN: dace.float64,
            ZEPSEC: dace.float64,
            RG: dace.float64,
            RTHOMO: dace.float64,
            ZALFAW: dace.float64,
            PLU_NF: dace.float64[KLEV, KLON, NBLOCKS],
            LDCUM_NF: dace.int32[KLON, NBLOCKS],
            PSNDE_NF: dace.float64[KLEV, KLON, NBLOCKS],
            PAPH_NF: dace.float64[KLEV+1, KLON, NBLOCKS],
            PSUPSAT_NF: dace.float64[KLEV, KLON, NBLOCKS],
            PT_NF: dace.float64[KLEV, KLON, NBLOCKS],
            tendency_tmp_t_NF: dace.float64[KLEV, KLON, NBLOCKS],
            PLUDE_NF: dace.float64[KLEV, KLON, NBLOCKS],
            ZSOLQA_NF: dace.float64[NCLV, NCLV, KLON, NBLOCKS]
        ):

    for JN in dace.map[0:NBLOCKS:KLON]:
        ZCONVSRCE = np.zeros([KLON, NCLV], dtype=np.float64)
        ZDTGDP = np.zeros([KLON], dtype=np.float64)
        ZDP = np.zeros([KLON], dtype=np.float64)
        ZGDP = np.zeros([KLON], dtype=np.float64)
        ZTP1 = np.zeros([KLON], dtype=np.float64)

        # The starting range should be NCLDTOP-1 and end KLEV but using these leads to very strange behaviour
        for JK in range(1, 137):
            # for JL in range(KIDIA-1, KFDIA):
            for JL in range(0, 1):
                ZTP1[JL] = PT_NF[JK, JL, JN] + PTSPHY * tendency_tmp_t_NF[JK, JL, JN]
                if PSUPSAT_NF[JK, JL, JN] > ZEPSEC:
                    if ZTP1[JL] > RTHOMO:
                        ZSOLQA_NF[NCLDQL, NCLDQL, JL, JN] = ZSOLQA_NF[NCLDQL, NCLDQL, JL, JN] + PSUPSAT_NF[JK, JL, JN]
                    else:
                        ZSOLQA_NF[NCLDQI, NCLDQI, JL, JN] = ZSOLQA_NF[NCLDQI, NCLDQI, JL, JN] + PSUPSAT_NF[JK, JL, JN]

            # for JL in range(KIDIA-1, KFDIA):
            for JL in range(0, 1):
                ZDP[JL] = PAPH_NF[JK+1, JL, JN]-PAPH_NF[JK, JL, JN]
                ZGDP[JL] = RG/ZDP[JL]
                ZDTGDP[JL] = PTSPHY*ZGDP[JL]

            # for JL in range(KIDIA-1, KFDIA):
            for JL in range(0, 1):
                PLUDE_NF[JK, JL, JN] = PLUDE_NF[JK, JL, JN]*ZDTGDP[JL]
                if LDCUM_NF[JL, JN] and PLUDE_NF[JK, JL, JN] > RLMIN and PLU_NF[JK+1, JL, JN] > ZEPSEC:
                    ZCONVSRCE[JL, NCLDQL] = ZALFAW*PLUDE_NF[JK, JL, JN]
                    ZCONVSRCE[JL, NCLDQI] = [1.0 - ZALFAW]*PLUDE_NF[JK, JL, JN]
                    ZSOLQA_NF[NCLDQL, NCLDQL, JL, JN] = ZSOLQA_NF[NCLDQL, NCLDQL, JL, JN] + ZCONVSRCE[JL, NCLDQL]
                    ZSOLQA_NF[NCLDQI, NCLDQI, JL, JN] = ZSOLQA_NF[NCLDQI, NCLDQI, JL, JN] + ZCONVSRCE[JL, NCLDQI]
                else:
                    PLUDE_NF[JK, JL, JN] = 0.0

                if LDCUM_NF[JL, JN]:
                    ZSOLQA_NF[NCLDQS, NCLDQS, JL, JN] = ZSOLQA_NF[NCLDQS, NCLDQS, JL, JN] + PSNDE_NF[JK, JL, JN] * ZDTGDP[JL]
