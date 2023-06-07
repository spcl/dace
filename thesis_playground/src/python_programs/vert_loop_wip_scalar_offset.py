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
def vert_loop_wip_scalar_offset(
            PTSPHY: dace.float64,
            ZEPSEC: dace.float64,
            RTHOMO: dace.float64,
            PSUPSAT_NF: dace.float64[KLEV, KLON, NBLOCKS],
            PT_NF: dace.float64[KLEV, KLON, NBLOCKS],
            tendency_tmp_t_NF: dace.float64[KLEV, KLON, NBLOCKS],
            ZSOLQA: dace.float64[KLON, NCLV, NCLV, NBLOCKS]
        ):

    for JN in dace.map[0:NBLOCKS:KLON]:
        ZTP1 = np.zeros([KLON, KLEV], dtype=np.float64)
        # ZTP1 = np.zeros([KLON, 137], dtype=np.float64)
        # ZSOLQA = np.zeros([KLON, NCLV, NCLV], dtype=np.float64)

        # for JK in range(1, 137):
        #     for JL in range(0, 1):
        #         ZTP1[JL, JK] = PT_NF[JK, JL, JN] + PTSPHY * tendency_tmp_t_NF[JK, JL, JN]

        # The starting range should be NCLDTOP-1 and end KLEV but using these leads to very strange behaviour
        for JK in range(1, 137):
            # for JL in range(KIDIA-1, KFDIA):
            for JL in range(0, 1):
                if PSUPSAT_NF[JK, JL, JN] > ZEPSEC:
                    ZTP1[JL, JK] = PT_NF[JK, JL, JN] + PTSPHY * tendency_tmp_t_NF[JK, JL, JN]
                    # ZTP1[JL] = PT_NF[JK, JL, JN] + PTSPHY * tendency_tmp_t_NF[JK, JL, JN]
                    if ZTP1[JL, JK] > RTHOMO:
                        ZSOLQA[JL, NCLDQL, NCLDQL, JN] = ZSOLQA[JL, NCLDQL, NCLDQL, JN] + PSUPSAT_NF[JK, JL, JN]
                    else:
                        ZSOLQA[JL, NCLDQI, NCLDQI, JN] = ZSOLQA[JL, NCLDQI, NCLDQI, JN] + PSUPSAT_NF[JK, JL, JN]
