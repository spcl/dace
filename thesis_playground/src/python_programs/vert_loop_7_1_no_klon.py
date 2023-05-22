import dace
import numpy as np


NBLOCKS = dace.symbol('NBLOCKS')
KLEV = dace.symbol('KLEV')
NCLV = dace.symbol('NCLV')
NCLDTOP = dace.symbol('NCLDTOP')
NCLDQL = dace.symbol('NCLDQL')
NCLDQI = dace.symbol('NCLDQI')
NCLDQS = dace.symbol('NCLDQS')


@dace.program
def vert_loop_7_1_no_klon(
            PTSPHY: dace.float64,
            RLMIN: dace.float64,
            ZEPSEC: dace.float64,
            RG: dace.float64,
            RTHOMO: dace.float64,
            ZALFAW: dace.float64,
            PLU_NF: dace.float64[KLEV, NBLOCKS],
            LDCUM_NF: dace.int32[NBLOCKS],
            PSNDE_NF: dace.float64[KLEV, NBLOCKS],
            PAPH_NF: dace.float64[KLEV+1, NBLOCKS],
            PSUPSAT_NF: dace.float64[KLEV, NBLOCKS],
            PT_NF: dace.float64[KLEV, NBLOCKS],
            tendency_tmp_t_NF: dace.float64[KLEV, NBLOCKS],
            PLUDE_NF: dace.float64[KLEV, NBLOCKS]
        ):

    for JN in dace.map[0:NBLOCKS]:
        ZCONVSRCE = np.zeros([NCLV], dtype=np.float64)
        ZSOLQA = np.zeros([NCLV, NCLV], dtype=np.float64)
        ZDTGDP = 0.0
        ZDP = 0.0
        ZGDP = 0.0
        ZTP1 = 0.0

        for JK in range(NCLDTOP, KLEV-1):
            ZTP1 = PT_NF[JK, JN] + PTSPHY * tendency_tmp_t_NF[JK, JN]
            if PSUPSAT_NF[JK, JN] > ZEPSEC:
                if ZTP1 > RTHOMO:
                    ZSOLQA[NCLDQL, NCLDQL] = ZSOLQA[NCLDQL, NCLDQL] + PSUPSAT_NF[JK, JN]
                else:
                    ZSOLQA[NCLDQI, NCLDQI] = ZSOLQA[NCLDQI, NCLDQI] + PSUPSAT_NF[JK, JN]

            ZDP = PAPH_NF[JK+1, JN]-PAPH_NF[JK, JN]
            ZGDP = RG/ZDP
            ZDTGDP = PTSPHY*ZGDP

            PLUDE_NF[JK, JN] = PLUDE_NF[JK, JN]*ZDTGDP
            if LDCUM_NF[JN] and PLUDE_NF[JK, JN] > RLMIN and PLU_NF[JK+1, JN] > ZEPSEC:
                ZCONVSRCE[NCLDQL] = ZALFAW*PLUDE_NF[JK, JN]
                ZCONVSRCE[NCLDQI] = [1.0 - ZALFAW]*PLUDE_NF[JK, JN]
                ZSOLQA[NCLDQL, NCLDQL] = ZSOLQA[NCLDQL, NCLDQL] + ZCONVSRCE[NCLDQL]
                ZSOLQA[NCLDQI, NCLDQI] = ZSOLQA[NCLDQI, NCLDQI] + ZCONVSRCE[NCLDQI]
            else:
                PLUDE_NF[JK, JN] = 0.0

            if LDCUM_NF[JN]:
                ZSOLQA[NCLDQS, NCLDQS] = ZSOLQA[NCLDQS, NCLDQS] + PSNDE_NF[JK, JN] * ZDTGDP
