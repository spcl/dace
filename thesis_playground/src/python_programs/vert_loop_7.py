import dace
import numpy as np


# NBLOCKS = dace.symbol('NBLOCKS')
# NCLV = dace.symbol('NCLV')
# KLEV = dace.symbol('KLEV')
# KLON = dace.symbol('KLON')
# NCLDTOP = dace.symbol('NCLDTOP')
# KFDIA = dace.symbol('KFDIA')
# KIDIA = dace.symbol('KIDIA')
# NCLDQL = dace.symbol('NCLDQL')
# NCLDQI = dace.symbol('NCLDQI')
# NCLDQS = dace.symbol('NCLDQS')
# TODO: Convert back to symbols
NCLDTOP = 2
KIDIA = 1
KFDIA = 1
NCLDQI = 2
NCLDQL = 3
NCLDQS = 5
NCLV = 10

KLEV = 4
KLON = 1
NBLOCKS = 5


@dace.program
def inner_loops_7(
        PTSPHY: dace.float64,
        RLMIN: dace.float64,
        ZEPSEC: dace.float64,
        RG: dace.float64,
        RTHOMO: dace.float64,
        ZALFAW: dace.float64,
        PLU_NF: dace.float64[KLEV, KLON],
        LDCUM_NF: dace.int32[KLON],
        PSNDE_NF: dace.float64[KLEV, KLON],
        PAPH_NF: dace.float64[KLEV+1, KLON],
        PSUPSAT_NF: dace.float64[KLEV, KLON],
        PT_NF: dace.float64[KLEV, KLON],
        tendency_tmp_t_NF: dace.float64[KLEV, KLON],
        PLUDE_NF: dace.float64[KLEV, KLON]
        ):

    # ZCONVSRCE = np.zeros([KLON, NCLV], dtype=np.float64)
    # ZSOLQA = np.zeros([KLON, NCLV, NCLV], dtype=np.float64)
    # ZDTGDP = np.zeros([KLON], dtype=np.float64)
    # ZDP = np.zeros([KLON], dtype=np.float64)
    # ZGDP = np.zeros([KLON], dtype=np.float64)
    # ZTP1 = np.zeros([KLON], dtype=np.float64)

    for JK in range(NCLDTOP-1, KLEV):
        # for JL in range(KIDIA-1, KFDIA):
            # ZTP1[JL] = PT_NF[JK, JL] + PTSPHY * tendency_tmp_t_NF[JK, JL]
            # if PSUPSAT_NF[JK, JL] > ZEPSEC:
            #     if ZTP1[JL] > RTHOMO:
            #         ZSOLQA[JL, NCLDQL, NCLDQL] = ZSOLQA[JL, NCLDQL, NCLDQL] + PSUPSAT_NF[JK, JL]
            #     else:
            #         ZSOLQA[JL, NCLDQI, NCLDQI] = ZSOLQA[JL, NCLDQI, NCLDQI] + PSUPSAT_NF[JK, JL]

        # for JL in range(KIDIA-1, KFDIA):
            # ZDP[JL] = PAPH_NF[JK+1, JL]-PAPH_NF[JK, JL]
            # PLUDE_NF[JK, JL] = 8.0
            # ZGDP[JL] = RG/ZDP[JL]
            # ZDTGDP[JL] = PTSPHY*ZGDP[JL]

        PLUDE_NF[JK, 0] = JK

        # for JL in range(KIDIA-1, KFDIA):
        #     PLUDE_NF[JK, JL] = PLUDE_NF[JK, JL]*ZDTGDP[JL]
        #     if LDCUM_NF[JL] and PLUDE_NF[JK, JL] > RLMIN and PLU_NF[JK+1, JL] > ZEPSEC:
        #         ZCONVSRCE[JL, NCLDQL] = ZALFAW*PLUDE_NF[JK, JL]
        #         ZCONVSRCE[JL, NCLDQI] = (1.0 - ZALFAW)*PLUDE_NF[JK, JL]
        #         ZSOLQA[JL, NCLDQL, NCLDQL] = ZSOLQA[JL, NCLDQL, NCLDQL]+ZCONVSRCE[JL, NCLDQL]
        #         ZSOLQA[JL, NCLDQI, NCLDQI] = ZSOLQA[JL, NCLDQI, NCLDQI]+ZCONVSRCE[JL, NCLDQI]
        #     else:
        #         PLUDE_NF[JK, JL] = 0.0

        #     if LDCUM_NF[JL]:
        #         ZSOLQA[JL, NCLDQS, NCLDQS] = ZSOLQA[JL, NCLDQS, NCLDQS] + PSNDE_NF[JK, JL]*ZDTGDP[JL]


@dace.program
def vert_loop_7(
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
            PLUDE_NF: dace.float64[KLEV, KLON, NBLOCKS]
        ):

    for JN in dace.map[0:NBLOCKS:KLON]:
        inner_loops_7(
            PTSPHY,  RLMIN,  ZEPSEC,  RG,  RTHOMO,  ZALFAW,  PLU_NF[:, :, JN],  LDCUM_NF[:, JN],  PSNDE_NF[:, :, JN],
            PAPH_NF[:, :, JN], PSUPSAT_NF[:, :, JN],  PT_NF[:, :, JN],  tendency_tmp_t_NF[:, :, JN], PLUDE_NF[:, :, JN],
            )
            # NCLV=NCLV)
            # NCLV=NCLV, NCLDTOP=NCLDTOP, KIDIA=KIDIA, KFDIA=KFDIA, NCLDQL=NCLDQL, NCLDQI=NCLDQI,
            # NCLDQS=NCLDQS)
