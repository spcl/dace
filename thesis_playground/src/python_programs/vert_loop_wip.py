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

# NCLDTOP = 2
# KIDIA = 1
# KFDIA = 1
# NCLDQI = 2
# NCLDQL = 3
# NCLDQS = 5
# NCLV = 10

# KLEV = 4
# KLON = 1
# NBLOCKS = 5

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

    for JK in range(NCLDTOP-1, KLEV):
        PLUDE_NF[JK, 0] = JK


@dace.program
def vert_loop_wip(
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
        # for JK in range(NCLDTOP-1, KLEV):
        for JK in range(2,4):
            PLUDE_NF[JK, 0, JN] = NCLDTOP
        # inner_loops_7(
        #     PTSPHY,  RLMIN,  ZEPSEC,  RG,  RTHOMO,  ZALFAW,  PLU_NF[:, :, JN],  LDCUM_NF[:, JN],  PSNDE_NF[:, :, JN],
        #     PAPH_NF[:, :, JN], PSUPSAT_NF[:, :, JN],  PT_NF[:, :, JN],  tendency_tmp_t_NF[:, :, JN], PLUDE_NF[:, :, JN],
        #     )
