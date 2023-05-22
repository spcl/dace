import dace

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


@dace.program
def vert_loop_7_1_no_temp(
            PTSPHY: dace.float64,
            RLMIN: dace.float64,
            ZEPSEC: dace.float64,
            RG: dace.float64,
            RTHOMO: dace.float64,
            ZALFAW: dace.float64,
            PLU_NF: dace.float64[KLON, KLEV, NBLOCKS],
            LDCUM_NF: dace.int32[KLON, NBLOCKS],
            PSNDE_NF: dace.float64[KLON, KLEV, NBLOCKS],
            PAPH_NF: dace.float64[KLON, KLEV+1, NBLOCKS],
            PSUPSAT_NF: dace.float64[KLON, KLEV, NBLOCKS],
            PT_NF: dace.float64[KLON, KLEV, NBLOCKS],
            tendency_tmp_t_NF: dace.float64[KLON, KLEV, NBLOCKS],
            ZCONVSRCE: dace.float64[KLON, NCLV, NBLOCKS],
            ZSOLQA: dace.float64[KLON, NCLV, NCLV, NBLOCKS],
            ZDTGDP: dace.float64[KLON, NBLOCKS],
            ZDP: dace.float64[KLON, NBLOCKS],
            ZGDP: dace.float64[KLON, NBLOCKS],
            ZTP1: dace.float64[KLON, NBLOCKS],
            PLUDE_NF: dace.float64[KLON, KLEV, NBLOCKS]
        ):

    ZCONVSRCE[:, :, :] = 0.0
    ZSOLQA[:, :, :, :] = 0.0
    ZDTGDP[:, :] = 0.0
    ZDP[:, :] = 0.0
    ZGDP[:, :] = 0.0
    ZTP1[:, :] = 0.0

    for JN in dace.map[0:NBLOCKS:KLON]:
        for JK in range(NCLDTOP, KLEV-1):
            for JL in range(KIDIA, KFDIA):
                ZTP1[JL, JN] = PT_NF[JL, JK, JN] + PTSPHY * tendency_tmp_t_NF[JL, JK, JN]
                if PSUPSAT_NF[JL, JK, JN] > ZEPSEC:
                    if ZTP1[JL, JN] > RTHOMO:
                        ZSOLQA[JL, NCLDQL, NCLDQL, JN] = ZSOLQA[JL, NCLDQL, NCLDQL, JN] + PSUPSAT_NF[JL, JK, JN]
                    else:
                        ZSOLQA[JL, NCLDQI, NCLDQI, JN] = ZSOLQA[JL, NCLDQI, NCLDQI, JN] + PSUPSAT_NF[JL, JK, JN]

            for JL in range(KIDIA, KFDIA):
                ZDP[JL] = PAPH_NF[JL, JK+1, JN]-PAPH_NF[JL, JK, JN]
                ZGDP[JL] = RG/ZDP[JL]
                ZDTGDP[JL] = PTSPHY*ZGDP[JL]

            for JL in range(KIDIA, KFDIA):
                PLUDE_NF[JL, JK, JN] = PLUDE_NF[JL, JK, JN]*ZDTGDP[JL, JN]
                if LDCUM_NF[JL, JN] and PLUDE_NF[JL, JK, JN] > RLMIN and PLU_NF[JL, JK+1, JN] > ZEPSEC:
                    ZCONVSRCE[JL, NCLDQL, JN] = ZALFAW*PLUDE_NF[JL, JK, JN]
                    ZCONVSRCE[JL, NCLDQI, JN] = [1.0 - ZALFAW]*PLUDE_NF[JL, JK, JN]
                    ZSOLQA[JL, NCLDQL, NCLDQL, JN] = ZSOLQA[JL, NCLDQL, NCLDQL, JN] + ZCONVSRCE[JL, NCLDQL, JN]
                    ZSOLQA[JL, NCLDQI, NCLDQI, JN] = ZSOLQA[JL, NCLDQI, NCLDQI, JN] + ZCONVSRCE[JL, NCLDQI, JN]
                else:
                    PLUDE_NF[JL, JK, JN] = 0.0

                if LDCUM_NF[JL, JN]:
                    ZSOLQA[JL, NCLDQS, NCLDQS] = ZSOLQA[JL, NCLDQS, NCLDQS] + PSNDE_NF[JL, JK, JN] * ZDTGDP[JL]
