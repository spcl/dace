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
NCLDQS = dace.symbol('NCLDQV')

@dace.program
def vert_loop_10(
            PTSPHY: dace.float64,
            RLMIN: dace.float64,
            ZEPSEC: dace.float64,
            RG: dace.float64,
            RTHOMO: dace.float64,
            ZALFAW: dace.float64,
            PLU: dace.float64[KLON, KLEV, NBLOCKS],
            LDCUM: dace.int32[KLON, NBLOCKS],
            PSNDE: dace.float64[KLON, KLEV, NBLOCKS],
            PAPH_N: dace.float64[KLON, KLEV+1, NBLOCKS],
            PSUPSAT_N: dace.float64[KLON, KLEV, NBLOCKS],
            PT_N: dace.float64[KLON, KLEV, NBLOCKS],
            tendency_tmp_t_N: dace.float64[KLON, KLEV, NBLOCKS],
            tendency_tmp_cld_N: dace.float64[KLON, KLEV, NCLV, NBLOCKS],
            PCLV_N: dace.float64[KLON, KLEV, NCLV, NBLOCKS],
            ZSOLQA: dace.float64[KLON, NCLV, NCLV, NBLOCKS],
            PLUDE: dace.float64[KLON, KLEV, NBLOCKS]):

    for JN in dace.map[0:NBLOCKS:KLON]:
        ZCONVSRCE = np.zeros([KLON, NCLV])
        ZDTGDP = np.zeros([KLON])
        ZDP = np.zeros([KLON])
        ZGDP = np.zeros([KLON])
        ZTP1 = np.zeros([KLON, KLEV])
        ZLIQFRAC = np.zeros([KLON, KLEV])
        ZICEFRAC = np.zeros([KLON, KLEV])
        ZQX = np.zeros([KLON, KLEV, NCLV])
        ZLI = np.zeros([KLON, KLEV])

        for JK in range(KLEV):
            for JL in range(KIDIA-1,KFDIA):
                ZTP1[JL,JK]        = PT_N[JL,JK,JN]+PTSPHY*tendency_tmp_t_N[JL,JK,JN]

        for JM in range(NCLV-1):
            for JK in range(KLEV):
                for JL in range(KIDIA-1,KFDIA):
                    ZQX[JL,JK,JM]  = PCLV_N[JL,JK,JM,JN]+PTSPHY*tendency_tmp_cld_N[JL,JK,JM,JN]

        for JK in range(KLEV):
            for JL in range(KIDIA-1,KFDIA):
                ZLI[JL,JK]=ZQX[JL,JK,NCLDQL]+ZQX[JL,JK,NCLDQI]
                if (ZLI[JL,JK]>RLMIN):
                    ZLIQFRAC[JL,JK]=ZQX[JL,JK,NCLDQL]/ZLI[JL,JK]
                    ZICEFRAC[JL,JK]=1.0 - ZLIQFRAC[JL,JK]
                else:
                    ZLIQFRAC[JL,JK]=0.0
                    ZICEFRAC[JL,JK]=0.0

        for JK in range(NCLDTOP-1,KLEV):
            for JL in range(KIDIA-1,KFDIA):
                if (PSUPSAT_N[JL,JK,JN] >ZEPSEC):
                    if (ZTP1[JL,JK] > RTHOMO):
                        ZSOLQA[JL,NCLDQL,NCLDQL,JN] = ZSOLQA[JL,NCLDQL,NCLDQL,JN]+PSUPSAT_N[JL,JK,JN]
                    else:
                        ZSOLQA[JL,NCLDQI,NCLDQI,JN] = ZSOLQA[JL,NCLDQI,NCLDQI,JN]+PSUPSAT_N[JL,JK,JN]

            for JL in range(KIDIA-1,KFDIA):
                ZDP[JL]     = PAPH_N[JL,JK+1,JN]-PAPH_N[JL,JK,JN]
                ZGDP[JL]    = RG/ZDP[JL]
                ZDTGDP[JL]  = PTSPHY*ZGDP[JL]
                ZDTGDP[JL]  = ZDTGDP[JL] + (ZTP1[JL,JK-1]+ZTP1[JL,JK])/PAPH_N[JL,JK,JN]

            for JL in range(KIDIA-1,KFDIA):
                PLUDE[JL,JK,JN]=PLUDE[JL,JK,JN]*ZDTGDP[JL]

                if(LDCUM[JL,JN] and PLUDE[JL,JK,JN] > RLMIN and PLU[JL,JK+1,JN]> ZEPSEC):
                    ZCONVSRCE[JL,NCLDQL] = ZALFAW*PLUDE[JL,JK,JN]
                    ZCONVSRCE[JL,NCLDQI] = (1.0 - ZALFAW)*PLUDE[JL,JK,JN]
                    ZSOLQA[JL,NCLDQL,NCLDQL,JN] = ZSOLQA[JL,NCLDQL,NCLDQL,JN]+ZCONVSRCE[JL,NCLDQL]
                    ZSOLQA[JL,NCLDQI,NCLDQI,JN] = ZSOLQA[JL,NCLDQI,NCLDQI,JN]+ZCONVSRCE[JL,NCLDQI]
                else:
                    PLUDE[JL,JK,JN]=0.0

                if (LDCUM[JL,JN]):
                    ZSOLQA[JL,NCLDQS,NCLDQS,JN] = ZSOLQA[JL,NCLDQS,NCLDQS,JN] + PSNDE[JL,JK,JN]*ZDTGDP[JL]

            for JL in range(KIDIA-1,KFDIA):
                if(ZLI[JL,JK] > ZEPSEC):
                    ZSOLQA[JL,NCLDQV,NCLDQL,JN] = ZSOLQA[JL,NCLDQV,NCLDQL,JN]+ZLIQFRAC[JL,JK]
                    ZSOLQA[JL,NCLDQL,NCLDQV,JN] = ZSOLQA[JL,NCLDQL,NCLDQV,JN]-ZLIQFRAC[JL,JK]
                    ZSOLQA[JL,NCLDQV,NCLDQI,JN] = ZSOLQA[JL,NCLDQV,NCLDQI,JN]+ZICEFRAC[JL,JK]
                    ZSOLQA[JL,NCLDQI,NCLDQV,JN] = ZSOLQA[JL,NCLDQI,NCLDQV,JN]-ZICEFRAC[JL,JK]
