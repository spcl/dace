PROGRAM vert_loop_4

    INTEGER, PARAMETER :: JPIM = SELECTED_INT_KIND(9)
    INTEGER, PARAMETER :: JPRB = SELECTED_REAL_KIND(13, 300)

    INTEGER(KIND=JPIM), PARAMETER  :: KLON = 100
    INTEGER(KIND=JPIM), PARAMETER  :: KLEV = 100
    INTEGER(KIND=JPIM), PARAMETER  :: NCLV = 100
    INTEGER(KIND=JPIM), PARAMETER  :: NBLOCKS = 100

    ! Parameters
    INTEGER(KIND=JPIM) KIDIA 
    INTEGER(KIND=JPIM) KFDIA 
    INTEGER(KIND=JPIM) NCLDQS 
    INTEGER(KIND=JPIM) NCLDQI 
    INTEGER(KIND=JPIM) NCLDQL 
    INTEGER(KIND=JPIM) NCLDTOP
    INTEGER(KIND=JPIM) NPROMA 

    ! input
    REAL(KIND=JPRB) PTSPHY
    REAL(KIND=JPRB) RLMIN
    REAL(KIND=JPRB) ZEPSEC
    REAL(KIND=JPRB) RG
    ! was a temporary scalar before, to complicated to include whole computation here
    REAL(KIND=JPRB) ZALFAW
    REAL(KIND=JPRB) RTHOMO
    REAL(KIND=JPRB) PLU(KLON, KLEV, NBLOCKS)
    LOGICAL LDCUM(KLON, NBLOCKS)
    REAL(KIND=JPRB) PSNDE(KLON, KLEV, NBLOCKS)
    REAL(KIND=JPRB) PAPH(KLON, KLEV+1, NBLOCKS)
    ! This could be different in memory
    REAL(KIND=JPRB) PSUPSAT(KLON, KLEV, NBLOCKS)
    REAL(KIND=JPRB) PCLV(KLON, KLEV, NCLV, NBLOCKS)
    REAL(KIND=JPRB) PT(KLON, KLEV, NBLOCKS)
    REAL(KIND=JPRB) tendency_tmp_T(KLON, KLEV, NBLOCKS)

    ! output
    REAL(KIND=JPRB) PLUDE(KLON, KLEV, NBLOCKS)


    CALL vert_loop_4_routine(&
        & KLON, KLEV, NCLV, KIDIA, KFDIA, NCLDQS, NCLDQI, NCLDQL, NCLDTOP, NPROMA, NBLOCKS, &
        & PTSPHY, RLMIN, ZEPSEC, RG, ZALFAW, RTHOMO, PLU, LDCUM, PSNDE, PAPH, PSUPSAT, PCLV, PT, tendency_tmp_T, &
        & PLUDE)

END PROGRAM
! Base on lines 1096 to 1120 and others
SUBROUTINE vert_loop_4_routine(&
    & KLON, KLEV, NCLV, KIDIA, KFDIA, NCLDQS, NCLDQI, NCLDQL, NCLDTOP, NPROMA, NBLOCKS, &
    & PTSPHY, RLMIN, ZEPSEC, RG, RTHOMO, ZALFAW, PLU, LDCUM, PSNDE, PAPH_N, PSUPSAT_N, PCLV_N, PT_N, tendency_tmp_t_N, &
    & PLUDE)

    INTEGER, PARAMETER :: JPIM = SELECTED_INT_KIND(9)
    INTEGER, PARAMETER :: JPRB = SELECTED_REAL_KIND(13, 300)

    ! Parameters
    INTEGER(KIND=JPIM) KLON
    INTEGER(KIND=JPIM) KLEV
    INTEGER(KIND=JPIM) NCLV
    INTEGER(KIND=JPIM) KIDIA 
    INTEGER(KIND=JPIM) KFDIA 
    INTEGER(KIND=JPIM) NCLDQS 
    INTEGER(KIND=JPIM) NCLDQI 
    INTEGER(KIND=JPIM) NCLDQL 
    INTEGER(KIND=JPIM) NCLDTOP
    ! NGPTOT == NPROMA
    INTEGER(KIND=JPIM) NPROMA 
    INTEGER(KIND=JPIM) NBLOCKS 

    ! input
    REAL(KIND=JPRB) PTSPHY
    REAL(KIND=JPRB) RLMIN
    REAL(KIND=JPRB) ZEPSEC
    REAL(KIND=JPRB) RG
    ! was a temporary scalar before, to complicated to include whole computation here
    REAL(KIND=JPRB) ZALFAW
    REAL(KIND=JPRB) RTHOMO
    REAL(KIND=JPRB) PLU(KLON, KLEV, NBLOCKS)
    LOGICAL LDCUM(KLON, NBLOCKS)
    REAL(KIND=JPRB) PSNDE(KLON, KLEV, NBLOCKS)
    REAL(KIND=JPRB) PAPH_N(KLON, KLEV+1, NBLOCKS)
    ! This could be different in memory
    REAL(KIND=JPRB) PSUPSAT_N(KLON, KLEV, NBLOCKS)
    REAL(KIND=JPRB) PCLV_N(KLON, KLEV, NCLV, NBLOCKS)
    REAL(KIND=JPRB) PT_N(KLON, KLEV, NBLOCKS)
    REAL(KIND=JPRB) tendency_tmp_t_N(KLON, KLEV, NBLOCKS)

    ! output
    REAL(KIND=JPRB) PLUDE(KLON, KLEV, NBLOCKS)

    DO JN=1,NBLOCKS,NPROMA
        CALL inner_loops(&
            & KLON, KLEV, NCLV, KIDIA, KFDIA, NCLDQS, NCLDQI, NCLDQL, NCLDTOP, &
            & PTSPHY, RLMIN, ZEPSEC, RG, RTHOMO, ZALFAW, PLU(:,:,JN), LDCUM(:,JN), PSNDE(:,:,JN), PAPH_N(:,:,JN), &
            & PSUPSAT_N(:,:,JN), PCLV_N(:,:,:,JN), PT_N(:,:,JN), tendency_tmp_t_N(:,:,JN), &
            & PLUDE(:,:,JN))

    ENDDO

END SUBROUTINE vert_loop_4_routine

SUBROUTINE inner_loops(&
    & KLON, KLEV, NCLV, KIDIA, KFDIA, NCLDQS, NCLDQI, NCLDQL, NCLDTOP, &
    & PTSPHY, RLMIN, ZEPSEC, RG, RTHOMO, ZALFAW, PLU, LDCUM, PSNDE, PAPH_N, PSUPSAT_N, PCLV_N, PT_N, tendency_tmp_t_N, &
    & PLUDE)

    INTEGER, PARAMETER :: JPIM = SELECTED_INT_KIND(9)
    INTEGER, PARAMETER :: JPRB = SELECTED_REAL_KIND(13, 300)

    REAL(KIND=JPRB) PTSPHY
    REAL(KIND=JPRB) RLMIN
    REAL(KIND=JPRB) ZEPSEC
    REAL(KIND=JPRB) RG
    ! was a temporary scalar before, to complicated to include whole computation here
    REAL(KIND=JPRB) ZALFAW
    REAL(KIND=JPRB) RTHOMO
    REAL(KIND=JPRB) PLU(KLON, KLEV)
    LOGICAL LDCUM(KLON)
    REAL(KIND=JPRB) PSNDE(KLON, KLEV)
    REAL(KIND=JPRB) PAPH_N(KLON, KLEV+1)
    ! This could be different in memory
    REAL(KIND=JPRB) PSUPSAT_N(KLON, KLEV)
    REAL(KIND=JPRB) PCLV_N(KLON, KLEV, NCLV)
    REAL(KIND=JPRB) PT_N(KLON, KLEV)
    REAL(KIND=JPRB) tendency_tmp_t_N(KLON, KLEV)

    ! output
    REAL(KIND=JPRB) PLUDE(KLON, KLEV)

    ! temporary scalars
    ! temporary arrays
    REAL(KIND=JPRB) ZSOLAC(KLON)
    REAL(KIND=JPRB) ZFOEALFA(KLON, KLEV)
    REAL(KIND=JPRB) ZCONVSRCE(KLON, NCLV)
    REAL(KIND=JPRB) ZSOLQA(KLON, NCLV, NCLV)
    REAL(KIND=JPRB) ZDTGDP(KLON)
    REAL(KIND=JPRB) ZDP(KLON)
    REAL(KIND=JPRB) ZGDP(KLON)
    REAL(KIND=JPRB) ZTP1(KLON, KLEV)

    ! Not sure if this causes problems
    ZSOLAC(:) = 0.0
    ZFOEALFA(:, :) = 0.0
    ZCONVSRCE(:, :) = 0.0
    ZSOLQA(:, :, :) = 0.0
    ZDTGDP(:) = 0.0
    ZDP(:) = 0.0
    ZGDP(:) = 0.0
    ZTP1(:, :) = 0.0



    ! Loop from line 657
    DO JK=1,KLEV
        DO JL=KIDIA,KFDIA
            ZTP1(JL,JK)        = PT_N(JL,JK)+PTSPHY*tendency_tmp_t_N(JL,JK)
        ENDDO
    ENDDO
    ! To line 665

    DO JK=NCLDTOP,KLEV
        DO JL=KIDIA,KFDIA
            ! Loop from line 1061
            IF (PSUPSAT_N(JL,JK)>ZEPSEC) THEN
                IF (ZTP1(JL,JK) > RTHOMO) THEN
                    ZSOLQA(JL,NCLDQL,NCLDQL) = ZSOLQA(JL,NCLDQL,NCLDQL)+PSUPSAT_N(JL,JK)
                ELSE
                    ZSOLQA(JL,NCLDQI,NCLDQI) = ZSOLQA(JL,NCLDQI,NCLDQI)+PSUPSAT_N(JL,JK)
                ENDIF
            ENDIF
        ENDDO
        ! to line 1081

        ! Loop from 907
        DO JL=KIDIA,KFDIA   ! LOOP CLASS 3
            ZDP(JL)     = PAPH_N(JL,JK+1)-PAPH_N(JL,JK)     ! dp
            ZGDP(JL)    = RG/ZDP(JL)                    ! g/dp
            ZDTGDP(JL)  = PTSPHY*ZGDP(JL)               ! dt g/dp
        ENDDO
        ! To 919

        DO JL=KIDIA,KFDIA   ! LOOP CLASS 3

            PLUDE(JL,JK)=PLUDE(JL,JK)*ZDTGDP(JL)

            IF(LDCUM(JL).AND.PLUDE(JL,JK) > RLMIN.AND.PLU(JL,JK+1)> ZEPSEC) THEN
                ZCONVSRCE(JL,NCLDQL) = ZALFAW*PLUDE(JL,JK)
                ZCONVSRCE(JL,NCLDQI) = (1.0 - ZALFAW)*PLUDE(JL,JK)
                ZSOLQA(JL,NCLDQL,NCLDQL) = ZSOLQA(JL,NCLDQL,NCLDQL)+ZCONVSRCE(JL,NCLDQL)
                ZSOLQA(JL,NCLDQI,NCLDQI) = ZSOLQA(JL,NCLDQI,NCLDQI)+ZCONVSRCE(JL,NCLDQI)
            ELSE

                PLUDE(JL,JK)=0.0

            ENDIF
            ! *convective snow detrainment source
            IF (LDCUM(JL)) ZSOLQA(JL,NCLDQS,NCLDQS) = ZSOLQA(JL,NCLDQS,NCLDQS) + PSNDE(JL,JK)*ZDTGDP(JL)

        ENDDO
    ENDDO ! on vertical level JK

END SUBROUTINE inner_loops
