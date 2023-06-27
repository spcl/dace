! Make the temp array ZTP1, which also has a KLEV dimension, being passed to inner_loops function
PROGRAM vert_loop_6_1

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

    ! input
    REAL(KIND=JPRB) PTSPHY
    REAL(KIND=JPRB) RLMIN
    REAL(KIND=JPRB) ZEPSEC
    REAL(KIND=JPRB) RG
    ! was a temporary scalar before, to complicated to include whole computation here
    REAL(KIND=JPRB) ZALFAW
    REAL(KIND=JPRB) RTHOMO
    REAL(KIND=JPRB) PLU(NBLOCKS, KLON, KLEV)
    INTEGER(KIND=JPIM) LDCUM(NBLOCKS, KLON)
    REAL(KIND=JPRB) PSNDE(NBLOCKS, KLON, KLEV)
    REAL(KIND=JPRB) PAPH(NBLOCKS, KLON, KLEV+1)
    ! This could be different in memory
    REAL(KIND=JPRB) PSUPSAT(NBLOCKS, KLON, KLEV)
    REAL(KIND=JPRB) PT(NBLOCKS, KLON, KLEV)
    REAL(KIND=JPRB) tendency_tmp_T(NBLOCKS, KLON, KLEV)

    ! output
    REAL(KIND=JPRB) PLUDE(NBLOCKS, KLON, KLEV)


    CALL vert_loop_6_1_routine(&
        & KLON, KLEV, NCLV, KIDIA, KFDIA, NCLDQS, NCLDQI, NCLDQL, NCLDTOP, NBLOCKS, &
        & PTSPHY, RLMIN, ZEPSEC, RG, RTHOMO, ZALFAW, PLU, LDCUM, PSNDE, PAPH, PSUPSAT, PT, tendency_tmp_T, &
        & PLUDE)

END PROGRAM
! Base on lines 1096 to 1120 and others
SUBROUTINE vert_loop_6_1_routine(&
    & KLON, KLEV, NCLV, KIDIA, KFDIA, NCLDQS, NCLDQI, NCLDQL, NCLDTOP, NBLOCKS, &
    & PTSPHY, RLMIN, ZEPSEC, RG, RTHOMO, ZALFAW, PLU_NF, LDCUM_NF, PSNDE_NF, PAPH_NF, PSUPSAT_NF, PT_NF, tendency_tmp_t_NF, &
    & PLUDE_NF)

    INTEGER, PARAMETER :: JPIM = SELECTED_INT_KIND(9)
    INTEGER, PARAMETER :: JPRB = SELECTED_REAL_KIND(13, 300)

    ! Parameters
    INTEGER(KIND=JPIM) KLEV
    INTEGER(KIND=JPIM) NCLV
    INTEGER(KIND=JPIM) KIDIA 
    INTEGER(KIND=JPIM) KFDIA 
    INTEGER(KIND=JPIM) NCLDQS 
    INTEGER(KIND=JPIM) NCLDQI 
    INTEGER(KIND=JPIM) NCLDQL 
    INTEGER(KIND=JPIM) NCLDTOP
    ! NGPTOT == NPROMA == KLON
    ! INTEGER(KIND=JPIM) NPROMA 
    INTEGER(KIND=JPIM) NBLOCKS 

    ! input
    REAL(KIND=JPRB) PTSPHY
    REAL(KIND=JPRB) RLMIN
    REAL(KIND=JPRB) ZEPSEC
    REAL(KIND=JPRB) RG
    ! was a temporary scalar before, to complicated to include whole computation here
    REAL(KIND=JPRB) ZALFAW
    REAL(KIND=JPRB) RTHOMO

    REAL(KIND=JPRB) PLU_NF(NBLOCKS, KLON, KLEV)
    INTEGER(KIND=JPIM) LDCUM_NF(NBLOCKS, KLON)
    REAL(KIND=JPRB) PSNDE_NF(NBLOCKS, KLON, KLEV)
    REAL(KIND=JPRB) PAPH_NF(NBLOCKS, KLON, KLEV+1)
    ! This could be different in memory
    REAL(KIND=JPRB) PSUPSAT_NF(NBLOCKS, KLON, KLEV)
    REAL(KIND=JPRB) PT_NF(NBLOCKS, KLON, KLEV)
    REAL(KIND=JPRB) tendency_tmp_t_NF(NBLOCKS, KLON, KLEV)

    ! output
    REAL(KIND=JPRB) PLUDE_NF(NBLOCKS, KLON, KLEV)

    ! temporary arrays
    REAL(KIND=JPRB) ZTP1(NBLOCKS, KLON, KLEV)

    DO JN=1,NBLOCKS,KLON
        CALL inner_loops(&
            & KLON, KLEV, NCLV, KIDIA, KFDIA, NCLDQS, NCLDQI, NCLDQL, NCLDTOP, &
            & PTSPHY, RLMIN, ZEPSEC, RG, RTHOMO, ZALFAW, PLU_NF(JN,:,:), LDCUM_NF(JN,:), PSNDE_NF(JN,:,:), PAPH_NF(JN,:,:), &
            & PSUPSAT_NF(JN,:,:), PT_NF(JN,:,:), tendency_tmp_t_NF(JN,:,:), &
            & PLUDE_NF(JN,:,:), ZTP1(JN,:,:))

    ENDDO

END SUBROUTINE vert_loop_6_1_routine

SUBROUTINE inner_loops(&
    & KLON, KLEV, NCLV, KIDIA, KFDIA, NCLDQS, NCLDQI, NCLDQL, NCLDTOP, &
    & PTSPHY, RLMIN, ZEPSEC, RG, RTHOMO, ZALFAW, PLU_NF, LDCUM_NF, PSNDE_NF, PAPH_NF, PSUPSAT_NF, PT_NF, tendency_tmp_t_NF, &
    & PLUDE_NF, ZTP1)

    INTEGER, PARAMETER :: JPIM = SELECTED_INT_KIND(9)
    INTEGER, PARAMETER :: JPRB = SELECTED_REAL_KIND(13, 300)

    REAL(KIND=JPRB) PTSPHY
    REAL(KIND=JPRB) RLMIN
    REAL(KIND=JPRB) ZEPSEC
    REAL(KIND=JPRB) RG
    ! was a temporary scalar before, to complicated to include whole computation here
    REAL(KIND=JPRB) ZALFAW
    REAL(KIND=JPRB) RTHOMO
    REAL(KIND=JPRB) PLU_NF(KLON, KLEV)
    LOGICAL LDCUM_NF(KLON)
    REAL(KIND=JPRB) PSNDE_NF(KLON, KLEV)
    REAL(KIND=JPRB) PAPH_NF(KLON, KLEV+1)
    ! This could be different in memory
    REAL(KIND=JPRB) PSUPSAT_NF(KLON, KLEV)
    REAL(KIND=JPRB) PT_NF(KLON, KLEV)
    REAL(KIND=JPRB) tendency_tmp_t_NF(KLON, KLEV)

    ! output
    REAL(KIND=JPRB) PLUDE_NF(KLON, KLEV)

    ! temporary scalars
    ! temporary arrays
    REAL(KIND=JPRB) ZCONVSRCE(KLON, NCLV)
    REAL(KIND=JPRB) ZSOLQA(KLON, NCLV, NCLV)
    REAL(KIND=JPRB) ZDTGDP(KLON)
    REAL(KIND=JPRB) ZDP(KLON)
    REAL(KIND=JPRB) ZGDP(KLON)
    REAL(KIND=JPRB) ZTP1(KLON, KLEV)

    ! Not sure if this causes problems
    ZCONVSRCE(:, :) = 0.0
    ZSOLQA(:, :, :) = 0.0
    ZDTGDP(:) = 0.0
    ZDP(:) = 0.0
    ZGDP(:) = 0.0
    ZTP1(:, :) = 0.0



    ! Loop from line 657
    DO JK=1,KLEV
        DO JL=KIDIA,KFDIA
            ZTP1(JL,JK)        = PT_NF(JL,JK)+PTSPHY*tendency_tmp_t_NF(JL,JK)
        ENDDO
    ENDDO
    ! To line 665

    DO JK=NCLDTOP,KLEV
        DO JL=KIDIA,KFDIA
            ! Loop from line 1061
            IF (PSUPSAT_NF(JL,JK)>ZEPSEC) THEN
                IF (ZTP1(JL,JK) > RTHOMO) THEN
                    ZSOLQA(JL,NCLDQL,NCLDQL) = ZSOLQA(JL,NCLDQL,NCLDQL)+PSUPSAT_NF(JL,JK)
                ELSE
                    ZSOLQA(JL,NCLDQI,NCLDQI) = ZSOLQA(JL,NCLDQI,NCLDQI)+PSUPSAT_NF(JL,JK)
                ENDIF
            ENDIF
        ENDDO
        ! to line 1081

        ! Loop from 907
        DO JL=KIDIA,KFDIA   ! LOOP CLASS 3
            ZDP(JL)     = PAPH_NF(JL,JK+1)-PAPH_NF(JL,JK)     ! dp
            ZGDP(JL)    = RG/ZDP(JL)                    ! g/dp
            ZDTGDP(JL)  = PTSPHY*ZGDP(JL)               ! dt g/dp
        ENDDO
        ! To 919

        IF (JK < KLEV .AND. JK>=NCLDTOP) THEN
            DO JL=KIDIA,KFDIA   ! LOOP CLASS 3

                PLUDE_NF(JL,JK)=PLUDE_NF(JL,JK)*ZDTGDP(JL)

                IF(LDCUM_NF(JL).AND.PLUDE_NF(JL,JK) > RLMIN.AND.PLU_NF(JL,JK+1)> ZEPSEC) THEN
                    ZCONVSRCE(JL,NCLDQL) = ZALFAW*PLUDE_NF(JL,JK)
                    ZCONVSRCE(JL,NCLDQI) = (1.0 - ZALFAW)*PLUDE_NF(JL,JK)
                    ZSOLQA(JL,NCLDQL,NCLDQL) = ZSOLQA(JL,NCLDQL,NCLDQL)+ZCONVSRCE(JL,NCLDQL)
                    ZSOLQA(JL,NCLDQI,NCLDQI) = ZSOLQA(JL,NCLDQI,NCLDQI)+ZCONVSRCE(JL,NCLDQI)
                ELSE

                    PLUDE_NF(JL,JK)=0.0

                ENDIF
                ! *convective snow detrainment source
                IF (LDCUM_NF(JL)) ZSOLQA(JL,NCLDQS,NCLDQS) = ZSOLQA(JL,NCLDQS,NCLDQS) + PSNDE_NF(JL,JK)*ZDTGDP(JL)

            ENDDO
        ENDIF
    ENDDO ! on vertical level JK

END SUBROUTINE inner_loops
