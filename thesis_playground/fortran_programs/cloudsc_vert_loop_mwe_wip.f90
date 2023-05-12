PROGRAM vert_loop_mwe_wip

    INTEGER, PARAMETER :: JPIM = SELECTED_INT_KIND(9)
    INTEGER, PARAMETER :: JPRB = SELECTED_REAL_KIND(13, 300)

    INTEGER(KIND=JPIM), PARAMETER  :: KLEV = 100
    INTEGER(KIND=JPIM), PARAMETER  :: NBLOCKS = 100

    REAL(KIND=JPRB) INPUT_F(NBLOCKS, KLEV)
    REAL(KIND=JPRB) OUTPUT_F(NBLOCKS, KLEV)

    CALL vert_loop_mwe_wip_routine(KLEV, NBLOCKS, INPUT_F, OUTPUT_F)

END PROGRAM

SUBROUTINE vert_loop_mwe_wip_routine(KLEV, NBLOCKS, INPUT_F, OUTPUT_F)

    INTEGER, PARAMETER :: JPIM = SELECTED_INT_KIND(9)
    INTEGER, PARAMETER :: JPRB = SELECTED_REAL_KIND(13, 300)

    ! Parameters
    INTEGER(KIND=JPIM) KLEV
    INTEGER(KIND=JPIM) NBLOCKS 

    REAL(KIND=JPRB) INPUT_F(NBLOCKS, KLEV)
    REAL(KIND=JPRB) OUTPUT_F(NBLOCKS, KLEV)

    DO JN=1,NBLOCKS
        CALL inner_loops(KLEV, NBLOCKS, INPUT_F(JN,:), OUTPUT_F(JN,:))
    ENDDO

END SUBROUTINE vert_loop_mwe_wip_routine

SUBROUTINE inner_loops(KLEV, NBLOCKS, INPUT_F, OUTPUT_F)

    INTEGER, PARAMETER :: JPIM = SELECTED_INT_KIND(9)
    INTEGER, PARAMETER :: JPRB = SELECTED_REAL_KIND(13, 300)

    REAL(KIND=JPRB) INPUT_F(KLEV)

    ! output
    REAL(KIND=JPRB) OUTPUT_F(KLEV)

    ! temporary arrays
    REAL(KIND=JPRB) TMP(KLEV)

    TMP(:) = 0

    DO JK=3 ,KLEV
        TMP(JK) = (INPUT_F(JK) + INPUT_F(JK-1) + INPUT_F(JK-2)) * 3
        OUTPUT_F = (TMP(JK) + TMP(JK-1) + TMP(JK-2)) * 3
    ENDDO

END SUBROUTINE inner_loops
