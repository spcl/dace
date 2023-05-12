PROGRAM vert_loop_mwe_no_klon

    INTEGER, PARAMETER :: JPIM = SELECTED_INT_KIND(9)
    INTEGER, PARAMETER :: JPRB = SELECTED_REAL_KIND(13, 300)

    INTEGER(KIND=JPIM), PARAMETER  :: KLEV = 100
    INTEGER(KIND=JPIM), PARAMETER  :: NCLV = 100
    INTEGER(KIND=JPIM), PARAMETER  :: NBLOCKS = 100

    ! Parameters
    INTEGER(KIND=JPIM) NCLDTOP

    ! input
    REAL(KIND=JPRB) PTSPHY
    REAL(KIND=JPRB) PAPH(KLEV+1, NBLOCKS)

    ! output
    REAL(KIND=JPRB) PLUDE(KLEV, NBLOCKS)

    CALL vert_loop_orig_mwe_no_klon_routine(&
        & KLEV, NCLV, NCLDTOP, NBLOCKS, &
        & PTSPHY, PAPH, &
        & PLUDE)

END PROGRAM
! Base on lines 1096 to 1120 and others
SUBROUTINE vert_loop_orig_mwe_no_klon_routine(&
    & KLEV, NCLV, NCLDTOP, NBLOCKS, &
    & PTSPHY, PAPH_NS, &
    & PLUDE_NS)

    INTEGER, PARAMETER :: JPIM = SELECTED_INT_KIND(9)
    INTEGER, PARAMETER :: JPRB = SELECTED_REAL_KIND(13, 300)

    ! Parameters
    INTEGER(KIND=JPIM) KLEV
    INTEGER(KIND=JPIM) NCLV
    INTEGER(KIND=JPIM) NCLDTOP
    ! NGPTOT == NPROMA == KLON
    ! INTEGER(KIND=JPIM) NPROMA 
    INTEGER(KIND=JPIM) NBLOCKS 

    ! input
    REAL(KIND=JPRB) PTSPHY
    REAL(KIND=JPRB) PAPH_NS(KLEV+1, NBLOCKS)

    ! output
    REAL(KIND=JPRB) PLUDE_NS(KLEV, NBLOCKS)

    DO JN=1,NBLOCKS
        CALL inner_loops(&
            & KLEV, NCLV, NCLDTOP, &
            & PTSPHY, PAPH_NS(:,JN), &
            & PLUDE_NS(:,JN))

    ENDDO

END SUBROUTINE vert_loop_orig_mwe_no_klon_routine

SUBROUTINE inner_loops(&
    & KLEV, NCLV, NCLDTOP, &
    & PTSPHY, PAPH_NS, &
    & PLUDE_NS)

    INTEGER, PARAMETER :: JPIM = SELECTED_INT_KIND(9)
    INTEGER, PARAMETER :: JPRB = SELECTED_REAL_KIND(13, 300)

    REAL(KIND=JPRB) PTSPHY
    REAL(KIND=JPRB) PAPH_NS(KLEV+1)

    ! output
    REAL(KIND=JPRB) PLUDE_NS(KLEV)

    ! temporary arrays
    REAL(KIND=JPRB) ZDTGDP

    ! Not sure if this causes problems
    ZDTGDP = 0.0

    DO JK=NCLDTOP,KLEV
        ZDTGDP  = PTSPHY*PAPH_NS(JK+1)-PAPH_NS(JK)
        PLUDE_NS(JK)=PLUDE_NS(JK)*ZDTGDP
    ENDDO ! on vertical level JK

END SUBROUTINE inner_loops
