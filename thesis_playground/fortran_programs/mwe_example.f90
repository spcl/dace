PROGRAM main
    IMPLICIT NONE
    REAL INP1(KLON, KLEV, NCLV)
    REAL INP2(KLON, KLEV, NCLV)

    INTEGER, PARAMETER  :: NCLV = 5
    INTEGER, PARAMETER  :: KLEV = 137
    INTEGER, PARAMETER  :: KIDIA = 1
    INTEGER, PARAMETER  :: KFDIA = 1
    INTEGER, PARAMETER  :: KLON = 1
    INTEGER, PARAMETER  :: NCLDQV = 5

    REAL :: ZQX(KLON,KLEV,NCLV)  ! water variables
    REAL :: ZLNEG(KLON,KLEV,NCLV)     ! for negative correction diagnostics
    REAL :: tendency_loc_q(KLON,KLEV),tendency_loc_T(KLON,KLEV)  ! GFL fields
    INTEGER :: IPHASE(NCLV) ! marker for water phase of each species
    REAL :: RALVDCP, ZQTMST, ZQADJ, RLMIN


    CALL work(&
        & NCLV, KLEV, KIDIA, KFDIA, KLON, &
        & RLMIN, RALVDCP, ZQTMST, &
        & ZQX, ZLNEG, tendency_loc_q, tendency_loc_T, IPHASE)

END PROGRAM

SUBROUTINE work(&
        & NCLV, KLEV, KIDIA, KFDIA, KLON, &
        & RLMIN, RALVDCP, ZQTMST, &
        & ZQX, ZLNEG, tendency_loc_q, tendency_loc_T, IPHASE)
    INTEGER, PARAMETER  :: NCLV = 5
    INTEGER, PARAMETER  :: KLEV = 137
    INTEGER, PARAMETER  :: KIDIA = 1
    INTEGER, PARAMETER  :: KFDIA = 1
    INTEGER, PARAMETER  :: KLON = 1

    REAL :: ZQX(KLON,KLEV,NCLV)  ! water variables
    REAL :: ZLNEG(KLON,KLEV,NCLV)     ! for negative correction diagnostics
    REAL :: tendency_loc_q(KLON,KLEV),tendency_loc_T(KLON,KLEV)  ! GFL fields
    INTEGER :: IPHASE(NCLV) ! marker for water phase of each species
    REAL :: RALVDCP, ZQTMST, ZQADJ, RLMIN

    DO JM=1,NCLV-1
      DO JK=1,KLEV
        DO JL=KIDIA,KFDIA
          IF (ZQX(JL,JK,JM)<RLMIN) THEN
            ZLNEG(JL,JK,JM) = ZLNEG(JL,JK,JM)+ZQX(JL,JK,JM)
            ZQADJ               = ZQX(JL,JK,JM)*ZQTMST
            tendency_loc_q(JL,JK)        = tendency_loc_q(JL,JK)+ZQADJ
            IF (IPHASE(JM)==1) tendency_loc_T(JL,JK) = tendency_loc_T(JL,JK)-RALVDCP*ZQADJ
            IF (IPHASE(JM)==2) tendency_loc_T(JL,JK) = tendency_loc_T(JL,JK)-RALSDCP*ZQADJ
            ZQX(JL,JK,NCLDQV)   = ZQX(JL,JK,NCLDQV)+ZQX(JL,JK,JM)
            ZQX(JL,JK,JM)       = 0.0
          ENDIF
        ENDDO
      ENDDO
    ENDDO
END SUBROUTINE work
