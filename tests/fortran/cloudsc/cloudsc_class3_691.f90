PROGRAM tidy_up_cloud_cover
    INTEGER, PARAMETER :: JPIM = SELECTED_INT_KIND(9)
    INTEGER, PARAMETER :: JPRB = SELECTED_REAL_KIND(13, 300)

    INTEGER(KIND=JPIM), PARAMETER  :: KLON = 100
    INTEGER(KIND=JPIM), PARAMETER  :: KLEV = 100
    INTEGER(KIND=JPIM), PARAMETER  :: NCLV = 100

    INTEGER(KIND=JPIM) KIDIA 
    INTEGER(KIND=JPIM) KFDIA 
    INTEGER(KIND=JPIM) NCLDQV 
    INTEGER(KIND=JPIM) NCLDQI 
    INTEGER(KIND=JPIM) NCLDQL 

    REAL(KIND=JPRB) RLMIN
    REAL(KIND=JPRB) RAMIN
    REAL(KIND=JPRB) ZQTMST
    REAL(KIND=JPRB) RALVDCP
    REAL(KIND=JPRB) RALSDCP
    REAL(KIND=JPRB) ZA(KLON, KLEV)

    REAL(KIND=JPRB) ZQX(KLON, KLEV, NCLV)
    REAL(KIND=JPRB) ZLNEG(KLON, KLEV, NCLV)
    REAL(KIND=JPRB) tendency_loc_q(KLON, KLEV)
    REAL(KIND=JPRB) tendency_loc_T(KLON, KLEV)

    CALL tidy_up_cloud_cover_routine(&
        & KLON, KLEV, NCLV, KIDIA, KFDIA, NCLDQV, NCLDQI, NCLDQL, &
        & RLMIN, RAMIN, ZQTMST, RALVDCP, RALSDCP, ZA, ZQX, ZLNEG, tendency_loc_q, tendency_loc_T)

END

SUBROUTINE tidy_up_cloud_cover_routine(&
        & KLON, KLEV, NCLV, KIDIA, KFDIA, NCLDQV, NCLDQI, NCLDQL, &
        & RLMIN, RAMIN, ZQTMST, RALVDCP, RALSDCP, ZA, ZQX, ZLNEG, tendency_loc_q, tendency_loc_T)

    INTEGER, PARAMETER :: JPIM = SELECTED_INT_KIND(9)
    INTEGER, PARAMETER :: JPRB = SELECTED_REAL_KIND(13, 300)

    INTEGER(KIND=JPIM) KLON
    INTEGER(KIND=JPIM) KLEV
    INTEGER(KIND=JPIM) NCLV
    INTEGER(KIND=JPIM) KIDIA 
    INTEGER(KIND=JPIM) KFDIA 
    INTEGER(KIND=JPIM) NCLDQV 
    INTEGER(KIND=JPIM) NCLDQI 
    INTEGER(KIND=JPIM) NCLDQL 

    REAL(KIND=JPRB) RLMIN
    REAL(KIND=JPRB) RAMIN
    REAL(KIND=JPRB) ZQTMST
    REAL(KIND=JPRB) RALVDCP
    REAL(KIND=JPRB) RALSDCP
    REAL(KIND=JPRB) ZA(KLON, KLEV)

    REAL(KIND=JPRB) ZQX(KLON, KLEV, NCLV)
    REAL(KIND=JPRB) ZLNEG(KLON, KLEV, NCLV)
    REAL(KIND=JPRB) tendency_loc_q(KLON, KLEV)
    REAL(KIND=JPRB) tendency_loc_T(KLON, KLEV)

    ! temporary variables
    REAL(KIND=JPRB) ZQADJ

    DO JK=1,KLEV
        DO JL=KIDIA,KFDIA
            IF (ZQX(JL,JK,NCLDQL)+ZQX(JL,JK,NCLDQI)<RLMIN.OR.ZA(JL,JK)<RAMIN) THEN

                ! Evaporate small cloud liquid water amounts
                ZLNEG(JL,JK,NCLDQL) = ZLNEG(JL,JK,NCLDQL)+ZQX(JL,JK,NCLDQL)
                ZQADJ               = ZQX(JL,JK,NCLDQL)*ZQTMST
                tendency_loc_q(JL,JK)        = tendency_loc_q(JL,JK)+ZQADJ
                tendency_loc_T(JL,JK)        = tendency_loc_T(JL,JK)-RALVDCP*ZQADJ
                ZQX(JL,JK,NCLDQV)   = ZQX(JL,JK,NCLDQV)+ZQX(JL,JK,NCLDQL)
                ZQX(JL,JK,NCLDQL)   = 0.0

                ! Evaporate small cloud ice water amounts
                ZLNEG(JL,JK,NCLDQI) = ZLNEG(JL,JK,NCLDQI)+ZQX(JL,JK,NCLDQI)
                ZQADJ               = ZQX(JL,JK,NCLDQI)*ZQTMST
                tendency_loc_q(JL,JK)        = tendency_loc_q(JL,JK)+ZQADJ
                tendency_loc_T(JL,JK)        = tendency_loc_T(JL,JK)-RALSDCP*ZQADJ
                ZQX(JL,JK,NCLDQV)   = ZQX(JL,JK,NCLDQV)+ZQX(JL,JK,NCLDQI)
                ZQX(JL,JK,NCLDQI)   = 0.0

                ! Set cloud cover to zero
                ZA(JL,JK)           = 0.0

            ENDIF
        ENDDO
    ENDDO

END SUBROUTINE tidy_up_cloud_cover_routine
