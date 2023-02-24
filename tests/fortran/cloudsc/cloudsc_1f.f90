PROGRAM liq_ice_fractions

    INTEGER, PARAMETER :: JPIM = SELECTED_INT_KIND(9)
    INTEGER, PARAMETER :: JPRB = SELECTED_REAL_KIND(13, 300)

    INTEGER(KIND=JPIM), PARAMETER  :: KLON = 100
    INTEGER(KIND=JPIM), PARAMETER  :: KLEV = 100
    INTEGER(KIND=JPIM), PARAMETER  :: NCLV = 100

    INTEGER(KIND=JPIM) KIDIA 
    INTEGER(KIND=JPIM) KFDIA 
    INTEGER(KIND=JPIM) NCLDQI
    INTEGER(KIND=JPIM) NCLDQL

    REAL(KIND=JPRB) RLMIN

    REAL(KIND=JPRB) ZA(KLON,KLEV)
    REAL(KIND=JPRB) ZLIQFRAC(KLON,KLEV)
    REAL(KIND=JPRB) ZICEFRAC(KLON,KLEV)
    REAL(KIND=JPRB) ZQX(KLON,KLEV,NCLV) 

    CALL liq_ice_fractions_routine( &
        & KLON, KLEV, NCLV, KIDIA, KFDIA, NCLDQI, NCLDQL, &
        & RLMIN, ZA, ZLIQFRAC, ZICEFRAC, ZQX)
END

SUBROUTINE liq_ice_fractions_routine( &
    & KLON, KLEV, NCLV, KIDIA, KFDIA, NCLDQI, NCLDQL, &
    & RLMIN, ZA, ZLIQFRAC, ZICEFRAC, ZQX)

    INTEGER, PARAMETER :: JPIM = SELECTED_INT_KIND(9)
    INTEGER, PARAMETER :: JPRB = SELECTED_REAL_KIND(13, 300)

    INTEGER(KIND=JPIM) KLON
    INTEGER(KIND=JPIM) KLEV
    INTEGER(KIND=JPIM) NCLV
    INTEGER(KIND=JPIM) KIDIA 
    INTEGER(KIND=JPIM) KFDIA 
    INTEGER(KIND=JPIM) NCLDQI
    INTEGER(KIND=JPIM) NCLDQL

    REAL(KIND=JPRB) RLMIN

    REAL(KIND=JPRB) ZLI(KLON,KLEV)
    REAL(KIND=JPRB) ZA(KLON,KLEV)
    REAL(KIND=JPRB) ZLIQFRAC(KLON,KLEV)
    REAL(KIND=JPRB) ZICEFRAC(KLON,KLEV)
    REAL(KIND=JPRB) ZQX(KLON,KLEV,NCLV) 

    DO JK=1,KLEV
        DO JL=KIDIA,KFDIA

            !------------------------------------------
            ! Ensure cloud fraction is between 0 and 1
            !------------------------------------------
            ZA(JL,JK)=MAX(0.0,MIN(1.0,ZA(JL,JK)))

            !-------------------------------------------------------------------
            ! Calculate liq/ice fractions (no longer a diagnostic relationship)
            !-------------------------------------------------------------------
            ZLI(JL,JK)=ZQX(JL,JK,NCLDQL)+ZQX(JL,JK,NCLDQI)
            IF (ZLI(JL,JK)>RLMIN) THEN
                ZLIQFRAC(JL,JK)=ZQX(JL,JK,NCLDQL)/ZLI(JL,JK)
                ZICEFRAC(JL,JK)=1.0-ZLIQFRAC(JL,JK)
            ELSE
                ZLIQFRAC(JL,JK)=0.0
                ZICEFRAC(JL,JK)=0.0
            ENDIF

        ENDDO
    ENDDO

END SUBROUTINE liq_ice_fractions_routine