PROGRAM precipitation_cover_overlap

    INTEGER, PARAMETER :: JPIM = SELECTED_INT_KIND(9)
    INTEGER, PARAMETER :: JPRB = SELECTED_REAL_KIND(13, 300)

    INTEGER(KIND=JPIM), PARAMETER  :: KLON = 100
    INTEGER(KIND=JPIM), PARAMETER  :: KLEV = 100
    INTEGER(KIND=JPIM), PARAMETER  :: NCLV = 100
    INTEGER(KIND=JPIM)  :: KIDIA 
    INTEGER(KIND=JPIM)  :: KFDIA 
    INTEGER(KIND=JPIM)  :: NCLDTOP
    INTEGER(KIND=JPIM)  :: NCLDQS
    INTEGER(KIND=JPIM)  :: NCLDQR

    ! Might consider setting RCOVPMIN not randomly
    REAL(KIND=JPRB)     :: RCOVPMIN
    REAL(KIND=JPRB)     :: ZEPSEC
    REAL(KIND=JPRB)     :: ZQXFG(KLON, NCLV)
    REAL(KIND=JPRB)     :: ZA(KLON, KLEV)
    REAL(KIND=JPRB)     :: ZQPRETOT(KLON)

    REAL(KIND=JPRB)     :: ZCOVPTOT(KLON)
    REAL(KIND=JPRB)     :: ZCOVPCLR(KLON)
    REAL(KIND=JPRB)     :: ZCOVPMAX(KLON)
    REAL(KIND=JPRB)     :: ZRAINCLD(KLON)
    REAL(KIND=JPRB)     :: ZSNOWCLD(KLON)


    CALL precipitation_cover_overlap_routine(&
        & KLON, KLEV, KIDIA, KFDIA, NCLV, NCLDTOP, NCLDQS, NCLDQR, &
        & RCOVPMIN, ZEPSEC, ZQXFG, ZA, ZQPRETOT, &
        & ZCOVPTOT, ZCOVPCLR, ZCOVPMAX, ZRAINCLD, ZSNOWCLD)

END

SUBROUTINE precipitation_cover_overlap_routine(&
        & KLON, KLEV, KIDIA, KFDIA, NCLV, NCLDTOP, NCLDQS, NCLDQR, &
        & RCOVPMIN, ZEPSEC, ZQXFG, ZA, ZQPRETOT, &
        & ZCOVPTOT, ZCOVPCLR, ZCOVPMAX, ZRAINCLD, ZSNOWCLD)

    INTEGER, PARAMETER :: JPIM = SELECTED_INT_KIND(9)
    INTEGER, PARAMETER :: JPRB = SELECTED_REAL_KIND(13, 300)

    INTEGER(KIND=JPIM)  :: KLON
    INTEGER(KIND=JPIM)  :: KLEV
    INTEGER(KIND=JPIM)  :: KIDIA 
    INTEGER(KIND=JPIM)  :: KFDIA 
    INTEGER(KIND=JPIM)  :: NCLV 
    INTEGER(KIND=JPIM)  :: NCLDTOP
    INTEGER(KIND=JPIM)  :: NCLDQS
    INTEGER(KIND=JPIM)  :: NCLDQR

    ! Might consider setting RCOVPMIN not randomly
    REAL(KIND=JPRB)     :: RCOVPMIN
    REAL(KIND=JPRB)     :: ZEPSEC
    REAL(KIND=JPRB)     :: ZQXFG(KLON, NCLV)
    REAL(KIND=JPRB)     :: ZA(KLON, KLEV)
    REAL(KIND=JPRB)     :: ZQPRETOT(KLON)

    REAL(KIND=JPRB)     :: ZCOVPTOT(KLON)
    REAL(KIND=JPRB)     :: ZCOVPCLR(KLON)
    REAL(KIND=JPRB)     :: ZCOVPMAX(KLON)
    REAL(KIND=JPRB)     :: ZRAINCLD(KLON)
    REAL(KIND=JPRB)     :: ZSNOWCLD(KLON)

    DO JK=NCLDTOP,KLEV
        DO JL=KIDIA,KFDIA     ! LOOP CLASS 2
            IF (ZQPRETOT(JL)>ZEPSEC) THEN
                ZCOVPTOT(JL) = 1.0 - ((1.0-ZCOVPTOT(JL))*&
                &            (1.0 - MAX(ZA(JL,JK),ZA(JL,JK-1)))/&
                &            (1.0 - MIN(ZA(JL,JK-1),1.0-1.E-06)) )  
                ZCOVPTOT(JL) = MAX(ZCOVPTOT(JL),RCOVPMIN)
                ZCOVPCLR(JL) = MAX(0.0,ZCOVPTOT(JL)-ZA(JL,JK)) ! clear sky proportion
                ZRAINCLD(JL) = ZQXFG(JL,NCLDQR)/ZCOVPTOT(JL)
                ZSNOWCLD(JL) = ZQXFG(JL,NCLDQS)/ZCOVPTOT(JL)
                ZCOVPMAX(JL) = MAX(ZCOVPTOT(JL),ZCOVPMAX(JL))
            ELSE
                ZRAINCLD(JL) = 0.0 
                ZSNOWCLD(JL) = 0.0 
                ZCOVPTOT(JL) = 0.0 ! no flux - reset cover
                ZCOVPCLR(JL) = 0.0   ! reset clear sky proportion 
                ZCOVPMAX(JL) = 0.0 ! reset max cover for ZZRH calc 
            ENDIF
        ENDDO
    ENDDO

END SUBROUTINE precipitation_cover_overlap_routine

