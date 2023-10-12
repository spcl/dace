PROGRAM rain_evaporation

    INTEGER, PARAMETER :: JPIM = SELECTED_INT_KIND(9)
    INTEGER, PARAMETER :: JPRB = SELECTED_REAL_KIND(13, 300)

    INTEGER(KIND=JPIM), PARAMETER  :: KLON = 100
    INTEGER(KIND=JPIM), PARAMETER  :: KLEV = 100
    INTEGER(KIND=JPIM), PARAMETER  :: NCLV = 100

    INTEGER(KIND=JPIM) KIDIA 
    INTEGER(KIND=JPIM) KFDIA 
    INTEGER(KIND=JPIM) NCLDQV 
    INTEGER(KIND=JPIM) NCLDQR 
    INTEGER(KIND=JPIM) NCLDTOP 

    ! inputs
    REAL(KIND=JPRB) RPRECRHMAX
    REAL(KIND=JPRB) ZEPSEC
    REAL(KIND=JPRB) ZEPSILON
    REAL(KIND=JPRB) RVRFACTOR
    REAL(KIND=JPRB) RG
    REAL(KIND=JPRB) RPECONS
    REAL(KIND=JPRB) PTSPHY
    REAL(KIND=JPRB) ZRG_R
    REAL(KIND=JPRB) RCOVPMIN
    REAL(KIND=JPRB) ZCOVPMAX(KLON)
    REAL(KIND=JPRB) ZA(KLON, KLEV)
    REAL(KIND=JPRB) ZQX(KLON, KLEV, NCLV)
    REAL(KIND=JPRB) ZQSLIQ(KLON, KLEV)
    REAL(KIND=JPRB) ZCOVPCLR(KLON)
    REAL(KIND=JPRB) ZDTGDP(KLON)
    REAL(KIND=JPRB) PAP(KLON, KLEV)
    REAL(KIND=JPRB) PAPH(KLON, KLEV+1)
    REAL(KIND=JPRB) ZCORQSLIQ(KLON)
    REAL(KIND=JPRB) ZDP(KLON)

    ! outputs
    REAL(KIND=JPRB) ZSOLQA2(KLON, KLEV, NCLV, NCLV)
    REAL(KIND=JPRB) ZCOVPTOT2(KLON,KLEV)
    REAL(KIND=JPRB) ZQXFG2(KLON, KLEV, NCLV)

    CALL rain_evaporation_routine(&
        & KLON, KLEV, NCLV, KIDIA , KFDIA , NCLDQV , NCLDQR , NCLDTOP, &
        & RPRECRHMAX, ZEPSEC, ZEPSILON, RVRFACTOR, RG, RPECONS, PTSPHY, ZRG_R, RCOVPMIN, &
        & ZCOVPMAX, ZA, ZQX, ZQSLIQ, ZCOVPCLR, ZDTGDP, PAP, PAPH, ZCORQSLIQ, ZDP, &
        & ZSOLQA2, ZCOVPTOT2, ZQXFG2)

END PROGRAM

SUBROUTINE rain_evaporation_routine(&
        & KLON, KLEV, NCLV, KIDIA , KFDIA , NCLDQV , NCLDQR , NCLDTOP, &
        & RPRECRHMAX, ZEPSEC, ZEPSILON, RVRFACTOR, RG, RPECONS, PTSPHY, ZRG_R, RCOVPMIN, &
        & ZCOVPMAX, ZA, ZQX, ZQSLIQ, ZCOVPCLR, ZDTGDP, PAP, PAPH, ZCORQSLIQ, ZDP, &
        & ZSOLQA2, ZCOVPTOT2, ZQXFG2)

    INTEGER, PARAMETER :: JPIM = SELECTED_INT_KIND(9)
    INTEGER, PARAMETER :: JPRB = SELECTED_REAL_KIND(13, 300)

    INTEGER(KIND=JPIM) KLON
    INTEGER(KIND=JPIM) KLEV
    INTEGER(KIND=JPIM) NCLV
    INTEGER(KIND=JPIM) KIDIA 
    INTEGER(KIND=JPIM) KFDIA 
    INTEGER(KIND=JPIM) NCLDQV 
    INTEGER(KIND=JPIM) NCLDQR 
    INTEGER(KIND=JPIM) NCLDTOP 

    ! inputs
    REAL(KIND=JPRB) RPRECRHMAX
    REAL(KIND=JPRB) ZEPSEC
    REAL(KIND=JPRB) ZEPSILON
    REAL(KIND=JPRB) RVRFACTOR
    REAL(KIND=JPRB) RG
    REAL(KIND=JPRB) RPECONS
    REAL(KIND=JPRB) PTSPHY
    REAL(KIND=JPRB) ZRG_R
    REAL(KIND=JPRB) RCOVPMIN
    REAL(KIND=JPRB) ZCOVPMAX(KLON)
    REAL(KIND=JPRB) ZA(KLON, KLEV)
    REAL(KIND=JPRB) ZQX(KLON, KLEV, NCLV)
    REAL(KIND=JPRB) ZQSLIQ(KLON, KLEV)
    REAL(KIND=JPRB) ZCOVPCLR(KLON)
    REAL(KIND=JPRB) ZDTGDP(KLON)
    REAL(KIND=JPRB) PAP(KLON, KLEV)
    REAL(KIND=JPRB) PAPH(KLON, KLEV+1)
    REAL(KIND=JPRB) ZCORQSLIQ(KLON)
    REAL(KIND=JPRB) ZDP(KLON)

    ! outputs
    REAL(KIND=JPRB) ZSOLQA2(KLON, KLEV, NCLV, NCLV)
    REAL(KIND=JPRB) ZCOVPTOT2(KLON, KLEV)
    REAL(KIND=JPRB) ZQXFG2(KLON, KLEV, NCLV)

    ! temporary variables
    REAL(KIND=JPRB) ZZRH
    REAL(KIND=JPRB) ZQE
    REAL(KIND=JPRB) ZPRECLR
    REAL(KIND=JPRB) ZBETA1
    REAL(KIND=JPRB) ZBETA
    REAL(KIND=JPRB) ZDENOM
    REAL(KIND=JPRB) ZDPR
    REAL(KIND=JPRB) ZDPEVAP
    REAL(KIND=JPRB) ZEVAP
    LOGICAL LLO1

    DO JK=NCLDTOP,KLEV
        DO JL=KIDIA,KFDIA

        ZZRH=RPRECRHMAX+(1.0-RPRECRHMAX)*ZCOVPMAX(JL)/MAX(ZEPSEC,1.0-ZA(JL,JK))
        ZZRH=MIN(MAX(ZZRH,RPRECRHMAX),1.0)

        ZQE=(ZQX(JL,JK,NCLDQV)-ZA(JL,JK)*ZQSLIQ(JL,JK))/&
        & MAX(ZEPSEC,1.0-ZA(JL,JK))  
        !---------------------------------------------
        ! humidity in moistest ZCOVPCLR part of domain
        !---------------------------------------------
        ZQE=MAX(0.0,MIN(ZQE,ZQSLIQ(JL,JK)))
        LLO1=ZCOVPCLR(JL)>ZEPSEC .AND. &
        & ZQXFG2(JL,JK,NCLDQR)>ZEPSEC .AND. &
        & ZQE<ZZRH*ZQSLIQ(JL,JK)

        IF(LLO1) THEN
            ! note: zpreclr is a rain flux
            ZPRECLR = ZQXFG2(JL,JK,NCLDQR)*ZCOVPCLR(JL)/ &
            & SIGN(MAX(ABS(ZCOVPTOT2(JL,JK)*ZDTGDP(JL)),ZEPSILON),ZCOVPTOT2(JL,JK)*ZDTGDP(JL))

            !--------------------------------------
            ! actual microphysics formula in zbeta
            !--------------------------------------

            ZBETA1 = SQRT(PAP(JL,JK)/&
            & PAPH(JL,KLEV+1))/RVRFACTOR*ZPRECLR/&
            & MAX(ZCOVPCLR(JL),ZEPSEC)

            ZBETA=RG*RPECONS*0.5*ZBETA1**0.5777

            ZDENOM  = 1.0+ZBETA*PTSPHY*ZCORQSLIQ(JL)
            ZDPR    = ZCOVPCLR(JL)*ZBETA*(ZQSLIQ(JL,JK)-ZQE)/ZDENOM*ZDP(JL)*ZRG_R
            ZDPEVAP = ZDPR*ZDTGDP(JL)

            !---------------------------------------------------------
            ! add evaporation term to explicit sink.
            ! this has to be explicit since if treated in the implicit
            ! term evaporation can not reduce rain to zero and model
            ! produces small amounts of rainfall everywhere. 
            !---------------------------------------------------------

            ! Evaporate rain
            ZEVAP = MIN(ZDPEVAP,ZQXFG2(JL,JK,NCLDQR))

            ZSOLQA2(JL,JK,NCLDQV,NCLDQR) = ZSOLQA2(JL,JK,NCLDQV,NCLDQR)+ZEVAP
            ZSOLQA2(JL,JK,NCLDQR,NCLDQV) = ZSOLQA2(JL,JK,NCLDQR,NCLDQV)-ZEVAP

            !-------------------------------------------------------------
            ! Reduce the total precip coverage proportional to evaporation
            ! to mimic the previous scheme which had a diagnostic
            ! 2-flux treatment, abandoned due to the new prognostic precip
            !-------------------------------------------------------------
            ZCOVPTOT2(JL,JK) = MAX(RCOVPMIN,ZCOVPTOT2(JL,JK)-MAX(0.0, &
            &            (ZCOVPTOT2(JL,JK)-ZA(JL,JK))*ZEVAP/ZQXFG2(JL,JK,NCLDQR)))

            ! Update fg field
            ZQXFG2(JL,JK,NCLDQR) = ZQXFG2(JL,JK,NCLDQR)-ZEVAP

            ENDIF
        ENDDO
    ENDDO

END SUBROUTINE rain_evaporation_routine
