PROGRAM ice_supersaturation_adjustment

    INTEGER, PARAMETER :: JPIM = SELECTED_INT_KIND(9)
    INTEGER, PARAMETER :: JPRB = SELECTED_REAL_KIND(13, 300)

    INTEGER(KIND=JPIM), PARAMETER  :: KLON = 100
    INTEGER(KIND=JPIM), PARAMETER  :: KLEV = 100
    INTEGER(KIND=JPIM), PARAMETER  :: NCLV = 100

    INTEGER(KIND=JPIM)  :: KIDIA 
    INTEGER(KIND=JPIM)  :: KFDIA
    INTEGER(KIND=JPIM)  :: NCLDTOP
    INTEGER(KIND=JPIM)  :: NCLDQL
    INTEGER(KIND=JPIM)  :: NCLDQI
    INTEGER(KIND=JPIM)  :: NCLDQV
    INTEGER(KIND=JPIM) :: NSSOPT

    REAL(KIND=JPRB) :: PTSPHY, RAMIN, RKOOP1, RKOOP2, RKOOPTAU, R2ES, R3LES, R3IES, R4LES, R4IES, RTHOMO, RTT
    REAL(KIND=JPRB) :: ZEPSEC, ZEPSILON

    REAL(KIND=JPRB) :: PSUPSAT(KLON,KLEV)
    REAL(KIND=JPRB) :: ZA(KLON,KLEV)
    REAL(KIND=JPRB) :: ZCORQSICE(KLON)
    REAL(KIND=JPRB) :: ZSOLQA(KLON,NCLV,NCLV)
    REAL(KIND=JPRB) :: ZQSICE(KLON,KLEV)
    REAL(KIND=JPRB) :: ZQX(KLON,KLEV,NCLV)
    REAL(KIND=JPRB) :: ZQXFG(KLON,NCLV) 
    REAL(KIND=JPRB) :: ZTP1(KLON,KLEV) 

    CALL ice_supersaturation_adjustment_routine( &
        & KLON, KLEV, KIDIA, KFDIA, NCLV, NCLDTOP, NCLDQL, NCLDQI, NCLDQV, NSSOPT, &
        & PTSPHY, RAMIN, RKOOP1, RKOOP2, RKOOPTAU, R2ES, R3LES, R3IES, R4LES, R4IES, RTHOMO, RTT, ZEPSEC, ZEPSILON, &
        & PSUPSAT, ZA, ZCORQSICE, ZSOLQA, ZQSICE, ZQX, ZQXFG, ZTP1)

END

SUBROUTINE ice_supersaturation_adjustment_routine( &
    & KLON, KLEV, KIDIA, KFDIA, NCLV, NCLDTOP, NCLDQL, NCLDQI, NCLDQV, NSSOPT, &
    & PTSPHY, RAMIN, RKOOP1, RKOOP2, RKOOPTAU, R2ES, R3LES, R3IES, R4LES, R4IES, RTHOMO, RTT, ZEPSEC, ZEPSILON, &
    & PSUPSAT, ZA, ZCORQSICE, ZSOLQA, ZQSICE, ZQX, ZQXFG, ZTP1)

    INTEGER, PARAMETER :: JPIM = SELECTED_INT_KIND(9)
    INTEGER, PARAMETER :: JPRB = SELECTED_REAL_KIND(13, 300)

    INTEGER(KIND=JPIM)  :: KLON
    INTEGER(KIND=JPIM)  :: KLEV
    INTEGER(KIND=JPIM)  :: KIDIA 
    INTEGER(KIND=JPIM)  :: KFDIA
    INTEGER(KIND=JPIM)  :: NCLV
    INTEGER(KIND=JPIM)  :: NCLDTOP
    INTEGER(KIND=JPIM)  :: NCLDQL
    INTEGER(KIND=JPIM)  :: NCLDQI
    INTEGER(KIND=JPIM)  :: NCLDQV
    INTEGER(KIND=JPIM) :: NSSOPT

    REAL(KIND=JPRB) :: PTSPHY, RAMIN, RKOOP1, RKOOP2, RKOOPTAU, R2ES, R3IES, R3LES, R4IES, R4LES, RTHOMO, RTT
    REAL(KIND=JPRB) :: ZEPSEC, ZEPSILON

    REAL(KIND=JPRB) :: PSUPSAT(KLON,KLEV)
    REAL(KIND=JPRB) :: ZA(KLON,KLEV)
    REAL(KIND=JPRB) :: ZCORQSICE(KLON)
    REAL(KIND=JPRB) :: ZSOLQA(KLON,NCLV,NCLV)
    REAL(KIND=JPRB) :: ZQSICE(KLON,KLEV)
    REAL(KIND=JPRB) :: ZQX(KLON,KLEV,NCLV)
    REAL(KIND=JPRB) :: ZQXFG(KLON,NCLV) 
    REAL(KIND=JPRB) :: ZTP1(KLON,KLEV) 

    REAL(KIND=JPRB) :: ZSOLAC(KLON)
    REAL(KIND=JPRB) :: ZFOKOOP(KLON)
    REAL(KIND=JPRB) :: ZPSUPSATSRCE(KLON,NCLV)
    REAL(KIND=JPRB) :: ZSUPSAT(KLON)

    REAL(KIND=JPRB) :: FOEELIQ, FOEEICE, FOKOOP, PTARE
    REAL(KIND=JPRB) :: ZFAC, ZFACI, ZQP1ENV

    INTEGER(KIND=JPIM)  :: JK, JL

    FOEELIQ( PTARE ) = R2ES*EXP(R3LES*(PTARE-RTT)/(PTARE-R4LES))
    FOEEICE( PTARE ) = R2ES*EXP(R3IES*(PTARE-RTT)/(PTARE-R4IES))
    FOKOOP (PTARE) = MIN(RKOOP1-RKOOP2*PTARE,FOEELIQ(PTARE)/FOEEICE(PTARE))

    DO JK=NCLDTOP,KLEV

        DO JL=KIDIA,KFDIA

        !-----------------------------------
        ! 3.1.1 Supersaturation limit (from Koop)
        !-----------------------------------
        ! Needs to be set for all temperatures
            ZFOKOOP(JL)=FOKOOP(ZTP1(JL,JK))
        ENDDO

        DO JL=KIDIA,KFDIA
    
            IF (ZTP1(JL,JK)>=RTT .OR. NSSOPT==0) THEN
                ZFAC  = 1.0
                ZFACI = 1.0
            ELSE
                ZFAC  = ZA(JL,JK)+ZFOKOOP(JL)*(1.0-ZA(JL,JK))
                ZFACI = PTSPHY/RKOOPTAU
            ENDIF
    
            !-------------------------------------------------------------------
            ! 3.1.2 Calculate supersaturation wrt Koop including dqs/dT 
            !       correction factor
            ! [#Note: QSICE or QSLIQ]
            !-------------------------------------------------------------------
        
            ! Calculate supersaturation to add to cloud
            IF (ZA(JL,JK) > 1.0-RAMIN) THEN
                ZSUPSAT(JL) = MAX((ZQX(JL,JK,NCLDQV)-ZFAC*ZQSICE(JL,JK))/ZCORQSICE(JL)&
                &                  ,0.0)
            ELSE
                ! Calculate environmental humidity supersaturation
                ZQP1ENV = (ZQX(JL,JK,NCLDQV) - ZA(JL,JK)*ZQSICE(JL,JK))/ &
                & MAX(1.0-ZA(JL,JK),ZEPSILON)
            !& SIGN(MAX(ABS(1.0-ZA(JL,JK)),ZEPSILON),1.0-ZA(JL,JK))
                ZSUPSAT(JL) = MAX((1.0-ZA(JL,JK))*(ZQP1ENV-ZFAC*ZQSICE(JL,JK))&
                &                  /ZCORQSICE(JL),0.0)
            ENDIF 
        
            !-------------------------------------------------------------------
            ! Here the supersaturation is turned into liquid water
            ! However, if the temperature is below the threshold for homogeneous
            ! freezing then the supersaturation is turned instantly to ice.
            !--------------------------------------------------------------------
        
            IF (ZSUPSAT(JL) > ZEPSEC) THEN
        
                IF (ZTP1(JL,JK) > RTHOMO) THEN
                ! Turn supersaturation into liquid water        
                ZSOLQA(JL,NCLDQL,NCLDQV) = ZSOLQA(JL,NCLDQL,NCLDQV)+ZSUPSAT(JL)
                ZSOLQA(JL,NCLDQV,NCLDQL) = ZSOLQA(JL,NCLDQV,NCLDQL)-ZSUPSAT(JL)
                ! Include liquid in first guess
                ZQXFG(JL,NCLDQL)=ZQXFG(JL,NCLDQL)+ZSUPSAT(JL)
                ELSE
                ! Turn supersaturation into ice water        
                ZSOLQA(JL,NCLDQI,NCLDQV) = ZSOLQA(JL,NCLDQI,NCLDQV)+ZSUPSAT(JL)
                ZSOLQA(JL,NCLDQV,NCLDQI) = ZSOLQA(JL,NCLDQV,NCLDQI)-ZSUPSAT(JL)
                ! Add ice to first guess for deposition term 
                ZQXFG(JL,NCLDQI)=ZQXFG(JL,NCLDQI)+ZSUPSAT(JL)
                ENDIF
        
                ! Increase cloud amount using RKOOPTAU timescale
                ZSOLAC(JL) = (1.0-ZA(JL,JK))*ZFACI
        
            ENDIF
    
            !-------------------------------------------------------
            ! 3.1.3 Include supersaturation from previous timestep
            ! (Calculated in sltENDIF semi-lagrangian LDSLPHY=T)
            !-------------------------------------------------------    
            IF (PSUPSAT(JL,JK)>ZEPSEC) THEN

                IF (ZTP1(JL,JK) > RTHOMO) THEN
                    ! Turn supersaturation into liquid water
                    ZSOLQA(JL,NCLDQL,NCLDQL) = ZSOLQA(JL,NCLDQL,NCLDQL)+PSUPSAT(JL,JK)
                    ZPSUPSATSRCE(JL,NCLDQL) = PSUPSAT(JL,JK)
                    ! Add liquid to first guess for deposition term 
                    ZQXFG(JL,NCLDQL)=ZQXFG(JL,NCLDQL)+PSUPSAT(JL,JK)
                    ! Store cloud budget diagnostics if required
                ELSE
                    ! Turn supersaturation into ice water
                    ZSOLQA(JL,NCLDQI,NCLDQI) = ZSOLQA(JL,NCLDQI,NCLDQI)+PSUPSAT(JL,JK)
                    ZPSUPSATSRCE(JL,NCLDQI) = PSUPSAT(JL,JK)
                    ! Add ice to first guess for deposition term 
                    ZQXFG(JL,NCLDQI)=ZQXFG(JL,NCLDQI)+PSUPSAT(JL,JK)
                    ! Store cloud budget diagnostics if required
                ENDIF
    
                ! Increase cloud amount using RKOOPTAU timescale
                ZSOLAC(JL)=(1.0-ZA(JL,JK))*ZFACI
            ! Store cloud budget diagnostics if required
            ENDIF
    
        ENDDO ! on JL

    ENDDO

END SUBROUTINE ice_supersaturation_adjustment_routine