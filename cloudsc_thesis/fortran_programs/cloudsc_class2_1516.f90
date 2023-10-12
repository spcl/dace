PROGRAM ice_growth_vapour_deposition
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

    ! input
    REAL(KIND=JPRB)     :: RTT
    REAL(KIND=JPRB)     :: R2ES
    REAL(KIND=JPRB)     :: R3IES
    REAL(KIND=JPRB)     :: R4IES
    REAL(KIND=JPRB)     :: RLMIN
    REAL(KIND=JPRB)     :: RV
    REAL(KIND=JPRB)     :: RD
    REAL(KIND=JPRB)     :: RG
    REAL(KIND=JPRB)     :: RLSTT
    REAL(KIND=JPRB)     :: RDEPLIQREFRATE
    REAL(KIND=JPRB)     :: RDEPLIQREFDEPTH
    REAL(KIND=JPRB)     :: RCLDTOPCF
    REAL(KIND=JPRB)     :: PTSPHY
    REAL(KIND=JPRB)     :: RICEINIT
    REAL(KIND=JPRB)     :: ZA(KLON, KLEV)
    REAL(KIND=JPRB)     :: ZCLDTOPDIST2(KLON,KLEV)
    REAL(KIND=JPRB)     :: ZDP(KLON)
    REAL(KIND=JPRB)     :: ZRHO(KLON)
    REAL(KIND=JPRB)     :: ZTP1(KLON,KLEV)
    REAL(KIND=JPRB)     :: ZFOKOOP(KLON)
    REAL(KIND=JPRB)     :: PAP(KLON, KLEV)
    REAL(KIND=JPRB)     :: ZICECLD(KLON)
    ! Is temporary variable
    ! REAL(KIND=JPRB)     :: ZICENUCLEI2(KLON)

    ! Is in and output
    REAL(KIND=JPRB)     :: ZSOLQA2(KLON, KLEV, NCLV, NCLV)
    ! Is in and output
    REAL(KIND=JPRB)     :: ZQXFG2(KLON, KLEV, NCLV)

    CALL ice_growth_vapour_deposition_routine(&
        & KLON, KLEV, KIDIA, KFDIA,  NCLV, NCLDTOP, NCLDQL, NCLDQI, &
        & RTT, R2ES, R3IES, R4IES, RLMIN, RV, RD, RG, RLSTT, RDEPLIQREFRATE, RDEPLIQREFDEPTH, RCLDTOPCF, PTSPHY, RICEINIT, &
        & ZA, ZCLDTOPDIST2, ZDP, ZRHO, ZTP1, ZFOKOOP, PAP, ZICECLD,  &
        & ZSOLQA2, ZQXFG2)

END


SUBROUTINE ice_growth_vapour_deposition_routine(&
        & KLON, KLEV, KIDIA, KFDIA,  NCLV, NCLDTOP, NCLDQL, NCLDQI, &
        & RTT, R2ES, R3IES, R4IES, RLMIN, RV, RD, RG, RLSTT, RDEPLIQREFRATE, RDEPLIQREFDEPTH, RCLDTOPCF, PTSPHY, RICEINIT, &
        & ZA, ZCLDTOPDIST2, ZDP, ZRHO, ZTP1, ZFOKOOP, PAP, ZICECLD,  &
        & ZSOLQA2, ZQXFG2)

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

    ! input
    REAL(KIND=JPRB)     :: RTT
    REAL(KIND=JPRB)     :: R2ES
    REAL(KIND=JPRB)     :: R3IES
    REAL(KIND=JPRB)     :: R4IES
    REAL(KIND=JPRB)     :: RLMIN
    REAL(KIND=JPRB)     :: RV
    REAL(KIND=JPRB)     :: RD
    REAL(KIND=JPRB)     :: RG
    REAL(KIND=JPRB)     :: RLSTT
    REAL(KIND=JPRB)     :: RDEPLIQREFRATE
    REAL(KIND=JPRB)     :: RDEPLIQREFDEPTH
    REAL(KIND=JPRB)     :: RCLDTOPCF
    REAL(KIND=JPRB)     :: PTSPHY
    REAL(KIND=JPRB)     :: RICEINIT
    REAL(KIND=JPRB)     :: ZA(KLON, KLEV)
    REAL(KIND=JPRB)     :: ZCLDTOPDIST2(KLON, KLEV)
    REAL(KIND=JPRB)     :: ZDP(KLON)
    REAL(KIND=JPRB)     :: ZRHO(KLON)
    REAL(KIND=JPRB)     :: ZTP1(KLON,KLEV)
    REAL(KIND=JPRB)     :: ZFOKOOP(KLON)
    REAL(KIND=JPRB)     :: PAP(KLON, KLEV)
    REAL(KIND=JPRB)     :: ZICECLD(KLON)

    ! Is in and output
    REAL(KIND=JPRB)     :: ZSOLQA2(KLON, KLEV, NCLV, NCLV)
    ! Is in and output
    REAL(KIND=JPRB)     :: ZQXFG2(KLON, KLEV, NCLV)

    ! temporary variables
    REAL(KIND=JPRB)     :: FOEEICE
    REAL(KIND=JPRB)     :: ZVPICE
    REAL(KIND=JPRB)     :: ZVPLIQ
    REAL(KIND=JPRB)     :: ZDEPOS
    REAL(KIND=JPRB)     :: ZINEW
    REAL(KIND=JPRB)     :: ZCVDS
    REAL(KIND=JPRB)     :: ZBDD
    REAL(KIND=JPRB)     :: ZADD
    REAL(KIND=JPRB)     :: ZINFACTOR
    REAL(KIND=JPRB)     :: ZICE0
    ! Is temporary variable
    REAL(KIND=JPRB)     :: ZICENUCLEI(KLON, KLEV)

    DO JK=NCLDTOP,KLEV
        DO JL=KIDIA,KFDIA

            !--------------------------------------------------------------
            ! Calculate distance from cloud top 
            ! defined by cloudy layer below a layer with cloud frac <0.01
            ! ZDZ = ZDP(JL)/(ZRHO(JL)*RG)
            !--------------------------------------------------------------

            IF (ZA(JL,JK-1) < RCLDTOPCF .AND. ZA(JL,JK) >= RCLDTOPCF) THEN
                ZCLDTOPDIST2(JL,JK) = 0.0
            ELSE
                ZCLDTOPDIST2(JL,JK) = ZCLDTOPDIST2(JL,JK) + ZDP(JL)/(ZRHO(JL)*RG)
            ENDIF

            !--------------------------------------------------------------
            ! only treat depositional growth if liquid present. due to fact 
            ! that can not model ice growth from vapour without additional 
            ! in-cloud water vapour variable
            !--------------------------------------------------------------
            IF (ZTP1(JL,JK)<RTT .AND. ZQXFG2(JL,JK,NCLDQL)>RLMIN) THEN  ! T<273K

                ! Manual inlince of FOEEICE function
                ! Has having it inside this function results in a KeyError: Internal_Subprogram_Part
                FOEEICE = R2ES*EXP(R3IES*(ZTP1(JL,JK)-RTT)/(ZTP1(JL,JK)-R4IES))
                ZVPICE=FOEEICE*RV/RD
                ZVPLIQ=ZVPICE*ZFOKOOP(JL) 
                ZICENUCLEI(JL,JK)=1000.0*EXP(12.96*(ZVPLIQ-ZVPICE)/ZVPLIQ-0.639)

                !------------------------------------------------
                !   2.4e-2 is conductivity of air
                !   8.8 = 700**1/3 = density of ice to the third
                !------------------------------------------------
                ZADD=RLSTT*(RLSTT/(RV*ZTP1(JL,JK))-1.0)/(2.4E-2*ZTP1(JL,JK))
                ZBDD=RV*ZTP1(JL,JK)*PAP(JL,JK)/(2.21*ZVPICE)
                ZCVDS=7.8*(ZICENUCLEI(JL,JK)/ZRHO(JL))**0.666*(ZVPLIQ-ZVPICE) / &
                 & (8.87*(ZADD+ZBDD)*ZVPICE)

                !-----------------------------------------------------
                ! RICEINIT=1.E-12_JPRB is initial mass of ice particle
                !-----------------------------------------------------
                ZICE0=MAX(ZICECLD(JL), ZICENUCLEI(JL,JK)*RICEINIT/ZRHO(JL))

                !------------------
                ! new value of ice:
                !------------------
                ZINEW=(0.666*ZCVDS*PTSPHY+ZICE0**0.666)**1.5

                !---------------------------
                ! grid-mean deposition rate:
                !--------------------------- 
                ZDEPOS=MAX(ZA(JL,JK)*(ZINEW-ZICE0),0.0)

                !--------------------------------------------------------------------
                ! Limit deposition to liquid water amount
                ! If liquid is all frozen, ice would use up reservoir of water 
                ! vapour in excess of ice saturation mixing ratio - However this 
                ! can not be represented without a in-cloud humidity variable. Using 
                ! the grid-mean humidity would imply a large artificial horizontal 
                ! flux from the clear sky to the cloudy area. We thus rely on the 
                ! supersaturation check to clean up any remaining supersaturation
                !--------------------------------------------------------------------
                ZDEPOS=MIN(ZDEPOS,ZQXFG2(JL,JK,NCLDQL)) ! limit to liquid water amount

                !--------------------------------------------------------------------
                ! At top of cloud, reduce deposition rate near cloud top to account for
                ! small scale turbulent processes, limited ice nucleation and ice fallout 
                !--------------------------------------------------------------------
                !      ZDEPOS = ZDEPOS*MIN(RDEPLIQREFRATE+ZCLDTOPDIST(JL)/RDEPLIQREFDEPTH,1.0_JPRB)
                ! Change to include dependence on ice nuclei concentration
                ! to increase deposition rate with decreasing temperatures 
                ZINFACTOR = MIN(ZICENUCLEI(JL,JK)/15000., 1.0)
                ZDEPOS = ZDEPOS*MIN(ZINFACTOR + (1.0-ZINFACTOR)* &
                          & (RDEPLIQREFRATE+ZCLDTOPDIST2(JL,JK)/RDEPLIQREFDEPTH),1.0)

                !--------------
                ! add to matrix 
                !--------------
                ZSOLQA2(JL,JK,NCLDQI,NCLDQL)=ZSOLQA2(JL,JK,NCLDQI,NCLDQL)+ZDEPOS
                ZSOLQA2(JL,JK,NCLDQL,NCLDQI)=ZSOLQA2(JL,JK,NCLDQL,NCLDQI)-ZDEPOS
                ZQXFG2(JL,JK,NCLDQI)=ZQXFG2(JL,JK,NCLDQI)+ZDEPOS
                ZQXFG2(JL,JK,NCLDQL)=ZQXFG2(JL,JK,NCLDQL)-ZDEPOS

            ENDIF
        ENDDO
    ENDDO ! on vertical level JK

    ! CONTAINS

    ! PURE ELEMENTAL FUNCTION FOEEICE(PTARE)
    !     REAL :: FOEEICE
    !     REAL(KIND=JPRB), VALUE, INTENT(IN) :: PTARE

    !     FOEEICE = R2ES*EXP(R3IES*(PTARE-RTT)/(PTARE-R4IES))
    !     RETURN
    ! END FUNCTION FOEEICE

END SUBROUTINE ice_growth_vapour_deposition_routine
