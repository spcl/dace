PROGRAM vert_loop_1

    INTEGER, PARAMETER :: JPIM = SELECTED_INT_KIND(9)
    INTEGER, PARAMETER :: JPRB = SELECTED_REAL_KIND(13, 300)

    INTEGER(KIND=JPIM), PARAMETER  :: KLON = 100
    INTEGER(KIND=JPIM), PARAMETER  :: KLEV = 100
    INTEGER(KIND=JPIM), PARAMETER  :: NCLV = 100

    ! Parameters
    INTEGER(KIND=JPIM) KIDIA 
    INTEGER(KIND=JPIM) KFDIA 
    INTEGER(KIND=JPIM) NCLDQS 
    INTEGER(KIND=JPIM) NCLDQI 
    INTEGER(KIND=JPIM) NCLDQL 
    INTEGER(KIND=JPIM) NGPTOT 
    INTEGER(KIND=JPIM) NPROMA 

    ! input
    REAL(KIND=JPRB) RCOVPMIN
    REAL(KIND=JPRB) ZEPSEC
    REAL(KIND=JPRB) RLCRITSNOW
    REAL(KIND=JPRB) RCLCRIT_SEA

    ! TODO: Change sizes of the input over NBLOCKS
    REAL(KIND=JPRB) ZA(KLON,KLEV)
    REAL(KIND=JPRB) ZQX(KLON,KLEV,NCLV)
    REAL(KIND=JPRB) ZTP1(KLON,KLEV)
    REAL(KIND=JPRB) ZPFPLSX(KLON,KLEV+1,NCLV)

    ! output
    REAL(KIND=JPRB) ZSOLQB(KLON,NCLV,NCLV)

    CALL vert_loop_1_routine(&
        & KLON, KLEV, NCLV, KIDIA, KFDIA, NCLDQS, NCDLQI, NCLDQL, NGPTOT, NPROMA, &
        & RCOVPMIN, ZEPSEC, RLCRITSNOW, RLCRIT_SEA, ZA, ZQX, ZTP1, ZPFPLSX, &
        & ZSOLQB)

END PROGRAM

SUBROUTINE vert_loop_1_routine(&
    & KLON, KLEV, NCLV, KIDIA, KFDIA, NCLDQS, NCDLQI, NCLDQL, NGPTOT, NPROMA, &
    & RCOVPMIN, ZEPSEC, RLCRITSNOW, RLCRIT_SEA, ZA, ZQX, ZTP1, ZPFPLSX, &
    & ZSOLQB)
    INTEGER, PARAMETER :: JPIM = SELECTED_INT_KIND(9)
    INTEGER, PARAMETER :: JPRB = SELECTED_REAL_KIND(13, 300)

    ! Parameters
    INTEGER(KIND=JPIM) KLON
    INTEGER(KIND=JPIM) KLEV
    INTEGER(KIND=JPIM) NCLV
    INTEGER(KIND=JPIM) KIDIA 
    INTEGER(KIND=JPIM) KFDIA 
    INTEGER(KIND=JPIM) NCLDQS 
    INTEGER(KIND=JPIM) NCLDQI 
    INTEGER(KIND=JPIM) NCLDQL 
    INTEGER(KIND=JPIM) NGPTOT 
    INTEGER(KIND=JPIM) NPROMA 

    ! input
    REAL(KIND=JPRB) RCOVPMIN
    REAL(KIND=JPRB) ZEPSEC
    REAL(KIND=JPRB) RLCRITSNOW
    REAL(KIND=JPRB) RCLCRIT_SEA

    REAL(KIND=JPRB) ZA(KLON,KLEV)
    REAL(KIND=JPRB) ZQX(KLON,KLEV,NCLV)
    REAL(KIND=JPRB) ZTP1(KLON,KLEV)
    REAL(KIND=JPRB) ZPFPLSX(KLON,KLEV+1,NCLV)

    ! output
    REAL(KIND=JPRB) ZSOLQB(KLON,NCLV,NCLV)

    ! temporary
    REAL(KIND=JPRB) ZTMPA
    REAL(KIND=JPRB) ZZCO
    REAL(KIND=JPRB) ZLCRIT
    REAL(KIND=JPRB) ZPRECIP
    REAL(KIND=JPRB) ZCFPR

    ! Should KLON not be 1 here?
    REAL(KIND=JPRB) ZLIQCLD(KLON)
    REAL(KIND=JPRB) ZICECLD(KLON)
    REAL(KIND=JPRB) ZCOVPTOT(KLON)
    REAL(KIND=JPRB) ZSNOWAUT(KLON)
    REAL(KIND=JPRB) ZRAINAUT(KLON)

    ZSOLQB(:,:,:) = 0.0
    DO JKGLO=1,NGPTOT,NPROMA
        DO JK=NCLDTOP,KLEV
            DO JL=KIDIA,KFDIA
                ZTMPA = 1.0/MAX(ZA(JL,JK),ZEPSEC)
                ZLIQCLD(JL) = ZQX(JL,JK,NCLDQL)*ZTMPA
                ZICECLD(JL) = ZQX(JL,JK,NCLDQI)*ZTMPA
            ENDDO
        
            DO JL=KIDIA,KFDIA
                ! Remove if/else here
                ZCOVPTOT(JL) = 1.0 - ((1.0-ZCOVPTOT(JL))*&
                &            (1.0 - MAX(ZA(JL,JK),ZA(JL,JK-1)))/&
                &            (1.0 - MIN(ZA(JL,JK-1),1.0-1.E-06)) )  
                ZCOVPTOT(JL) = MAX(ZCOVPTOT(JL),RCOVPMIN)
                ! to here
            ENDDO
              
            DO JL=KIDIA,KFDIA
                IF(ZTP1(JL,JK) <= RTT) THEN
                    IF (ZICECLD(JL)>ZEPSEC) THEN
                        ZZCO=PTSPHY*RSNOWLIN1*EXP(RSNOWLIN2*(ZTP1(JL,JK)-RTT))
                        ! Removed if here
                        ZLCRIT=RLCRITSNOW
                        ! to here
                        ZSNOWAUT(JL)=ZZCO*(1.0-EXP(-(ZICECLD(JL)/ZLCRIT)**2))
                        ZSOLQB(JL,NCLDQS,NCLDQI)=ZSOLQB(JL,NCLDQS,NCLDQI)+ZSNOWAUT(JL)
                    ENDIF
                ENDIF 
              
                IF (ZLIQCLD(JL)>ZEPSEC) THEN
                    ZZCO=RKCONV*PTSPHY
                    ! Removed some ifs around ZLCRIT
                    ZLCRIT = RCLCRIT_SEA  ! ocean
                    ZPRECIP=(ZPFPLSX(JL,JK,NCLDQS)+ZPFPLSX(JL,JK,NCLDQR))/MAX(ZEPSEC,ZCOVPTOT(JL))
                    ZCFPR=1.0 + RPRC1*SQRT(MAX(ZPRECIP,0.0))
                    ZLCRIT=ZLCRIT/MAX(ZCFPR,ZEPSEC)
              
                    IF(ZLIQCLD(JL)/ZLCRIT < 20.0 )THEN ! Security for exp for some compilers
                        ZRAINAUT(JL)=ZZCO*(1.0-EXP(-(ZLIQCLD(JL)/ZLCRIT)**2))
                    ELSE
                        ZRAINAUT(JL)=ZZCO
                    ENDIF

                    ! rain freezes instantly
                    IF(ZTP1(JL,JK) <= RTT) THEN
                        ZSOLQB(JL,NCLDQS,NCLDQL)=ZSOLQB(JL,NCLDQS,NCLDQL)+ZRAINAUT(JL)
                    ELSE
                        ZSOLQB(JL,NCLDQR,NCLDQL)=ZSOLQB(JL,NCLDQR,NCLDQL)+ZRAINAUT(JL)
                    ENDIF
                ENDIF
            ENDDO
        ENDDO
    ENDDO

END SUBROUTINE vert_loop_1_routine

