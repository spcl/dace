PROGRAM update_tendancies

    INTEGER, PARAMETER :: JPIM = SELECTED_INT_KIND(9)
    INTEGER, PARAMETER :: JPRB = SELECTED_REAL_KIND(13, 300)

    INTEGER(KIND=JPIM), PARAMETER  :: KLON = 100
    INTEGER(KIND=JPIM), PARAMETER  :: KLEV = 100
    INTEGER(KIND=JPIM), PARAMETER  :: NCLV = 100

    INTEGER(KIND=JPIM) KIDIA 
    INTEGER(KIND=JPIM) KFDIA
    INTEGER(KIND=JPIM) NCLDQV
    INTEGER(KIND=JPIM) NCLDTOP

    REAL(KIND=JPRB) RALSDCP
    REAL(KIND=JPRB) RALVDCP
    REAL(KIND=JPRB) ZQTMST

    REAL(KIND=JPRB) ZQX(KLON,KLEV,NCLV)
    REAL(KIND=JPRB) ZQX0(KLON,KLEV,NCLV)

    REAL(KIND=JPRB) ZDA(KLON)
    REAL(KIND=JPRB) ZFLUXQ(KLON,NCLV) 
    REAL(KIND=JPRB) ZPSUPSATSRCE(KLON,NCLV)
    REAL(KIND=JPRB) ZCONVSRCE(KLON,NCLV)
    REAL(KIND=JPRB) ZFALLSINK(KLON,NCLV) 
    REAL(KIND=JPRB) ZFALLSRCE(KLON,NCLV)
    REAL(KIND=JPRB) ZCONVSINK(KLON,NCLV)
    REAL(KIND=JPRB) ZQXN(KLON,NCLV)

    REAL(KIND=JPRB) IPHASE(NCLV)
    REAL(KIND=JPRB) tendency_loc_a(KLON,KLEV)
    REAL(KIND=JPRB) tendency_loc_T(KLON,KLEV)
    REAL(KIND=JPRB) tendency_loc_q(KLON,KLEV)
    REAL(KIND=JPRB) tendency_loc_cld(KLON,KLEV,NCLV)

    REAL(KIND=JPRB) PCOVPTOT(KLON,KLEV)
    REAL(KIND=JPRB) ZCOVPTOT(KLON)

    CALL update_tendancies_routine( &
        & KLON, KLEV, KIDIA, KFDIA, NCLV, NCLDQV, NCLDTOP, &
        & RALSDCP, RALVDCP, ZQTMST, &
        & ZQX, ZQX0, ZDA, ZFLUXQ, ZPSUPSATSRCE, ZCONVSRCE, ZFALLSINK, ZFALLSRCE, ZCONVSINK, ZQXN, &
        & IPHASE, tendency_loc_a, tendency_loc_T, tendency_loc_q, tendency_loc_cld, &
        & PCOVPTOT, ZCOVPTOT)

END

SUBROUTINE update_tendancies_routine( &
    & KLON, KLEV, KIDIA, KFDIA, NCLV, NCLDQV, NCLDTOP, &
    & RALSDCP, RALVDCP, ZQTMST, &
    & ZQX, ZQX0, ZDA, ZFLUXQ, ZPSUPSATSRCE, ZCONVSRCE, ZFALLSINK, ZFALLSRCE, ZCONVSINK, ZQXN, &
    & IPHASE, tendency_loc_a, tendency_loc_T, tendency_loc_q, tendency_loc_cld, &
    & PCOVPTOT, ZCOVPTOT)

    INTEGER, PARAMETER :: JPIM = SELECTED_INT_KIND(9)
    INTEGER, PARAMETER :: JPRB = SELECTED_REAL_KIND(13, 300)

    INTEGER(KIND=JPIM) KLON
    INTEGER(KIND=JPIM) KLEV
    INTEGER(KIND=JPIM) KIDIA 
    INTEGER(KIND=JPIM) KFDIA
    INTEGER(KIND=JPIM) NCLV
    INTEGER(KIND=JPIM) NCLDQV
    INTEGER(KIND=JPIM) NCLDTOP

    REAL(KIND=JPRB) RALSDCP
    REAL(KIND=JPRB) RALVDCP
    REAL(KIND=JPRB) ZQTMST

    REAL(KIND=JPRB) ZQX(KLON,KLEV,NCLV)
    REAL(KIND=JPRB) ZQX0(KLON,KLEV,NCLV)

    REAL(KIND=JPRB) ZDA(KLON)
    REAL(KIND=JPRB) ZFLUXQ(KLON,NCLV) 
    REAL(KIND=JPRB) ZPSUPSATSRCE(KLON,NCLV)
    REAL(KIND=JPRB) ZCONVSRCE(KLON,NCLV)
    REAL(KIND=JPRB) ZFALLSINK(KLON,NCLV) 
    REAL(KIND=JPRB) ZFALLSRCE(KLON,NCLV)
    REAL(KIND=JPRB) ZCONVSINK(KLON,NCLV)
    REAL(KIND=JPRB) ZQXN(KLON,NCLV)

    REAL(KIND=JPRB) IPHASE(NCLV)
    REAL(KIND=JPRB) tendency_loc_a(KLON,KLEV)
    REAL(KIND=JPRB) tendency_loc_T(KLON,KLEV)
    REAL(KIND=JPRB) tendency_loc_q(KLON,KLEV)
    REAL(KIND=JPRB) tendency_loc_cld(KLON,KLEV,NCLV)

    REAL(KIND=JPRB) PCOVPTOT(KLON,KLEV)
    REAL(KIND=JPRB) ZCOVPTOT(KLON)

    INTEGER(KIND=JPIM)  :: JK, JL, JM

    DO JK=NCLDTOP,KLEV

        !--------------------------------
        ! 6.1 Temperature and CLV budgets 
        !--------------------------------

        DO JM=1,NCLV-1
            DO JL=KIDIA,KFDIA

                ! calculate fluxes in and out of box for conservation of TL
                ZFLUXQ(JL,JM)=ZPSUPSATSRCE(JL,JM)+ZCONVSRCE(JL,JM)+ZFALLSRCE(JL,JM)-&
                    & (ZFALLSINK(JL,JM)+ZCONVSINK(JL,JM))*ZQXN(JL,JM)
            ENDDO

            IF (IPHASE(JM)==1) THEN
                DO JL=KIDIA,KFDIA
                    tendency_loc_T(JL,JK)=tendency_loc_T(JL,JK)+ &
                        & RALVDCP*(ZQXN(JL,JM)-ZQX(JL,JK,JM)-ZFLUXQ(JL,JM))*ZQTMST
                ENDDO
            ENDIF

            IF (IPHASE(JM)==2) THEN
                DO JL=KIDIA,KFDIA
                    tendency_loc_T(JL,JK)=tendency_loc_T(JL,JK)+ &
                        & RALSDCP*(ZQXN(JL,JM)-ZQX(JL,JK,JM)-ZFLUXQ(JL,JM))*ZQTMST
                ENDDO
            ENDIF

            !----------------------------------------------------------------------
            ! New prognostic tendencies - ice,liquid rain,snow 
            ! Note: CLV arrays use PCLV in calculation of tendency while humidity
            !       uses ZQX. This is due to clipping at start of cloudsc which
            !       include the tendency already in tendency_loc_T and tendency_loc_q. ZQX was reset
            !----------------------------------------------------------------------
            DO JL=KIDIA,KFDIA
                tendency_loc_cld(JL,JK,JM)=tendency_loc_cld(JL,JK,JM)+(ZQXN(JL,JM)-ZQX0(JL,JK,JM))*ZQTMST
            ENDDO

        ENDDO

        DO JL=KIDIA,KFDIA
            !----------------------
            ! 6.2 Humidity budget
            !----------------------
            tendency_loc_q(JL,JK)=tendency_loc_q(JL,JK)+(ZQXN(JL,NCLDQV)-ZQX(JL,JK,NCLDQV))*ZQTMST

            !-------------------
            ! 6.3 cloud cover 
            !-----------------------
            tendency_loc_a(JL,JK)=tendency_loc_a(JL,JK)+ZDA(JL)*ZQTMST
        ENDDO

        !--------------------------------------------------
        ! Copy precipitation fraction into output variable
        !-------------------------------------------------
        DO JL=KIDIA,KFDIA
            PCOVPTOT(JL,JK) = ZCOVPTOT(JL)
        ENDDO
    
    ENDDO

END SUBROUTINE update_tendancies_routine