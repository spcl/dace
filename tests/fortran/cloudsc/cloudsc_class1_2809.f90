! Copied from tests/fortran/cloudsc/cloudsc_8c.f90

PROGRAM sedimentation_falling_all

    INTEGER, PARAMETER :: JPIM = SELECTED_INT_KIND(9)
    INTEGER, PARAMETER :: JPRB = SELECTED_REAL_KIND(13, 300)

    INTEGER(KIND=JPIM), PARAMETER  :: KLON = 100
    INTEGER(KIND=JPIM), PARAMETER  :: KLEV = 100
    INTEGER(KIND=JPIM), PARAMETER  :: NCLV = 5

    INTEGER(KIND=JPIM)  :: KIDIA
    INTEGER(KIND=JPIM)  :: KFDIA
    INTEGER(KIND=JPIM)  :: NCLDQI
    INTEGER(KIND=JPIM)  :: NCLDTOP

    LOGICAL :: LAERICESED
    LOGICAL :: LLFALL(NCLV)
    REAL(KIND=JPRB) :: ZFALLSRCE(KLON,NCLV)
    REAL(KIND=JPRB) :: ZPFPLSX(KLON,KLEV+1,NCLV)
    REAL(KIND=JPRB) :: ZDTGDP(KLON)
    REAL(KIND=JPRB) :: ZSOLQA(KLON,NCLV,NCLV)
    REAL(KIND=JPRB) :: ZQXFG(KLON,NCLV)
    REAL(KIND=JPRB) :: ZQPRETOT(KLON)
    REAL(KIND=JPRB) :: ZVQX(NCLV)
    REAL(KIND=JPRB) :: ZRHO(KLON)
    REAL(KIND=JPRB) :: PRE_ICE(KLON,KLEV)
    REAL(KIND=JPRB) :: ZFALL
    REAL(KIND=JPRB) :: ZFALLSINK(KLON,NCLV)

    CALL sedimentation_falling_all_routine( &
      & NCLV, KLON, KLEV, NCLDQI, &
      & KIDIA, KFDIA, NCLDTOP, LAERICESED, LLFALL, &
      & ZFALLSRCE, ZPFPLSX, ZDTGDP, ZSOLQA, ZQXFG, ZQPRETOT, &
      & ZVQX, ZRHO, PRE_ICE, ZFALLSINK)

END

SUBROUTINE sedimentation_falling_all_routine ( &
    & NCLV, KLON, KLEV, NCLDQI, NCLDTOP, &
    & KIDIA, KFDIA, LAERICESED, LLFALL, &
    & ZFALLSRCE, ZPFPLSX, ZDTGDP, ZSOLQA, ZQXFG, ZQPRETOT, &
    & ZVQX, ZRHO, PRE_ICE, ZFALLSINK)

    INTEGER, PARAMETER :: JPIM = SELECTED_INT_KIND(9)
    INTEGER, PARAMETER :: JPRB = SELECTED_REAL_KIND(13, 300)

    INTEGER(KIND=JPIM)  :: NCLV
    INTEGER(KIND=JPIM)  :: KLON
    INTEGER(KIND=JPIM)  :: KLEV
    INTEGER(KIND=JPIM)  :: KIDIA
    INTEGER(KIND=JPIM)  :: KFDIA
    INTEGER(KIND=JPIM)  :: NCLDQI
    INTEGER(KIND=JPIM)  :: NCLDTOP

    LOGICAL :: LAERICESED
    LOGICAL :: LLFALL(NCLV)
    REAL(KIND=JPRB) :: ZFALLSRCE(KLON,NCLV)
    REAL(KIND=JPRB) :: ZPFPLSX(KLON,KLEV+1,NCLV)
    REAL(KIND=JPRB) :: ZDTGDP(KLON)
    REAL(KIND=JPRB) :: ZSOLQA(KLON,NCLV,NCLV)
    REAL(KIND=JPRB) :: ZQXFG(KLON,NCLV)
    REAL(KIND=JPRB) :: ZQPRETOT(KLON)
    REAL(KIND=JPRB) :: ZVQX(NCLV)
    REAL(KIND=JPRB) :: ZRHO(KLON)
    REAL(KIND=JPRB) :: ZRE_ICE
    REAL(KIND=JPRB) :: PRE_ICE(KLON,KLEV)
    REAL(KIND=JPRB) :: ZFALL
    REAL(KIND=JPRB) :: ZFALLSINK(KLON,NCLV)

    INTEGER(KIND=JPIM)  :: JM, JL

    DO JK=NCLDTOP,KLEV
      DO JM = 1,NCLV
          IF (LLFALL(JM) .OR. JM == NCLDQI) THEN
          DO JL=KIDIA,KFDIA
              !------------------------
              ! source from layer above
              !------------------------
              IF (JK > NCLDTOP) THEN
              ZFALLSRCE(JL,JM) = ZPFPLSX(JL,JK,JM)*ZDTGDP(JL)
              ZSOLQA(JL,JM,JM) = ZSOLQA(JL,JM,JM)+ZFALLSRCE(JL,JM)
              ZQXFG(JL,JM)     = ZQXFG(JL,JM)+ZFALLSRCE(JL,JM)
              ! use first guess precip----------V
              ZQPRETOT(JL)     = ZQPRETOT(JL)+ZQXFG(JL,JM)
              ENDIF
              !-------------------------------------------------
              ! sink to next layer, constant fall speed
              !-------------------------------------------------
              ! if aerosol effect then override
              !  note that for T>233K this is the same as above.
              IF (LAERICESED .AND. JM == NCLDQI) THEN
              ZRE_ICE=PRE_ICE(JL,JK)
              ! The exponent value is from
              ! Morrison et al. JAS 2005 Appendix
              ZVQX(NCLDQI) = 0.002*ZRE_ICE**1.0
              ENDIF
              ZFALL=ZVQX(JM)*ZRHO(JL)
              !-------------------------------------------------
              ! modified by Heymsfield and Iaquinta JAS 2000
              !-------------------------------------------------
              ! ZFALL = ZFALL*((PAP(JL,JK)*RICEHI1)**(-0.178)) &
              !            &*((ZTP1(JL,JK)*RICEHI2)**(-0.394))

              ZFALLSINK(JL,JM)=ZDTGDP(JL)*ZFALL
              ! Cloud budget diagnostic stored at end as implicit
          ENDDO ! jl
          ENDIF ! LLFALL
      ENDDO ! jm
    ENDDO

END SUBROUTINE sedimentation_falling_all_routine
