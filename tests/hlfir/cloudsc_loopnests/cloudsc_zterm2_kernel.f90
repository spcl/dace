SUBROUTINE zterm2_kernel(n, zpr02, zcorrfac, zrho, zcorrfac2, &
                         rcl_const3s, rcl_const4s, rcl_const5s, rcl_const6s, &
                         zterm2)
  ! Exact extract of cloudsc.F90 line 3304 (4.5b SNOW evap, IEVAPSNOW=1):
  !
  !   ZTERM2 = 0.65*RCL_CONST6S*ZPR02**RCL_CONST4S+RCL_CONST3S*ZCORRFAC**0.5 &
  !          *ZRHO(JL)**0.5*ZPR02**RCL_CONST5S/ZCORRFAC2**0.5
  !
  ! In cloudsc.F90 every literal here is BARE DEFAULT-REAL (no _JPRB
  ! suffix): ``0.65``, ``0.5``.  This kernel preserves that exact form
  ! so we can compare bridge-vs-gfortran on the unsuffixed-literal
  ! compound expression -- which is what the bottom_upper xfail
  ! actually executes.
  IMPLICIT NONE
  INTEGER(KIND = 4), VALUE :: n
  REAL(KIND = 8), INTENT(IN)  :: zpr02(n), zcorrfac(n), zrho(n), zcorrfac2(n)
  REAL(KIND = 8), VALUE       :: rcl_const3s, rcl_const4s, rcl_const5s, rcl_const6s
  REAL(KIND = 8), INTENT(OUT) :: zterm2(n)
  INTEGER(KIND = 4) :: i
  DO i = 1, n
    zterm2(i) = 0.65*rcl_const6s*zpr02(i)**rcl_const4s + rcl_const3s*zcorrfac(i)**0.5 &
              & *zrho(i)**0.5*zpr02(i)**rcl_const5s/zcorrfac2(i)**0.5
  END DO
END SUBROUTINE zterm2_kernel

SUBROUTINE zbeta_kernel(n, zqsliq, ztp1, zesatliq, zcorr2, zevap_denom, &
                        zlambda, zrho, zfallcorr, &
                        rcl_const1r, rcl_const2r, rcl_const3r, rcl_const4r, &
                        zbeta)
  ! Exact extract of cloudsc.F90 lines 3158-3161 (4.5a RAIN evap,
  ! IEVAPRAIN=2, Abel-Boutle):
  !
  !   ZBETA = (0.5_JPRB/ZQSLIQ(JL,JK))*ZTP1(JL,JK)**2*ZESATLIQ* &
  !         RCL_CONST1R*(ZCORR2/ZEVAP_DENOM)*(0.78_JPRB/(ZLAMBDA**RCL_CONST4R)+ &
  !         RCL_CONST2R*(ZRHO(JL)*ZFALLCORR)**0.5_JPRB/ &
  !         (ZCORR2**0.5_JPRB*ZLAMBDA**RCL_CONST3R))
  !
  ! All literals here ARE properly _JPRB-suffixed (this is Abel-Boutle,
  ! the modern branch).  Tests the deeply-nested mixed-power expression.
  IMPLICIT NONE
  INTEGER(KIND = 4), VALUE :: n
  REAL(KIND = 8), INTENT(IN)  :: zqsliq(n), ztp1(n), zesatliq(n), zcorr2(n), zevap_denom(n)
  REAL(KIND = 8), INTENT(IN)  :: zlambda(n), zrho(n), zfallcorr(n)
  REAL(KIND = 8), VALUE       :: rcl_const1r, rcl_const2r, rcl_const3r, rcl_const4r
  REAL(KIND = 8), INTENT(OUT) :: zbeta(n)
  INTEGER(KIND = 4) :: i
  DO i = 1, n
    zbeta(i) = (0.5_8/zqsliq(i))*ztp1(i)**2*zesatliq(i)* &
             & rcl_const1r*(zcorr2(i)/zevap_denom(i))*(0.78_8/(zlambda(i)**rcl_const4r)+ &
             & rcl_const2r*(zrho(i)*zfallcorr(i))**0.5_8/ &
             & (zcorr2(i)**0.5_8*zlambda(i)**rcl_const3r))
  END DO
END SUBROUTINE zbeta_kernel

SUBROUTINE zaplusb_kernel(n, zvpice, ztp1, pap, rcl_apb1, rcl_apb2, rcl_apb3, zaplusb)
  ! Exact extract of cloudsc.F90 line 3295 (4.5b SNOW evap):
  !
  !   ZAPLUSB = RCL_APB1*ZVPICE-RCL_APB2*ZVPICE*ZTP1(JL,JK)+ &
  !           PAP(JL,JK)*RCL_APB3*ZTP1(JL,JK)**3
  !
  ! ``ZTP1**3`` is integer-exponent.  Otherwise simple FMA-chain
  ! arithmetic with three runtime constants.
  IMPLICIT NONE
  INTEGER(KIND = 4), VALUE :: n
  REAL(KIND = 8), INTENT(IN)  :: zvpice(n), ztp1(n), pap(n)
  REAL(KIND = 8), VALUE       :: rcl_apb1, rcl_apb2, rcl_apb3
  REAL(KIND = 8), INTENT(OUT) :: zaplusb(n)
  INTEGER(KIND = 4) :: i
  DO i = 1, n
    zaplusb(i) = rcl_apb1*zvpice(i) - rcl_apb2*zvpice(i)*ztp1(i) + pap(i)*rcl_apb3*ztp1(i)**3
  END DO
END SUBROUTINE zaplusb_kernel
