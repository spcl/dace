SUBROUTINE rain_evaporation_abel_boutle(kidia, kfdia, klon, ztp1, zqx_ncldqv, za, zqsliq, zqxfg_ncldqr, zcovptot, zcovpclr, zcovpmax, zrho, pap, zsolqa, zevap_out, rtt, rv, rd, rprecrhmax, rcovpmin, rdensref, ptsphy, zepsec, rcl_fac1, rcl_fac2, rcl_cdenom1, rcl_cdenom2, rcl_cdenom3, rcl_ka273, rcl_const1r, rcl_const2r, rcl_const3r, rcl_const4r, ncldqv, ncldqr, nclv)
  IMPLICIT NONE
  INTEGER(KIND = 4), VALUE :: kidia, kfdia, klon
  INTEGER(KIND = 4), VALUE :: nclv
  INTEGER(KIND = 4), VALUE :: ncldqv, ncldqr
  REAL(KIND = 8), INTENT(IN) :: ztp1(klon)
  REAL(KIND = 8), INTENT(IN) :: zqx_ncldqv(klon)
  REAL(KIND = 8), INTENT(IN) :: za(klon)
  REAL(KIND = 8), INTENT(IN) :: zqsliq(klon)
  REAL(KIND = 8), INTENT(IN) :: zrho(klon)
  REAL(KIND = 8), INTENT(IN) :: pap(klon)
  REAL(KIND = 8), INTENT(INOUT) :: zqxfg_ncldqr(klon)
  REAL(KIND = 8), INTENT(INOUT) :: zcovptot(klon)
  REAL(KIND = 8), INTENT(INOUT) :: zcovpclr(klon)
  REAL(KIND = 8), INTENT(IN) :: zcovpmax(klon)
  REAL(KIND = 8), INTENT(INOUT) :: zsolqa(klon, nclv, nclv)
  REAL(KIND = 8), INTENT(OUT) :: zevap_out(klon)
  REAL(KIND = 8), VALUE :: rtt, rv, rd
  REAL(KIND = 8), VALUE :: rprecrhmax, rcovpmin, rdensref, ptsphy, zepsec
  REAL(KIND = 8), VALUE :: rcl_fac1, rcl_fac2
  REAL(KIND = 8), VALUE :: rcl_cdenom1, rcl_cdenom2, rcl_cdenom3
  REAL(KIND = 8), VALUE :: rcl_ka273
  REAL(KIND = 8), VALUE :: rcl_const1r, rcl_const2r, rcl_const3r, rcl_const4r
  INTEGER(KIND = 4) :: jl
  REAL(KIND = 8) :: zzrh, zqe, zpreclr, zfallcorr, zesatliq
  REAL(KIND = 8) :: zlambda, zevap_denom, zcorr2, zka, zsubsat
  REAL(KIND = 8) :: zbeta, zdenom, zdpevap, zevap
  LOGICAL :: llo1
  REAL(KIND = 8) :: r2es_local, r3les_local, r4les_local
  r2es_local = 611.21D0
  r3les_local = 17.502D0
  r4les_local = 32.19D0
  zevap_out(:) = 0.0D0
  DO jl = 1, klon
    zzrh = rprecrhmax + (1.0D0 - rprecrhmax) * zcovpmax(jl) / MAX(zepsec, 1.0D0 - za(jl))
    zzrh = MIN(MAX(zzrh, rprecrhmax), 1.0D0)
    zzrh = MIN(0.8D0, zzrh)
    zqe = MAX(0.0D0, MIN(zqx_ncldqv(jl), zqsliq(jl)))
    llo1 = (zcovpclr(jl) > zepsec) .AND. (zqxfg_ncldqr(jl) > zepsec) .AND. (zqe < zzrh * zqsliq(jl))
    IF (llo1) THEN
      zpreclr = zqxfg_ncldqr(jl) / zcovptot(jl)
      zfallcorr = (rdensref / zrho(jl)) ** 0.4D0
      zesatliq = rv / rd * 611.21D0 * EXP(17.502D0 * (ztp1(jl) - rtt) / (ztp1(jl) - 32.19D0))
      zlambda = (rcl_fac1 / (zrho(jl) * zpreclr)) ** rcl_fac2
      zevap_denom = rcl_cdenom1 * zesatliq - rcl_cdenom2 * ztp1(jl) * zesatliq + rcl_cdenom3 * ztp1(jl) ** 3 * pap(jl)
      zcorr2 = (ztp1(jl) / 273.0D0) ** 1.5D0 * 393.0D0 / (ztp1(jl) + 120.0D0)
      zka = rcl_ka273 * zcorr2
      zsubsat = MAX(zzrh * zqsliq(jl) - zqe, 0.0D0)
      zbeta = (0.5D0 / zqsliq(jl)) * ztp1(jl) ** 2 * zesatliq * rcl_const1r * (zcorr2 / zevap_denom) * (0.78D0 / (zlambda ** rcl_const4r) + rcl_const2r * (zrho(jl) * zfallcorr) ** 0.5D0 / (zcorr2 ** 0.5D0 * zlambda ** rcl_const3r))
      zdenom = 1.0D0 + zbeta * ptsphy
      zdpevap = zcovpclr(jl) * zbeta * ptsphy * zsubsat / zdenom
      zevap = MIN(zdpevap, zqxfg_ncldqr(jl))
      zevap_out(jl) = zevap
      zsolqa(jl, ncldqv, ncldqr) = zsolqa(jl, ncldqv, ncldqr) + zevap
      zsolqa(jl, ncldqr, ncldqv) = zsolqa(jl, ncldqr, ncldqv) - zevap
      zcovptot(jl) = MAX(rcovpmin, zcovptot(jl) - MAX(0.0D0, (zcovptot(jl) - za(jl)) * zevap / zqxfg_ncldqr(jl)))
      zqxfg_ncldqr(jl) = zqxfg_ncldqr(jl) - zevap
    END IF
  END DO
END SUBROUTINE rain_evaporation_abel_boutle
