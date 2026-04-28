SUBROUTINE compute_saturation_values(kidia, kfdia, klon, klev, ztp1, pap, zfoealfa, zfoeewmt, zqsmix, zfoeew, zqsice, zfoeeliqt, zqsliq, rtt, retv, r2es, r3les, r3ies, r4les, r4ies, rtice, rtwat, rtwat_rtice_r)
  IMPLICIT NONE
  INTEGER(KIND = 4), VALUE :: kidia
  INTEGER(KIND = 4), VALUE :: kfdia
  INTEGER(KIND = 4), VALUE :: klon
  INTEGER(KIND = 4), VALUE :: klev
  REAL(KIND = 8), INTENT(IN) :: ztp1(klon, klev)
  REAL(KIND = 8), INTENT(IN) :: pap(klon, klev)
  REAL(KIND = 8), INTENT(OUT) :: zfoealfa(klon, klev)
  REAL(KIND = 8), INTENT(OUT) :: zfoeewmt(klon, klev)
  REAL(KIND = 8), INTENT(OUT) :: zqsmix(klon, klev)
  REAL(KIND = 8), INTENT(OUT) :: zfoeew(klon, klev)
  REAL(KIND = 8), INTENT(OUT) :: zqsice(klon, klev)
  REAL(KIND = 8), INTENT(OUT) :: zfoeeliqt(klon, klev)
  REAL(KIND = 8), INTENT(OUT) :: zqsliq(klon, klev)
  REAL(KIND = 8), VALUE :: rtt
  REAL(KIND = 8), VALUE :: retv
  REAL(KIND = 8), VALUE :: r2es
  REAL(KIND = 8), VALUE :: r3les
  REAL(KIND = 8), VALUE :: r3ies
  REAL(KIND = 8), VALUE :: r4les
  REAL(KIND = 8), VALUE :: r4ies
  REAL(KIND = 8), VALUE :: rtice
  REAL(KIND = 8), VALUE :: rtwat
  REAL(KIND = 8), VALUE :: rtwat_rtice_r
  INTEGER(KIND = 4) :: jk, jl
  REAL(KIND = 8) :: ptare
  REAL(KIND = 8) :: zdelta
  REAL(KIND = 8) :: zfoealfa_loc
  REAL(KIND = 8) :: zfoeewm_loc
  REAL(KIND = 8) :: zfoeeliq_loc
  REAL(KIND = 8) :: zfoeeice_loc
  DO jk = 1, klev
    DO jl = 1, klon
      ptare = ztp1(jl, jk)
      zfoealfa_loc = ((MAX(rtice, MIN(rtwat, ptare)) - rtice) * rtwat_rtice_r) ** 2
      zfoealfa_loc = MIN(1.0D0, zfoealfa_loc)
      zfoealfa(jl, jk) = zfoealfa_loc
      zfoeeliq_loc = r2es * EXP(r3les * (ptare - rtt) / (ptare - r4les))
      zfoeeice_loc = r2es * EXP(r3ies * (ptare - rtt) / (ptare - r4ies))
      zfoeewm_loc = r2es * (zfoealfa_loc * EXP(r3les * (ptare - rtt) / (ptare - r4les)) + (1.0D0 - zfoealfa_loc) * EXP(r3ies * (ptare - rtt) / (ptare - r4ies)))
      zfoeewmt(jl, jk) = MIN(zfoeewm_loc / pap(jl, jk), 0.5D0)
      zqsmix(jl, jk) = zfoeewmt(jl, jk)
      zqsmix(jl, jk) = zqsmix(jl, jk) / (1.0D0 - retv * zqsmix(jl, jk))
      zdelta = MAX(0.0D0, SIGN(1.0D0, ptare - rtt))
      zfoeew(jl, jk) = (zdelta * zfoeeliq_loc + (1.0D0 - zdelta) * zfoeeice_loc) / pap(jl, jk)
      zfoeew(jl, jk) = MIN(0.5D0, zfoeew(jl, jk))
      zqsice(jl, jk) = zfoeew(jl, jk) / (1.0D0 - retv * zfoeew(jl, jk))
      zfoeeliqt(jl, jk) = MIN(zfoeeliq_loc / pap(jl, jk), 0.5D0)
      zqsliq(jl, jk) = zfoeeliqt(jl, jk)
      zqsliq(jl, jk) = zqsliq(jl, jk) / (1.0D0 - retv * zqsliq(jl, jk))
    END DO
  END DO
END SUBROUTINE compute_saturation_values
