SUBROUTINE ice_supersaturation_adjustment(kidia, kfdia, klon, ztp1, za, zqx_ncldqv, zqsice, zcorqsice, zfokoop, zsolqa, zsolac, zqxfg, rtt, ramin, rthomo, nssopt, rkooptau, ptsphy, zepsec, ncldql, ncldqi, ncldqv, nclv)
  IMPLICIT NONE
  INTEGER(KIND = 4), VALUE :: kidia, kfdia, klon
  INTEGER(KIND = 4), VALUE :: nclv
  INTEGER(KIND = 4), VALUE :: ncldql, ncldqi, ncldqv
  INTEGER(KIND = 4), VALUE :: nssopt
  REAL(KIND = 8), INTENT(IN) :: ztp1(klon)
  REAL(KIND = 8), INTENT(IN) :: za(klon)
  REAL(KIND = 8), INTENT(IN) :: zqx_ncldqv(klon)
  REAL(KIND = 8), INTENT(IN) :: zqsice(klon)
  REAL(KIND = 8), INTENT(IN) :: zcorqsice(klon)
  REAL(KIND = 8), INTENT(IN) :: zfokoop(klon)
  REAL(KIND = 8), VALUE :: rtt, ramin, rthomo, rkooptau, ptsphy, zepsec
  REAL(KIND = 8), INTENT(INOUT) :: zsolqa(klon, nclv, nclv)
  REAL(KIND = 8), INTENT(INOUT) :: zsolac(klon)
  REAL(KIND = 8), INTENT(INOUT) :: zqxfg(klon, nclv)
  INTEGER(KIND = 4) :: jl
  REAL(KIND = 8) :: zfac, zfaci
  REAL(KIND = 8) :: zsupsat, zqp1env
  REAL(KIND = 8) :: zepsilon
  zepsilon = 1D-14
  DO jl = 1, klon
    IF (ztp1(jl) >= rtt .OR. nssopt == 0) THEN
      zfac = 1.0D0
      zfaci = 1.0D0
    ELSE
      zfac = za(jl) + zfokoop(jl) * (1.0D0 - za(jl))
      zfaci = ptsphy / rkooptau
    END IF
    IF (za(jl) > 1.0D0 - ramin) THEN
      zsupsat = MAX((zqx_ncldqv(jl) - zfac * zqsice(jl)) / zcorqsice(jl), 0.0D0)
    ELSE
      zqp1env = (zqx_ncldqv(jl) - za(jl) * zqsice(jl)) / MAX(1.0D0 - za(jl), zepsilon)
      zsupsat = MAX((1.0D0 - za(jl)) * (zqp1env - zfac * zqsice(jl)) / zcorqsice(jl), 0.0D0)
    END IF
    IF (zsupsat > zepsec) THEN
      IF (ztp1(jl) > rthomo) THEN
        zsolqa(jl, ncldql, ncldqv) = zsolqa(jl, ncldql, ncldqv) + zsupsat
        zsolqa(jl, ncldqv, ncldql) = zsolqa(jl, ncldqv, ncldql) - zsupsat
        zqxfg(jl, ncldql) = zqxfg(jl, ncldql) + zsupsat
      ELSE
        zsolqa(jl, ncldqi, ncldqv) = zsolqa(jl, ncldqi, ncldqv) + zsupsat
        zsolqa(jl, ncldqv, ncldqi) = zsolqa(jl, ncldqv, ncldqi) - zsupsat
        zqxfg(jl, ncldqi) = zqxfg(jl, ncldqi) + zsupsat
      END IF
      zsolac(jl) = (1.0D0 - za(jl)) * zfaci
    END IF
  END DO
END SUBROUTINE ice_supersaturation_adjustment
