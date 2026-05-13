SUBROUTINE autoconversion_snow(kidia, kfdia, klon, nclv, ztp1, zicecld, pnice, zsolqb, zsnowaut, rtt, rlcritsnow, rsnowlin1, rsnowlin2, rnice, ptsphy, zepsec, laericeauto, ncldqs, ncldqi)
  IMPLICIT NONE
  INTEGER(KIND = 4), VALUE :: kidia, kfdia, klon, nclv
  INTEGER(KIND = 4), VALUE :: ncldqs, ncldqi
  REAL(KIND = 8), INTENT(IN) :: ztp1(klon)
  REAL(KIND = 8), INTENT(IN) :: zicecld(klon)
  REAL(KIND = 8), INTENT(IN) :: pnice(klon)
  REAL(KIND = 8), VALUE :: rtt
  REAL(KIND = 8), VALUE :: rlcritsnow
  REAL(KIND = 8), VALUE :: rsnowlin1
  REAL(KIND = 8), VALUE :: rsnowlin2
  REAL(KIND = 8), VALUE :: rnice
  REAL(KIND = 8), VALUE :: ptsphy
  REAL(KIND = 8), VALUE :: zepsec
  INTEGER(KIND=4), VALUE :: laericeauto
  REAL(KIND = 8), INTENT(OUT) :: zsnowaut(klon)
  REAL(KIND = 8), INTENT(INOUT) :: zsolqb(klon, nclv, nclv)
  INTEGER(KIND = 4) :: jl
  REAL(KIND = 8) :: zzco, zlcrit
  zsnowaut(:) = 0.0D0
  DO jl = 1, klon
    IF (ztp1(jl) <= rtt) THEN
      IF (zicecld(jl) > zepsec) THEN
        zzco = ptsphy * rsnowlin1 * EXP(rsnowlin2 * (ztp1(jl) - rtt))
        IF (laericeauto /= 0) THEN
          zlcrit = rlcritsnow
          zzco = zzco * (rnice / pnice(jl)) ** 0.333D0
        ELSE
          zlcrit = rlcritsnow
        END IF
        zsnowaut(jl) = zzco * (1.0D0 - EXP(- (zicecld(jl) / zlcrit) ** 2))
        zsolqb(jl, ncldqs, ncldqi) = zsolqb(jl, ncldqs, ncldqi) + zsnowaut(jl)
      END IF
    END IF
  END DO
END SUBROUTINE autoconversion_snow
