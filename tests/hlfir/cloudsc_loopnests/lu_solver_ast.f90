SUBROUTINE lu_solver_microphysics(kidia, kfdia, klon, nclv, zqlhs, zqxn)
  IMPLICIT NONE
  INTEGER(KIND = 4), VALUE :: kidia, kfdia, klon, nclv
  REAL(KIND = 8), INTENT(INOUT) :: zqlhs(klon, nclv, nclv)
  REAL(KIND = 8), INTENT(INOUT) :: zqxn(klon, nclv)
  INTEGER(KIND = 4) :: jl, jn, jm, ik
  DO jn = 1, nclv - 1
    DO jm = jn + 1, nclv
      DO jl = 1, klon
        zqlhs(jl, jm, jn) = zqlhs(jl, jm, jn) / zqlhs(jl, jn, jn)
      END DO
      DO ik = jn + 1, nclv
        DO jl = 1, klon
          zqlhs(jl, jm, ik) = zqlhs(jl, jm, ik) - (zqlhs(jl, jm, jn) * zqlhs(jl, jn, ik))
        END DO
      END DO
    END DO
  END DO
  DO jn = 2, nclv
    DO jm = 1, jn - 1
      DO jl = 1, klon
        zqxn(jl, jn) = zqxn(jl, jn) - (zqlhs(jl, jn, jm) * zqxn(jl, jm))
      END DO
    END DO
  END DO
  DO jl = 1, klon
    zqxn(jl, nclv) = zqxn(jl, nclv) / zqlhs(jl, nclv, nclv)
  END DO
  DO jn = nclv - 1, 1, - 1
    DO jm = jn + 1, nclv
      DO jl = 1, klon
        zqxn(jl, jn) = zqxn(jl, jn) - (zqlhs(jl, jn, jm) * zqxn(jl, jm))
      END DO
    END DO
    DO jl = 1, klon
      zqxn(jl, jn) = zqxn(jl, jn) / zqlhs(jl, jn, jn)
    END DO
  END DO
END SUBROUTINE lu_solver_microphysics
