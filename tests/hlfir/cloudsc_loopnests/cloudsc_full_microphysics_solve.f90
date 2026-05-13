SUBROUTINE full_microphysics_solve(kidia, kfdia, klon, klev, nclv, ncldqv, jk_idx, &
                                   zfallsink, zsolqa, zsolqb, zqx, zqlhs, zqxn, zepsec)
  ! Extracted from cloudsc.F90 lines 3494-3592 -- Section 5.2.2 of CLOUDSC:
  !   (a) build LHS matrix ZQLHS from ZFALLSINK + sum(ZSOLQB) on diagonals
  !       and -ZSOLQB off-diagonals
  !   (b) build RHS vector ZQXN from ZQX(:,jk_idx,:) + sum(ZSOLQA along JN)
  !   (c) LU factorization (non-pivoting)
  !   (d) back-substitution (two-step)
  !   (e) clip species with values < zepsec into vapor (NCLDQV)
  !
  ! The bare ``cloudsc_lu_solver`` test (which uses already-factorized
  ! ZQLHS + RHS as input) passes bit-exactly via the bridge.  This
  ! bigger loopnest extends the same scope to the LHS/RHS assembly,
  ! which is the suspect for the 1-9 ulp ZQXN drift observed at
  ! JK=NCLDTOP=15 in the full-CLOUDSC run.  Stresses:
  !   - JM/JN/JO triple-nested loop with branching
  !   - per-JL inner sum (ZEXPLICIT accumulator)
  !   - vector-section assigns (ZQLHS(KIDIA:KFDIA,...))
  !   - LU factor + back-sub combined
  IMPLICIT NONE
  INTEGER(KIND = 4), VALUE :: kidia, kfdia, klon, klev, nclv, ncldqv, jk_idx
  REAL(KIND = 8), INTENT(IN)    :: zfallsink(klon, nclv)
  REAL(KIND = 8), INTENT(IN)    :: zsolqa(klon, nclv, nclv)
  REAL(KIND = 8), INTENT(IN)    :: zsolqb(klon, nclv, nclv)
  REAL(KIND = 8), INTENT(IN)    :: zqx(klon, klev, nclv)
  REAL(KIND = 8), INTENT(INOUT) :: zqlhs(klon, nclv, nclv)
  REAL(KIND = 8), INTENT(OUT)   :: zqxn(klon, nclv)
  REAL(KIND = 8), VALUE         :: zepsec

  INTEGER(KIND = 4) :: jl, jn, jm, jo, ik
  REAL(KIND = 8)    :: zexplicit

  ! (a) build LHS
  DO jm = 1, nclv
    DO jn = 1, nclv
      IF (jn == jm) THEN
        DO jl = kidia, kfdia
          zqlhs(jl, jn, jm) = 1.0_8 + zfallsink(jl, jm)
          DO jo = 1, nclv
            zqlhs(jl, jn, jm) = zqlhs(jl, jn, jm) + zsolqb(jl, jo, jn)
          END DO
        END DO
      ELSE
        DO jl = kidia, kfdia
          zqlhs(jl, jn, jm) = -zsolqb(jl, jn, jm)
        END DO
      END IF
    END DO
  END DO

  ! (b) build RHS = ZQXN
  DO jm = 1, nclv
    DO jl = kidia, kfdia
      zexplicit = 0.0_8
      DO jn = 1, nclv
        zexplicit = zexplicit + zsolqa(jl, jm, jn)
      END DO
      zqxn(jl, jm) = zqx(jl, jk_idx, jm) + zexplicit
    END DO
  END DO

  ! (c) LU factorization (non-pivoting)
  DO jn = 1, nclv - 1
    DO jm = jn + 1, nclv
      zqlhs(kidia:kfdia, jm, jn) = zqlhs(kidia:kfdia, jm, jn) / zqlhs(kidia:kfdia, jn, jn)
      DO ik = jn + 1, nclv
        DO jl = kidia, kfdia
          zqlhs(jl, jm, ik) = zqlhs(jl, jm, ik) - zqlhs(jl, jm, jn) * zqlhs(jl, jn, ik)
        END DO
      END DO
    END DO
  END DO

  ! (d) back-substitution: forward step
  DO jn = 2, nclv
    DO jm = 1, jn - 1
      zqxn(kidia:kfdia, jn) = zqxn(kidia:kfdia, jn) - zqlhs(kidia:kfdia, jn, jm) * zqxn(kidia:kfdia, jm)
    END DO
  END DO
  ! (d) back-substitution: backward step
  zqxn(kidia:kfdia, nclv) = zqxn(kidia:kfdia, nclv) / zqlhs(kidia:kfdia, nclv, nclv)
  DO jn = nclv - 1, 1, -1
    DO jm = jn + 1, nclv
      zqxn(kidia:kfdia, jn) = zqxn(kidia:kfdia, jn) - zqlhs(kidia:kfdia, jn, jm) * zqxn(kidia:kfdia, jm)
    END DO
    zqxn(kidia:kfdia, jn) = zqxn(kidia:kfdia, jn) / zqlhs(kidia:kfdia, jn, jn)
  END DO

  ! (e) clip small values into vapor
  DO jn = 1, nclv - 1
    DO jl = kidia, kfdia
      IF (zqxn(jl, jn) < zepsec) THEN
        zqxn(jl, ncldqv) = zqxn(jl, ncldqv) + zqxn(jl, jn)
        zqxn(jl, jn) = 0.0_8
      END IF
    END DO
  END DO
END SUBROUTINE full_microphysics_solve
