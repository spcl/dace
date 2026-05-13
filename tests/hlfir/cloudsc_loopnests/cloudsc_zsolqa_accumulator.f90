SUBROUTINE zsolqa_accumulator(n, nclv, kidia, kfdia, ncldqv, ncldqr, ncldqs, &
                              zevap, zsnowsrc, zsolqa)
  ! Minimal reproducer for the CLOUDSC 4.5a/4.5b accumulator pattern:
  !
  !   ZSOLQA(JL, NCLDQV, NCLDQR) = ZSOLQA(JL, NCLDQV, NCLDQR) + ZEVAP(JL)
  !   ZSOLQA(JL, NCLDQR, NCLDQV) = ZSOLQA(JL, NCLDQR, NCLDQV) - ZEVAP(JL)
  !   ZSOLQA(JL, NCLDQV, NCLDQS) = ZSOLQA(JL, NCLDQV, NCLDQS) + ZSNOWSRC(JL)
  !   ZSOLQA(JL, NCLDQS, NCLDQV) = ZSOLQA(JL, NCLDQS, NCLDQV) - ZSNOWSRC(JL)
  !
  ! Hypothesis: when the bridge lowers ``X = X + Y`` as a Python
  ! ``X += Y`` tasklet, it gets a WCR (write-conflict-resolution =
  ! atomic-add) edge instead of an explicit read-modify-write.  WCR on
  ! a scalar element may reorder accumulations vs gfortran's strict
  ! left-to-right, producing ulp-level drift -- which compounds across
  ! the full vertical column and triggers the cloudsc_full 26/548
  ! PCOVPTOT mismatch.
  !
  ! If this test fails at rtol=atol=1e-15, the WCR-vs-RMW hypothesis is
  ! confirmed.  The fix would be to emit explicit read+tasklet+write
  ! (no WCR) for any ``A(i,j) = A(i,j) + expr`` pattern with a single
  ! producer.
  IMPLICIT NONE
  INTEGER(KIND = 4), VALUE :: n, nclv, kidia, kfdia
  INTEGER(KIND = 4), VALUE :: ncldqv, ncldqr, ncldqs
  REAL(KIND = 8), INTENT(IN)    :: zevap(n)
  REAL(KIND = 8), INTENT(IN)    :: zsnowsrc(n)
  REAL(KIND = 8), INTENT(INOUT) :: zsolqa(n, nclv, nclv)
  INTEGER(KIND = 4) :: jl
  DO jl = kidia, kfdia
    zsolqa(jl, ncldqv, ncldqr) = zsolqa(jl, ncldqv, ncldqr) + zevap(jl)
    zsolqa(jl, ncldqr, ncldqv) = zsolqa(jl, ncldqr, ncldqv) - zevap(jl)
    zsolqa(jl, ncldqv, ncldqs) = zsolqa(jl, ncldqv, ncldqs) + zsnowsrc(jl)
    zsolqa(jl, ncldqs, ncldqv) = zsolqa(jl, ncldqs, ncldqv) - zsnowsrc(jl)
  END DO
END SUBROUTINE zsolqa_accumulator
