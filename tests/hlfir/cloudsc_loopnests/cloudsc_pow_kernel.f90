SUBROUTINE pow_kernel(n, x, exponent_val, y)
  ! Minimal reproducer for the `rho**0.78` pattern in CLOUDSC's 4.5a
  ! RAIN evaporation (Abel-Boutle 2012).  Tests whether the bridge's
  ! lowering of `x ** non_integer_exponent` produces identical results
  ! to gfortran -- specifically whether libm's pow() is called from
  ! the same code path in both runs.
  !
  ! In CLOUDSC: `ZCORRFAC = (RDENSREF / ZRHO(JL)) ** 0.4_JPRB` etc.
  ! Non-integer exponents trigger libm pow().  If the bridge lowers
  ! this differently than gfortran (different libm entry / different
  ! intermediate representation), the ulp-level drift in 4.5a RAIN
  ! evap is explained.
  IMPLICIT NONE
  INTEGER(KIND = 4), VALUE :: n
  REAL(KIND = 8), INTENT(IN)  :: x(n)
  REAL(KIND = 8), VALUE       :: exponent_val
  REAL(KIND = 8), INTENT(OUT) :: y(n)
  INTEGER(KIND = 4) :: i
  DO i = 1, n
    y(i) = x(i) ** exponent_val
  END DO
END SUBROUTINE pow_kernel
