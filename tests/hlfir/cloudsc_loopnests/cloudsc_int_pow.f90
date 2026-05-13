SUBROUTINE int_pow_kernel(n, x, y2, y3)
  ! Tests integer-exponent power lowering: y2 = x**2, y3 = x**3.  Both
  ! gfortran and the bridge can lower these as either repeated multiply
  ! (x*x, (x*x)*x) or as libm pow(x, 2.0) / ipow().  Cloudsc 4.5b
  ! contains ``ZTP1**2`` and ``ZTP1**3`` and ``ZTERM1 = ... * ZTP1**2 * ...``
  ! which feeds into the divergent precip chain.
  !
  ! If repeated-multiply vs pow() differ in this combination -- or worse,
  ! if one backend uses pow() with INTEGER exponent (which is undefined
  ! per C standard for pow(double, int)) -- we get the kind of large
  ! divergence the cloudsc_full failure shows (max abs 0.69, far worse
  ! than 1-ulp drift).
  IMPLICIT NONE
  INTEGER(KIND = 4), VALUE :: n
  REAL(KIND = 8), INTENT(IN)  :: x(n)
  REAL(KIND = 8), INTENT(OUT) :: y2(n)
  REAL(KIND = 8), INTENT(OUT) :: y3(n)
  INTEGER(KIND = 4) :: i
  DO i = 1, n
    y2(i) = x(i) ** 2
    y3(i) = x(i) ** 3
  END DO
END SUBROUTINE int_pow_kernel
