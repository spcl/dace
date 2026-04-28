! Quantum ESPRESSO ``addusxx_g`` -- complex phase factor computation.
!
! Translated from:
!     for na in range(nat):
!         arg = 2 * np.pi * np.sum((xk - xkq) * tau[na, :])
!         eigqts[na] = np.cos(arg) - 1j * np.sin(arg)
!
! Pattern: per-atom complex(8) phase factor
! ``cos(arg) - i*sin(arg)``, with ``arg`` a 3-component dot product.
! Exercises real -> complex constant construction and SIN/COS on real(8).
subroutine kernel(nat, xk, xkq, tau, eigqts)
  implicit none
  integer,    intent(in)    :: nat
  real(8),    intent(in)    :: xk(3), xkq(3)
  real(8),    intent(in)    :: tau(3, nat)
  complex(8), intent(out)   :: eigqts(nat)
  real(8), parameter :: TWOPI = 6.283185307179586d0
  real(8) :: arg
  integer :: na
  do na = 1, nat
    arg = TWOPI * ((xk(1) - xkq(1)) * tau(1, na) &
                 + (xk(2) - xkq(2)) * tau(2, na) &
                 + (xk(3) - xkq(3)) * tau(3, na))
    eigqts(na) = cmplx(cos(arg), -sin(arg), kind=8)
  end do
end subroutine kernel
