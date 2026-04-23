! Two hlfir.assign shapes that should go straight to library nodes:
!   * whole-array copy (``b = a``)   - lowers to hlfir.assign on two boxes
!   * scalar-zero fill (``c = 0.0``) - lowers to hlfir.assign with a scalar src
subroutine copy_and_memset(a, b, c, n)
  implicit none
  integer, intent(in)    :: n
  real(8), intent(in)    :: a(n)
  real(8), intent(inout) :: b(n)
  real(8), intent(inout) :: c(n)
  b = a
  c = 0.0d0
end subroutine copy_and_memset
