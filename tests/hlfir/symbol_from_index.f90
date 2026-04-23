! Local scalar ``ix`` is used exclusively as an array index (``a(ix)``),
! not in scalar arithmetic.  The HLFIR frontend should classify it as a
! symbol so an assignment to it (``ix = 1``) triggers a state change,
! which is how DaCe keeps array-index values live across states.
subroutine idx_sym(a, b, n)
  implicit none
  integer, intent(in)    :: n
  real(8), intent(in)    :: a(n)
  real(8), intent(inout) :: b(n)
  integer :: ix, i
  ix = 1
  do i = 1, n
    b(i) = a(ix)
    ix = ix + 1
    if (ix > n) ix = 1
  end do
end subroutine idx_sym
