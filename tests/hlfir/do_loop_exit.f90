! Fortran EXIT inside a counted DO.  After lift-cf-to-scf Flang's
! cf.cond_br exit edge folds into the scf.while's scf.condition, so the
! bridge should see a ``kind="while"`` AST node whose condition encodes
! both the iteration bound and the early-exit predicate.
subroutine do_exit(a, b, n)
  implicit none
  integer, intent(in)    :: n
  real(8), intent(in)    :: a(n)
  real(8), intent(inout) :: b(n)
  integer :: i
  do i = 1, n
    if (a(i) > 100.0d0) exit
    b(i) = a(i) * 2.0d0
  end do
end subroutine do_exit
