! IF / ELSE branching on an element-wise condition.  The ``if`` body and
! ``else`` body each write b(i), and an IF with no else clause writes c(i).
! After the frontend implements ``_emit_cond`` the two branches should be
! emitted as separate states joined by interstate edges on ``a(i) > 0`` and
! ``not (a(i) > 0)``.
subroutine if_else_branch(a, b, c, n)
  implicit none
  integer, intent(in)    :: n
  real(8), intent(in)    :: a(n)
  real(8), intent(inout) :: b(n)
  real(8), intent(inout) :: c(n)
  integer :: i
  do i = 1, n
    if (a(i) > 0.0d0) then
      b(i) = a(i) * 2.0d0
    else
      b(i) = -a(i)
    end if
    if (a(i) > 1.0d0) then
      c(i) = 100.0d0
    end if
  end do
end subroutine if_else_branch
