! Scalar struct with four 1-D array members of DIFFERENT extents.  The
! ``hlfir-flatten-structs`` pass should pack these into a single ELLPACK-
! style 2-D companion of shape ``[4 x max(10,20,15,5)] = [4 x 20]`` and
! rename the function to ``<orig>_soa``.  Member accesses rewrite into
! row-sliced views of the combined array.
subroutine jagged_demo(st)
  implicit none
  type :: jagged_t
    real(8) :: a(10)
    real(8) :: b(20)
    real(8) :: c(15)
    real(8) :: d(5)
  end type jagged_t
  type(jagged_t), intent(inout) :: st
  integer :: i
  do i = 1, 5
    st%a(i) = real(i,     8)
    st%b(i) = real(2 * i, 8)
    st%c(i) = real(3 * i, 8)
    st%d(i) = real(4 * i, 8)
  end do
end subroutine jagged_demo
