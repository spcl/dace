! Scalar struct with four uniformly-shaped 2-D array members — the minimal
! analogue of the ICON velocity-tendencies state struct.  After the
! ``hlfir-flatten-structs`` pass, the function signature must replace the
! struct dummy with four individual 2-D array arguments, and the function
! must be renamed to ``<orig>_soa``.
subroutine velocity_demo(st)
  implicit none
  integer, parameter :: nx = 4, ny = 4
  type :: state_t
    real(8) :: u(nx, ny)
    real(8) :: v(nx, ny)
    real(8) :: w(nx, ny)
    real(8) :: p(nx, ny)
  end type state_t
  type(state_t), intent(inout) :: st
  integer :: i, j
  do j = 1, ny
    do i = 1, nx
      st%u(i, j) = st%u(i, j) + st%v(i, j)
      st%p(i, j) = st%w(i, j) * 2.0d0
    end do
  end do
end subroutine velocity_demo
