! E1 -- MatrixAdd (translated from
! ``SC26-Layout-AD/Experiments/E1_MatrixAdd/bench_cpu.cpp``).
!
! Hot kernel:  C(i, j) = C(i, j) + A(i, j) + B(i, j)
!
! In Fortran column-major terms this is just elementwise array
! addition with accumulation.  The original C++ benchmark mixes
! row-major and col-major B to study layout effects; we drop that
! complication and let the bridge see plain Fortran semantics.
subroutine kernel(m, n, a, b, c)
  implicit none
  integer, intent(in)    :: m, n
  real(8), intent(in)    :: a(m, n), b(m, n)
  real(8), intent(inout) :: c(m, n)
  integer :: i, j
  do j = 1, n
    do i = 1, m
      c(i, j) = c(i, j) + a(i, j) + b(i, j)
    end do
  end do
end subroutine kernel
