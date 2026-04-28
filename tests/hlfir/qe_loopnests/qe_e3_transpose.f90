! E3 -- Transpose (translated from
! ``SC26-Layout-AD/Experiments/E3_Transpose/run_transpose.py``'s
! GPU sweep).
!
! Hot kernel:  B(j, i) = A(i, j)   for square real(8) matrices.
!
! The original benchmark sweeps GPU kernel variants (smem, swizzle,
! padding, vectorisation); we drop all that and ship the canonical
! Fortran loop pair so the bridge sees the plain dependence pattern.
subroutine kernel(n, a, b)
  implicit none
  integer, intent(in)  :: n
  real(8), intent(in)  :: a(n, n)
  real(8), intent(out) :: b(n, n)
  integer :: i, j
  do j = 1, n
    do i = 1, n
      b(j, i) = a(i, j)
    end do
  end do
end subroutine kernel
