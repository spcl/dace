! E2 -- Conjugation (translated from
! ``SC26-Layout-AD/Experiments/E2_Conjugation/conjugate_inplace.cpp``).
!
! Hot kernel:  b(i) = conjg(b(i))   for an array of complex(8).
!
! In the original C++ this is an in-place imaginary-part negation
! across an interleaved (re, im) buffer; the Fortran equivalent is
! ``b(:) = conjg(b(:))`` which lowers to the same elementwise op
! through the bridge's complex arithmetic path.
subroutine kernel(n, b)
  implicit none
  integer,    intent(in)    :: n
  complex(8), intent(inout) :: b(n)
  integer :: i
  do i = 1, n
    b(i) = conjg(b(i))
  end do
end subroutine kernel
