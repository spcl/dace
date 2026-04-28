! Quantum ESPRESSO ``addusxx_g`` -- accumulating scatter inner loop.
!
! Translated from the Python @dace.program in
! ``npbench/quantum_espresso/usxx.py`` (final scatter):
!     indices = nl[offset:offset+blocksize] - 1
!     for i in range(blocksize):
!         rhoc_out[indices[i]] += aux2[i]
!
! Pattern: noncontiguous accumulating-scatter into a complex array
! via a 1-based index map.  Exercises the bridge's
! ``hlfir-expand-region-assign`` plus complex(8) arithmetic.
!
! Note: Fortran does NOT have ``+=`` on a vector subscript, so we
! write the equivalent ``do`` form directly.  This avoids triggering
! the (currently xfailed) aliased self-assignment path even when
! ``rhoc_out`` happens to be the same buffer as a partial sum.
subroutine kernel(blocksize, nrxxs, nl, aux2, rhoc_out)
  implicit none
  integer,    intent(in)    :: blocksize, nrxxs
  integer,    intent(in)    :: nl(blocksize)         ! 1-based indices
  complex(8), intent(in)    :: aux2(blocksize)
  complex(8), intent(inout) :: rhoc_out(nrxxs)
  integer :: i
  do i = 1, blocksize
    rhoc_out(nl(i)) = rhoc_out(nl(i)) + aux2(i)
  end do
end subroutine kernel
