! Elementwise intrinsic applied over a 1-D array.  Flang lowers this to a
! composition of hlfir.elemental ops whose body ends in ``math.sin``
! plus an ``arith.mulf`` / ``arith.addf`` composition.  The HLFIR-to-SDFG
! bridge now recognises that shape and emits a loop + Python tasklet with
! body ``_out = sin(_in_a_0) + (2.0 * _in_a_1)``.
subroutine elemwise_sin(a, b, n)
  implicit none
  integer, intent(in)    :: n
  real(8), intent(in)    :: a(n)
  real(8), intent(inout) :: b(n)
  b = sin(a) + 2.0d0 * a
end subroutine elemwise_sin
