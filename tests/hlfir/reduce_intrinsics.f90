! Whole-array scalar reductions (sum / product / minval / maxval).  Each
! call lowers to a dedicated hlfir.<name> op whose source is the input
! array box; the bridge maps them onto DaCe ``standard.Reduce`` library
! nodes via ``state.add_reduce(wcr, axes=None, identity)``.
subroutine reduce_scalar(a, total, prod, lo, hi, n)
  implicit none
  integer, intent(in)    :: n
  real(8), intent(in)    :: a(n)
  real(8), intent(inout) :: total
  real(8), intent(inout) :: prod
  real(8), intent(inout) :: lo
  real(8), intent(inout) :: hi
  total = sum(a)
  prod  = product(a)
  lo    = minval(a)
  hi    = maxval(a)
end subroutine reduce_scalar
