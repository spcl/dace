! Local 1-D array of a user-defined derived type with two scalar real(8)
! members.  The ``hlfir-flatten-structs`` pass should split ``z`` into two
! flat f64 arrays (``_QFcomplex_struct_demoEz_re`` / ``..._im``) and drop
! the original struct-backed declare.
subroutine complex_struct_demo(out_re, out_im)
  implicit none
  integer, parameter :: n = 8
  real(8), intent(out)   :: out_re(n)
  real(8), intent(out)   :: out_im(n)
  type :: complex_t
    real(8) :: re
    real(8) :: im
  end type complex_t
  type(complex_t) :: z(n)
  integer :: i
  do i = 1, n
    z(i)%re =  real(i, 8)
    z(i)%im = -real(i, 8)
    out_re(i) = z(i)%re
    out_im(i) = z(i)%im
  end do
end subroutine complex_struct_demo
