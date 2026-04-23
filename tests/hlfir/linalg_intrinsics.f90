! Three Fortran linalg intrinsics that each lower to a first-class HLFIR op:
!   * matmul(A, B)      - hlfir.matmul      -> blas.MatMul (GEMM shape)
!   * transpose(A)      - hlfir.transpose   -> standard.Transpose
!   * matmul(A, v)      - hlfir.matmul      -> blas.MatMul (GEMV shape)
!   * dot_product(u, u) - hlfir.dot_product -> blas.Dot
subroutine linalg_ops(a, b, c, at, v, w, u, s, n, m, k)
  implicit none
  integer, intent(in)    :: n, m, k
  real(8), intent(in)    :: a(n, m)
  real(8), intent(in)    :: b(m, k)
  real(8), intent(inout) :: c(n, k)
  real(8), intent(inout) :: at(m, n)
  real(8), intent(in)    :: v(m)
  real(8), intent(inout) :: w(n)
  real(8), intent(in)    :: u(n)
  real(8), intent(inout) :: s
  c  = matmul(a, b)
  at = transpose(a)
  w  = matmul(a, v)
  s  = dot_product(u, u)
end subroutine linalg_ops
