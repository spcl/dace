! E4 -- GAS / zaxpy (translated from
! ``SC26-Layout-AD/Experiments/E4_GAS/zaxpy.cpp``).
!
! Hot kernel:  Y(i) = a * X(i) + Y(i)   for complex(8) X, Y and
!              complex(8) scalar a.
!
! Same shape as the BLAS ``zaxpy``.  The original C++ sweeps GPU
! versions and tests grid-aware-storage layouts; we drop those and
! ship the elementwise loop.
subroutine kernel(n, a, x, y)
  implicit none
  integer,    intent(in)    :: n
  ! ``a`` is a length-1 array rather than a plain ``complex(8)``
  ! scalar to dodge a DaCe-core gap: ``ctypes`` on Python 3.12 has
  ! no ``c_double_complex``, so DaCe's ``as_ctypes`` mapping for
  ! ``complex128`` truncates the imaginary part on by-value passes.
  ! Length-1 arrays go through the pointer ABI which is bit-identical.
  complex(8), intent(in)    :: a(1)
  complex(8), intent(in)    :: x(n)
  complex(8), intent(inout) :: y(n)
  integer :: i
  do i = 1, n
    y(i) = a(1) * x(i) + y(i)
  end do
end subroutine kernel
