import numpy as np

from dace.frontend.fortran.fortran_parser import create_singular_sdfg_from_string
from tests.fortran.fortran_test_helper import SourceCodeBuilder


def test_fortran_frontend_view_reshape():
    """
    Tests that the Fortran frontend can parse complex initializations.
    """
    sources, main = SourceCodeBuilder().add_file("""
module lib1
  implicit none
  real :: outside_init = 1
end module lib1

module lib2
contains
  subroutine view_reshape_test_function(dd)
    use lib1, only: outside_init
    double precision dd(16)
    real:: bob = epsilon(1.0)

    dd(2) = 5.5 + bob

  end subroutine view_reshape_test_function
end module lib2

subroutine main(d)
  use lib2, only: view_reshape_test_function
  implicit none
  integer :: i
  integer :: j
  double precision d(4,4,2)

  i=2
  j=1
  call view_reshape_test_function(d(:,:,1))
end subroutine main
""").check_with_gfortran().get()
    sdfg = create_singular_sdfg_from_string(sources, 'main')
    sdfg.simplify(verbose=True)
    a = np.full([4, 4, 2], 42, order="F", dtype=np.float64)
    sdfg(d=a, outside_init=0)
    assert (a[0, 0, 0] == 42)
    assert (a[1, 0, 0] == 5.5)
    assert (a[2, 0, 0] == 42)


if __name__ == "__main__":

    test_fortran_frontend_view_reshape()
