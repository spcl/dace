import numpy as np

from dace.frontend.fortran.fortran_parser import create_singular_sdfg_from_string
from tests.fortran.fortran_test_helper import SourceCodeBuilder


def test_fortran_frontend_init():
    """
    Tests that the Fortran frontend can parse complex initializations.
    """
    sources, main = SourceCodeBuilder().add_file("""
module lib1
  implicit none
  real :: outside_init = epsilon(1.0)
end module lib1
""").add_file("""
module lib2
contains
  subroutine init_test_function(d)
    use lib1, only: outside_init
    double precision d(4)
    real:: bob = epsilon(1.0)
    d(2) = 5.5 + bob + outside_init
  end subroutine init_test_function
end module lib2
""").add_file("""
subroutine main(d)
  use lib2, only: init_test_function
  implicit none
  double precision d(4)
  call init_test_function(d)
end subroutine main
""").check_with_gfortran().get()
    sdfg = create_singular_sdfg_from_string(sources, 'main')
    sdfg.simplify(verbose=True)
    a = np.full([4], 42, order="F", dtype=np.float64)
    sdfg(d=a, outside_init=0)
    assert (a[0] == 42)
    assert (a[1] == 5.5)
    assert (a[2] == 42)


def test_fortran_frontend_init2():
    """
    Tests that the Fortran frontend can parse complex initializations.
    """
    sources, main = SourceCodeBuilder().add_file("""
module lib1
  implicit none
  real, parameter :: TORUS_MAX_LAT = 4.0/18.0*atan(1.0)
end module lib1
""").add_file("""
module lib2
contains
  subroutine init2_test_function(d)
    use lib1, only: TORUS_MAX_LAT
    double precision d(4)
    d(2) = 5.5 + TORUS_MAX_LAT
  end subroutine init2_test_function
end module lib2
""").add_file("""
subroutine main(d)
  use lib2, only: init2_test_function
  implicit none
  double precision d(4)
  call init2_test_function(d)
end subroutine main
""").check_with_gfortran().get()
    sdfg = create_singular_sdfg_from_string(sources, 'main')
    sdfg.simplify(verbose=True)
    a = np.full([4], 42, order="F", dtype=np.float64)
    sdfg(d=a, torus_max_lat=4.0 / 18.0 * np.arctan(1.0))
    assert (a[0] == 42)
    assert (a[1] == 5.674532920122147)
    assert (a[2] == 42)


if __name__ == "__main__":

    test_fortran_frontend_init()
    test_fortran_frontend_init2()