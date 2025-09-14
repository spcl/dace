# Copyright 2019-2025 ETH Zurich and the DaCe authors. All rights reserved.

import numpy as np

from dace.frontend.fortran.fortran_parser import create_singular_sdfg_from_string
from tests.fortran.fortran_test_helper import SourceCodeBuilder


def test_fortran_frontend_simplify():
    """
    Test that the DaCe simplify works with the input SDFG provided by the Fortran frontend.
    """
    sources, main = SourceCodeBuilder().add_file("""
subroutine main(d)
  double precision d(2, 3)
  integer a, b
  a = 1
  b = 2
  d(:, :) = 0.0
  d(a, b) = 5
end subroutine main
""").check_with_gfortran().get()
    sdfg = create_singular_sdfg_from_string(sources, 'main')
    sdfg.simplify(verbose=True)
    a = np.full([2, 3], 42, order="F", dtype=np.float64)
    sdfg(d=a)
    assert (a[0, 0] == 0)
    assert (a[0, 1] == 5)
    assert (a[1, 2] == 0)


if __name__ == "__main__":
    test_fortran_frontend_simplify()
