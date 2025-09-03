# Copyright 2019-2023 ETH Zurich and the DaCe authors. All rights reserved.

import numpy as np
import pytest

from dace.frontend.fortran.fortran_parser import create_singular_sdfg_from_string
from tests.fortran.fortran_test_helper import SourceCodeBuilder


@pytest.mark.skip(reason="This requires Deferred Allocation support on DaCe, which we do not have yet.")
def test_fortran_frontend_basic_allocate():
    """
    Tests that the Fortran frontend can parse array accesses and that the accessed indices are correct.
    """
    sources, main = SourceCodeBuilder().add_file("""
subroutine main(d)
  double precision, allocatable, intent(out) :: d(:, :)
  allocate (d(4, 5))
  d(2, 1) = 5.5
end subroutine main""").check_with_gfortran().get()
    sdfg = create_singular_sdfg_from_string(sources, 'main')
    sdfg.simplify(verbose=True)
    a = np.full([4, 5], 42, order="F", dtype=np.float64)
    sdfg(d=a)
    assert (a[0, 0] == 42)
    assert (a[1, 0] == 5.5)
    assert (a[2, 0] == 42)


if __name__ == "__main__":
    test_fortran_frontend_basic_allocate()
