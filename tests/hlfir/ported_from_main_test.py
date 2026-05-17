"""Drop-in HLFIR ports of the simplest tests that ship on ``origin/main``
under ``tests/fortran/``.

Each test keeps the original numerical assertions and uses the HLFIR
``create_sdfg_from_string`` so that swapping the import line is the only
change required to run the existing test against the new frontend.

We port the tests one at a time  --  this file picks up the very short
ones; more intricate cases (``allocate``-based entry points, PROGRAM
wrappers, etc.) wait until the matching HLFIR lowerings land.
"""

import numpy as np
import pytest

from _util import have_flang

pytestmark = pytest.mark.skipif(not have_flang(), reason="flang-new-21 not available")

# ---------------------------------------------------------------------------
# tests/fortran/fortran_loops_test.py  --  simplest nested-loop case.
# ---------------------------------------------------------------------------


def test_fortran_frontend_loop_region_basic_loop():
    from dace.frontend.hlfir.fortran_parser import create_sdfg_from_string

    # The legacy version wraps the subroutine in a PROGRAM + CALL; the HLFIR
    # frontend runs on the subroutine directly (cross-subroutine lowering is
    # not yet implemented).  The compute body is identical.
    test_string = """
subroutine loop_test_function(a, b, c)
  implicit none
  real(8) :: a(10, 10), b(10, 10), c(10, 10)
  integer :: jk, jl
  do jk = 1, 10
    do jl = 1, 10
      c(jk, jl) = a(jk, jl) + b(jk, jl)
    end do
  end do
end subroutine loop_test_function
"""
    sdfg = create_sdfg_from_string(test_string, "loop_test", use_explicit_cf=True)

    a_test = np.full((10, 10), 2.0, dtype=np.float64)
    b_test = np.full((10, 10), 3.0, dtype=np.float64)
    c_test = np.zeros((10, 10), dtype=np.float64)
    sdfg(a=a_test, b=b_test, c=c_test)

    validate = np.full((10, 10), 5.0, dtype=np.float64)
    assert np.allclose(c_test, validate)
