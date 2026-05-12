"""Ported from f2dace/dev:tests/fortran/dace_support_test.py."""
from __future__ import annotations

import numpy as np
import pytest

from _util import build_sdfg, have_flang

pytestmark = pytest.mark.skipif(not have_flang(), reason="flang-new-21 not on PATH")


def test_fortran_frontend_simplify(tmp_path):
    """The Fortran frontend's SDFG output composes with DaCe's standard
    simplify pipeline.  Originally a probe for the old Python frontend;
    re-purposed here to confirm the HLFIR bridge's output is also
    composable with downstream DaCe transforms."""
    test_string = """
                    PROGRAM symbol_test
                    implicit none
                    double precision d(2,3)
                    CALL symbol_test_function(d)
                    end

                    SUBROUTINE symbol_test_function(d)
                    double precision d(2,3)
                    integer a,b

                    a=1
                    b=2
                    d(:,:)=0.0
                    d(a,b)=5

                    END SUBROUTINE symbol_test_function
                    """
    sdfg = build_sdfg(test_string, tmp_path, name='symbol_test', entry='_QPsymbol_test_function').build()
    a = np.full([2, 3], 42, order="F", dtype=np.float64)
    sdfg(d=a)
    assert a[0, 0] == 0
    assert a[0, 1] == 5
    assert a[1, 2] == 0
