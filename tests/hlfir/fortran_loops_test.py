"""Ported from f2dace/dev:tests/fortran/fortran_loops_test.py."""
from __future__ import annotations

import numpy as np
import pytest

from _util import build_sdfg, have_flang

pytestmark = pytest.mark.skipif(not have_flang(), reason="flang-new-21 not on PATH")


def test_fortran_frontend_loop_region_basic_loop(tmp_path):
    test_string = """
    PROGRAM loop_test_program
        implicit none
        double precision a(10,10)
        double precision b(10,10)
        double precision c(10,10)

        CALL loop_test_function(a,b,c)
    end

    SUBROUTINE loop_test_function(a,b,c)
        double precision :: a(10,10)
        double precision :: b(10,10)
        double precision :: c(10,10)

        INTEGER :: JK,JL
        DO JK=1,10
            DO JL=1,10
                c(JK,JL) = a(JK,JL) + b(JK,JL)
            ENDDO
        ENDDO
    end SUBROUTINE loop_test_function
    """
    sdfg = build_sdfg(test_string, tmp_path, name='loop_test', entry='_QPloop_test_function').build()

    a_test = np.full([10, 10], 2, order="F", dtype=np.float64)
    b_test = np.full([10, 10], 3, order="F", dtype=np.float64)
    c_test = np.zeros([10, 10], order="F", dtype=np.float64)
    sdfg(a=a_test, b=b_test, c=c_test)

    validate = np.full([10, 10], 5, order="F", dtype=np.float64)
    assert np.allclose(c_test, validate)
