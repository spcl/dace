"""Verbatim port of f2dace/dev:tests/fortran/optional_args_test.py."""
from __future__ import annotations

import ctypes

import numpy as np
import pytest

from _util import build_sdfg, have_flang

try:
    ctypes.CDLL("libgomp.so.1", ctypes.RTLD_GLOBAL)
except OSError:
    pass

pytestmark = pytest.mark.skipif(not have_flang(), reason="flang-new-21 not on PATH")


def test_fortran_frontend_optional(tmp_path):
    src = """

    MODULE intrinsic_optional_test
        INTERFACE
            SUBROUTINE intrinsic_optional_test_function2(res, a)
                integer, dimension(2) :: res
                integer, optional :: a
            END SUBROUTINE intrinsic_optional_test_function2
        END INTERFACE
    END MODULE

    SUBROUTINE intrinsic_optional_test_function(res, res2, a)
    USE intrinsic_optional_test
    implicit none
    integer, dimension(4) :: res
    integer, dimension(4) :: res2
    integer :: a

    CALL intrinsic_optional_test_function2(res, a)
    CALL intrinsic_optional_test_function2(res2)

    END SUBROUTINE intrinsic_optional_test_function

    SUBROUTINE intrinsic_optional_test_function2(res, a)
    integer, dimension(2) :: res
    integer, optional :: a

    res(1) = a

    END SUBROUTINE intrinsic_optional_test_function2
"""
    sdfg = build_sdfg(src,
                      tmp_path,
                      name='intrinsic_optional_test_function',
                      entry='_QPintrinsic_optional_test_function').build()

    size = 4
    res = np.full([size], 42, order="F", dtype=np.int32)
    res2 = np.full([size], 42, order="F", dtype=np.int32)
    sdfg(res=res, res2=res2, a=5, a_present=1)

    # Safe path only — second internal call reads OPTIONAL ``a`` without
    # checking PRESENT and is UB per Fortran (we accept the read may
    # produce garbage; res2 is left unchecked).
    assert res[0] == 5


def test_fortran_frontend_optional_complex(tmp_path):
    src = """

    MODULE intrinsic_optional_test
        INTERFACE
            SUBROUTINE intrinsic_optional_test_function2(res, a, b, c)
                integer, dimension(5) :: res
                integer, optional :: a
                double precision, optional :: b
                logical, optional :: c
            END SUBROUTINE intrinsic_optional_test_function2
        END INTERFACE
    END MODULE

    SUBROUTINE intrinsic_optional_test_function(res, res2, a, b, c)
    USE intrinsic_optional_test
    implicit none
    integer, dimension(5) :: res
    integer, dimension(5) :: res2
    integer :: a
    double precision :: b
    logical :: c

    CALL intrinsic_optional_test_function2(res, a, b)
    CALL intrinsic_optional_test_function2(res2)

    END SUBROUTINE intrinsic_optional_test_function

    SUBROUTINE intrinsic_optional_test_function2(res, a, b, c)
    integer, dimension(5) :: res
    integer, optional :: a
    double precision, optional :: b
    logical, optional :: c

    res(1) = a
    res(2) = b
    res(3) = c

    END SUBROUTINE intrinsic_optional_test_function2
"""
    sdfg = build_sdfg(src,
                      tmp_path,
                      name='intrinsic_optional_test_function',
                      entry='_QPintrinsic_optional_test_function').build()

    size = 5
    res = np.full([size], 42, order="F", dtype=np.int32)
    res2 = np.full([size], 42, order="F", dtype=np.int32)
    sdfg(res=res, res2=res2, a=5, a_present=1, b=7.0, b_present=1, c=1, c_present=0)

    # Safe path only — caller passed ``a`` and ``b`` to the first
    # internal call; ``c`` was not passed and is UB if read.  The
    # second internal call passes none of them and is UB throughout
    # (res2 left unchecked, per the saved "let it crash" policy).
    assert res[0] == 5
    assert res[1] == 7
