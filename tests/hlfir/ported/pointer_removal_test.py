"""Verbatim port of f2dace/dev:tests/fortran/pointer_removal_test.py."""
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


def test_fortran_frontend_ptr_assignment_removal(tmp_path):
    src = """
MODULE lib
    TYPE simple_type
        INTEGER :: a
    END TYPE simple_type
END MODULE lib

SUBROUTINE type_in_call_test_function(d)
    USE lib
    REAL d(5,5)
    TYPE(simple_type), TARGET :: s
    INTEGER,POINTER :: tmp
    tmp=>s%a

    tmp = 13
    d(2,1) = max(1.0, tmp)
END SUBROUTINE type_in_call_test_function
"""
    sdfg = build_sdfg(src, tmp_path, name='type_in_call_test_function').build()
    a = np.full([5, 5], 42, order="F", dtype=np.float32)
    sdfg(d=a)
    assert (a[0, 0] == 42)
    assert (a[1, 0] == 13)
    assert (a[2, 0] == 42)


def test_fortran_frontend_ptr_assignment_removal_array(tmp_path):
    src = """
MODULE lib
    TYPE simple_type
        REAL :: w(5,5,5)
    END TYPE simple_type
END MODULE lib

SUBROUTINE type_in_call_test_function(d)
    USE lib
    REAL d(5,5)
    TYPE(simple_type), TARGET :: s
    REAL,POINTER :: tmp(:,:,:)
    tmp=>s%w

    tmp(1,1,1) = 11.0
    d(2,1) = max(1.0, tmp(1,1,1))
END SUBROUTINE type_in_call_test_function
"""
    sdfg = build_sdfg(src, tmp_path, name='type_in_call_test_function').build()
    a = np.full([5, 5], 42, order="F", dtype=np.float32)
    sdfg(d=a)
    assert (a[0, 0] == 42)
    assert (a[1, 0] == 11)
    assert (a[2, 0] == 42)


def test_fortran_frontend_ptr_assignment_removal_array_assumed(tmp_path):
    src = """
MODULE lib
    TYPE simple_type
        REAL :: w(5,5,5)
    END TYPE simple_type
CONTAINS
    SUBROUTINE type_in_call_test_function2(tmp)
        REAL,POINTER :: tmp(:,:,:)
        tmp(2,1,1) = 1410
    END SUBROUTINE type_in_call_test_function2
END MODULE lib

SUBROUTINE type_in_call_test_function(d)
    USE lib
    REAL d(5,5)
    TYPE(simple_type), TARGET :: s
    REAL,POINTER :: tmp(:,:,:)
    tmp=>s%w

    tmp(1,1,1) = 11.0
    d(2,1) = max(1.0, tmp(1,1,1))

    CALL type_in_call_test_function2(tmp)
    d(3,1) = max(1.0, tmp(2,1,1))

END SUBROUTINE type_in_call_test_function
"""
    sdfg = build_sdfg(src, tmp_path, name='type_in_call_test_function', entry='_QPtype_in_call_test_function').build()
    a = np.full([5, 5], 42, order="F", dtype=np.float32)
    sdfg(d=a)
    assert (a[0, 0] == 42)
    assert (a[1, 0] == 11)
    assert (a[2, 0] == 1410)


def test_fortran_frontend_ptr_assignment_removal_array_nested(tmp_path):
    src = """
MODULE lib
    TYPE inner_type
        REAL :: w(5,5,5)
    END TYPE inner_type
    TYPE simple_type
        TYPE(inner_type) :: val1
    END TYPE simple_type
END MODULE lib

SUBROUTINE type_in_call_test_function(d)
    USE lib
    REAL d(5,5)
    TYPE(simple_type), TARGET :: s
    REAL,POINTER :: tmp(:,:,:)
    !tmp=>s%val1%val%w
    tmp=>s%val1%w

    tmp(1,1,1) = 11.0
    d(2,1) = tmp(1,1,1)
END SUBROUTINE type_in_call_test_function
"""
    sdfg = build_sdfg(src, tmp_path, name='type_in_call_test_function').build()
    a = np.full([5, 5], 42, order="F", dtype=np.float32)
    sdfg(d=a)
    assert (a[0, 0] == 42)
    assert (a[1, 0] == 11)
    assert (a[2, 0] == 42)
