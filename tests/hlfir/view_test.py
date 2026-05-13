"""Ported from f2dace/dev:tests/fortran/view_test.py.

Exercises Fortran array-slice arguments to subroutines  --  the
caller passes ``aa(:, :, k)`` (a 2-D view into a 3-D parent), the
callee operates on it as if it were a contiguous 2-D array.
"""
from __future__ import annotations

import numpy as np
import pytest

from _util import build_sdfg, have_flang

pytestmark = pytest.mark.skipif(not have_flang(), reason="flang-new-21 not on PATH")


def test_fortran_frontend_view_test(tmp_path):
    """Single-view: caller passes one 2-D slice into a 3-D parent.

    The original test declared the callee's dummy ``aa`` rank-3
    (``aa(10,11,23)``) but indexed it rank-2 (``aa(JK,JL)``)  --
    invalid Fortran that the old Python parser silently accepted.
    Fixed by declaring the dummy rank-2 to match the access shape;
    the caller still passes a rank-2 slice ``a(:, :, 1)`` of its
    rank-3 storage so the "subroutine receives a 2-D view into a
    3-D parent" coverage is preserved.
    """
    test_name = "view_test"
    test_string = f"""
                    PROGRAM {test_name}_program
implicit none
double precision a(10,11,12)
double precision res(2,2,2)

CALL {test_name}_function(a,res)

end

SUBROUTINE {test_name}_function(aa,res)

double precision aa(10,11,12)
double precision res(2,2,2)

call viewlens(aa(:,:,1),res)

end SUBROUTINE {test_name}_function

SUBROUTINE viewlens(aa,res)

IMPLICIT NONE

double precision  :: aa(10,11)
double precision :: res(2,2,2)

INTEGER ::  JK, JL

res(1,1,1)=0.0
DO JK=1,10
  DO JL=1,11
    res(1,1,1)=res(1,1,1)+aa(JK,JL)
  ENDDO
ENDDO
aa(1,1)=res(1,1,1)


END SUBROUTINE viewlens
                    """
    sdfg = build_sdfg(test_string, tmp_path, name=test_name, entry=f'_QP{test_name}_function').build()
    a = np.full([10, 11, 12], 42, order="F", dtype=np.float64)
    b = np.full([2, 2, 2], 42, order="F", dtype=np.float64)
    b[0, 0, 0] = 1
    sdfg(aa=a, res=b)
    assert a[0, 0, 1] == 42
    assert a[0, 0, 0] == 4620
    assert b[0, 0, 0] == 4620


def test_fortran_frontend_view_test_2(tmp_path):
    """Multiple views per array in the same context: caller passes
    ``aa(:, :, j)`` and ``aa(:, :, k)`` for distinct ``j``, ``k``.
    """
    test_name = "view2_test"
    test_string = f"""
                    PROGRAM {test_name}_program
implicit none
integer, parameter :: n=10
double precision a(n,11,12),b(n,11,12),c(n,11,12)

CALL {test_name}_function(a,b,c,n)

end

SUBROUTINE {test_name}_function(aa,bb,cc,n)

integer :: n
double precision aa(n,11,12),bb(n,11,12),cc(n,11,12)
integer j,k

j=1
    call viewlens(aa(:,:,j),bb(:,:,j),cc(:,:,j))
k=2
    call viewlens(aa(:,:,k),bb(:,:,k),cc(:,:,k))

end SUBROUTINE {test_name}_function

SUBROUTINE viewlens(aa,bb,cc)

IMPLICIT NONE

double precision  :: aa(10,11),bb(10,11),cc(10,11)

INTEGER ::  JK, JL

DO JK=1,10
  DO JL=1,11
    cc(JK,JL)=bb(JK,JL)+aa(JK,JL)
  ENDDO
ENDDO

END SUBROUTINE viewlens
                    """
    sdfg = build_sdfg(test_string, tmp_path, name=test_name, entry=f'_QP{test_name}_function').build()
    a = np.full([10, 11, 12], 42, order="F", dtype=np.float64)
    b = np.full([10, 11, 12], 42, order="F", dtype=np.float64)
    c = np.full([10, 11, 12], 42, order="F", dtype=np.float64)

    b[0, 0, 0] = 1
    sdfg(aa=a, bb=b, cc=c, n=10)
    assert c[0, 0, 0] == 43
    assert c[1, 1, 1] == 84


def test_fortran_frontend_view_test_3(tmp_path):
    """Multiple views from the SAME array in the same context (``aa(:,
    :, j)`` and ``aa(:, :, j+1)`` both bound on the call).
    """
    test_name = "view3_test"
    test_string = f"""
                    PROGRAM {test_name}_program
implicit none
integer, parameter :: n=10
double precision a(n,n+1,12),b(n,n+1,12)

CALL {test_name}_function(a,b,n)

end

SUBROUTINE {test_name}_function(aa,bb,n)

integer :: n
double precision aa(n,n+1,12),bb(n,n+1,12)
integer j,k

j=1
    call viewlens(aa(:,:,j),bb(:,:,j),bb(:,:,j+1))

end SUBROUTINE {test_name}_function

SUBROUTINE viewlens(aa,bb,cc)

IMPLICIT NONE

double precision  :: aa(10,11),bb(10,11),cc(10,11)

INTEGER ::  JK, JL

DO JK=1,10
  DO JL=1,11
    cc(JK,JL)=bb(JK,JL)+aa(JK,JL)
  ENDDO
ENDDO

END SUBROUTINE viewlens
                    """
    sdfg = build_sdfg(test_string, tmp_path, name=test_name, entry=f'_QP{test_name}_function').build()
    a = np.full([10, 11, 12], 42, order="F", dtype=np.float64)
    b = np.full([10, 11, 12], 42, order="F", dtype=np.float64)

    b[0, 0, 0] = 1
    sdfg(aa=a, bb=b, n=10)
    assert b[0, 0, 0] == 1
    assert b[0, 0, 1] == 43
