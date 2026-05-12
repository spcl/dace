"""Boolean and integer argument dtype correctness in the bridge bindings.

Regression guard for the cloudsc_full PLUDE diagnosis: the registry
used to allocate a Fortran ``LOGICAL`` input as a numpy ``int32``
array even though the bridge SDFG declared the argument as ``bool *``.
The byte-layout mismatch made the SDFG read byte offsets 0..3 of one
int32 as four separate "booleans", silently corrupting the input
across element boundaries.

These tests fix one Fortran kernel each (boolean + integer) and
verify the bridge SDFG reads each element correctly when called
with a numpy array of the matching dtype.  If a future change widens
``np.bool_`` to int32 or otherwise re-pessimises the binding, these
tests catch it before the value gets used in a comparison or branch.
"""
from __future__ import annotations

import numpy as np
import pytest

from _util import build_sdfg, have_flang

pytestmark = pytest.mark.skipif(not have_flang(), reason="flang-new-21 not on PATH")


def test_bool_logical_array_pass_through(tmp_path):
    """``LOGICAL`` 1-D input read element-by-element + written to a
    numeric output.  If the bridge reads bytes instead of LOGICAL
    elements, the int output for an alternating-bool input will be
    wrong on every other element.
    """
    test_string = """
                    SUBROUTINE bool_pass(flags, out, n)
                    integer :: n
                    logical, intent(in) :: flags(n)
                    integer, intent(out) :: out(n)
                    integer i
                    DO i = 1, n
                        IF (flags(i)) THEN
                            out(i) = 1
                        ELSE
                            out(i) = 0
                        ENDIF
                    ENDDO
                    END SUBROUTINE bool_pass
                    """
    sdfg = build_sdfg(test_string, tmp_path, name='bool_pass', entry='_QPbool_pass').build()

    n = 8
    # Alternating True/False — any byte-stride bug surfaces here.
    flags = np.array([True, False, True, False, True, False, True, False], dtype=np.bool_)
    out = np.zeros(n, dtype=np.int32)
    sdfg(flags=flags, out=out, n=n)
    expected = np.array([1, 0, 1, 0, 1, 0, 1, 0], dtype=np.int32)
    assert np.array_equal(out, expected), f"got {out} expected {expected}"


def test_bool_logical_array_2d_pass_through(tmp_path):
    """2-D ``LOGICAL`` input (shape (klon, nblocks)) — the exact shape
    cloudsc's ``LDCUM`` uses.  Verifies indexing across BOTH dims
    when klon is small and nblocks is small (the cloudsc parameter
    regime where the PLUDE byte-stride bug fired).
    """
    test_string = """
                    SUBROUTINE bool_2d_pass(flags, out, klon, nblocks)
                    integer :: klon, nblocks
                    logical, intent(in) :: flags(klon, nblocks)
                    integer, intent(out) :: out(klon, nblocks)
                    integer i, j
                    DO j = 1, nblocks
                        DO i = 1, klon
                            IF (flags(i, j)) THEN
                                out(i, j) = 1
                            ELSE
                                out(i, j) = 0
                            ENDIF
                        ENDDO
                    ENDDO
                    END SUBROUTINE bool_2d_pass
                    """
    sdfg = build_sdfg(test_string, tmp_path, name='bool_2d_pass', entry='_QPbool_2d_pass').build()

    klon, nblocks = 1, 4
    flags = np.array([[False, True, True, False]], dtype=np.bool_, order='F')
    out = np.zeros((klon, nblocks), dtype=np.int32, order='F')
    sdfg(flags=flags, out=out, klon=klon, nblocks=nblocks)
    expected = np.array([[0, 1, 1, 0]], dtype=np.int32)
    assert np.array_equal(out, expected), f"got {out} expected {expected}"


def test_int32_array_pass_through(tmp_path):
    """Plain INTEGER input element-by-element.  Sanity-check that
    ``int32`` arguments aren't accidentally truncated to int8/int16
    or widened to int64 anywhere in the binding path.
    """
    test_string = """
                    SUBROUTINE int_double(inp, out, n)
                    integer :: n
                    integer, intent(in) :: inp(n)
                    integer, intent(out) :: out(n)
                    integer i
                    DO i = 1, n
                        out(i) = inp(i) * 2
                    ENDDO
                    END SUBROUTINE int_double
                    """
    sdfg = build_sdfg(test_string, tmp_path, name='int_double', entry='_QPint_double').build()

    n = 6
    inp = np.array([1, 2, 3, 100, -5, 1000000], dtype=np.int32)
    out = np.zeros(n, dtype=np.int32)
    sdfg(inp=inp, out=out, n=n)
    assert np.array_equal(out, inp * 2)
