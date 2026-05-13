"""Boolean and integer argument dtype correctness in the bridge bindings.

Regression guard for the cloudsc_full PLUDE diagnosis: the registry
used to allocate a Fortran LOGICAL input as a numpy ``int32`` array
even though the bridge SDFG declared the argument as ``bool *``.
The byte-layout mismatch silently corrupted LDCUM across element
boundaries.

E2e against an f2py-compiled reference of the same Fortran source.
"""
from __future__ import annotations

import numpy as np
import pytest

from _util import build_sdfg, have_flang
from _helpers import f2py

pytestmark = pytest.mark.skipif(not have_flang(), reason="flang-new-21 not on PATH")


def test_bool_logical_array_pass_through(tmp_path):
    """``LOGICAL`` 1-D input read element-by-element + written to a
    numeric output.  Alternating True/False  --  any byte-stride bug
    surfaces here.
    """
    src = """
SUBROUTINE bool_pass(flags, out, n)
integer, intent(in) :: n
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
    ref = f2py(src, tmp_path / 'ref', 'bool_pass_ref')
    sdfg_dir = tmp_path / 'sdfg'
    sdfg_dir.mkdir(parents=True, exist_ok=True)
    sdfg = build_sdfg(src, sdfg_dir, name='bool_pass', entry='_QPbool_pass').build()

    flags = np.array([True, False, True, False, True, False, True, False], dtype=np.bool_)
    n = flags.size
    # f2py returns intent(out) arrays.
    out_ref = ref.bool_pass(flags)

    out = np.zeros(n, dtype=np.int32)
    sdfg(flags=flags, out=out, n=n)
    np.testing.assert_array_equal(out, out_ref)


def test_bool_logical_array_2d_pass_through(tmp_path):
    """2-D ``LOGICAL`` input (the LDCUM shape)."""
    src = """
SUBROUTINE bool_2d_pass(flags, out, klon, nblocks)
integer, intent(in) :: klon, nblocks
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
    ref = f2py(src, tmp_path / 'ref', 'bool_2d_pass_ref')
    sdfg_dir = tmp_path / 'sdfg'
    sdfg_dir.mkdir(parents=True, exist_ok=True)
    sdfg = build_sdfg(src, sdfg_dir, name='bool_2d_pass', entry='_QPbool_2d_pass').build()

    klon, nblocks = 1, 4
    flags = np.array([[False, True, True, False]], dtype=np.bool_, order='F')
    out_ref = ref.bool_2d_pass(flags)

    out = np.zeros((klon, nblocks), dtype=np.int32, order='F')
    sdfg(flags=flags, out=out, klon=klon, nblocks=nblocks)
    np.testing.assert_array_equal(out, out_ref)


def test_int32_array_pass_through(tmp_path):
    """Plain INTEGER input pass-through."""
    src = """
SUBROUTINE int_double(inp, out, n)
integer, intent(in) :: n
integer, intent(in) :: inp(n)
integer, intent(out) :: out(n)
integer i
DO i = 1, n
    out(i) = inp(i) * 2
ENDDO
END SUBROUTINE int_double
"""
    ref = f2py(src, tmp_path / 'ref', 'int_double_ref')
    sdfg_dir = tmp_path / 'sdfg'
    sdfg_dir.mkdir(parents=True, exist_ok=True)
    sdfg = build_sdfg(src, sdfg_dir, name='int_double', entry='_QPint_double').build()

    inp = np.array([1, 2, 3, 100, -5, 1000000], dtype=np.int32)
    n = inp.size
    out_ref = ref.int_double(inp)

    out = np.zeros(n, dtype=np.int32)
    sdfg(inp=inp, out=out, n=n)
    np.testing.assert_array_equal(out, out_ref)


def test_bool_scalar_logical_pass_through(tmp_path):
    """Scalar LOGICAL passed as a length-1 array argument.

    Closes the audit gap that let the cloudsc registry pass
    ``np.int32(np.bool_(True))`` to a ``bool *`` parameter for six
    of its scalar LOGICAL inputs (``LAERLIQCOLL``, ``LDMAINCALL``,
    ``LDSLPHY``, etc.).  Functionally it worked because the int32
    LSB for value ``1`` happens to read as ``true`` in C bool, but
    any value with bit-0 = 0 (e.g. 256) would silently corrupt to
    ``false``.  Pin the contract here: scalar LOGICAL must be
    routed as a length-1 ``np.bool_`` array.
    """
    src = """
SUBROUTINE bool_scalar(flag, out, n)
integer, intent(in) :: n
logical, intent(in) :: flag
integer, intent(out) :: out(n)
integer i
DO i = 1, n
    IF (flag) THEN
        out(i) = 1
    ELSE
        out(i) = 0
    ENDIF
ENDDO
END SUBROUTINE bool_scalar
"""
    ref = f2py(src, tmp_path / 'ref', 'bool_scalar_ref')
    sdfg_dir = tmp_path / 'sdfg'
    sdfg_dir.mkdir(parents=True, exist_ok=True)
    sdfg = build_sdfg(src, sdfg_dir, name='bool_scalar', entry='_QPbool_scalar').build()

    from dace.data import Scalar
    desc = sdfg.arglist().get('flag')

    def _route_bool(v):
        """Route scalar LOGICAL to whatever the bridge declared:
        ``Scalar(bool)`` (intent(in) scalar) takes a plain Python
        bool; ``Array(1,) bool`` takes a length-1 ``np.bool_``
        array.  Routing as ``np.int32`` would mis-type the C ABI
        and the dtype-mismatch warning would fire (silent corruption
        on values with bit-0 = 0)."""
        if isinstance(desc, Scalar):
            return bool(v)
        return np.array([bool(v)], dtype=np.bool_)

    n = 5
    out_ref = ref.bool_scalar(True, n=n)
    out = np.zeros(n, dtype=np.int32)
    sdfg(flag=_route_bool(True), out=out, n=n)
    np.testing.assert_array_equal(out, out_ref)

    out_ref = ref.bool_scalar(False, n=n)
    out = np.zeros(n, dtype=np.int32)
    sdfg(flag=_route_bool(False), out=out, n=n)
    np.testing.assert_array_equal(out, out_ref)
