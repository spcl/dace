"""Verbatim port of f2dace/dev:tests/fortran/non-interactive/fortran_int_init_test.py."""

import numpy as np
import pytest

from _util import build_sdfg, have_flang

pytestmark = pytest.mark.skipif(not have_flang(), reason="flang-new-21 not on PATH")


def test_fortran_frontend_int_init(tmp_path):
    src = """
subroutine main(d)
  integer, parameter :: i8 = selected_int_kind(15)
  integer(kind=i8) d(2)
  d(1) = int(z'000000ffffffffff', i8)
end subroutine main
"""
    sdfg = build_sdfg(src, tmp_path, name='main').build()
    d = np.full([2], 42, order="F", dtype=np.int64)
    sdfg(d=d)
    assert d[0] == int("000000ffffffffff", 16)
