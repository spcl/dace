"""Ported from f2dace/dev:tests/fortran/call_extract_test.py."""
from __future__ import annotations

import numpy as np
import pytest

from _util import build_sdfg, have_flang

pytestmark = pytest.mark.skipif(not have_flang(), reason="flang-new-21 not on PATH")


def test_fortran_frontend_call_extract(tmp_path):
    test_string = """
                    PROGRAM intrinsic_call_extract
                    implicit none
                    real, dimension(2) :: d
                    real, dimension(2) :: res
                    CALL intrinsic_call_extract_test_function(d,res)
                    end

                    SUBROUTINE intrinsic_call_extract_test_function(d,res)
                    real, dimension(2) :: d
                    real, dimension(2) :: res

                    res(1) = SQRT(SIGN(EXP(d(1)), LOG(d(1))))
                    res(2) = MIN(SQRT(EXP(d(1))), SQRT(EXP(d(1))) - 1)

                    END SUBROUTINE intrinsic_call_extract_test_function
                    """

    sdfg = build_sdfg(test_string,
                      tmp_path,
                      name='intrinsic_call_extract',
                      entry='_QPintrinsic_call_extract_test_function').build()

    inp = np.full([2], 42, order="F", dtype=np.float32)
    res = np.full([2], 42, order="F", dtype=np.float32)
    sdfg(d=inp, res=res)
    assert np.allclose(res, [np.sqrt(np.exp(inp[0])), np.sqrt(np.exp(inp[0])) - 1])
