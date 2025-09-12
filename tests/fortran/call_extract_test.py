# Copyright 2019-2023 ETH Zurich and the DaCe authors. All rights reserved.

import numpy as np
import pytest

from dace.frontend.fortran import fortran_parser


def test_fortran_frontend_call_extract():
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

    sdfg = fortran_parser.create_sdfg_from_string(test_string, "intrinsic_call_extract_test", normalize_offsets=True)
    sdfg.simplify(verbose=True)
    sdfg.compile()

    input = np.full([2], 42, order="F", dtype=np.float32)
    res = np.full([2], 42, order="F", dtype=np.float32)
    sdfg(d=input, res=res)
    assert np.allclose(res, [np.sqrt(np.exp(input[0])), np.sqrt(np.exp(input[0])) - 1])


if __name__ == "__main__":

    test_fortran_frontend_call_extract()
