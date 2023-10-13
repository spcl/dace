# Copyright 2019-2023 ETH Zurich and the DaCe authors. All rights reserved.

import numpy as np

from dace.frontend.fortran import ast_transforms, fortran_parser

def test_fortran_frontend_any_array():
    """
    Tests that the generated array map correctly handles offsets.
    """
    test_string = """
                    PROGRAM intrinsic_any_test
                    implicit none
                    logical, dimension(5) :: d
                    logical, dimension(2) :: res
                    CALL intrinsic_any_test_function(d, res)
                    end

                    SUBROUTINE intrinsic_any_test_function(d, res)
                    logical, dimension(5) :: d
                    logical, dimension(2) :: res

                    res(1) = ANY(d)

                    !res(1) = ANY(d == .True.)
                    !d(3) = .False.
                    !res(2) = ANY(d == .True.)

                    !res(1) = ANY(d == e)
                    !d(3) = .False.
                    !res(2) = ANY(d == 

                    END SUBROUTINE intrinsic_any_test_function
                    """

    sdfg = fortran_parser.create_sdfg_from_string(test_string, "intrinsic_any_test", False)
    sdfg.simplify(verbose=True)
    sdfg.compile()

    size = 5
    d = np.full([size], False, order="F", dtype=np.int32)
    res = np.full([2], 42, order="F", dtype=np.int32)

    d[2] = True
    sdfg(d=d, res=res)
    assert res[0] == True

    d[2] = False
    sdfg(d=d, res=res)
    assert res[0] == False

if __name__ == "__main__":

    test_fortran_frontend_any_array()
