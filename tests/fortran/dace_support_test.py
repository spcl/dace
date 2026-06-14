# Copyright 2023 ETH Zurich and the DaCe authors. All rights reserved.

import numpy as np

from dace.frontend.fortran import fortran_parser


def test_fortran_frontend_simplify():
    """
    Test that the DaCe simplify works with the input SDFG provided by the Fortran frontend.
    """
    test_string = """
                    PROGRAM symbol_test
                    implicit none
                    double precision d(2,3)
                    CALL symbol_test_function(d)
                    end

                    SUBROUTINE symbol_test_function(d)
                    double precision d(2,3)
                    integer a,b

                    a=1
                    b=2
                    d(:,:)=0.0
                    d(a,b)=5

                    END SUBROUTINE symbol_test_function
                    """
    sdfg = fortran_parser.create_sdfg_from_string(test_string, "symbol_test")
    sdfg.simplify(verbose=True)
    a = np.full([2, 3], 42, order="F", dtype=np.float64)
    sdfg(d=a)
    assert (a[0, 0] == 0)
    assert (a[0, 1] == 5)
    assert (a[1, 2] == 0)


if __name__ == "__main__":
    test_fortran_frontend_simplify()
