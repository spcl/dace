# Copyright 2019-2023 ETH Zurich and the DaCe authors. All rights reserved.

import numpy as np
import pytest

from dace.frontend.fortran import fortran_parser


def test_fortran_frontend_while():
    test_string = """
                    PROGRAM while
                    implicit none
                    real, dimension(2) :: d
                    real, dimension(2) :: res
                    CALL while_test_function(d,res)
                    end

                    SUBROUTINE while_test_function(d,res)
                    real, dimension(2) :: d
                    real, dimension(2) :: res


                    integer :: i
                    i=0
                    res(1)=d(1)*2
                    do while (i<10)
                        res(1)=res(1)+1
                        i=i+1
                    end do

                    END SUBROUTINE while_test_function
                    """

    sdfg = fortran_parser.create_sdfg_from_string(test_string, "while_test", normalize_offsets=True)
    sdfg.simplify(verbose=True)
    sdfg.compile()

    input = np.full([2], 42, order="F", dtype=np.float32)
    res = np.full([2], 42, order="F", dtype=np.float32)
    sdfg(d=input, res=res)
    assert np.allclose(res, [94, 42])


if __name__ == "__main__":

    test_fortran_frontend_while()
