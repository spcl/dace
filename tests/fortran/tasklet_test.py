# Copyright 2019-2023 ETH Zurich and the DaCe authors. All rights reserved.

import numpy as np
import pytest

from dace.frontend.fortran import fortran_parser

def test_fortran_frontend_tasklet():
    test_string = """
                    PROGRAM tasklet
                    implicit none
                    real, dimension(2) :: d
                    real, dimension(2) :: res
                    CALL tasklet_test_function(d,res)
                    end

                    SUBROUTINE tasklet_test_function(d,res)
                    real, dimension(2) :: d
                    real, dimension(2) :: res
                    real :: temp


                    integer :: i
                    i=1
                    temp = 88
                    d(1)=d(1)*2
                    temp = MIN(d(i), temp)
                    res(1) = temp + 10

                    END SUBROUTINE tasklet_test_function
                    """

    sdfg = fortran_parser.create_sdfg_from_string(test_string, "tasklet_test", normalize_offsets=True)
    sdfg.simplify(verbose=True)
    
    sdfg.compile()
    
    input = np.full([2], 42, order="F", dtype=np.float32)
    res = np.full([2], 42, order="F", dtype=np.float32)
    sdfg(d=input, res=res)
    assert np.allclose(res, [94, 42])


if __name__ == "__main__":

    test_fortran_frontend_tasklet()
