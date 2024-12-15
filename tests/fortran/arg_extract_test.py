# Copyright 2019-2023 ETH Zurich and the DaCe authors. All rights reserved.

import numpy as np
import pytest

from dace.frontend.fortran import fortran_parser

def test_fortran_frontend_arg_extract():
    test_string = """
                    PROGRAM arg_extract
                    implicit none
                    real, dimension(2) :: d
                    real, dimension(2) :: res
                    CALL arg_extract_test_function(d,res)
                    end

                    SUBROUTINE arg_extract_test_function(d,res)
                    real, dimension(2) :: d
                    real, dimension(2) :: res

                    if (MIN(d(1),1) .EQ. 1 ) then
                        res(1) = 3
                        res(2) = 7
                    else
                        res(1) = 5
                        res(2) = 10
                    endif

                    END SUBROUTINE arg_extract_test_function
                    """

    sdfg = fortran_parser.create_sdfg_from_string(test_string, "arg_extract", normalize_offsets=True)
    sdfg.simplify(verbose=True)
    sdfg.compile()
    

    
    input = np.full([2], 42, order="F", dtype=np.float32)
    res = np.full([2], 42, order="F", dtype=np.float32)
    sdfg(d=input, res=res)
    assert np.allclose(res, [3,7])


def test_fortran_frontend_arg_extract2():
    test_string = """
                    PROGRAM arg_extract2
                    implicit none
                    real, dimension(2) :: d
                    real, dimension(2) :: res
                    CALL arg_extract2_test_function(d,res)
                    end

                    SUBROUTINE arg_extract2_test_function(d,res)
                    real, dimension(2) :: d
                    real, dimension(2) :: res

                    if (ALLOCATED(res)) then
                        res(1) = 3
                        res(2) = 7
                    else
                        res(1) = 5
                        res(2) = 10
                    endif

                    END SUBROUTINE arg_extract2_test_function
                    """

    sdfg = fortran_parser.create_sdfg_from_string(test_string, "arg_extract2", normalize_offsets=True)
    sdfg.simplify(verbose=True)
    sdfg.compile()
    

    
    input = np.full([2], 42, order="F", dtype=np.float32)
    res = np.full([2], 42, order="F", dtype=np.float32)
    sdfg(d=input, res=res)
    assert np.allclose(res, [3,7])

if __name__ == "__main__":

    test_fortran_frontend_arg_extract()
    test_fortran_frontend_arg_extract2()
