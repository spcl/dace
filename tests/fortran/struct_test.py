# Copyright 2019-2023 ETH Zurich and the DaCe authors. All rights reserved.

import numpy as np
import pytest

from dace.frontend.fortran import fortran_parser

def test_fortran_struct():
    test_string = """
                    PROGRAM struct_test_range
                    implicit none

                    type test_type
                        integer :: start
                        integer :: end
                    end type

                    integer, dimension(6) :: res
                    integer :: start
                    integer :: end
                    CALL struct_test_range_test_function(res, start, end)
                    end

                    SUBROUTINE struct_test_range_test_function(res, start, end)
                    integer, dimension(6) :: res
                    integer :: start
                    integer :: end
                    type(test_type) :: indices

                    indices = test_type(start, end)

                    CALL struct_test_range2_test_function(res, indices)

                    END SUBROUTINE struct_test_range_test_function

                    SUBROUTINE struct_test_range2_test_function(res, idx)
                    integer, dimension(6) :: res
                    type(test_type) :: idx

                    res(idx%start:idx%end) = 42

                    END SUBROUTINE struct_test_range2_test_function
                    """
    sources={}
    sdfg = fortran_parser.create_sdfg_from_string(test_string, "res", False, sources=sources)
    sdfg.save('before.sdfg')
    sdfg.simplify(verbose=True)
    sdfg.save('after.sdfg')
    sdfg.compile()

    size = 6
    res = np.full([size], 42, order="F", dtype=np.int32)
    res[:] = 0
    sdfg(res=res, start=2, end=5)
    print(res)

if __name__ == "__main__":
    test_fortran_struct()
