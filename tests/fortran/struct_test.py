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
                    integer :: startidx
                    integer :: endidx
                    CALL struct_test_range_test_function(res, startidx, endidx)
                    end

                    SUBROUTINE struct_test_range_test_function(res, startidx, endidx)
                    integer, dimension(6) :: res
                    integer :: startidx
                    integer :: endidx
                    type(test_type) :: indices

                    indices%start=startidx
                    indices%end=endidx

                    CALL struct_test_range2_test_function(res, indices)

                    END SUBROUTINE struct_test_range_test_function

                    SUBROUTINE struct_test_range2_test_function(res, idx)
                    integer, dimension(6) :: res
                    type(test_type) :: idx

                    res(idx%start:idx%end) = 42

                    END SUBROUTINE struct_test_range2_test_function
                    """
    sources={}
    sdfg = fortran_parser.create_sdfg_from_string(test_string, "struct_test_range_test", False, sources=sources)
    sdfg.save('before.sdfg')
    sdfg.simplify(verbose=True)
    sdfg.save('after.sdfg')
    sdfg.compile()

    size = 6
    res = np.full([size], 42, order="F", dtype=np.int32)
    res[:] = 0
    sdfg(res=res, start=2, end=5)
    print(res)

def test_fortran_struct_lhs():
    test_string = """
                    PROGRAM struct_test_range
                    implicit none

                    type test_type
                        integer, dimension(6) :: res
                        integer :: start
                        integer :: end
                    end type

                    type test_type2
                        type(test_type) :: var
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
                    type(test_type2) :: val

                    indices = test_type(res, start, end)
                    val = test_type2(indices)

                    CALL struct_test_range2_test_function(val)

                    END SUBROUTINE struct_test_range_test_function

                    SUBROUTINE struct_test_range2_test_function(idx)
                    type(test_type2) :: idx

                    idx%var%res(idx%var%start:idx%var%end) = 42

                    END SUBROUTINE struct_test_range2_test_function
                    """
    sources={}
    sdfg = fortran_parser.create_sdfg_from_string(test_string, "struct_test_range_test", False, sources=sources)
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
    test_fortran_struct_lhs()
