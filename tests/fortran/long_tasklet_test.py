# Copyright 2023 ETH Zurich and the DaCe authors. All rights reserved.

from dace.frontend.fortran import fortran_parser

import dace.frontend.fortran.ast_transforms as ast_transforms
import dace.frontend.fortran.ast_internal_classes as ast_internal_classes
import numpy as np

def test_fortran_frontend_long_tasklet():
    """
    Tests that the Fortran frontend can parse array accesses and that the accessed indices are correct.
    """
    test_string = """
                    PROGRAM long_tasklet_test
                    implicit none

                    
                    type test_type
                        integer :: indices(5)
                        integer :: start
                        integer :: end
                    end type

                    double precision d(5)
                    double precision, dimension(5) :: arr
                    double precision, dimension(50:54) :: arr3
                    CALL long_tasklet_test_function(d)
                    end

                    SUBROUTINE long_tasklet_test_function(d)
                    double precision d(5)
                    double precision, dimension(50:54) :: arr4
                    double precision, dimension(5) :: arr
                    type(test_type) :: ind

                    arr(:)=2.0
                    ind%indices(:)=1
                    d(2)=5.5
                    d(1)=arr(1)*arr(ind%indices(1))!+arr(2,2,2)*arr(ind%indices(2,2,2),2,2)!+arr(3,3,3)*arr(ind%indices(3,3,3),3,3)

                    END SUBROUTINE long_tasklet_test_function
                    """
    sources={}
    sources["long_tasklet_test"]=test_string
    sdfg = fortran_parser.create_sdfg_from_string(test_string, "long_tasklet_test",sources=sources)
    sdfg.simplify(verbose=True)
    a = np.full([5], 42, order="F", dtype=np.float64)
    sdfg(d=a)
    assert (a[1] == 5.5)
    assert (a[0] == 4)
 
if __name__ == "__main__":

    test_fortran_frontend_long_tasklet()
