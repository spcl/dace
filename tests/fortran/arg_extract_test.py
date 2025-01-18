# Copyright 2019-2023 ETH Zurich and the DaCe authors. All rights reserved.

import numpy as np

from dace.frontend.fortran import fortran_parser
from dace.frontend.fortran.fortran_parser import create_singular_sdfg_from_string
from tests.fortran.fortran_test_helper import SourceCodeBuilder


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

    sdfg = fortran_parser.create_sdfg_from_string(test_string, "arg_extract_test", normalize_offsets=True)
    sdfg.simplify(verbose=True)
    sdfg.compile()
    

    
    input = np.full([2], 42, order="F", dtype=np.float32)
    res = np.full([2], 42, order="F", dtype=np.float32)
    sdfg(d=input, res=res)
    assert np.allclose(res, [3,7])



def test_fortran_frontend_arg_extract3():
    sources, main = SourceCodeBuilder().add_file("""
subroutine main(d, res)
  implicit none
  real, dimension(2) :: d
  real, dimension(2) :: res
  integer :: jg
  logical, dimension(2) :: is_cloud
  jg = 1
  is_cloud(1) = .true.
  d(1) = 10
  d(2) = 20
  res(1) = merge(merge(d(1), d(2), d(1) < d(2) .and. is_cloud(jg)), 0.0, is_cloud(jg))
  res(2) = 52
end subroutine main
""").check_with_gfortran().get()

    sdfg = create_singular_sdfg_from_string(sources, 'main', True)
    #sdfg.simplify(verbose=True)
    sdfg.compile()
    

    
    input = np.full([2], 42, order="F", dtype=np.float32)
    res = np.full([2], 42, order="F", dtype=np.float32)
    sdfg(d=input, res=res)
    assert np.allclose(res, [10,52])    


def test_fortran_frontend_arg_extract4():
    test_string = """
                    PROGRAM arg_extract4
                    implicit none
                    real, dimension(2) :: d
                    real, dimension(2) :: res
                    CALL arg_extract4_test_function(d,res)
                    end

                    SUBROUTINE arg_extract4_test_function(d,res)
                    real, dimension(2) :: d
                    real, dimension(2) :: res
                    real :: merge_val
                    real :: merge_val2

                    integer :: jg
                    logical, dimension(2) :: is_cloud

                    jg = 1
                    is_cloud(1) = .true.
                    d(1)=10
                    d(2)=20
                    merge_val = MERGE(d(1), d(2), d(1) < d(2) .AND. is_cloud(jg))
                    merge_val2 = MERGE(merge_val, 0.0D0, is_cloud(jg))
                    res(1)=merge_val2
                    res(2) = 52

                    END SUBROUTINE arg_extract4_test_function
                    """

    sdfg = fortran_parser.create_sdfg_from_string(test_string, "arg_extract4_test", normalize_offsets=True)
    sdfg.simplify(verbose=True)
    sdfg.compile()
    

    
    input = np.full([2], 42, order="F", dtype=np.float32)
    res = np.full([2], 42, order="F", dtype=np.float32)
    sdfg(d=input, res=res)
    assert np.allclose(res, [10,52])       

if __name__ == "__main__":
    test_fortran_frontend_arg_extract()
    test_fortran_frontend_arg_extract3()
    test_fortran_frontend_arg_extract4()
          
