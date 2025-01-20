# Copyright 2019-2023 ETH Zurich and the DaCe authors. All rights reserved.

from fparser.common.readfortran import FortranStringReader
from fparser.common.readfortran import FortranFileReader
from fparser.two.parser import ParserFactory
import sys, os
import numpy as np
import pytest

from dace import SDFG, SDFGState, nodes, dtypes, data, subsets, symbolic
from dace.frontend.fortran import fortran_parser
from fparser.two.symbol_table import SymbolTable

from dace.frontend.fortran.fortran_parser import create_singular_sdfg_from_string
from dace.sdfg import utils as sdutil

import dace.frontend.fortran.ast_components as ast_components
import dace.frontend.fortran.ast_transforms as ast_transforms
import dace.frontend.fortran.ast_utils as ast_utils
import dace.frontend.fortran.ast_internal_classes as ast_internal_classes

from dace.transformation.passes.lift_struct_views import LiftStructViews
from dace.transformation import pass_pipeline as ppl
from tests.fortran.fortran_test_helper import SourceCodeBuilder


def test_fortran_frontend_missing_func():
    """
    Tests that the Fortran frontend can parse the simplest type declaration and make use of it in a computation.
    """
    test_string = """
        PROGRAM missing_test
            implicit none


            REAL :: d(5,5)

            CALL missing_test_function(d)            
        end

        
  SUBROUTINE missing_test_function(d)
                    REAL d(5,5)
                    REAL z(5)
                    
                    CALL init_zero_contiguous_dp(z, 5, opt_acc_async=.TRUE.,lacc=.FALSE.)
                    d(2,1) = 5.5 + z(1)

    END SUBROUTINE missing_test_function     
       
  SUBROUTINE init_contiguous_dp(var, n, v, opt_acc_async, lacc)
    INTEGER, INTENT(in) :: n
    REAL, INTENT(out) :: var(n)
    REAL, INTENT(in) :: v
    LOGICAL, INTENT(in), OPTIONAL :: opt_acc_async
    LOGICAL, INTENT(in), OPTIONAL :: lacc

    INTEGER :: i
    LOGICAL :: lzacc

    CALL set_acc_host_or_device(lzacc, lacc)

    DO i = 1, n
      var(i) = v
    END DO

    CALL acc_wait_if_requested(1, opt_acc_async)
  END SUBROUTINE init_contiguous_dp

  SUBROUTINE init_zero_contiguous_dp(var, n, opt_acc_async, lacc)
    INTEGER, INTENT(in) :: n
    REAL, INTENT(out) :: var(n)
    LOGICAL, INTENT(IN), OPTIONAL :: opt_acc_async
    LOGICAL, INTENT(IN), OPTIONAL :: lacc
    
    
    CALL init_contiguous_dp(var, n, 0.0, opt_acc_async, lacc)
    var(1)=var(1)+1.0

  END SUBROUTINE init_zero_contiguous_dp

  
    SUBROUTINE set_acc_host_or_device(lzacc, lacc)
    LOGICAL, INTENT(out) :: lzacc
    LOGICAL, INTENT(in), OPTIONAL :: lacc

    lzacc = .FALSE.

     END SUBROUTINE set_acc_host_or_device

  SUBROUTINE acc_wait_if_requested(acc_async_queue, opt_acc_async)
    INTEGER, INTENT(IN) :: acc_async_queue
    LOGICAL, INTENT(IN), OPTIONAL :: opt_acc_async


  END SUBROUTINE acc_wait_if_requested
    """
    sources = {}
    sources["missing_test"] = test_string
    sdfg = fortran_parser.create_sdfg_from_string(test_string, "missing_test", True, sources=sources)
    sdfg.simplify(verbose=True)
    a = np.full([5, 5], 42, order="F", dtype=np.float32)
    sdfg(d=a)
    assert (a[0, 0] == 42)
    assert (a[1, 0] == 6.5)
    assert (a[2, 0] == 42)


def test_fortran_frontend_missing_extraction():
    """
    Tests that the Fortran frontend can parse the simplest type declaration and make use of it in a computation.
    """
    sources, main = SourceCodeBuilder().add_file("""
subroutine main(d)
  real d(5, 5)
  real z(5)
  integer :: jk = 5
  integer :: nrdmax_jg = 3
  do jk = max(0, nrdmax_jg - 2), 2
    d(jk, jk) = 17
  end do
  d(2, 1) = 5.5
end subroutine main
""").check_with_gfortran().get()
    sdfg = create_singular_sdfg_from_string(sources, 'main', )
    sdfg.simplify(verbose=True)
    a = np.full([5, 5], 42, order="F", dtype=np.float32)
    sdfg(d=a)
    assert (a[0, 0] == 17)
    assert (a[1, 0] == 5.5)
    assert (a[2, 0] == 42)


if __name__ == "__main__":
    test_fortran_frontend_missing_func()
    test_fortran_frontend_missing_extraction()
