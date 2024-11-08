# Copyright 2023 ETH Zurich and the DaCe authors. All rights reserved.

from fparser.common.readfortran import FortranStringReader
from fparser.common.readfortran import FortranFileReader
from fparser.two.parser import ParserFactory
import sys, os
import numpy as np
import pytest


from dace import SDFG, SDFGState, nodes, dtypes, data, subsets, symbolic
from dace.frontend.fortran import fortran_parser
from fparser.two.symbol_table import SymbolTable
from dace.sdfg import utils as sdutil

import dace.frontend.fortran.ast_components as ast_components
import dace.frontend.fortran.ast_transforms as ast_transforms
import dace.frontend.fortran.ast_utils as ast_utils
import dace.frontend.fortran.ast_internal_classes as ast_internal_classes




def test_fortran_frontend_class():
    """
    Tests that whether clasess are translated correctly
    """
    test_string = """
                    PROGRAM class_test
                    
                    TYPE, ABSTRACT :: t_comm_pattern

  CONTAINS

    PROCEDURE(interface_setup_comm_pattern), DEFERRED :: setup
    PROCEDURE(interface_exchange_data_r3d), DEFERRED :: exchange_data_r3d
END TYPE t_comm_pattern

TYPE, EXTENDS(t_comm_pattern) :: t_comm_pattern_orig
   INTEGER :: n_pnts  ! Number of points we output into local array;
                      ! this may be bigger than n_recv due to
                      ! duplicate entries

   INTEGER, ALLOCATABLE :: recv_limits(:)

  CONTAINS

    PROCEDURE :: setup => setup_comm_pattern
    PROCEDURE :: exchange_data_r3d => exchange_data_r3d

END TYPE t_comm_pattern_orig



                    implicit none
                    integer d(2)
                    CALL class_test_function(d)
                    end


SUBROUTINE setup_comm_pattern(p_pat, dst_n_points)

    CLASS(t_comm_pattern_orig), TARGET, INTENT(OUT) :: p_pat

    INTEGER, INTENT(IN) :: dst_n_points        ! Total number of points

    p_pat%n_pnts = dst_n_points
  END SUBROUTINE setup_comm_pattern

  SUBROUTINE exchange_data_r3d(p_pat, recv)

    CLASS(t_comm_pattern_orig), TARGET, INTENT(INOUT) :: p_pat
    REAL, INTENT(INOUT), TARGET           :: recv(:,:,:)
  
  recv(1,1,1)=recv(1,1,1)+p_pat%n_pnts

  END SUBROUTINE exchange_data_r3d                    

                    SUBROUTINE class_test_function(d)
                    integer d(2)
                    real recv(2,2,2)

                    CLASS(t_comm_pattern_orig) :: p_pat

                    CALL setup_comm_pattern(p_pat, 42)
                    CALL exchange_data_r3d(p_pat, recv)
                   d(1)=p_pat%n_pnts               
                   END SUBROUTINE class_test_function
                    """
    sdfg = fortran_parser.create_sdfg_from_string(test_string, "class_test",False,False)
    for node, parent in sdfg.all_nodes_recursive():
        if isinstance(node, nodes.NestedSDFG):
            if node.sdfg is not None:
                if 'test_function' in node.sdfg.name:
                    sdfg = node.sdfg
                    break
    sdfg.parent = None
    sdfg.parent_sdfg = None
    sdfg.parent_nsdfg_node = None
    sdfg.reset_sdfg_list()                
    sdfg.simplify(verbose=True)
    sdfg.view()
    sdfg.compile()
    # sdfg = fortran_parser.create_sdfg_from_string(test_string, "int_init_test")
    # sdfg.simplify(verbose=True)
    # d = np.full([2], 42, order="F", dtype=np.int64)
    # sdfg(d=d)
    # assert (d[0] == 400)



if __name__ == "__main__":

 

    test_fortran_frontend_class()

