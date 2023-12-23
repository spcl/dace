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

  PRIVATE

   ! Number of points we receive in communication,
   ! this is the same as recv_limits

   INTEGER :: n_recv  ! Number of points we receive from other PEs
   INTEGER :: n_pnts  ! Number of points we output into local array;
                      ! this may be bigger than n_recv due to
                      ! duplicate entries
   INTEGER :: n_send  ! Number of points we send to other PEs

   INTEGER :: np_recv ! Number of PEs from which data have to be received
   INTEGER :: np_send ! Number of PEs to which data have to be sent

   !> which communicator to apply this pattern to
   INTEGER :: comm

   ! "recv_limits":
   !
   ! All data that is received from PE np is buffered in the receive
   ! buffer between start index "p_pat%recv_limits(np)+1" and the end
   ! index "p_pat%recv_limits(np+1)".
   INTEGER, ALLOCATABLE :: recv_limits(:)

   ! "recv_src", "recv_dst_blk/idx":
   !
   ! For all points i=1,n_pnts the data received at index recv_src(i)
   ! in the receiver buffer is copied to the destination array at
   ! position recv_dst_idx/blk(i)
   INTEGER, ALLOCATABLE :: recv_src(:)
   INTEGER, ALLOCATABLE :: recv_dst_blk(:)
   INTEGER, ALLOCATABLE :: recv_dst_idx(:)

   ! "send_limits":
   !
   ! All data that is sent to PE np is buffered by the local PE in the
   ! send buffer between start index "p_pat%send_limits(np)+1" and the
   ! end index "p_pat%send_limits(np+1)".
   INTEGER, ALLOCATABLE :: send_limits(:)

   ! "send_src_idx/blk":
   !
   ! For all points i=1,n_send the data in the send buffer at the ith
   ! position is copied from the source array at position
   ! send_src_idx/blk(i)
   INTEGER, ALLOCATABLE :: send_src_blk(:)
   INTEGER, ALLOCATABLE :: send_src_idx(:)

   ! "pelist_send", "pelist_recv":
   !
   ! list of PEs where to send the data to, and from where to receive
   ! the data
   INTEGER, ALLOCATABLE :: pelist_send(:)
   INTEGER, ALLOCATABLE :: pelist_recv(:)

   ! "send_startidx", "send_count":
   !
   ! The local PE sends send_count(i) data items to PE pelist_send(i),
   ! starting at send_startidx(i) in the send buffer.
   INTEGER, ALLOCATABLE :: send_startidx(:)
   INTEGER, ALLOCATABLE :: send_count(:)

   ! "recv_startidx", "recv_count":
   !
   ! The local PE recvs recv_count(i) data items from PE pelist_recv(i),
   ! starting at recv_startidx(i) in the receiver buffer.
   INTEGER, ALLOCATABLE :: recv_startidx(:)
   INTEGER, ALLOCATABLE :: recv_count(:)

  CONTAINS

    PROCEDURE :: setup => setup_comm_pattern
    PROCEDURE :: exchange_data_r3d => exchange_data_r3d

END TYPE t_comm_pattern_orig



                    implicit none
                    integer d(2)
                    CALL class_test_function(d)
                    end


SUBROUTINE setup_comm_pattern(p_pat, dst_n_points, dst_owner, &
                                dst_global_index, send_glb2loc_index, &
                                src_n_points, src_owner, src_global_index, &
                                inplace, comm)

    CLASS(t_comm_pattern_orig), TARGET, INTENT(OUT) :: p_pat

    INTEGER, INTENT(IN) :: dst_n_points        ! Total number of points
    INTEGER, INTENT(IN) :: dst_owner(:)        ! Owner of every point
    INTEGER, INTENT(IN) :: dst_global_index(:) ! Global index of every point
    TYPE(t_glb2loc_index_lookup), INTENT(IN) :: send_glb2loc_index
                                               ! global to local index
                                               ! lookup information
                                               ! of the SENDER array
    INTEGER, INTENT(IN) :: src_n_points        ! Total number of points
    INTEGER, INTENT(IN) :: src_owner(:)        ! Owner of every point
    INTEGER, INTENT(IN) :: src_global_index(:) ! Global index of every point

    LOGICAL, OPTIONAL, INTENT(IN) :: inplace
    INTEGER, OPTIONAL, INTENT(in) :: comm

    


  END SUBROUTINE setup_comm_pattern

  SUBROUTINE exchange_data_r3d(p_pat, recv, send, add)

    CLASS(t_comm_pattern_orig), TARGET, INTENT(INOUT) :: p_pat
    REAL(dp), INTENT(INOUT), TARGET           :: recv(:,:,:)
    REAL(dp), INTENT(IN), OPTIONAL, TARGET    :: send(:,:,:)
    REAL(dp), INTENT(IN), OPTIONAL, TARGET    :: add (:,:,:)


  END SUBROUTINE exchange_data_r3d                    

                    SUBROUTINE class_test_function(d)
                    integer d(2)
                    CLASS(t_comm_pattern_orig) :: p_pat
                    p_pat%src_n_points=12
                   d(1)=p_pat%src_n_points               
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

