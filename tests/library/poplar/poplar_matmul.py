# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
import dace
from dace.memlet import Memlet
import dace.libraries.poplar as poplar
import numpy as np
import pytest

###############################################################################

def make_sdfg(dtype):

    sdfg = dace.SDFG("poplar_matmul")
    state = sdfg.add_state("matmul_state")

    sdfg.add_array('A', [10], dtype)
    sdfg.add_array('B', [10], dtype)  
    sdfg.add_array('C', [10], dtype)
    
    a = state.add_access("A")
    b = state.add_access("B")
    c = state.add_access("C")
    
    poplar_mm_node = poplar.nodes.popmm.IPUMatMul("MATMUL")
    poplar_mm_node.implementation = "MM"
    
    state.add_memlet_path(a, poplar_mm_node, dst_conn="_inbufferA", memlet=dace.Memlet(f"A"))
    state.add_memlet_path(b, poplar_mm_node, dst_conn="_inbufferB", memlet=dace.Memlet(f"B"))
    state.add_memlet_path(poplar_mm_node, c, src_conn="_outbufferC", memlet=dace.Memlet(f"C"))

    return sdfg


###############################################################################


# def _test_poplar(info, sdfg, dtype):
    
#     poplar_sdfg = sdfg.compile()



@pytest.mark.poplar
def test_poplar():
    sdfg = make_sdfg(np.float64)
    sdfg.compile()
    print("Success!")

###############################################################################

# N = dace.symbol('N', dtype=dace.int64)


# @dace.program
# def dace_bcast(A: dace.float32[N]):
#     dace.comm.Bcast(A, root=0)


# @pytest.mark.mpi
# def test_dace_bcast():
#     from mpi4py import MPI as MPI4PY
#     comm = MPI4PY.COMM_WORLD
#     rank = comm.Get_rank()
#     commsize = comm.Get_size()
#     mpi_sdfg = None
#     if commsize < 2:
#         raise ValueError("This test is supposed to be run with at least two processes!")
#     for r in range(0, commsize):
#         if r == rank:
#             mpi_sdfg = dace_bcast.compile()
#         comm.Barrier()

#     length = 128
#     if rank == 0:
#         A = np.full([length], np.pi, dtype=np.float32)
#     else:
#         A = np.random.randn(length).astype(np.float32)

#     mpi_sdfg(A=A, N=length)

#     assert (np.allclose(A, np.full([length], np.pi, dtype=np.float32)))


###############################################################################

if __name__ == "__main__":
    test_poplar()
    # test_dace_bcast()
###############################################################################
