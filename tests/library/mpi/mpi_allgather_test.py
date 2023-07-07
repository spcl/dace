# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
import dace
from dace.memlet import Memlet
import dace.libraries.mpi as mpi
import numpy as np
import pytest

###############################################################################


def make_sdfg(dtype):

    n = dace.symbol("n")
    p = dace.symbol("p")

    sdfg = dace.SDFG("mpi_allgather")
    state = sdfg.add_state("dataflow")

    sdfg.add_array("inA", [n], dtype, transient=False)
    sdfg.add_array("outA", [n * p], dtype, transient=False)
    inA = state.add_access("inA")
    outA = state.add_access("outA")
    allgather_node = mpi.nodes.allgather.Allgather("allgather")

    state.add_memlet_path(inA,
                          allgather_node,
                          dst_conn="_inbuffer",
                          memlet=Memlet.simple(inA, "0:n", num_accesses=n))
    state.add_memlet_path(allgather_node,
                          outA,
                          src_conn="_outbuffer",
                          memlet=Memlet.simple(outA, "0:n*p", num_accesses=1))

    return sdfg


###############################################################################


def _test_mpi(info, sdfg, dtype):
    from mpi4py import MPI as MPI4PY
    comm = MPI4PY.COMM_WORLD
    rank = comm.Get_rank()
    commsize = comm.Get_size()
    mpi_sdfg = None
    if commsize < 2:
        raise ValueError("This test is supposed to be run with at least two processes!")
    for r in range(0, commsize):
        if r == rank:
            mpi_sdfg = sdfg.compile()
        comm.Barrier()

    size = 128
    A = np.full(size, 42, dtype=dtype)
    B = np.full(size * commsize, 23, dtype=dtype)
    mpi_sdfg(inA=A, outA=B, n=size, p=commsize)
    # now B should be an array of size*commsize, containing 42
    if not np.allclose(B, np.full(size * commsize, 42, dtype=dtype)):
        raise (ValueError("The received values are not what I expected."))


@pytest.mark.mpi
def test_mpi():
    _test_mpi("MPI Bcast", make_sdfg(np.float64), np.float64)


###############################################################################

###############################################################################

if __name__ == "__main__":
    test_mpi()
###############################################################################
