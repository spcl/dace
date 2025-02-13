# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
import dace
from dace.memlet import Memlet
import dace.libraries.mpi as mpi
import numpy as np
import pytest

###############################################################################


def make_sdfg(dtype):

    n = dace.symbol("n")

    sdfg = dace.SDFG("mpi_allreduce")
    state = sdfg.add_state("dataflow")

    sdfg.add_array("inbuf", [n], dtype, transient=False)
    sdfg.add_array("outbuf", [n], dtype, transient=False)
    inbuf = state.add_access("inbuf")
    outbuf = state.add_access("outbuf")
    allreduce_node = mpi.nodes.allreduce.Allreduce("allreduce")

    state.add_memlet_path(inbuf,
                          allreduce_node,
                          dst_conn="_inbuffer",
                          memlet=Memlet.simple(inbuf, "0:n", num_accesses=n))
    state.add_memlet_path(allreduce_node,
                          outbuf,
                          src_conn="_outbuffer",
                          memlet=Memlet.simple(outbuf, "0:n", num_accesses=n))

    return sdfg


###############################################################################


@pytest.mark.parametrize("implementation, dtype", [
    pytest.param("MPI", dace.float32, marks=pytest.mark.mpi),
    pytest.param("MPI", dace.float64, marks=pytest.mark.mpi)
])
def test_mpi(implementation, dtype):
    from mpi4py import MPI as MPI4PY
    np_dtype = getattr(np, dtype.to_string())
    comm = MPI4PY.COMM_WORLD
    rank = comm.Get_rank()
    commsize = comm.Get_size()
    mpi_sdfg = None
    if commsize < 2:
        raise ValueError("This test is supposed to be run with at least two processes!")
    for r in range(0, commsize):
        if r == rank:
            sdfg = make_sdfg(dtype)
            mpi_sdfg = sdfg.compile()
        comm.Barrier()

    size = 8
    A = np.full(size, 1, dtype=np_dtype)
    B = np.full(size, 42, dtype=np_dtype)
    root = np.array([0], dtype=np.int32)
    mpi_sdfg(inbuf=A, outbuf=B, root=root, n=size)
    # now B should be an array of size, containing commsize
    if (not np.allclose(B, np.full(size, commsize, dtype=np_dtype))):
        raise (ValueError("The received values are not what I expected on root."))


###############################################################################

if __name__ == "__main__":
    test_mpi("MPI", dace.float32)
    test_mpi("MPI", dace.float64)

###############################################################################
