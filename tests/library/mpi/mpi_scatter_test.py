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

    sdfg = dace.SDFG("mpi_scatter")
    state = sdfg.add_state("dataflow")

    sdfg.add_array("inbuf", [n * p], dtype, transient=False)
    sdfg.add_array("outbuf", [n], dtype, transient=False)
    sdfg.add_array("root", [1], dace.dtypes.int32, transient=False)
    inbuf = state.add_access("inbuf")
    outbuf = state.add_access("outbuf")
    root = state.add_access("root")
    scatter_node = mpi.nodes.scatter.Scatter("scatter")

    state.add_memlet_path(inbuf,
                          scatter_node,
                          dst_conn="_inbuffer",
                          memlet=Memlet.simple(inbuf, "0:n*p", num_accesses=n))
    state.add_memlet_path(root, scatter_node, dst_conn="_root", memlet=Memlet.simple(root, "0:1", num_accesses=1))
    state.add_memlet_path(scatter_node,
                          outbuf,
                          src_conn="_outbuffer",
                          memlet=Memlet.simple(outbuf, "0:n", num_accesses=1))

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
    A = np.full(size * commsize, 7, dtype=np_dtype)
    B = np.full(size, 42, dtype=np_dtype)
    root = np.array([0], dtype=np.int32)
    mpi_sdfg(inbuf=A, outbuf=B, root=root, n=size, p=commsize)
    # now B should be an array of size, containing 0
    if not np.allclose(B, np.full(size, 7, dtype=np_dtype)):
        raise (ValueError("The received values are not what I expected."))


###############################################################################

N = dace.symbol('N', dtype=dace.int64)
P = dace.symbol('P', dtype=dace.int64)


@dace.program
def dace_scatter_gather(A: dace.float32[N * P]):
    tmp = np.empty_like(A, shape=[N])
    dace.comm.Scatter(A, tmp, root=0)
    tmp[:] = np.pi
    dace.comm.Gather(tmp, A, root=0)


@pytest.mark.mpi
def test_dace_scatter_gather():
    from mpi4py import MPI as MPI4PY
    comm = MPI4PY.COMM_WORLD
    rank = comm.Get_rank()
    commsize = comm.Get_size()
    mpi_sdfg = None
    if commsize < 2:
        raise ValueError("This test is supposed to be run with at least two processes!")
    for r in range(0, commsize):
        if r == rank:
            mpi_sdfg = dace_scatter_gather.compile()
        comm.Barrier()

    length = 128
    if rank == 0:
        A = np.full([length * commsize], np.pi, dtype=np.float32)
    else:
        A = np.random.randn(length * commsize).astype(np.float32)

    mpi_sdfg(A=A, N=length, P=commsize)

    if rank == 0:
        assert (np.allclose(A, np.full([length * commsize], np.pi, dtype=np.float32)))
    else:
        assert (True)

###############################################################################

if __name__ == "__main__":
    test_mpi("MPI", dace.float32)
    test_mpi("MPI", dace.float64)
    test_dace_scatter_gather()
###############################################################################
