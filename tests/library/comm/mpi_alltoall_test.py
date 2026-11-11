# Copyright 2019-2023 ETH Zurich and the DaCe authors. All rights reserved.
import numpy as np
import pytest

import dace
from dace.libraries import comm
from dace.memlet import Memlet

###############################################################################


def make_sdfg(dtype):

    n = dace.symbol("n")

    sdfg = dace.SDFG("mpi_alltoall")
    state = sdfg.add_state("dataflow")

    sdfg.add_array("inbuf", [n], dtype, transient=False)
    sdfg.add_array("outbuf", [n], dtype, transient=False)
    inbuf = state.add_access("inbuf")
    outbuf = state.add_access("outbuf")
    alltoall_node = comm.nodes.alltoall.Alltoall("alltoall")

    state.add_memlet_path(
        inbuf, alltoall_node, dst_conn="_inbuffer", memlet=Memlet.simple(inbuf, "0:n", num_accesses=n)
    )
    state.add_memlet_path(
        alltoall_node, outbuf, src_conn="_outbuffer", memlet=Memlet.simple(outbuf, "0:n", num_accesses=n)
    )

    return sdfg


###############################################################################


@pytest.mark.parametrize(
    "implementation, dtype",
    [
        pytest.param("MPI", dace.float32, marks=pytest.mark.mpi),
        pytest.param("MPI", dace.float64, marks=pytest.mark.mpi),
    ],
)
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

    size = 128
    size_per_proc = int(size / commsize)
    A = np.arange(0, size, dtype=np_dtype)
    B = np.full(size, 0, dtype=np_dtype)
    mpi_sdfg(inbuf=A, outbuf=B, n=size)

    # now B should be an array of size,
    # containing (size / size_per_proc) repeated chunked_data
    chunked_data = A[rank * size_per_proc : (rank + 1) * size_per_proc]
    correct_data = np.tile(chunked_data, int(size / size_per_proc))
    if not np.allclose(B, correct_data):
        raise (ValueError("The received values are not what I expected on root."))


def make_compound_extents_sdfg():
    sdfg = dace.SDFG("mpi_alltoall_compound_extents")
    state = sdfg.add_state("dataflow")
    shape = [dace.symbol("p"), dace.symbolic.pystr_to_symbolic("(m + 1) // 2 + 3")]
    for name in ("inbuf", "outbuf"):
        sdfg.add_array(name, shape, dace.float64)
    node = comm.nodes.alltoall.Alltoall("alltoall")
    whole = {name: Memlet.from_array(name, sdfg.arrays[name]) for name in ("inbuf", "outbuf")}
    state.add_memlet_path(state.add_access("inbuf"), node, dst_conn="_inbuffer", memlet=whole["inbuf"])
    state.add_memlet_path(node, state.add_access("outbuf"), src_conn="_outbuffer", memlet=whole["outbuf"])
    return sdfg


@pytest.mark.mpi
def test_mpi_counts_of_compound_extents():
    """Extents with a sum and a floor division: the count is one C++ expression of the whole buffer."""
    from mpi4py import MPI as MPI4PY

    comm = MPI4PY.COMM_WORLD
    rank, commsize = comm.Get_rank(), comm.Get_size()
    if commsize < 2:
        raise ValueError("This test is supposed to be run with at least two processes!")
    mpi_sdfg = None
    for r in range(commsize):
        if r == rank:
            mpi_sdfg = make_compound_extents_sdfg().compile()
        comm.Barrier()

    m = 5
    width = (m + 1) // 2 + 3
    A = np.arange(commsize * width, dtype=np.float64).reshape(commsize, width) + 100 * rank
    B = np.zeros((commsize, width))
    mpi_sdfg(inbuf=A, outbuf=B, p=commsize, m=m)
    expected = np.stack([np.arange(rank * width, (rank + 1) * width) + 100.0 * src for src in range(commsize)])
    assert np.array_equal(B, expected)


###############################################################################

if __name__ == "__main__":
    test_mpi("MPI", dace.float32)
    test_mpi("MPI", dace.float64)
    test_mpi_counts_of_compound_extents()

###############################################################################
