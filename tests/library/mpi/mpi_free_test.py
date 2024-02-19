# Copyright 2019-2023 ETH Zurich and the DaCe authors. All rights reserved.
import dace
from dace.sdfg import utils
import dace.dtypes as dtypes
from dace.memlet import Memlet
import dace.libraries.mpi as mpi
import dace.frontend.common.distr as comm
import numpy as np
import pytest


###############################################################################


def make_sdfg(dtype):
    n = dace.symbol("n")

    sdfg = dace.SDFG("mpi_win_free")
    window_state = sdfg.add_state("create_window")

    sdfg.add_array("win_buffer", [n], dtype=dtype, transient=False)
    win_buffer = window_state.add_access("win_buffer")

    window_name = sdfg.add_window()
    win_create_node = mpi.nodes.win_create.Win_create(window_name)

    window_state.add_edge(win_buffer,
                   None,
                   win_create_node,
                   '_win_buffer',
                   Memlet.simple(win_buffer, "0:n", num_accesses=n))

    # for other nodes depends this window to connect
    _, scal = sdfg.add_scalar(window_name, dace.int32, transient=True)
    wnode = window_state.add_write(window_name)
    window_state.add_edge(win_create_node,
                          "_out",
                          wnode,
                          None,
                          Memlet.from_array(window_name, scal))

###############################################################################

    free_state = sdfg.add_state("win_free")

    sdfg.add_edge(window_state, free_state, dace.InterstateEdge())

    free_name = sdfg.add_rma_ops(window_name, "free")
    win_free_node = mpi.nodes.win_free.Win_free(free_name, window_name)

    # pseudo access for ordering
    window_node = free_state.add_access(window_name)
    window_desc = sdfg.arrays[window_name]

    free_state.add_edge(window_node,
                         None,
                         win_free_node,
                         "_in",
                         Memlet.from_array(window_name, window_desc))

    _, scal = sdfg.add_scalar(free_name, dace.int32, transient=True)
    wnode = free_state.add_write(free_name)
    free_state.add_edge(win_free_node,
                         "_out",
                         wnode,
                         None,
                         Memlet.from_array(free_name, scal))

    return sdfg


###############################################################################

@pytest.mark.parametrize("implementation, dtype", [
    pytest.param("MPI", dace.float32, marks=pytest.mark.mpi),
    pytest.param("MPI", dace.int32, marks=pytest.mark.mpi)
])
def test_win_free(dtype):
    from mpi4py import MPI
    np_dtype = getattr(np, dtype.to_string())
    comm_world = MPI.COMM_WORLD
    comm_rank = comm_world.Get_rank()
    comm_size = comm_world.Get_size()

    if comm_size < 2:
        raise ValueError("This test is supposed to be run with at least two processes!")

    mpi_func = None
    for r in range(0, comm_size):
        if r == comm_rank:
            sdfg = make_sdfg(dtype)
            mpi_func = sdfg.compile()
        comm_world.Barrier()

    window_size = 10
    win_buffer = np.arange(0, window_size, dtype=np_dtype)

    mpi_func(win_buffer=win_buffer, n=window_size)

if __name__ == "__main__":
    test_win_free(dace.int32)
    test_win_free(dace.float32)
