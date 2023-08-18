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

    sdfg = dace.SDFG("mpi_win_create")
    state = sdfg.add_state("start")

    sdfg.add_array("win_buffer", [n], dtype=dtype, transient=False)
    win_buffer = state.add_access("win_buffer")

    window_name = sdfg.add_window()
    win_create_node = mpi.nodes.win_create.Win_create(window_name)

    state.add_edge(win_buffer,
                   None,
                   win_create_node,
                   '_win_buffer',
                   Memlet.simple(win_buffer, "0:n", num_accesses=n))
    
    # for other nodes depends this window to connect
    _, scal = sdfg.add_scalar(window_name, dace.int32, transient=True)
    wnode = state.add_write(window_name)
    state.add_edge(win_create_node,
                   "_out",
                   wnode,
                   None,
                   Memlet.from_array(window_name, scal))

    return sdfg


###############################################################################


@pytest.mark.parametrize("implementation, dtype", [
    pytest.param("MPI", dace.float32, marks=pytest.mark.mpi),
    pytest.param("MPI", dace.int32, marks=pytest.mark.mpi)
])
def test_win_create(dtype):
    from mpi4py import MPI
    np_dtype = getattr(np, dtype.to_string())
    comm_world = MPI.COMM_WORLD
    comm_rank = comm_world.Get_rank()
    comm_size = comm_world.Get_size()

    if comm_size < 2:
        raise ValueError("This test is supposed to be run with at least two processes!")
    
    sdfg = make_sdfg(dtype)
    mpi_func = utils.distributed_compile(sdfg, comm_world)

    window_size = 10
    win_buffer = np.arange(0, window_size, dtype=np_dtype)

    mpi_func(win_buffer=win_buffer, n=window_size)


###############################################################################


if __name__ == "__main__":
    test_win_create(dace.float32)
    test_win_create(dace.int32)
