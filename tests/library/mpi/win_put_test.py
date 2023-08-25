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

    sdfg = dace.SDFG("mpi_win_put")
    window_state = sdfg.add_state("create_window")

    sdfg.add_array("assertion", [1], dtype=dace.int32, transient=False)
    sdfg.add_array("win_buffer", [n], dtype=dtype, transient=False)
    sdfg.add_array("send_buffer", [n], dtype=dtype, transient=False)
    sdfg.add_array("target_rank", [1], dace.dtypes.int32, transient=False)

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

    fence_state_1 = sdfg.add_state("win_fence_1")

    sdfg.add_edge(window_state, fence_state_1, dace.InterstateEdge())

    fence_name = sdfg.add_rma_ops()
    win_fence_node = mpi.nodes.win_fence.Win_fence(fence_name, window_name)

    # pseudo access for ordering
    window_node = fence_state_1.add_access(window_name)
    window_desc = sdfg.arrays[window_name]

    fence_state_1.add_edge(window_node,
                         None,
                         win_fence_node,
                         None,
                         Memlet.from_array(window_name, window_desc))

    assertion_node = fence_state_1.add_access("assertion")

    fence_state_1.add_edge(assertion_node,
                           None,
                           win_fence_node,
                           '_assertion',
                           Memlet.simple(assertion_node, "0:1", num_accesses=1))

    _, scal = sdfg.add_scalar(fence_name, dace.int32, transient=True)
    wnode = fence_state_1.add_write(fence_name)
    fence_state_1.add_edge(win_fence_node,
                           "_out",
                           wnode,
                           None,
                           Memlet.from_array(fence_name, scal))

###############################################################################

    put_state = sdfg.add_state("win_put")

    sdfg.add_edge(fence_state_1, put_state, dace.InterstateEdge())

    put_name = sdfg.add_rma_ops()
    win_put_node = mpi.nodes.win_put.Win_put(put_name, window_name)

    # pseudo access for ordering
    fence_node = put_state.add_access(fence_name)
    fence_desc = sdfg.arrays[fence_name]

    send_buffer = put_state.add_access("send_buffer")
    
    target_rank = put_state.add_access("target_rank")

    put_state.add_edge(fence_node,
                       None,
                       win_put_node,
                       "_in",
                       Memlet.from_array(fence_name, fence_desc))

    put_state.add_edge(send_buffer,
                       None,
                       win_put_node,
                       "_inbuffer",
                       Memlet.simple(send_buffer, "0:n", num_accesses=n))

    put_state.add_edge(target_rank,
                       None,
                       win_put_node,
                       "_target_rank",
                       Memlet.simple(target_rank, "0:1", num_accesses=1))
    
    _, scal = sdfg.add_scalar(put_name, dace.int32, transient=True)
    wnode = put_state.add_write(put_name)
    put_state.add_edge(win_put_node,
                       "_out",
                       wnode,
                       None,
                       Memlet.from_array(put_name, scal))

###############################################################################

    fence_state_2 = sdfg.add_state("win_fence_2")

    sdfg.add_edge(put_state, fence_state_2, dace.InterstateEdge())

    fence_name = sdfg.add_rma_ops()
    win_fence_node = mpi.nodes.win_fence.Win_fence(fence_name, window_name)

    # pseudo access for ordering
    put_node = fence_state_2.add_access(put_name)
    put_desc = sdfg.arrays[put_name]

    fence_state_2.add_edge(put_node,
                         None,
                         win_fence_node,
                         None,
                         Memlet.from_array(put_name, put_desc))
    
    assertion_node = fence_state_2.add_access("assertion")

    fence_state_2.add_edge(assertion_node,
                         None,
                         win_fence_node,
                         '_assertion',
                         Memlet.simple(assertion_node, "0:1", num_accesses=1))

    _, scal = sdfg.add_scalar(fence_name, dace.int32, transient=True)
    wnode = fence_state_2.add_write(fence_name)
    fence_state_2.add_edge(win_fence_node,
                         "_out",
                         wnode,
                         None,
                         Memlet.from_array(fence_name, scal))

    return sdfg


###############################################################################

@pytest.mark.parametrize("implementation, dtype", [
    pytest.param("MPI", dace.float32, marks=pytest.mark.mpi),
    pytest.param("MPI", dace.int32, marks=pytest.mark.mpi)
])
def test_win_put(dtype):
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
    win_buffer = np.full(window_size, comm_rank, dtype=np_dtype)
    send_buffer = np.full(window_size, comm_rank, dtype=np_dtype)

    target_rank = np.array([(comm_rank + 1) % comm_size], dtype=np.int32)

    assertion = np.full([1], 0, dtype=np.int32)

    mpi_func(assertion=assertion,
             win_buffer=win_buffer,
             send_buffer=send_buffer,
             target_rank=target_rank,
             n=window_size)

    correct_data = np.full(window_size, (comm_rank - 1) % comm_size, dtype=np_dtype)
    if (not np.allclose(win_buffer, correct_data)):
        raise (ValueError("The received values are not what I expected on root."))

if __name__ == "__main__":
    test_win_put(dace.int32)
    test_win_put(dace.float32)
