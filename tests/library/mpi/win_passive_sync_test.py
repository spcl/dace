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

    sdfg = dace.SDFG("mpi_win_passive_sync")
    window_state = sdfg.add_state("create_window")

    sdfg.add_array("lock_type", [1], dtype=dace.int32, transient=False)
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

    lock_state = sdfg.add_state("win_lock")

    sdfg.add_edge(window_state, lock_state, dace.InterstateEdge())

    lock_name = sdfg.add_rma_ops(window_name, "lock")
    win_lock_node = mpi.nodes.win_lock.Win_lock(lock_name, window_name)

    # pseudo access for ordering
    window_node = lock_state.add_access(window_name)
    window_desc = sdfg.arrays[window_name]

    lock_state.add_edge(window_node,
                        None,
                        win_lock_node,
                        None,
                        Memlet.from_array(window_name, window_desc))

    lock_type_node = lock_state.add_access("lock_type")

    target_rank_node = lock_state.add_access("target_rank")

    assertion_node = lock_state.add_access("assertion")

    lock_state.add_edge(lock_type_node,
                        None,
                        win_lock_node,
                        '_lock_type',
                        Memlet.simple(lock_type_node, "0:1", num_accesses=1))

    lock_state.add_edge(target_rank_node,
                        None,
                        win_lock_node,
                        '_rank',
                        Memlet.simple(target_rank_node, "0:1", num_accesses=1))

    lock_state.add_edge(assertion_node,
                        None,
                        win_lock_node,
                        '_assertion',
                        Memlet.simple(assertion_node, "0:1", num_accesses=1))

    _, scal = sdfg.add_scalar(lock_name, dace.int32, transient=True)
    wnode = lock_state.add_write(lock_name)
    lock_state.add_edge(win_lock_node,
                           "_out",
                           wnode,
                           None,
                           Memlet.from_array(lock_name, scal))

###############################################################################

    put_state = sdfg.add_state("win_put")

    sdfg.add_edge(lock_state, put_state, dace.InterstateEdge())

    put_name = sdfg.add_rma_ops(window_name, "put")
    win_put_node = mpi.nodes.win_put.Win_put(put_name, window_name)

    # pseudo access for ordering
    lock_node = put_state.add_access(lock_name)
    lock_desc = sdfg.arrays[lock_name]

    send_buffer = put_state.add_access("send_buffer")

    target_rank = put_state.add_access("target_rank")

    put_state.add_edge(lock_node,
                       None,
                       win_put_node,
                       "_in",
                       Memlet.from_array(lock_name, lock_desc))

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

    flush_state = sdfg.add_state("win_flush")

    sdfg.add_edge(put_state, flush_state, dace.InterstateEdge())

    flush_name = sdfg.add_rma_ops(window_name, "flush")
    win_flush_node = mpi.nodes.win_flush.Win_flush(flush_name, window_name)

    # pseudo access for ordering
    put_node = flush_state.add_access(put_name)
    put_desc = sdfg.arrays[put_name]

    flush_state.add_edge(put_node,
                         None,
                         win_flush_node,
                         None,
                         Memlet.from_array(put_name, put_desc))

    target_rank_node = flush_state.add_access("target_rank")

    flush_state.add_edge(target_rank_node,
                        None,
                        win_flush_node,
                        '_rank',
                        Memlet.simple(target_rank_node, "0:1", num_accesses=1))

    _, scal = sdfg.add_scalar(flush_name, dace.int32, transient=True)
    wnode = flush_state.add_write(flush_name)
    flush_state.add_edge(win_flush_node,
                         "_out",
                         wnode,
                         None,
                         Memlet.from_array(flush_name, scal))

###############################################################################

    unlock_state = sdfg.add_state("win_unlock")

    sdfg.add_edge(flush_state, unlock_state, dace.InterstateEdge())

    unlock_name = sdfg.add_rma_ops(window_name, "unlock")
    win_unlock_node = mpi.nodes.win_unlock.Win_unlock(unlock_name, window_name)

    # pseudo access for ordering
    flush_node = unlock_state.add_access(flush_name)
    flush_desc = sdfg.arrays[flush_name]

    unlock_state.add_edge(flush_node,
                         None,
                         win_unlock_node,
                         None,
                         Memlet.from_array(flush_name, flush_desc))

    target_rank_node = unlock_state.add_access("target_rank")

    unlock_state.add_edge(target_rank_node,
                        None,
                        win_unlock_node,
                        '_rank',
                        Memlet.simple(target_rank_node, "0:1", num_accesses=1))

    _, scal = sdfg.add_scalar(unlock_name, dace.int32, transient=True)
    wnode = unlock_state.add_write(unlock_name)
    unlock_state.add_edge(win_unlock_node,
                         "_out",
                         wnode,
                         None,
                         Memlet.from_array(unlock_name, scal))

# added these two fences as Barrier to ensure that every rank has completed
# since every rank are running independently
# some ranks might exit(since they completed) the transmission
# while others are still transmitting
###############################################################################

    fence_state_1 = sdfg.add_state("win_fence")

    sdfg.add_edge(unlock_state, fence_state_1, dace.InterstateEdge())

    fence_name_1 = sdfg.add_rma_ops(window_name, "fence")
    win_fence_node = mpi.nodes.win_fence.Win_fence(fence_name_1, window_name)

    # pseudo access for ordering
    unlock_node = fence_state_1.add_access(unlock_name)
    unlock_desc = sdfg.arrays[unlock_name]

    fence_state_1.add_edge(unlock_node,
                         None,
                         win_fence_node,
                         None,
                         Memlet.from_array(unlock_name, unlock_desc))

    assertion_node = fence_state_1.add_access("assertion")

    fence_state_1.add_edge(assertion_node,
                         None,
                         win_fence_node,
                         '_assertion',
                         Memlet.simple(assertion_node, "0:1", num_accesses=1))

    _, scal = sdfg.add_scalar(fence_name_1, dace.int32, transient=True)
    wnode = fence_state_1.add_write(fence_name_1)
    fence_state_1.add_edge(win_fence_node,
                         "_out",
                         wnode,
                         None,
                         Memlet.from_array(fence_name_1, scal))

###############################################################################

    fence_state_2 = sdfg.add_state("win_fence")

    sdfg.add_edge(fence_state_1, fence_state_2, dace.InterstateEdge())

    fence_name_2 = sdfg.add_rma_ops(window_name, "fence")
    win_fence_node = mpi.nodes.win_fence.Win_fence(fence_name_2, window_name)

    # pseudo access for ordering
    fence_node = fence_state_2.add_access(fence_name_1)
    fence_desc = sdfg.arrays[fence_name_1]

    fence_state_2.add_edge(fence_node,
                         None,
                         win_fence_node,
                         None,
                         Memlet.from_array(fence_name_1, fence_desc))

    assertion_node = fence_state_2.add_access("assertion")

    fence_state_2.add_edge(assertion_node,
                         None,
                         win_fence_node,
                         '_assertion',
                         Memlet.simple(assertion_node, "0:1", num_accesses=1))

    _, scal = sdfg.add_scalar(fence_name_2, dace.int32, transient=True)
    wnode = fence_state_2.add_write(fence_name_2)
    fence_state_2.add_edge(win_fence_node,
                         "_out",
                         wnode,
                         None,
                         Memlet.from_array(fence_name_2, scal))

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

    mpi_func = None
    for r in range(0, comm_size):
        if r == comm_rank:
            sdfg = make_sdfg(dtype)
            mpi_func = sdfg.compile()
        comm_world.Barrier()

    window_size = 10
    win_buffer = np.full(window_size, comm_rank, dtype=np_dtype)
    send_buffer = np.full(window_size, comm_rank, dtype=np_dtype)

    target_rank = np.array([(comm_rank + 1) % comm_size], dtype=np.int32)
    lock_type = np.full([1], MPI.LOCK_SHARED, dtype=np.int32)
    assertion = np.full([1], 0, dtype=np.int32)

    mpi_func(lock_type=lock_type,
             assertion=assertion,
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
