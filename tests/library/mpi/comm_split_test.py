# Copyright 2019-2022 ETH Zurich and the DaCe authors. All rights reserved.
import dace
from dace.sdfg import utils
import dace.dtypes as dtypes
from dace.memlet import Memlet
import dace.libraries.mpi as mpi
import dace.frontend.common.distr as comm
import numpy as np
import pytest


@pytest.mark.mpi
def test_comm_split():
    from mpi4py import MPI
    comm_world = MPI.COMM_WORLD
    comm_rank = comm_world.Get_rank()
    comm_size = comm_world.Get_size()

    if comm_size < 2:
        raise ValueError("Please run this test with at least two processes.")
    
    sdfg = dace.SDFG("mpi_comm_split")
    state = sdfg.add_state("start")

    sdfg.add_scalar("color", dace.dtypes.int32, transient=False)
    sdfg.add_scalar("key", dace.dtypes.int32, transient=False)
    sdfg.add_array("new_rank", [1], dtype=dace.int32, transient=False)
    sdfg.add_array("new_size", [1], dtype=dace.int32, transient=False)

    color = state.add_read("color")
    key = state.add_read("key")

    # color and key needs to be variable
    comm_name = sdfg.add_comm()
    comm_split_node = mpi.nodes.comm_split.Comm_split(comm_name)

    state.add_edge(color, None, comm_split_node, '_color', Memlet.simple(color, "0:1", num_accesses=1))
    state.add_edge(key, None, comm_split_node, '_key', Memlet.simple(key, "0:1", num_accesses=1))
    
    # Pseudo-writing for newast.py #3195 check and complete Processcomm creation
    _, scal = sdfg.add_scalar(comm_name, dace.int32, transient=True)
    wnode = state.add_write(comm_name)
    state.add_edge(comm_split_node, "_out", wnode, None, Memlet.from_array(comm_name, scal))

    state2 = sdfg.add_state("main")
    
    sdfg.add_edge(state, state2, dace.InterstateEdge())

    tasklet = state2.add_tasklet(
        "new_comm_get",
        {},
        {'_rank', '_size'},
        f"_rank = __state->{comm_name}_rank;\n_size = __state->{comm_name}_size;",
        dtypes.Language.CPP)

    new_rank = state2.add_write("new_rank")
    new_size = state2.add_write("new_size")

    state2.add_edge(tasklet, '_rank', new_rank, None, Memlet.simple(new_rank, "0:1", num_accesses=1))
    state2.add_edge(tasklet, '_size', new_size, None, Memlet.simple(new_size, "0:1", num_accesses=1))

    func = utils.distributed_compile(sdfg, comm_world)

    # split world
    color = comm_rank % 2
    key = comm_rank
    new_rank = np.zeros((1, ), dtype=np.int32)
    new_size = np.zeros((1, ), dtype=np.int32)

    func(color=color, key=key, new_rank=new_rank, new_size=new_size)

    correct_new_rank = np.arange(0, comm_size, dtype=np.int32) // 2
    assert (correct_new_rank[comm_rank] == new_rank[0])

    # reverse rank order
    color = 0
    key = comm_size - comm_rank
    new_rank = np.zeros((1, ), dtype=np.int32)
    new_size = np.zeros((1, ), dtype=np.int32)

    func(color=color, key=key, new_rank=new_rank, new_size=new_size)

    correct_new_rank = np.flip(np.arange(0, comm_size, dtype=np.int32), 0)
    assert (correct_new_rank[comm_rank] == new_rank[0])

if __name__ == "__main__":
    test_comm_split()
