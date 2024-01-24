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
def test_comm_free():
    from mpi4py import MPI
    comm_world = MPI.COMM_WORLD
    comm_rank = comm_world.Get_rank()
    comm_size = comm_world.Get_size()

    if comm_size < 2:
        raise ValueError("Please run this test with at least two processes.")

    sdfg = dace.SDFG("mpi_free_test")
    start_state = sdfg.add_state("start")

    sdfg.add_scalar("color", dace.dtypes.int32, transient=False)
    sdfg.add_scalar("key", dace.dtypes.int32, transient=False)

    color = start_state.add_read("color")
    key = start_state.add_read("key")

    # color and key needs to be variable
    comm_name = sdfg.add_comm()
    comm_split_node = mpi.nodes.comm_split.Comm_split(comm_name)

    start_state.add_edge(color, None, comm_split_node, '_color', Memlet.simple(color, "0:1", num_accesses=1))
    start_state.add_edge(key, None, comm_split_node, '_key', Memlet.simple(key, "0:1", num_accesses=1))

    # Pseudo-writing for newast.py #3195 check and complete Processcomm creation
    _, scal = sdfg.add_scalar(comm_name, dace.int32, transient=True)
    wnode = start_state.add_write(comm_name)
    start_state.add_edge(comm_split_node, "_out", wnode, None, Memlet.from_array(comm_name, scal))

    main_state = sdfg.add_state("main")

    sdfg.add_edge(start_state, main_state, dace.InterstateEdge())

    comm_free_node = mpi.nodes.comm_free.Comm_free("_Comm_free_", comm_name)

    comm_node = main_state.add_read(comm_name)
    comm_desc = sdfg.arrays[comm_name]
    main_state.add_edge(comm_node, None, comm_free_node, "_in", Memlet.from_array(comm_name, comm_desc))

    func = utils.distributed_compile(sdfg, comm_world)

    # split world
    color = comm_rank % 2
    key = comm_rank

    func(color=color, key=key)


if __name__ == "__main__":
    test_comm_free()
