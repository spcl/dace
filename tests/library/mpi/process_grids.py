import dace
from dace.codegen.compiled_sdfg import CompiledSDFG, ReloadableDLL
import dace.dtypes as dtypes
import dace.frontend.common.distr as comm
import numpy as np
import pytest


@pytest.mark.mpi
def test_process_grid():

    P = dace.symbol('P', dace.int32)

    sdfg = dace.SDFG("process_grid_test")
    sdfg.add_symbol('P', dace.int32)
    _, darr = sdfg.add_array("dims", (2, ), dtype=dace.int32)
    _, parr = sdfg.add_array("periods", (2, ), dtype=dace.int32)
    _, carr = sdfg.add_array("coords", (2, ), dtype=dace.int32)
    _, varr = sdfg.add_array("valid", (1, ), dtype=dace.bool_)

    state = sdfg.add_state("start")
    pgrid_name = comm._cart_create(None, sdfg, state, [1, P])

    state2 = sdfg.add_state("main")
    sdfg.add_edge(state, state2, dace.InterstateEdge())
    tasklet = state2.add_tasklet(
        "MPI_Cart_get", {}, {'d', 'p', 'c', 'v'},
        f"MPI_Cart_get(__state->{pgrid_name}_comm, P, d, p, c);\nv = __state->{pgrid_name}_valid;", dtypes.Language.CPP)
    dims = state2.add_write("dims")
    periods = state2.add_write("periods")
    coords = state2.add_write("coords")
    valid = state2.add_write("valid")
    state2.add_edge(tasklet, 'd', dims, None, dace.Memlet.from_array("dims", darr))
    state2.add_edge(tasklet, 'p', periods, None, dace.Memlet.from_array("periods", parr))
    state2.add_edge(tasklet, 'c', coords, None, dace.Memlet.from_array("coords", carr))
    state2.add_edge(tasklet, 'v', valid, None, dace.Memlet("valid[0]"))

    from mpi4py import MPI
    commworld = MPI.COMM_WORLD
    rank = commworld.Get_rank()
    size = commworld.Get_size()

    if size < 2:
        raise ValueError("Please run this test with at least two processes.")

    if rank == 0:
        func = sdfg.compile()
    commworld.Barrier()
    if rank > 0:
        func = CompiledSDFG(sdfg, ReloadableDLL(".dacecache/{n}/build/lib{n}.so".format(n=sdfg.name), sdfg.name))
    commworld.Barrier()

    dims = np.zeros((2, ), dtype=np.int32)
    periods = np.zeros((2, ), dtype=np.int32)
    coords = np.zeros((2, ), dtype=np.int32)
    valid = np.zeros((1, ), dtype=np.bool_)
    func(dims=dims, periods=periods, coords=coords, valid=valid, P=size)

    assert (np.array_equal(dims, [1, size]))
    assert (np.array_equal(periods, [0, 0]))
    assert (np.array_equal(coords, [0, rank]))
    assert (valid[0])


if __name__ == "__main__":
    test_process_grid()
