# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
""" A test for the ElementWiseArrayOperation transformation. """

import dace
import numpy as np
from dace.transformation.dataflow import ElementWiseArrayOperation
import pytest

N = dace.symbol('N', dtype=dace.int64)


@dace.program
def eao_mpi(A: dace.float64[N], B: dace.float64[N]):
    return A * B


@pytest.mark.mpi
def test_eao_mpi():
    from mpi4py import MPI as MPI4PY
    comm = MPI4PY.COMM_WORLD
    rank = comm.Get_rank()
    commsize = comm.Get_size()
    mpi_sdfg = None
    if commsize < 2:
        raise ValueError("This test is supposed to be run with at least two processes!")
    for r in range(0, commsize):
        if r == rank:
            mpi_sdfg = eao_mpi.to_sdfg(simplify=True)
            mpi_sdfg.apply_transformations(ElementWiseArrayOperation)
            mpi_exec = mpi_sdfg.compile()
        comm.Barrier()

    length = 128 * commsize
    A = np.random.randn(length)
    B = np.random.randn(length)
    C = mpi_exec(A=A, B=B, N=length, commsize=commsize)
    if rank == 0:
        assert (np.allclose(C, A * B))
    else:
        assert (True)


def test_ordering_edge_into_map_exit_is_not_a_write():
    """The MapExit scan must skip ordering edges instead of looking up ``arrays[None]``."""
    sdfg = dace.SDFG('eao_ordering')
    sdfg.add_array('A', [8], dace.float64)
    sdfg.add_array('B', [8], dace.float64)
    state = sdfg.add_state()

    me, mx = state.add_map('m', dict(i='0:8'))
    tasklet = state.add_tasklet('t', {'a': None}, {'b': None}, 'b = a * 2.0')
    state.add_memlet_path(state.add_access('A'), me, tasklet, dst_conn='a', memlet=dace.Memlet('A[i]'))
    state.add_memlet_path(tasklet, mx, state.add_access('B'), src_conn='b', memlet=dace.Memlet('B[i]'))
    side = state.add_tasklet('side', {}, {}, 'pass')
    state.add_nedge(me, side, dace.Memlet())
    state.add_nedge(side, mx, dace.Memlet())
    sdfg.validate()

    xform = ElementWiseArrayOperation()
    xform.setup_match(sdfg, sdfg.cfg_id, sdfg.node_id(state), {ElementWiseArrayOperation.map_entry: state.node_id(me)},
                      0)
    assert xform.can_be_applied(state, 0, sdfg) is True  # used to raise KeyError: None


if __name__ == '__main__':
    test_eao_mpi()
    test_ordering_edge_into_map_exit_is_not_a_write()
