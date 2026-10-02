# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
""" A test for the ElementWiseArrayOperation transformation. """

import dace
import numpy as np
from dace.transformation.dataflow import ElementWiseArrayOperation, ElementWiseArrayOperation2D
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


H, W, Px, Py = (dace.symbol(s, dtype=dace.int64) for s in ('H', 'W', 'Px', 'Py'))


def _elementwise_2d_with_nested_body():
    """A 2D element-wise map whose body is a nested SDFG describing the whole containers."""
    body = dace.SDFG('body')
    body.add_array('a', [H, W], dace.float64)
    body.add_array('b', [H, W], dace.float64)
    body.add_symbol('i', dace.int64)
    body.add_symbol('j', dace.int64)
    bstate = body.add_state()
    tasklet = bstate.add_tasklet('t', {'x'}, {'y'}, 'y = x * 2')
    bstate.add_edge(bstate.add_read('a'), None, tasklet, 'x', dace.Memlet('a[i, j]'))
    bstate.add_edge(tasklet, 'y', bstate.add_write('b'), None, dace.Memlet('b[i, j]'))

    sdfg = dace.SDFG('eao2d_nested')
    sdfg.add_array('A', [H, W], dace.float64)
    sdfg.add_array('B', [H, W], dace.float64)
    state = sdfg.add_state()
    entry, exit_ = state.add_map('m', dict(i='0:H', j='0:W'))
    node = state.add_nested_sdfg(body, {'a'}, {'b'}, {'i': 'i', 'j': 'j'})
    state.add_memlet_path(state.add_read('A'), entry, node, dst_conn='a', memlet=dace.Memlet('A[i, j]'))
    state.add_memlet_path(node, exit_, state.add_write('B'), src_conn='b', memlet=dace.Memlet('B[i, j]'))
    return sdfg, state, node


def test_eao2d_nested_body_connectors_follow_the_block():
    """The map walks this rank's block, so the connectors below describe the block, not the array.

    Under the nested SDFG contract (see ``dace.sdfg.dealias.integrate_nested_sdfg``) a connector is
    the container it is connected to; leaving it describing the whole array would have the memlets
    inside stride over the array's rows instead of the block's.
    """
    sdfg, state, node = _elementwise_2d_with_nested_body()
    sdfg.validate()

    assert sdfg.apply_transformations(ElementWiseArrayOperation2D) == 1

    for edge in state.all_edges(node):
        connector = edge.dst_conn if edge.dst is node else edge.src_conn
        assert node.sdfg.arrays[connector].is_equivalent(sdfg.arrays[edge.data.data])
    sdfg.validate()


if __name__ == '__main__':
    test_eao_mpi()
    test_eao2d_nested_body_connectors_follow_the_block()
