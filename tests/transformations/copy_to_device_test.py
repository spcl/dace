# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""``CopyToDevice`` puts a copy of the data behind a nested SDFG's connectors.

The copy holds only the part of the container the edge selects and starts at its own origin. Under
the nested SDFG contract (see ``dace.sdfg.dealias.integrate_nested_sdfg``) a connector is the
container it is connected to and the memlets inside are written in that container's coordinates, so
both have to follow the copy.
"""
import numpy as np

import dace
from dace.transformation.dataflow import CopyToDevice

SHAPE = (4, 5)


def _elementwise_body():
    """``b[i, j] = 2 * a[i, j]``, with both connectors describing the whole container."""
    sdfg = dace.SDFG('body')
    sdfg.add_array('a', SHAPE, dace.float64)
    sdfg.add_array('b', SHAPE, dace.float64)
    sdfg.add_symbol('i', dace.int64)
    sdfg.add_symbol('j', dace.int64)
    state = sdfg.add_state()
    tasklet = state.add_tasklet('t', {'x'}, {'y'}, 'y = x * 2')
    state.add_edge(state.add_read('a'), None, tasklet, 'x', dace.Memlet('a[i, j]'))
    state.add_edge(tasklet, 'y', state.add_write('b'), None, dace.Memlet('b[i, j]'))
    return sdfg


def _mapped_nested_sdfg():
    """Two nested maps whose body is a nested SDFG reading ``A[i, j]`` and writing ``B[i, j]``."""
    sdfg = dace.SDFG('copy_to_device')
    sdfg.add_array('A', SHAPE, dace.float64)
    sdfg.add_array('B', SHAPE, dace.float64)
    state = sdfg.add_state()
    outer_entry, outer_exit = state.add_map('outer', dict(i='0:%d' % SHAPE[0]))
    inner_entry, inner_exit = state.add_map('inner', dict(j='0:%d' % SHAPE[1]))
    node = state.add_nested_sdfg(_elementwise_body(), {'a'}, {'b'}, {'i': 'i', 'j': 'j'})
    state.add_memlet_path(state.add_read('A'),
                          outer_entry,
                          inner_entry,
                          node,
                          dst_conn='a',
                          memlet=dace.Memlet('A[i, j]'))
    state.add_memlet_path(node,
                          inner_exit,
                          outer_exit,
                          state.add_write('B'),
                          src_conn='b',
                          memlet=dace.Memlet('B[i, j]'))
    return sdfg, state, node


def test_copy_to_device_moves_the_connectors_with_it():
    sdfg, state, node = _mapped_nested_sdfg()
    sdfg.validate()

    CopyToDevice.apply_to(sdfg, dict(storage=dace.StorageType.CPU_Heap), nested_sdfg=node, verify=False, save=False)

    for edge in state.all_edges(node):
        connector = edge.dst_conn if edge.dst is node else edge.src_conn
        assert edge.data.data.startswith('device_'), edge.data
        assert node.sdfg.arrays[connector].is_equivalent(sdfg.arrays[edge.data.data])
    sdfg.validate()

    A = np.arange(SHAPE[0] * SHAPE[1], dtype=np.float64).reshape(SHAPE).copy()
    B = np.zeros(SHAPE)
    sdfg(A=A, B=B)
    assert np.allclose(B, A * 2)


if __name__ == '__main__':
    test_copy_to_device_moves_the_connectors_with_it()
