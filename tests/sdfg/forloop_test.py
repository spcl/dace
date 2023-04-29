# Copyright 2019-2023 ETH Zurich and the DaCe authors. All rights reserved.
""" Tests ForLoop, ForLoopEntry, and ForLoopExit nodes in SDFGs."""
import dace
import numpy as np


def test_simple_forloop():

    sdfg = dace.SDFG('simple_forloop_sdfg')
    sdfg.add_array('a', [1], dace.int32)

    state = sdfg.add_state('simple_forloop_state')
    loop = dace.nodes.ForLoop('simple_forloop', ['i'], dace.subsets.Range([(0, 9, 1)]))
    inp_acc = dace.nodes.AccessNode('a')
    le, lx = dace.nodes.ForLoopEntry(loop), dace.nodes.ForLoopExit(loop)
    task = dace.nodes.Tasklet('simple_task', {'__inp'}, {'__out'}, '__out = __inp + i')
    out_acc = dace.nodes.AccessNode('a')
    state.add_memlet_path(inp_acc, le, task, memlet=dace.Memlet('a[0]'), dst_conn='__inp')
    state.add_memlet_path(task, lx, out_acc, memlet=dace.Memlet('a[0]'), src_conn='__out')

    val = np.array([0], dtype=np.int32)
    sdfg(a=val)
    assert val[0] == (9 * 10) // 2


def test_sdfg_state_add_forloop():

    sdfg = dace.SDFG('simple_forloop_sdfg')
    sdfg.add_array('a', [1], dace.int32)

    state = sdfg.add_state('simple_forloop_state')
    inp_acc = dace.nodes.AccessNode('a')
    le, lx = state.add_forloop('simple_forloop', {'i': dace.subsets.Range([(0, 9, 1)])})
    task = dace.nodes.Tasklet('simple_task', {'__inp'}, {'__out'}, '__out = __inp + i')
    out_acc = dace.nodes.AccessNode('a')
    state.add_memlet_path(inp_acc, le, task, memlet=dace.Memlet('a[0]'), dst_conn='__inp')
    state.add_memlet_path(task, lx, out_acc, memlet=dace.Memlet('a[0]'), src_conn='__out')

    val = np.array([0], dtype=np.int32)
    sdfg(a=val)
    assert val[0] == (9 * 10) // 2


def test_sdfg_state_add_forlooped_tasklet():

    sdfg = dace.SDFG('simple_forloop_sdfg')
    sdfg.add_array('a', [1], dace.int32)

    state = sdfg.add_state('simple_forloop_state')
    state.add_forlooped_tasklet('simple_forloop', {'i': dace.subsets.Range([(0, 9, 1)])},
                                {'__inp': dace.Memlet('a[0]')},
                                '__out = __inp + i', {'__out': dace.Memlet('a[0]')},
                                external_edges=True)

    val = np.array([0], dtype=np.int32)
    sdfg(a=val)
    assert val[0] == (9 * 10) // 2


if __name__ == '__main__':
    test_simple_forloop()
    test_sdfg_state_add_forloop()
    test_sdfg_state_add_forlooped_tasklet()
