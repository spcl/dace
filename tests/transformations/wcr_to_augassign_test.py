# Copyright 2019-2024 ETH Zurich and the DaCe authors. All rights reserved.
""" Tests WCRToAugAssign. """

import dace
import numpy as np
from dace.transformation.dataflow import WCRToAugAssign


def test_tasklet():

    @dace.program
    def test():
        a = np.zeros((10, ))
        for i in dace.map[1:9]:
            a[i - 1] += 1
        return a

    sdfg = test.to_sdfg(simplify=False)
    sdfg.apply_transformations(WCRToAugAssign)

    val = sdfg()
    ref = test.f()
    assert (np.allclose(val, ref))


def test_mapped_tasklet():

    @dace.program
    def test():
        a = np.zeros((10, ))
        for i in dace.map[1:9]:
            a[i - 1] += 1
        return a

    sdfg = test.to_sdfg(simplify=True)
    sdfg.apply_transformations(WCRToAugAssign)

    val = sdfg()
    ref = test.f()
    assert (np.allclose(val, ref))


def test_noncommutative_operand_order():
    """A subtraction WCR ``a[i] = a[i] - v[i]`` must lower to ``old - new``, not
    ``new - old``. Binding the WCR operands by argument name (not body position)
    keeps this correct; the prior position-based wiring silently produced
    ``new - old`` for non-commutative ops. The write is injective (``a[i]``), so
    the conversion's soundness gate allows it inside the parallel Map.
    """
    N = 16
    sdfg = dace.SDFG('wcr_sub_order')
    sdfg.add_array('a', [N], dace.float64)
    sdfg.add_array('v', [N], dace.float64)
    state = sdfg.add_state()
    me, mx = state.add_map('m', dict(i=f'0:{N}'))
    tasklet = state.add_tasklet('t', {'inp'}, {'out'}, 'out = inp')
    v_node = state.add_read('v')
    a_node = state.add_write('a')
    state.add_memlet_path(v_node, me, tasklet, dst_conn='inp', memlet=dace.Memlet('v[i]'))
    state.add_memlet_path(tasklet,
                          mx,
                          a_node,
                          src_conn='out',
                          memlet=dace.Memlet(data='a', subset='i', wcr='lambda a, b: a - b'))

    rng = np.random.default_rng(0)
    a0 = rng.random(N)
    v0 = rng.random(N)
    ref = a0 - v0  # a[i] = a[i] - v[i]

    applied = sdfg.apply_transformations(WCRToAugAssign)
    assert applied == 1
    assert all(e.data.wcr is None for s in sdfg.all_states() for e in s.edges())

    a = a0.copy()
    sdfg(a=a, v=v0.copy())
    assert np.allclose(a, ref), f"operand order wrong: got {a[:3]} expected {ref[:3]}"


if __name__ == '__main__':
    test_tasklet()
    test_mapped_tasklet()
    test_noncommutative_operand_order()
