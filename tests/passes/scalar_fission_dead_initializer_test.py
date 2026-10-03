# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Scalar fission of a loop temporary whose only other access is a dead initializer before the loop.

The Python frontend needs ``t = 0.0`` before a loop whose branches assign ``t``; once the branches
fold, the loop rewrites ``t`` every iteration and the initializer is dead. With one dominating write
the pass used to skip the container, so the loop kept sharing ``t`` with the outer scope and
LoopToMap refused it.
"""
import numpy as np

import dace
from dace.sdfg.state import LoopRegion
from tests.passes.scalar_fission_sibling_loops_test import M, N, add_assign, add_inner_loop, new_sdfg, privatize_and_map


def loop_after_a_dead_initializer() -> dace.SDFG:
    """``z = 0; for k: for i: z = a[k,i]*2; o1[k,i] = z + o1[k-1,i]``."""
    sdfg = new_sdfg('loop_after_a_dead_initializer')
    init = sdfg.add_state('init', is_start_block=True)
    tasklet = init.add_tasklet('zero', [], ['y'], 'y = 0.0')
    init.add_edge(tasklet, 'y', init.add_write('z'), None, dace.Memlet('z[0]'))
    outer = LoopRegion('outer', 'k < M', 'k', 'k = 1', 'k = k + 1')
    sdfg.add_node(outer)
    sdfg.add_edge(init, outer, dace.InterstateEdge())

    def body(inner, it):
        write = inner.add_state('write', is_start_block=True)
        add_assign(write, 'y = x * 2.0', {'x': f'a[k, {it}]'}, ('z', 'z[0]'))
        read = inner.add_state_after(write, 'read')
        add_assign(read, 'y = x + p', {'x': 'z[0]', 'p': f'o1[k - 1, {it}]'}, ('o1', f'o1[k, {it}]'))

    add_inner_loop(outer, 0, body)
    return sdfg


def test_a_loop_temporary_is_split_from_a_dead_initializer_before_the_loop():
    sdfg = loop_after_a_dead_initializer()
    left = privatize_and_map(sdfg)
    assert left == ['outer'], f'the inner loop should map once it owns its temporary, left sequential: {left}'
    a = np.random.default_rng(0).random((M, N))
    o1 = np.zeros((M, N))
    sdfg(a=a, o1=o1, o2=np.zeros((M, N)), M=M, N=N)
    ref = np.zeros((M, N))
    for k in range(1, M):
        ref[k] = a[k] * 2.0 + ref[k - 1]
    np.testing.assert_allclose(o1, ref, rtol=1e-14, atol=0)
