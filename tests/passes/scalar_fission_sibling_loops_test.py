# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Scalar fission of a temporary that sibling loops share inside a sequential outer loop.

CloudSC's horizontal ``jl`` loops sit inside the vertical ``jk`` loop, and its species unrolling
leaves several sibling loops writing one scalar temporary (``zevap_0``, ``zmelt_0``) before reading
it. The outer loop's back edge makes every write reach every read, so the write-shadow analysis
used to merge all the loops' scopes into one and nothing was renamed, which kept 19 CloudSC loops
from becoming maps.
"""
import numpy as np

import dace
from dace.sdfg.state import LoopRegion
from dace.transformation.interstate import LoopToMap
from dace.transformation.passes import PrivatizeScalars

M, N = 5, 33


def add_inner_loop(outer: LoopRegion, index: int, body) -> LoopRegion:
    inner = LoopRegion(f'inner{index}', f'i{index} < N', f'i{index}', f'i{index} = 0', f'i{index} = i{index} + 1')
    outer.add_node(inner, is_start_block=outer.number_of_nodes() == 0)
    body(inner, f'i{index}')
    return inner


def add_assign(state: dace.SDFGState, code: str, reads: dict, write: tuple) -> None:
    tasklet = state.add_tasklet('t', sorted(reads), ['y'], code)
    for conn, memlet in reads.items():
        state.add_edge(state.add_read(memlet.split('[')[0]), None, tasklet, conn, dace.Memlet(memlet))
    state.add_edge(tasklet, 'y', state.add_write(write[0]), None, dace.Memlet(write[1]))


def new_sdfg(name: str) -> dace.SDFG:
    sdfg = dace.SDFG(name)
    sdfg.add_symbol('M', dace.int64)
    sdfg.add_symbol('N', dace.int64)
    for array in ('a', 'o1', 'o2'):
        sdfg.add_array(array, ['M', 'N'], dace.float64)
    sdfg.add_scalar('z', dace.float64, transient=True)
    return sdfg


def sibling_loops_sharing_a_temporary() -> dace.SDFG:
    """``for k: {for i: z = a[k,i]*2; o1[k,i] = z + o1[k-1,i]}  {same with 3 into o2}``."""
    sdfg = new_sdfg('sibling_loops_sharing_a_temporary')
    outer = LoopRegion('outer', 'k < M', 'k', 'k = 1', 'k = k + 1')
    sdfg.add_node(outer, is_start_block=True)
    loops = []
    for index, (out, scale) in enumerate((('o1', 2.0), ('o2', 3.0))):

        def body(inner, it, out=out, scale=scale):
            write = inner.add_state('write', is_start_block=True)
            add_assign(write, f'y = x * {scale}', {'x': f'a[k, {it}]'}, ('z', 'z[0]'))
            read = inner.add_state_after(write, 'read')
            add_assign(read, 'y = x + p', {'x': 'z[0]', 'p': f'{out}[k - 1, {it}]'}, (out, f'{out}[k, {it}]'))

        loops.append(add_inner_loop(outer, index, body))
    outer.add_edge(loops[0], loops[1], dace.InterstateEdge())
    return sdfg


def temporary_carried_around_the_outer_loop() -> dace.SDFG:
    """``z = 0; for k: {for i: o1[k,i] = z + a[k,i]}  {for i: z = a[k,i]*3; o2[k,i] = z}``: the first
    loop reads the value the second loop left in the previous ``k`` iteration."""
    sdfg = new_sdfg('temporary_carried_around_the_outer_loop')
    init = sdfg.add_state('init', is_start_block=True)
    tasklet = init.add_tasklet('zero', [], ['y'], 'y = 0.0')
    init.add_edge(tasklet, 'y', init.add_write('z'), None, dace.Memlet('z[0]'))
    outer = LoopRegion('outer', 'k < M', 'k', 'k = 1', 'k = k + 1')
    sdfg.add_node(outer)
    sdfg.add_edge(init, outer, dace.InterstateEdge())

    def reader(inner, it):
        state = inner.add_state('read', is_start_block=True)
        add_assign(state, 'y = x + p', {'x': 'z[0]', 'p': f'a[k, {it}]'}, ('o1', f'o1[k, {it}]'))

    def writer(inner, it):
        write = inner.add_state('write', is_start_block=True)
        add_assign(write, 'y = x * 3.0', {'x': f'a[k, {it}]'}, ('z', 'z[0]'))
        read = inner.add_state_after(write, 'read')
        add_assign(read, 'y = x', {'x': 'z[0]'}, ('o2', f'o2[k, {it}]'))

    first, second = add_inner_loop(outer, 0, reader), add_inner_loop(outer, 1, writer)
    outer.add_edge(first, second, dace.InterstateEdge())
    return sdfg


def privatize_and_map(sdfg: dace.SDFG) -> list:
    sdfg.validate()
    PrivatizeScalars().apply_pass(sdfg, {})
    sdfg.apply_transformations_repeated(LoopToMap)
    sdfg.validate()
    return sorted(r.label for r in sdfg.all_control_flow_regions() if isinstance(r, LoopRegion))


def run(sdfg: dace.SDFG):
    a = np.random.default_rng(0).random((M, N))
    o1, o2 = np.zeros((M, N)), np.zeros((M, N))
    sdfg(a=a, o1=o1, o2=o2, M=M, N=N)
    return a, o1, o2


def test_sibling_loops_each_get_their_own_copy_of_a_shared_temporary():
    sdfg = sibling_loops_sharing_a_temporary()
    left = privatize_and_map(sdfg)
    assert left == ['outer'], f'the inner loops should map once each owns its temporary, left sequential: {left}'
    a, o1, o2 = run(sdfg)
    r1, r2 = np.zeros((M, N)), np.zeros((M, N))
    for k in range(1, M):
        r1[k] = a[k] * 2.0 + r1[k - 1]
        r2[k] = a[k] * 3.0 + r2[k - 1]
    np.testing.assert_allclose(o1, r1, rtol=1e-14, atol=0)
    np.testing.assert_allclose(o2, r2, rtol=1e-14, atol=0)


def test_a_temporary_carried_around_the_outer_loop_keeps_its_value():
    """The value crosses the outer back edge without passing any other write, so the scopes must
    stay merged: splitting them would leave the first loop reading a container nobody writes."""
    sdfg = temporary_carried_around_the_outer_loop()
    privatize_and_map(sdfg)
    a, o1, o2 = run(sdfg)
    r1, z = np.zeros((M, N)), 0.0
    for k in range(1, M):
        r1[k] = z + a[k]
        z = a[k, N - 1] * 3.0
    np.testing.assert_allclose(o1, r1, rtol=1e-14, atol=0)
    np.testing.assert_allclose(o2[1:], a[1:] * 3.0, rtol=1e-14, atol=0)
