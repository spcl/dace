# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Soundness strengthening for :class:`~dace.transformation.passes.buffer_expansion.BufferExpansion`.

``_defined_before_read`` credits a buffer as loop-private only when the writes that provably precede
a read TOGETHER cover the read region. The coverage test must be *precise*: a write that merely
shares a bounding box with the read (a STRIDED write ``buf[0:2*H:2]`` touches only the even slots but
its bounding box spans all of ``buf[0:2*H]``) leaves the untouched slots carrying whatever a previous
iteration left there. Crediting such a write privatises a genuinely loop-carried buffer, and every
iteration then reads its own never-written slice instead of the carried value.
"""
import sys

import numpy as np
import pytest
from numpy.testing import assert_array_equal

import dace
from dace.properties import CodeBlock
from dace.sdfg.state import ConditionalBlock, ControlFlowRegion, LoopRegion
from dace.transformation.passes.buffer_expansion import BufferExpansion

N = dace.symbol('N', nonnegative=True)
H = dace.symbol('H', nonnegative=True)


def _carried_strided_sdfg():
    """``buf``'s EVEN slots are refilled every iteration; its ODD slots are seeded once (at ``i == 0``)
    and then carried across every later iteration.

    Per iteration: ``buf[0:2*H:2] = 2*a[i, :]`` (a strided write whose bounding box spans all of
    ``buf``) then ``out[i] = sum(buf[0:2*H])``. The odd slots are written only by the ``i == 0``
    branch, so from ``i >= 1`` the read observes a value produced by an EARLIER iteration -- ``buf``
    is loop-carried and must NOT be privatised.
    """
    sdfg = dace.SDFG('carried_strided')
    sdfg.add_array('a', [N, H], dace.float64)
    sdfg.add_array('out', [N], dace.float64)
    sdfg.add_transient('buf', [2 * H], dace.float64)
    sdfg.add_transient('half', [H], dace.float64)
    loop = LoopRegion('lp', 'i < N', 'i', 'i = 0', 'i = i + 1')
    sdfg.add_node(loop, is_start_block=True)

    # Seed the odd slots with 7.0, on iteration 0 only -- they carry that value ever after.
    cb = ConditionalBlock('seed')
    loop.add_node(cb, is_start_block=True)
    branch = ControlFlowRegion('branch', sdfg=sdfg)
    cb.add_branch(CodeBlock('i == 0'), branch)
    ss = branch.add_state('seed_s', is_start_block=True)
    sh, sb = ss.add_access('half'), ss.add_access('buf')
    sme, smx = ss.add_map('sm', dict(j='0:H'))
    stk = ss.add_tasklet('s', {}, {'y'}, 'y = 7.0')
    ss.add_edge(sme, None, stk, None, dace.Memlet())
    ss.add_memlet_path(stk, smx, sh, src_conn='y', memlet=dace.Memlet('half[j]'))
    ss.add_edge(sh, None, sb, None, dace.Memlet(data='buf', subset='1:2*H:2', other_subset='0:H'))

    # Unconditional STRIDED write of the EVEN slots only; dominates the read below.
    ws = loop.add_state('wr')
    loop.add_edge(cb, ws, dace.InterstateEdge())
    ar, hw, bw = ws.add_read('a'), ws.add_access('half'), ws.add_access('buf')
    me, mx = ws.add_map('wm', dict(j='0:H'))
    t = ws.add_tasklet('w', {'x'}, {'y'}, 'y = 2.0 * x')
    ws.add_memlet_path(ar, me, t, dst_conn='x', memlet=dace.Memlet('a[i, j]'))
    ws.add_memlet_path(t, mx, hw, src_conn='y', memlet=dace.Memlet('half[j]'))
    ws.add_edge(hw, None, bw, None, dace.Memlet(data='buf', subset='0:2*H:2', other_subset='0:H'))

    # Unconditional read of the WHOLE buffer: fresh evens + carried odds.
    rs = loop.add_state('rd')
    loop.add_edge(ws, rs, dace.InterstateEdge())
    br, ow = rs.add_read('buf'), rs.add_access('out')
    rme, rmx = rs.add_map('rm', dict(j='0:2*H'))
    rt = rs.add_tasklet('r', {'x'}, {'y'}, 'y = x')
    rs.add_memlet_path(br, rme, rt, dst_conn='x', memlet=dace.Memlet('buf[j]'))
    rs.add_memlet_path(rt, rmx, ow, src_conn='y', memlet=dace.Memlet('out[i]', wcr='lambda p, q: p + q'))
    sdfg.validate()
    return sdfg, loop


def test_strided_partial_fill_is_not_loop_private():
    """A strided write only *bounding-box* covers the read; the slots it skips are loop-carried.

    The pass must either refuse ``buf`` or leave the result bit-exact -- privatising it makes every
    iteration read its own never-written odd slots instead of the carried 7.0.
    """
    n, h = 4, 3
    # Integer-valued doubles: the reduction is exact whatever order the WCR sums in, so the
    # comparison against numpy is bit-exact rather than tolerance-masked.
    a = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0], [10.0, 11.0, 12.0]])
    ref = (2.0 * a).sum(axis=1) + 7.0 * h  # odds hold the seeded 7.0 on every iteration

    baseline, _ = _carried_strided_sdfg()
    out_base = np.zeros(n)
    baseline(a=a.copy(), out=out_base, N=n, H=h)
    assert_array_equal(out_base, ref, err_msg='the sequential program itself must match numpy')

    sdfg, loop = _carried_strided_sdfg()
    if 'buf' in BufferExpansion()._privatizable_buffers(sdfg, loop):
        index, size = BufferExpansion._loop_index(loop)
        BufferExpansion()._expand(sdfg, loop, 'buf', index, size, BufferExpansion._ambient_order(sdfg))
        sdfg.validate()
    out = np.zeros(n)
    sdfg(a=a.copy(), out=out, N=n, H=h)
    assert_array_equal(out, ref, err_msg='privatising a strided-fill (carried) buffer changed the result')

    # The sharp statement: the guard must not credit a strided write as a full fill.
    fresh, floop = _carried_strided_sdfg()
    assert BufferExpansion._defined_before_read(floop, 'buf') is False
    assert 'buf' not in BufferExpansion()._privatizable_buffers(fresh, floop)


if __name__ == '__main__':
    sys.exit(pytest.main([__file__, '-v']))
