# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
"""Regression test: a transient whose shape depends on a symbol assigned inside
a ``ConditionalBlock`` must be allocated *after* that symbol is defined.

The transient ``a`` has shape ``(s,)``; ``s`` is assigned on the interstate
edges inside both branches of a ``ConditionalBlock``, so it is only defined
after the conditional. ``a`` is then accessed in a loop body and again after
the loop. ``determine_allocation_lifetime`` places such a (symbol-dependent,
multi-block) transient at the common dominator of its accesses -- the
pre-conditional state -- where ``s`` is not yet defined, so ``new double[s]``
ran with an uninitialised ``s`` (wrong-size allocation -> out-of-bounds writes).
Codegen now inserts a guard state after the symbol-defining region so the
allocation lands where ``s`` is defined.
"""
import numpy as np

import dace
from dace.properties import CodeBlock
from dace.sdfg.state import ConditionalBlock, ControlFlowRegion, LoopRegion


def _make_sdfg():
    sdfg = dace.SDFG('symdep_alloc_in_conditional')
    for sym in ('cond', 'n', 'm'):
        sdfg.add_symbol(sym, dace.int64)
    sdfg.add_symbol('s', dace.int64)
    sdfg.add_array('out', [10], dace.float64)
    sdfg.add_transient('a', ['s'], dace.float64)

    pre = sdfg.add_state('pre', is_start_block=True)

    cb = ConditionalBlock('cb')
    sdfg.add_node(cb)
    sdfg.add_edge(pre, cb, dace.InterstateEdge())
    then_reg = ControlFlowRegion('then_reg', sdfg=sdfg)
    t0 = then_reg.add_state('t0', is_start_block=True)
    t1 = then_reg.add_state('t1')
    then_reg.add_edge(t0, t1, dace.InterstateEdge(assignments={'s': 'n'}))
    cb.add_branch(CodeBlock('cond > 0'), then_reg)
    else_reg = ControlFlowRegion('else_reg', sdfg=sdfg)
    e0 = else_reg.add_state('e0', is_start_block=True)
    e1 = else_reg.add_state('e1')
    else_reg.add_edge(e0, e1, dace.InterstateEdge(assignments={'s': 'm'}))
    cb.add_branch(CodeBlock('not (cond > 0)'), else_reg)

    loop = LoopRegion('loop', loop_var='i', initialize_expr='i = 0', condition_expr='i < s', update_expr='i = i + 1')
    sdfg.add_node(loop)
    sdfg.add_edge(cb, loop, dace.InterstateEdge())
    body = loop.add_state('body', is_start_block=True)
    w = body.add_tasklet('w', {}, {'o'}, 'o = 1.0')
    body.add_edge(w, 'o', body.add_access('a'), None, dace.Memlet('a[i]'))

    post = sdfg.add_state('post')
    sdfg.add_edge(loop, post, dace.InterstateEdge())
    r = post.add_tasklet('r', {'x'}, {'o'}, 'o = x')
    post.add_edge(post.add_access('a'), None, r, 'x', dace.Memlet('a[0]'))
    post.add_edge(r, 'o', post.add_access('out'), None, dace.Memlet('out[0]'))

    sdfg.validate()
    return sdfg


def test_symbol_dependent_alloc_in_conditional():
    csdfg = _make_sdfg().compile()
    for cond, n, m in [(1, 5, 3), (0, 5, 3), (1, 17, 2), (0, 4, 13)]:
        out = np.zeros(10, dtype=np.float64)
        csdfg(out=out, cond=np.int64(cond), n=np.int64(n), m=np.int64(m))
        assert out[0] == 1.0


if __name__ == '__main__':
    test_symbol_dependent_alloc_in_conditional()
