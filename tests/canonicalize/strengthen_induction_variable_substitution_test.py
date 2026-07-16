# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Strengthening tests for InductionVariableSubstitution.

Every hunt here is the same one-sided assumption: the pass used to admit only ONE of the
two legal answers about where an IV's value comes from, and refuse otherwise.

1. The branch-uniform IV hoist (:func:`_hoist_branch_uniform_iv`) moves an IV increment
   that EVERY branch of a body conditional performs out of the conditional. The position
   it moves to must match where the branches READ the IV: a branch that USES the IV
   before incrementing it (``a[j] = ...; j += 1``) wants the PRE-increment value, so the
   increment has to land AFTER the conditional -- hoisting it BEFORE would make that use
   see the post-increment value -> off-by-one array index -> wrong numbers. This mirrors
   TSVC ``s124`` (``j = j + 1`` BEFORE ``a[j] = ...``), which hoists to the other side.

2. NO USE AT ALL is not ambiguity. ``_consistent_use_side`` used to answer ``None`` both
   for "the uses straddle the increment" (truly undecidable -> refuse) and for "there are
   no uses" (both answers correct -> lift). The second refused a loop whose only obstacle
   to parallelizing was a dead loop-carried counter.

3. A use BEFORE a derived-IV definition (``a[i] = b[k]; k = i + 1``) reads the PREVIOUS
   iteration's value -- ``f(i - stride)``, a closed form as well, not an impossibility.
   It is sound exactly when the value carried INTO the loop equals ``f(start - stride)``,
   because the first iteration has no previous iteration to read from. Both the lift and
   its guard are tested: a mismatching seed must still refuse.

All shapes must lose the loop-carried symbol and stay bit-exact.
"""

import numpy as np
import pytest

import dace
from dace.properties import CodeBlock
from dace.sdfg.state import ConditionalBlock, ControlFlowRegion, LoopRegion
from dace.transformation.passes.canonicalize.induction_variable_substitution import InductionVariableSubstitution


def _build_use_before_increment_sdfg(n: int) -> dace.SDFG:
    """Build ``j = 0; for i in range(n): if i%2==0: a[j]=b[i]; j+=1  else: a[j]=c[i]; j+=1``.

    Note the USE ``a[j]`` PRECEDES the increment ``j += 1`` in each branch -- so at
    iteration ``i`` the write hits ``a[i]`` (pre-increment j). Both branches increment
    unconditionally, so ``j`` tracks ``i`` exactly. ``a`` is sized ``n + 1`` so an
    off-by-one write (``a[i+1]``) stays in-bounds instead of segfaulting.
    """
    sdfg = dace.SDFG('use_before_inc')
    sdfg.add_array('a', [n + 1], dace.float64)
    sdfg.add_array('b', [n], dace.float64)
    sdfg.add_array('c', [n], dace.float64)
    sdfg.add_symbol('j', dace.int64)

    init = sdfg.add_state('init', is_start_block=True)

    loop = LoopRegion('loop', condition_expr=f'i < {n}', loop_var='i', initialize_expr='i = 0',
                      update_expr='i = i + 1')
    sdfg.add_node(loop)
    # Seed the IV symbol ``j`` before the loop.
    sdfg.add_edge(init, loop, dace.InterstateEdge(assignments={'j': '0'}))

    cb = ConditionalBlock('cb')
    loop.add_node(cb, is_start_block=True)

    def _populate_branch(br: ControlFlowRegion, name: str, src_array: str) -> None:
        s1 = br.add_state(name + '_use', is_start_block=True)
        s2 = br.add_state(name + '_end')
        rd = s1.add_read(src_array)
        wr = s1.add_write('a')
        tlt = s1.add_tasklet(name + '_t', {'__in'}, {'__out'}, '__out = __in',
                             language=dace.dtypes.Language.Python)
        s1.add_edge(rd, None, tlt, '__in', dace.Memlet(data=src_array, subset='i'))
        s1.add_edge(tlt, '__out', wr, None, dace.Memlet(data='a', subset='j'))
        # Increment AFTER the use.
        br.add_edge(s1, s2, dace.InterstateEdge(assignments={'j': 'j + 1'}))

    then_br = ControlFlowRegion('then')
    els_br = ControlFlowRegion('els')
    cb.add_branch(CodeBlock('(i % 2) == 0'), then_br)
    cb.add_branch(None, els_br)
    _populate_branch(then_br, 'then', 'b')
    _populate_branch(els_br, 'els', 'c')

    sdfg.validate()
    return sdfg


def _reference(n: int, b: np.ndarray, c: np.ndarray) -> np.ndarray:
    a = np.zeros(n + 1, dtype=np.float64)
    j = 0
    for i in range(n):
        if i % 2 == 0:
            a[j] = b[i]
        else:
            a[j] = c[i]
        j = j + 1
    return a


def _carries_symbol(sdfg: dace.SDFG, sym: str) -> bool:
    """Whether any loop body still assigns ``sym`` -- i.e. the loop-carried dependency
    on the IV survived (the loop cannot parallelize)."""
    return any(sym in (e.data.assignments or {}) for r in sdfg.all_control_flow_regions()
               if isinstance(r, LoopRegion) for e in r.all_interstate_edges())


def test_branch_uniform_hoist_use_before_increment():
    n = 8
    rng = np.random.default_rng(0)
    b = rng.random(n)
    c = rng.random(n)

    sdfg = _build_use_before_increment_sdfg(n)
    assert InductionVariableSubstitution().apply_pass(sdfg, {}) is not None, \
        "read-before-increment branch-uniform IV must be hoisted + closed, not refused"
    sdfg.validate()

    # The lift's whole point: ``j`` is no longer loop-carried, so the body is a pure
    # function of the loop variable and LoopToMap can parallelize it.
    assert not _carries_symbol(sdfg, 'j'), "loop-carried ``j`` survived -> loop still sequential"

    got = np.zeros(n + 1, dtype=np.float64)
    sdfg.compile()(a=got, b=b.copy(), c=c.copy())

    ref = _reference(n, b, c)
    # Bit-exact: the body only copies, so the lift must reproduce the reference exactly.
    assert np.array_equal(ref, got), f"value mismatch:\nref={ref}\ngot={got}"


def _build_unused_iv_sdfg() -> dace.SDFG:
    """``j = 0; for i in range(N): a[i] = b[i]; j += 1; a2[i] = c[i]`` then ``out[0] = d[j]``.

    The increment sits BETWEEN two content states (so neither the TOP nor the BOTTOM
    shortcut applies) and NOTHING in the body reads ``j`` -- only the post-loop ``d[j]``
    does. Both offsets are therefore correct in the body, and the substitution's only
    effect is to drop the loop-carried ``j`` so the loop can parallelize. ``N`` is kept
    SYMBOLIC on purpose.
    """
    sdfg = dace.SDFG('unused_iv')
    n = dace.symbol('N', dace.int64)
    sdfg.add_symbol('N', dace.int64)
    sdfg.add_symbol('j', dace.int64)
    for name in ('a', 'a2', 'b', 'c'):
        sdfg.add_array(name, [n], dace.float64)
    sdfg.add_array('d', [n + 1], dace.float64)
    sdfg.add_array('out', [1], dace.float64)

    init = sdfg.add_state('init', is_start_block=True)
    loop = LoopRegion('loop', condition_expr='i < N', loop_var='i', initialize_expr='i = 0', update_expr='i = i + 1')
    sdfg.add_node(loop)
    sdfg.add_edge(init, loop, dace.InterstateEdge(assignments={'j': '0'}))

    def _copy_state(state, dst, src):
        rd = state.add_read(src)
        wr = state.add_write(dst)
        tlt = state.add_tasklet(dst + '_t', {'__in'}, {'__out'}, '__out = __in', language=dace.dtypes.Language.Python)
        state.add_edge(rd, None, tlt, '__in', dace.Memlet(data=src, subset='i'))
        state.add_edge(tlt, '__out', wr, None, dace.Memlet(data=dst, subset='i'))

    s0 = loop.add_state('s0', is_start_block=True)
    s1 = loop.add_state('s1')
    _copy_state(s0, 'a', 'b')
    _copy_state(s1, 'a2', 'c')
    loop.add_edge(s0, s1, dace.InterstateEdge(assignments={'j': 'j + 1'}))

    # Post-loop reader: the ONLY use of ``j``, so the closed form's exit value is observable.
    post = sdfg.add_state('post')
    sdfg.add_edge(loop, post, dace.InterstateEdge())
    rd = post.add_read('d')
    wr = post.add_write('out')
    tlt = post.add_tasklet('post_t', {'__in'}, {'__out'}, '__out = __in', language=dace.dtypes.Language.Python)
    post.add_edge(rd, None, tlt, '__in', dace.Memlet(data='d', subset='j'))
    post.add_edge(tlt, '__out', wr, None, dace.Memlet(data='out', subset='0'))

    sdfg.validate()
    return sdfg


def test_unused_iv_is_liftable_not_ambiguous():
    """No use of the IV in the body is NOT ambiguity -- both offsets are correct, so the
    loop-carried increment must be stripped rather than refused."""
    n = 8
    rng = np.random.default_rng(1)
    b, c, d = rng.random(n), rng.random(n), rng.random(n + 1)

    sdfg = _build_unused_iv_sdfg()
    assert InductionVariableSubstitution().apply_pass(sdfg, {}) is not None, \
        "an IV with NO body use must lift (either offset is correct), not be refused as ambiguous"
    sdfg.validate()
    assert not _carries_symbol(sdfg, 'j'), "loop-carried ``j`` survived -> loop still sequential"

    a, a2, out = np.zeros(n), np.zeros(n), np.zeros(1)
    sdfg.compile()(a=a, a2=a2, b=b.copy(), c=c.copy(), d=d.copy(), out=out, N=n)

    assert np.array_equal(b, a) and np.array_equal(c, a2), "body copies must be unchanged"
    # ``j`` ends at N: the post-loop closed form has to reproduce that exactly.
    assert np.array_equal(np.array([d[n]]), out), f"post-loop j wrong: out={out}, expected d[{n}]={d[n]}"


def _build_use_before_definition_sdfg(seed: str) -> dace.SDFG:
    """``k = <seed>; for i in range(N): a[i] = b[k]; k = i + 1``.

    The gather ``b[k]`` PRECEDES the derived definition ``k := i + 1``, so it reads the
    PREVIOUS iteration's ``k``, i.e. ``f(i - 1) = i``. Iteration 0 has no previous
    iteration and reads the seed instead -- so the lagged closed form is correct exactly
    when ``seed == f(start - stride) = f(-1) = 0``. ``seed='0'`` must lift to the fully
    parallel ``a[i] = b[i]``; any other seed must refuse.
    """
    sdfg = dace.SDFG('use_before_def_' + seed)
    n = dace.symbol('N', dace.int64)
    sdfg.add_symbol('N', dace.int64)
    sdfg.add_symbol('k', dace.int64)
    sdfg.add_array('a', [n], dace.float64)
    sdfg.add_array('b', [n], dace.float64)

    init = sdfg.add_state('init', is_start_block=True)
    loop = LoopRegion('loop', condition_expr='i < N', loop_var='i', initialize_expr='i = 0', update_expr='i = i + 1')
    sdfg.add_node(loop)
    sdfg.add_edge(init, loop, dace.InterstateEdge(assignments={'k': seed}))

    s0 = loop.add_state('s0', is_start_block=True)
    tail = loop.add_state('tail')
    rd = s0.add_read('b')
    wr = s0.add_write('a')
    tlt = s0.add_tasklet('g_t', {'__in'}, {'__out'}, '__out = __in', language=dace.dtypes.Language.Python)
    s0.add_edge(rd, None, tlt, '__in', dace.Memlet(data='b', subset='k'))
    s0.add_edge(tlt, '__out', wr, None, dace.Memlet(data='a', subset='i'))
    loop.add_edge(s0, tail, dace.InterstateEdge(assignments={'k': 'i + 1'}))

    sdfg.validate()
    return sdfg


def _use_before_def_reference(n: int, b: np.ndarray, seed: int) -> np.ndarray:
    a = np.zeros(n, dtype=np.float64)
    k = seed
    for i in range(n):
        a[i] = b[k]
        k = i + 1
    return a


def test_derived_iv_use_before_definition_lifts():
    """A use BEFORE a derived definition reads the previous iteration's value -- a closed
    form too. With a seed that agrees at iteration 0 it must lift to ``a[i] = b[i]``."""
    n = 8
    b = np.random.default_rng(2).random(n)

    sdfg = _build_use_before_definition_sdfg('0')
    assert InductionVariableSubstitution().apply_pass(sdfg, {}) is not None, \
        "use-before-definition derived IV with an agreeing seed must lift, not be refused"
    sdfg.validate()
    assert not _carries_symbol(sdfg, 'k'), "loop-carried ``k`` survived -> loop still sequential"

    got = np.zeros(n)
    sdfg.compile()(a=got, b=b.copy(), N=n)
    ref = _use_before_def_reference(n, b, 0)
    assert np.array_equal(ref, got), f"value mismatch:\nref={ref}\ngot={got}"


def test_derived_iv_use_before_definition_disagreeing_seed_refuses():
    """The guard on the lift above: a seed that does NOT equal ``f(start - stride)`` makes
    iteration 0 read a value the lagged closed form mispredicts -> must stay sequential."""
    n = 8
    b = np.random.default_rng(3).random(n)

    sdfg = _build_use_before_definition_sdfg('3')  # f(-1) == 0 != 3
    assert InductionVariableSubstitution().apply_pass(sdfg, {}) is None, \
        "a seed disagreeing with the lagged closed form must be refused, not lifted"
    assert _carries_symbol(sdfg, 'k'), "``k`` must stay loop-carried when the lift is refused"

    got = np.zeros(n)
    sdfg.compile()(a=got, b=b.copy(), N=n)
    ref = _use_before_def_reference(n, b, 3)
    assert np.array_equal(ref, got), f"refusal must preserve semantics:\nref={ref}\ngot={got}"


if __name__ == '__main__':
    pytest.main([__file__, '-q'])
