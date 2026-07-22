# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Strengthening test for ArgMaxLift tie-breaking on non-strict comparisons.

The sequential source ``if a[i] OP x: x = a[i]; index = i`` picks a different one
of several equal extremes depending on the guard's strictness: a strict ``>`` /
``<`` never updates on a tie, so it keeps the FIRST occurrence; a non-strict
``>=`` / ``<=`` DOES update, so it keeps the LAST. The lifted ``ArgReduce`` scans
with a strict comparison (``_OP_CPP['max'] == '>'``) and so is first-occurrence
only, while ``ArgMaxLift`` maps BOTH ``ast.Gt`` and ``ast.GtE`` to Max (see
``_CMP_AST_TO_RTYPE``).

ArgMaxLift resolves this by arg-reducing over the REVERSED gather for the
non-strict shape (first-of-reversed == last-of-forward), so both forms lift and
each keeps its own sequential tie semantics. These tests build argmax/argmin
loops over arrays whose extreme appears at MORE THAN ONE position and assert the
lift fires and the value AND index are bit-exact with the sequential reference.
"""
import numpy as np
import pytest

import dace
from dace.sdfg.state import LoopRegion, ConditionalBlock, ControlFlowRegion, BreakBlock
from dace.properties import CodeBlock
from dace.libraries.standard.nodes import Reduce
from dace.transformation.passes.canonicalize.arg_max_lift import ArgMaxLift

N = dace.symbol('N')


def _num_loops(sdfg):
    return sum(1 for r in sdfg.all_control_flow_regions() if isinstance(r, LoopRegion) and r.loop_variable)


def _build_symbol_argmax_index_ge(label: str, op: str = '>='):
    """Symbol-carrier argmax WITH index (TSVC s315 shape) under a parametric guard.

    Mirrors the existing test's ``_build_symbol_argmax_index_sdfg`` but with a
    parametric operator so we can exercise ``>=`` / ``<=`` tie-breaking.
    """
    sdfg = dace.SDFG(label)
    sdfg.add_array('a', [N], dace.float64)
    sdfg.add_array('result', [1], dace.float64)
    sdfg.add_array('idx_result', [1], dace.int64)
    sdfg.add_symbol('x', dace.float64)
    sdfg.add_symbol('index', dace.int64)
    sdfg.add_symbol('a_index', dace.float64)

    init_state = sdfg.add_state('init', is_start_block=True)
    loop = LoopRegion(label + '_loop',
                      initialize_expr='i = 1',
                      condition_expr='i < N',
                      update_expr='i = i + 1',
                      loop_var='i')
    sdfg.add_node(loop)
    sdfg.add_edge(init_state, loop, dace.InterstateEdge(assignments={'x': 'a[0]', 'index': '0'}))

    start_blk = loop.add_state('start', is_start_block=True)
    cond_prep = loop.add_state('cond_prep')
    cond_block = ConditionalBlock('cond_block')
    loop.add_node(cond_block)
    loop.add_edge(start_blk, cond_prep, dace.InterstateEdge(assignments={'a_index': 'a[i]'}))
    loop.add_edge(cond_prep, cond_block, dace.InterstateEdge())
    cond_code = f'(a_index {op} x)'

    true_branch = ControlFlowRegion(label + '_true')
    cond_block.add_branch(CodeBlock(cond_code), true_branch)
    t1 = true_branch.add_state('t1', is_start_block=True)
    t2 = true_branch.add_state('t2')
    true_branch.add_edge(t1, t2, dace.InterstateEdge(assignments={'x': 'a[i]', 'index': 'i'}))

    post = sdfg.add_state('post')
    sdfg.add_edge(loop, post, dace.InterstateEdge())
    wv = post.add_write('result')
    tv = post.add_tasklet('write_val', {}, {'__o'}, '__o = x', language=dace.dtypes.Language.Python)
    post.add_edge(tv, '__o', wv, None, dace.Memlet(data='result', subset='0'))
    wi = post.add_write('idx_result')
    ti = post.add_tasklet('write_idx', {}, {'__o'}, '__o = index', language=dace.dtypes.Language.Python)
    post.add_edge(ti, '__o', wi, None, dace.Memlet(data='idx_result', subset='0'))
    return sdfg


def _sequential_argextreme(a, op: str):
    """Reference for ``x=a[0]; index=0; for i in 1..: if a[i] OP x: x=a[i]; index=i``."""
    x = a[0]
    index = 0
    for i in range(1, len(a)):
        hit = {'>': a[i] > x, '>=': a[i] >= x, '<': a[i] < x, '<=': a[i] <= x}[op]
        if hit:
            x = a[i]
            index = i
    return x, index


# ``a`` has its max 3.0 at 0 AND 2, and its min -1.0 at 1 AND 3, so every guard's
# tie choice is observable: strict keeps the first, non-strict the last.
_TIE_ARRAY = np.array([3.0, -1.0, 3.0, -1.0])


@pytest.mark.parametrize('op,expected_idx', [('>', 0), ('>=', 2), ('<', 1), ('<=', 3)])
def test_index_tie_breaking_matches_sequential(op, expected_idx):
    """An argmax/argmin-with-index loop over an array with a REPEATED extreme must
    lift AND report the index its own sequential guard would: FIRST occurrence for
    the strict ``>`` / ``<``, LAST for the non-strict ``>=`` / ``<=``.
    """
    from dace.libraries.standard.nodes import ArgReduce
    a = _TIE_ARRAY.copy()
    n = a.shape[0]
    exp_val, exp_idx = _sequential_argextreme(a, op)
    assert exp_idx == expected_idx  # sanity: the reference itself picks the tie we expect

    sdfg = _build_symbol_argmax_index_ge(f'tie_{ArgMaxLift.__name__}_{expected_idx}', op=op)
    sdfg.validate()
    res = ArgMaxLift().apply_pass(sdfg, {})
    assert res == 1, f'argmax-with-index under {op!r} must lift (a refusal costs the parallelism)'
    sdfg.validate()
    assert _num_loops(sdfg) == 0, 'the sequential loop must be gone'
    assert sum(1 for nd, _ in sdfg.all_nodes_recursive() if isinstance(nd, ArgReduce)) == 1

    val = np.zeros(1)
    idx = np.zeros(1, dtype=np.int64)
    sdfg(a=a, result=val, idx_result=idx, N=n)
    assert val[0] == exp_val, f'value: got {val[0]}, expected {exp_val}'
    assert idx[0] == exp_idx, f'index: got {idx[0]}, expected {exp_idx} (guard {op!r})'


@pytest.mark.parametrize('op', ['>', '>=', '<', '<='])
def test_index_lift_bit_exact_on_random_and_all_equal(op):
    """The lifted arg-reduce is bit-exact with the sequential loop on random data
    (no ties) and on an ALL-EQUAL array (maximal ties -- every position is an
    extreme, so the tie rule alone decides: index 0 strict, N-1 non-strict)."""
    rng = np.random.default_rng(315)
    tag = {'>': 'gt', '>=': 'ge', '<': 'lt', '<=': 'le'}[op]
    cases = (rng.standard_normal(24), np.full(9, 2.5), np.array([1.0, 1.0, 5.0, 5.0, 5.0, 1.0]))
    for k, a in enumerate(cases):
        exp_val, exp_idx = _sequential_argextreme(a, op)
        # A distinct SDFG name per case -- same-named builds share a .dacecache dir.
        sdfg = _build_symbol_argmax_index_ge(f'bitexact_{tag}_{k}', op=op)
        assert ArgMaxLift().apply_pass(sdfg, {}) == 1
        sdfg.validate()
        val = np.zeros(1)
        idx = np.zeros(1, dtype=np.int64)
        sdfg(a=a.copy(), result=val, idx_result=idx, N=a.shape[0])
        assert val[0] == exp_val, f'{op}: value {val[0]} != {exp_val}'
        assert idx[0] == exp_idx, f'{op}: index {idx[0]} != {exp_idx}'


_AL = dace.symbol('AL')


def _build_strided_abs_argmax_index_sdfg(label: str, op: str):
    """TSVC s318 shape: abs-transformed argmax/argmin WITH index over the strided
    gather ``a[k + (i-1)*inc]`` that ``InductionVariableSubstitution`` leaves
    (``k`` bound pre-loop to ``inc``, so the gather is ``a[inc*i]``). Mirrors the
    existing test's builder, with a parametric guard for the tie semantics."""
    sdfg = dace.SDFG(label)
    sdfg.add_array('a', [_AL], dace.float64)
    sdfg.add_array('result', [1], dace.float64)
    sdfg.add_array('idx_result', [1], dace.int64)
    sdfg.add_symbol('N', dace.int64)
    for s, t in (('x', dace.float64), ('index', dace.int64), ('a_index', dace.float64), ('inc', dace.int32),
                 ('k', dace.int32)):
        sdfg.add_symbol(s, t)

    init = sdfg.add_state('init', is_start_block=True)
    loop = LoopRegion(label + '_loop',
                      initialize_expr='i = 1',
                      condition_expr='i < N',
                      update_expr='i = i + 1',
                      loop_var='i')
    sdfg.add_node(loop)
    sdfg.add_edge(init, loop, dace.InterstateEdge(assignments={'x': 'abs(a[0])', 'index': '0', 'k': 'inc'}))

    sb = loop.add_state('start', is_start_block=True)
    cp = loop.add_state('cond_prep')
    cb = ConditionalBlock('cb')
    loop.add_node(cb)
    loop.add_edge(sb, cp, dace.InterstateEdge(assignments={'a_index': 'a[k + (i - 1) * inc]'}))
    loop.add_edge(cp, cb, dace.InterstateEdge())
    tb = ControlFlowRegion(label + '_true')
    cb.add_branch(CodeBlock(f'(abs(a_index) {op} x)'), tb)
    t1 = tb.add_state('t1', is_start_block=True)
    t2 = tb.add_state('t2')
    tb.add_edge(t1, t2, dace.InterstateEdge(assignments={'x': 'abs(a_index)', 'index': 'i'}))

    post = sdfg.add_state('post')
    sdfg.add_edge(loop, post, dace.InterstateEdge())
    tv = post.add_tasklet('wv', {}, {'__o'}, '__o = x', language=dace.dtypes.Language.Python)
    post.add_edge(tv, '__o', post.add_write('result'), None, dace.Memlet('result[0]'))
    ti = post.add_tasklet('wi', {}, {'__o'}, '__o = index', language=dace.dtypes.Language.Python)
    post.add_edge(ti, '__o', post.add_write('idx_result'), None, dace.Memlet('idx_result[0]'))
    return sdfg


@pytest.mark.parametrize('op,expected_idx', [('>', 0), ('>=', 4), ('<', 1), ('<=', 5)])
def test_strided_abs_index_tie_breaking_s318(op, expected_idx):
    """The strided abs+index path (TSVC s318) lifts under BOTH guards: the buffer
    it materialises is filled in reverse iteration order for the non-strict guard,
    so the ArgReduce's first-wins scan yields the sequential LAST occurrence."""
    from dace.libraries.standard.nodes import ArgReduce
    inc, n = 2, 6
    al = inc * (n - 1) + 4
    # |a[inc*j]| for j in 0..5 == [3, 1, 3, 1, 3, 1]: the max 3 repeats at j=0,2,4
    # and the min 1 at j=1,3,5, so every guard's tie choice is observable.
    a = np.zeros(al)
    for j, v in enumerate([3.0, -1.0, 3.0, 1.0, -3.0, 1.0]):
        a[inc * j] = v

    sdfg = _build_strided_abs_argmax_index_sdfg(f's318_tie_{expected_idx}', op=op)
    sdfg.validate()
    assert ArgMaxLift().apply_pass(sdfg, {}) == 1, f'strided abs-argmax-with-index under {op!r} must lift'
    sdfg.validate()
    assert _num_loops(sdfg) == 0
    assert sum(1 for nd, _ in sdfg.all_nodes_recursive() if isinstance(nd, ArgReduce)) == 1

    val = np.zeros(1)
    idx = np.zeros(1, dtype=np.int64)
    sdfg(a=a.copy(), result=val, idx_result=idx, N=n, inc=inc, AL=al)

    # Sequential reference over the same reduction set (seed j=0, then i=1..n-1).
    strided = np.abs(a[[inc * j for j in range(n)]])
    exp_val, exp_idx = _sequential_argextreme(strided, op)
    assert exp_idx == expected_idx  # sanity: the reference itself picks the tie we expect
    assert val[0] == exp_val, f'value: got {val[0]}, expected {exp_val}'
    assert idx[0] == exp_idx, f'index: got {idx[0]}, expected {exp_idx} (guard {op!r})'


@pytest.mark.parametrize('op,strict', [('>', True), ('>=', False)])
def test_2d_index_tie_breaking_matches_sequential(op, strict):
    """The 2-D nest (TSVC s3110) must lift under BOTH guards, with the flat
    ArgReduce's decomposed (xindex, yindex) matching the sequential nest's
    row-major tie choice: first occurrence for ``>``, last for ``>=``."""
    from dace.libraries.standard.nodes import ArgReduce
    from dace.transformation.passes.canonicalize.pipeline import canonicalize
    M = dace.symbol('M')

    if strict:

        @dace.program
        def argmax2d(aa: dace.float64[M, M], out: dace.float64[3]):
            maxv = aa[0, 0]
            xindex = 0
            yindex = 0
            for i in range(M):
                for j in range(M):
                    if aa[i, j] > maxv:
                        maxv = aa[i, j]
                        xindex = i
                        yindex = j
            out[0] = maxv
            out[1] = float(xindex)
            out[2] = float(yindex)
    else:

        @dace.program
        def argmax2d(aa: dace.float64[M, M], out: dace.float64[3]):
            maxv = aa[0, 0]
            xindex = 0
            yindex = 0
            for i in range(M):
                for j in range(M):
                    if aa[i, j] >= maxv:
                        maxv = aa[i, j]
                        xindex = i
                        yindex = j
            out[0] = maxv
            out[1] = float(xindex)
            out[2] = float(yindex)

    sdfg = argmax2d.to_sdfg(simplify=True)
    canonicalize(sdfg, validate=True, peel_limit=4, break_anti_dependence=True)
    assert sum(1 for nd, _ in sdfg.all_nodes_recursive() if isinstance(nd, ArgReduce)) == 1, '2-D nest must lift'
    assert _num_loops(sdfg) == 0

    m = 5
    # A repeated max placed at two row-major positions, so the tie rule is observable.
    aa = np.zeros((m, m))
    aa[1, 3] = 9.0
    aa[3, 2] = 9.0
    out = np.zeros(3)
    sdfg(aa=aa.copy(), out=out, M=m)

    # Sequential reference for the exact nest above.
    maxv, xi, yi = aa[0, 0], 0, 0
    for i in range(m):
        for j in range(m):
            if (aa[i, j] > maxv) if strict else (aa[i, j] >= maxv):
                maxv, xi, yi = aa[i, j], i, j
    assert (xi, yi) == ((1, 3) if strict else (3, 2))  # sanity: the guards disagree here
    assert out[0] == maxv, f'value: got {out[0]}, expected {maxv}'
    assert (int(out[1]), int(out[2])) == (xi, yi), f'index: got ({out[1]}, {out[2]}), expected ({xi}, {yi})'


# -----------------------------------------------------------------------------
# The ``tie_break`` knob: BOTH tie rules selectable, 'infer' the default.
# -----------------------------------------------------------------------------


@pytest.mark.parametrize('op,expected_idx', [('>', 0), ('>=', 2), ('<', 1), ('<=', 3)])
def test_default_tie_break_is_infer_and_reproduces_the_guard(op, expected_idx):
    """The knob's DEFAULT must leave today's behaviour untouched: ``tie_break``
    defaults to ``'infer'``, and an explicit ``'infer'`` resolves ``last_wins``
    to exactly the guard-strictness inference (non-strict -> last-wins)."""
    assert ArgMaxLift().tie_break == 'infer', 'the default must not move the corpus'
    inferred = op in ('>=', '<=')
    for pass_instance in (ArgMaxLift(), ArgMaxLift(tie_break='infer')):
        sdfg = _build_symbol_argmax_index_ge(f'infer_{expected_idx}', op=op)
        loop = next(r for r in sdfg.all_control_flow_regions() if isinstance(r, LoopRegion) and r.loop_variable)
        m = pass_instance._match(loop, sdfg)
        assert m is not None
        assert m.last_wins is inferred, f'{op!r}: inferred last_wins {m.last_wins}, expected {inferred}'


# Each case pins the tie rule AGAINST the guard's own strictness, so the knob is
# the only thing that can produce the expected index: ``ref_op`` is the guard
# whose sequential semantics the knob-pinned lift must now match (same reduction
# direction, opposite strictness).
@pytest.mark.parametrize('op,knob,ref_op', [
    ('>=', 'first', '>'),
    ('>', 'last', '>='),
    ('<=', 'first', '<'),
    ('<', 'last', '<='),
])
def test_tie_break_knob_overrides_the_inferred_rule(op, knob, ref_op):
    """``tie_break='first'`` / ``'last'`` select the tie semantics explicitly,
    overriding the inference. On an array with a REPEATED extreme the two rules
    disagree, so the knob is directly observable in the returned index."""
    from dace.libraries.standard.nodes import ArgReduce
    a = _TIE_ARRAY.copy()
    exp_val, exp_idx = _sequential_argextreme(a, ref_op)
    inferred_val, inferred_idx = _sequential_argextreme(a, op)
    assert exp_idx != inferred_idx  # sanity: the knob really is what decides here

    sdfg = _build_symbol_argmax_index_ge(f'knob_{knob}_{exp_idx}', op=op)
    sdfg.validate()
    assert ArgMaxLift(tie_break=knob).apply_pass(sdfg, {}) == 1, f'{op!r} under tie_break={knob!r} must still lift'
    sdfg.validate()
    assert _num_loops(sdfg) == 0
    assert sum(1 for nd, _ in sdfg.all_nodes_recursive() if isinstance(nd, ArgReduce)) == 1

    val = np.zeros(1)
    idx = np.zeros(1, dtype=np.int64)
    sdfg(a=a, result=val, idx_result=idx, N=a.shape[0])
    assert val[0] == exp_val, f'value: got {val[0]}, expected {exp_val}'
    assert idx[0] == exp_idx, f'tie_break={knob!r} on guard {op!r}: index {idx[0]} != {exp_idx}'


@pytest.mark.parametrize('knob', ['first', 'last'])
def test_tie_break_knob_is_a_noop_for_the_value_only_shape(knob):
    """Without an index carrier the tie rule is unobservable -- both choices yield
    the same extreme VALUE -- so ``last_wins`` stays False and the rewrite skips
    the reversal whatever the knob says."""

    @dace.program
    def s314(a: dace.float64[N], result: dace.float64[1]):
        x = a[0]
        for i in range(1, N):
            if a[i] >= x:
                x = a[i]
        result[0] = x

    sdfg = s314.to_sdfg(simplify=True)
    loop = next(r for r in sdfg.all_control_flow_regions() if isinstance(r, LoopRegion) and r.loop_variable)
    m = ArgMaxLift(tie_break=knob)._match(loop, sdfg)
    assert m is not None and m.idx_carrier_name is None
    assert m.last_wins is False, 'no index tracked -> the tie rule is unobservable, so no reversal'


# -----------------------------------------------------------------------------
# Break / early-exit loops: refused, under EVERY tie rule.
# -----------------------------------------------------------------------------

NB = dace.symbol('NB')


@dace.program
def _break_argmax_value_only(a: dace.float64[NB], result: dace.float64[1]):
    """A find-FIRST search, NOT a reduction: it stops at the first element that
    beats the seed, so ``x`` ends up holding that element -- not ``max(a)``."""
    x = a[0]
    for i in range(1, NB):
        if a[i] > x:
            x = a[i]
            break
    result[0] = x


# ``a[1] == 1.0`` is the first element exceeding the seed ``a[0] == 0.0``, so the
# sequential loop exits there with x == 1.0; a whole-range Reduce(Max) would say
# 5.0. The two answers differ, so a wrong lift cannot hide.
_BREAK_ARRAY = np.array([0.0, 1.0, 5.0, 2.0])


def _sequential_break_argmax(a):
    """Reference for ``x = a[0]; for i in 1..: if a[i] > x: x = a[i]; break``."""
    x = a[0]
    for i in range(1, a.shape[0]):
        if a[i] > x:
            x = a[i]
            break
    return x


def test_break_loop_is_not_lifted_to_a_whole_range_reduce():
    """A break/early-exit loop must NOT be lifted by ArgMaxLift (DEFAULT pass, no
    knob involved). It is a find-first search whose carrier holds the value at the
    EXIT iteration, while every arg-reduce this pass emits scans the whole range.

    Before the break guard this loop LIFTED and miscompiled the VALUE: the
    data-carrier true-branch check counts only non-empty ``SDFGState`` s, so the
    ``BreakBlock`` slipped through and the loop became ``max(a) == 5.0`` instead
    of the sequential ``1.0``. This is a wrong VALUE, not merely a wrong tie.

    The break shape's parallel lift is ``EarlyExitToFindIndex``, which runs
    earlier in the canonicalize pipeline; refusing here costs no parallelism.
    """
    sdfg = _break_argmax_value_only.to_sdfg(simplify=True)
    assert any(isinstance(nd, BreakBlock) for nd, _ in sdfg.all_nodes_recursive()), 'the break must survive to the pass'
    assert _num_loops(sdfg) == 1

    assert ArgMaxLift().apply_pass(sdfg, {}) is None, 'a break loop is a find-first, not a reduction'
    assert _num_loops(sdfg) == 1, 'the sequential break loop must be left alone'
    assert sum(1 for nd, _ in sdfg.all_nodes_recursive() if isinstance(nd, Reduce)) == 0
    sdfg.validate()

    # And it still computes the sequential answer.
    a = _BREAK_ARRAY.copy()
    exp = _sequential_break_argmax(a)
    assert exp == 1.0 and np.max(a) == 5.0  # sanity: the two candidate answers differ
    out = np.zeros(1)
    sdfg(a=a.copy(), result=out, NB=a.shape[0])
    assert out[0] == exp, f'got {out[0]}, expected the sequential {exp} (a whole-range Reduce(Max) gives 5.0)'


@pytest.mark.parametrize('knob', ['infer', 'first', 'last'])
def test_break_loop_is_refused_under_every_tie_break(knob):
    """The knob cannot buy a break loop a lift: the break refusal precedes any
    tie-rule resolution, so ``tie_break='last'`` can never force a last-wins scan
    onto a search that is first-wins by definition."""
    sdfg = _break_argmax_value_only.to_sdfg(simplify=True)
    assert ArgMaxLift(tie_break=knob).apply_pass(sdfg, {}) is None
    assert _num_loops(sdfg) == 1
    out = np.zeros(1)
    sdfg(a=_BREAK_ARRAY.copy(), result=out, NB=_BREAK_ARRAY.shape[0])
    assert out[0] == _sequential_break_argmax(_BREAK_ARRAY)


def _build_symbol_argmax_index_with_break(label: str, op: str):
    """The s315 argmax-with-index shape plus a ``break`` in the true branch --
    a find-first-that-beats-the-seed that also records its position."""
    sdfg = _build_symbol_argmax_index_ge(label, op=op)
    true_branch = next(br for r in sdfg.all_control_flow_regions() if isinstance(r, ConditionalBlock)
                       for _c, br in r.branches)
    brk = BreakBlock('brk')
    true_branch.add_node(brk)
    true_branch.add_edge(next(n for n in true_branch.nodes() if n.label == 't2'), brk, dace.InterstateEdge())
    return sdfg


@pytest.mark.parametrize('knob', ['infer', 'first', 'last'])
@pytest.mark.parametrize('op', ['>', '>='])
def test_break_with_index_loop_is_refused_under_every_tie_break(op, knob):
    """The index-tracking break shape is refused too, so ``last_wins`` is never
    resolved for a break-derived loop -- the knob cannot force a last-wins scan
    onto a search that is first-wins by definition.

    (This shape was ALREADY refused before the break guard: the symbol-carrier
    true-branch check rejects any non-``SDFGState`` node. The test pins that
    property so it cannot regress -- the value-only path above is the one that
    actually miscompiled.)
    """
    tag = {'>': 'gt', '>=': 'ge'}[op]
    sdfg = _build_symbol_argmax_index_with_break(f'brkidx_{tag}_{knob}', op=op)
    loop = next(r for r in sdfg.all_control_flow_regions() if isinstance(r, LoopRegion) and r.loop_variable)
    assert ArgMaxLift(tie_break=knob)._match(loop, sdfg) is None, 'a break-derived loop must never reach a tie rule'


if __name__ == '__main__':
    for _op, _idx in (('>', 0), ('>=', 2), ('<', 1), ('<=', 3)):
        test_index_tie_breaking_matches_sequential(_op, _idx)
        test_index_lift_bit_exact_on_random_and_all_equal(_op)
    for _op, _idx in (('>', 0), ('>=', 4), ('<', 1), ('<=', 5)):
        test_strided_abs_index_tie_breaking_s318(_op, _idx)
    test_2d_index_tie_breaking_matches_sequential('>', True)
    test_2d_index_tie_breaking_matches_sequential('>=', False)
    print("ok")
