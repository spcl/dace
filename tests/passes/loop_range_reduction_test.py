# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Tests for the ``LoopRangeReduction`` pass."""
from typing import Dict, List, Optional

import numpy as np
import pytest

import dace
from dace.properties import CodeBlock
from dace.sdfg.state import ConditionalBlock, ControlFlowRegion, LoopRegion
from dace.transformation.passes.loop_range_reduction import LoopRangeReduction

N = dace.symbol('N')
M = dace.symbol('M')

_ONE_LOOP = {'reduced_loops': 1, 'reduced_maps': 0}
_ONE_MAP = {'reduced_loops': 0, 'reduced_maps': 1}


def _loops(sdfg: dace.SDFG) -> List[LoopRegion]:
    return [n for n in sdfg.all_control_flow_blocks(recursive=True) if isinstance(n, LoopRegion)]


def _conditionals(sdfg: dace.SDFG) -> List[ConditionalBlock]:
    return [n for n in sdfg.all_control_flow_blocks(recursive=True) if isinstance(n, ConditionalBlock)]


def _header(loop: LoopRegion) -> str:
    return f'{loop.init_statement.as_string} ; {loop.loop_condition.as_string} ; {loop.update_statement.as_string}'


def _make_sdfg(guard: str,
               init: str = 'i = 0',
               condition: str = 'i < N',
               update: str = 'i = i + 1',
               size: int = 20,
               constants: Optional[Dict[str, np.ndarray]] = None,
               with_else: bool = False,
               empty_states: bool = False) -> dace.SDFG:
    """``for (init; condition; update): if guard: B[i] = A[i] + 1`` with symbols ``N``, ``M`` and constants."""
    sdfg = dace.SDFG('reduce_test')
    sdfg.add_symbol('N', dace.int64)
    sdfg.add_symbol('M', dace.int64)
    sdfg.add_array('A', [size], dace.float64)
    sdfg.add_array('B', [size], dace.float64)
    for name, value in (constants or {}).items():
        sdfg.add_constant(name, value)
    loop = LoopRegion('loop', condition, 'i', init, update, sdfg=sdfg)
    sdfg.add_node(loop, is_start_block=True)
    guard_block = ConditionalBlock('guard', sdfg=sdfg, parent=loop)
    if empty_states:
        pre = loop.add_state('pre', is_start_block=True)
        loop.add_node(guard_block)
        loop.add_edge(pre, guard_block, dace.InterstateEdge())
        post = loop.add_state('post')
        loop.add_edge(guard_block, post, dace.InterstateEdge())
    else:
        loop.add_node(guard_block, is_start_block=True)
    body = ControlFlowRegion('body', sdfg=sdfg, parent=guard_block)
    guard_block.add_branch(CodeBlock(guard), body)
    state = body.add_state('compute', is_start_block=True)
    tasklet = state.add_tasklet('t', {'a'}, {'b'}, 'b = a + 1')
    state.add_edge(state.add_read('A'), None, tasklet, 'a', dace.Memlet('A[i]'))
    state.add_edge(tasklet, 'b', state.add_write('B'), None, dace.Memlet('B[i]'))
    if with_else:
        els = ControlFlowRegion('els', sdfg=sdfg, parent=guard_block)
        guard_block.add_branch(None, els)
        els.add_state('empty_else', is_start_block=True)
    return sdfg


def _run(sdfg: dace.SDFG, size: int = 20, **symbols) -> np.ndarray:
    a = np.arange(size, dtype=np.float64)
    b = np.zeros(size, dtype=np.float64)
    sdfg(A=a, B=b, **symbols)
    return b


def _reference(selected: np.ndarray, size: int = 20) -> np.ndarray:
    a = np.arange(size, dtype=np.float64)
    b = np.zeros(size, dtype=np.float64)
    b[selected] = a[selected] + 1
    return b


def _apply(sdfg: dace.SDFG) -> Optional[Dict[str, int]]:
    sdfg.validate()
    result = LoopRangeReduction().apply_pass(sdfg, {})
    sdfg.validate()
    return result


# ---------------------------------------------------------------------------------------------------------------------
# Symbolic guards
# ---------------------------------------------------------------------------------------------------------------------


def test_symbolic_lower_and_upper_bound():
    sdfg = _make_sdfg('i >= 1 and i < M')
    assert _apply(sdfg) == _ONE_LOOP
    assert len(_conditionals(sdfg)) == 0
    (loop, ) = _loops(sdfg)
    assert loop.init_statement.as_string == 'i = 1'
    assert 'Min' in loop.loop_condition.as_string
    for n, m in ((20, 11), (20, 30), (20, 0), (20, 1)):
        expected = _reference(np.arange(max(1, 0), min(n, m)) if m > 1 else np.array([], dtype=int))
        assert np.allclose(_run(sdfg, N=n, M=m), expected)


@pytest.mark.parametrize('guard, selected', [
    ('i > 4', range(5, 20)),
    ('4 < i', range(5, 20)),
    ('i <= 4', range(0, 5)),
    ('i == 7', [7]),
    ('i != 7', [k for k in range(20) if k != 7]),
    ('not (i < 3)', range(3, 20)),
    ('not (i == 3)', [k for k in range(20) if k != 3]),
    ('3 <= i < 6', range(3, 6)),
    ('i < 2 or i > 15', [0, 1] + list(range(16, 20))),
    ('i < 2 or i > 15 or i == 8', [0, 1, 8] + list(range(16, 20))),
    ('not (i < 2 or i > 15)', range(2, 16)),
    ('(i > 1 and i < 4) or (i > 10 and i != 12 and i < 15)', [2, 3, 11, 13, 14]),
    ('True', range(0, 20)),
])
def test_symbolic_guard_shapes(guard, selected):
    sdfg = _make_sdfg(guard, condition='i < 20')
    assert _apply(sdfg) is not None
    assert len(_conditionals(sdfg)) == 0
    assert np.allclose(_run(sdfg, N=20, M=0), _reference(np.array(list(selected), dtype=int)))


def test_symbolic_guard_split_count():
    sdfg = _make_sdfg('i != 7', condition='i < 20')
    _apply(sdfg)
    loops = _loops(sdfg)
    assert len(loops) == 2
    assert loops[0].init_statement.as_string == 'i = 0' and loops[0].loop_condition.as_string == '(i < 7)'
    assert loops[1].init_statement.as_string == 'i = 8' and loops[1].loop_condition.as_string == '(i < 20)'


def test_contradiction_removes_loop():
    sdfg = _make_sdfg('i > 100', condition='i < 20')
    assert _apply(sdfg) == _ONE_LOOP
    assert len(_loops(sdfg)) == 0 and len(_conditionals(sdfg)) == 0
    assert np.allclose(_run(sdfg, N=20, M=0), np.zeros(20))


def test_symbolic_descending_loop():
    sdfg = _make_sdfg('i < M and i >= 2', init='i = N - 1', condition='i >= 0', update='i = i - 1')
    assert _apply(sdfg) == _ONE_LOOP
    (loop, ) = _loops(sdfg)
    assert 'Min' in loop.init_statement.as_string
    assert loop.loop_condition.as_string == '(i >= 2)'
    for n, m in ((20, 11), (20, 30), (20, 1)):
        assert np.allclose(_run(sdfg, N=n, M=m), _reference(np.arange(2, min(n, m)) if m > 2 else np.array([], int)))


def test_symbolic_strided_loop():
    sdfg = _make_sdfg('i >= 3 and i < M', init='i = 1', condition='i < N', update='i = i + 2')
    assert _apply(sdfg) == _ONE_LOOP
    (loop, ) = _loops(sdfg)
    assert loop.init_statement.as_string == 'i = 3'
    for n, m in ((20, 11), (20, 30), (20, 12)):
        assert np.allclose(_run(sdfg, N=n, M=m), _reference(np.arange(3, min(n, m), 2)))


def test_symbolic_lower_bound_start_symbolic_stride():
    sdfg = _make_sdfg('i >= 6', init='i = M', condition='i < N', update='i = i + 3')
    assert _apply(sdfg) == _ONE_LOOP
    (loop, ) = _loops(sdfg)
    assert 'int_ceil' in loop.init_statement.as_string
    for n, m in ((20, 1), (20, 2), (20, 7), (20, 19)):
        assert np.allclose(_run(sdfg, N=n, M=m), _reference(np.array([k for k in range(m, n, 3) if k >= 6], int)))


def test_residual_data_guard_is_kept():
    sdfg = _make_sdfg('i >= 2 and A[i] > 4.5 and i < M')
    assert _apply(sdfg) == _ONE_LOOP
    (loop, ) = _loops(sdfg)
    (guard, ) = _conditionals(sdfg)
    assert loop.init_statement.as_string == 'i = 2'
    assert guard.branches[0][0].as_string == '(A[i] > 4.5)'
    assert np.allclose(_run(sdfg, N=20, M=15), _reference(np.arange(5, 15)))


def test_scalar_constant_bound():
    sdfg = _make_sdfg('i < K', condition='i < 20', constants={'K': np.int64(6)})
    assert _apply(sdfg) == _ONE_LOOP
    (loop, ) = _loops(sdfg)
    assert loop.loop_condition.as_string == '(i < 6)'
    assert np.allclose(_run(sdfg, N=20, M=0), _reference(np.arange(6)))


# ---------------------------------------------------------------------------------------------------------------------
# Constant-array guards
# ---------------------------------------------------------------------------------------------------------------------

CSTARR = np.array([0, 0, 0, 1, 1, 0, 0, 2], dtype=np.int32)


def test_constant_array_guard_constant_loop():
    sdfg = _make_sdfg('cstarr[i] > 0', condition='i < 8', size=8, constants={'cstarr': CSTARR})
    assert _apply(sdfg) == _ONE_LOOP
    loops = _loops(sdfg)
    assert len(loops) == 2 and len(_conditionals(sdfg)) == 0
    assert _header(loops[0]) == 'i = 3 ; (i < 5) ; i = (i + 1)'
    assert _header(loops[1]) == 'i = 7 ; (i < 8) ; i = (i + 1)'
    assert np.allclose(_run(sdfg, size=8, N=8, M=0), _reference(np.array([3, 4, 7]), size=8))


def test_constant_array_guard_symbolic_loop():
    """With a symbolic trip count the domain comes from the array bounds and the loop range is clipped in."""
    sdfg = _make_sdfg('cstarr[i] > 0', size=8, constants={'cstarr': CSTARR})
    assert _apply(sdfg) == _ONE_LOOP
    assert len(_loops(sdfg)) == 2 and len(_conditionals(sdfg)) == 0
    for n in (8, 5, 4, 2):
        assert np.allclose(_run(sdfg, size=8, N=n, M=0), _reference(np.array([k for k in (3, 4, 7) if k < n], int), 8))


def test_constant_array_guard_forms():
    for guard, selected in (('cstarr[i] == 2', [7]), ('0 < cstarr[i] and i != 4', [3, 7]),
                            ('not cstarr[i]', [0, 1, 2, 5,
                                               6]), ('cstarr[i] > 0 or i == 0', [0, 3, 4,
                                                                                 7]), ('cstarr[i + 1] > 0', [2, 3, 6]),
                            ('cstarr[i] > K', [7]), ('cstarr[i] + cstarr[7 - i] > 0', [0, 3, 4, 7])):
        sdfg = _make_sdfg(guard, condition='i < 8', size=8, constants={'cstarr': CSTARR, 'K': np.int32(1)})
        assert _apply(sdfg) is not None, guard
        assert len(_conditionals(sdfg)) == 0, guard
        assert np.allclose(_run(sdfg, size=8, N=8, M=0), _reference(np.array(selected), size=8)), guard


def test_constant_array_guard_strided_loop():
    arr = np.array([1, 0, 1, 0, 1, 1, 0, 1, 0, 0, 0, 1], dtype=np.int64)
    sdfg = _make_sdfg('cst[i] > 0', condition='i < 12', update='i = i + 2', size=12, constants={'cst': arr})
    assert _apply(sdfg) == _ONE_LOOP
    loops = _loops(sdfg)
    # Only visited iterates matter: 0, 2, 4 form one run although cst[1] == cst[3] == 0.
    assert [_header(l) for l in loops] == ['i = 0 ; (i < 5) ; i = (i + 2)']
    assert np.allclose(_run(sdfg, size=12, N=12, M=0), _reference(np.array([0, 2, 4]), size=12))


def test_constant_2d_array_guard():
    arr = np.zeros((3, 6), dtype=np.int32)
    arr[1, 2:4] = 1
    sdfg = _make_sdfg('cst[1, i] == 1', condition='i < 6', size=6, constants={'cst': arr})
    assert _apply(sdfg) == _ONE_LOOP
    assert [_header(l) for l in _loops(sdfg)] == ['i = 2 ; (i < 4) ; i = (i + 1)']


def test_constant_array_all_false_removes_loop():
    sdfg = _make_sdfg('cstarr[i] > 5', condition='i < 8', size=8, constants={'cstarr': CSTARR})
    assert _apply(sdfg) == _ONE_LOOP
    assert len(_loops(sdfg)) == 0
    assert np.allclose(_run(sdfg, size=8, N=8, M=0), np.zeros(8))


def test_too_many_ranges_bails():
    arr = np.arange(20) % 2
    sdfg = _make_sdfg('cst[i] > 0', condition='i < 20', constants={'cst': arr})
    p = LoopRangeReduction()
    p.max_ranges = 4
    sdfg.validate()
    assert p.apply_pass(sdfg, {}) is None
    assert len(_loops(sdfg)) == 1 and len(_conditionals(sdfg)) == 1


# ---------------------------------------------------------------------------------------------------------------------
# Structural variants
# ---------------------------------------------------------------------------------------------------------------------


def test_empty_states_around_guard():
    sdfg = _make_sdfg('i >= 3', condition='i < 20', empty_states=True)
    assert _apply(sdfg) == _ONE_LOOP
    assert len(_conditionals(sdfg)) == 0
    assert np.allclose(_run(sdfg, N=20, M=0), _reference(np.arange(3, 20)))


def test_empty_else_branch():
    sdfg = _make_sdfg('i >= 3', condition='i < 20', with_else=True)
    assert _apply(sdfg) == _ONE_LOOP
    assert len(_conditionals(sdfg)) == 0
    assert np.allclose(_run(sdfg, N=20, M=0), _reference(np.arange(3, 20)))


def test_live_else_branch():
    """``if i < 3: pass else: body`` runs the body iff ``not (i < 3)``."""
    sdfg = _make_sdfg('i < 3', condition='i < 20', with_else=True)
    guard = _conditionals(sdfg)[0]
    # Swap the bodies: the else arm now carries the computation and the if arm is empty.
    (cond, body), (_, els) = guard.branches
    guard._branches = [(cond, els), (None, body)]
    assert _apply(sdfg) == _ONE_LOOP
    assert len(_conditionals(sdfg)) == 0
    (loop, ) = _loops(sdfg)
    assert loop.init_statement.as_string == 'i = 3'
    assert np.allclose(_run(sdfg, N=20, M=0), _reference(np.arange(3, 20)))


def test_frontend_generated_loop():

    @dace.program
    def prog(A: dace.float64[N], B: dace.float64[N]):
        for i in range(N):
            if i >= 1 and i < M:
                B[i] = A[i] * 2.0

    sdfg = prog.to_sdfg(simplify=True)
    if 'M' not in sdfg.symbols:
        # The Python frontend does not register a symbol that only appears in an ``if`` condition.
        sdfg.add_symbol('M', dace.int64)
    assert len(_conditionals(sdfg)) == 1
    assert _apply(sdfg) == _ONE_LOOP
    assert len(_conditionals(sdfg)) == 0
    for n, m in ((16, 9), (16, 40)):
        a = np.random.rand(n)
        b = np.zeros(n)
        sdfg(A=a, B=b, N=n, M=m)
        expected = np.zeros(n)
        expected[1:min(n, m)] = a[1:min(n, m)] * 2.0
        assert np.allclose(b, expected)


def test_nested_loops_inner_and_outer():

    @dace.program
    def prog(A: dace.float64[N, N], B: dace.float64[N, N]):
        for i in range(N):
            if i >= 2:
                for j in range(N):
                    if j < i:
                        B[i, j] = A[i, j] + 1.0

    sdfg = prog.to_sdfg(simplify=True)
    assert _apply(sdfg) == {'reduced_loops': 2, 'reduced_maps': 0}
    assert len(_conditionals(sdfg)) == 0
    n = 7
    a = np.random.rand(n, n)
    b = np.zeros((n, n))
    sdfg(A=a, B=b, N=n)
    expected = np.zeros((n, n))
    for i in range(2, n):
        expected[i, :i] = a[i, :i] + 1.0
    assert np.allclose(b, expected)


def test_multi_range_with_nested_sdfg_body():
    """Splitting into several loops deep-copies the body; a nested SDFG in it must stay consistent."""

    @dace.program
    def inner(a: dace.float64[20], b: dace.float64[20], k: dace.int64):
        b[k] = a[k] + 1.0

    @dace.program
    def prog(A: dace.float64[20], B: dace.float64[20]):
        for i in range(20):
            if i != 7:
                inner(A, B, i)

    sdfg = prog.to_sdfg(simplify=False)
    assert _apply(sdfg) == _ONE_LOOP
    assert len(_conditionals(sdfg)) == 0
    assert len(_loops(sdfg)) == 2
    a = np.random.rand(20)
    b = np.zeros(20)
    sdfg(A=a, B=b)
    expected = a + 1.0
    expected[7] = 0.0
    assert np.allclose(b, expected)


# ---------------------------------------------------------------------------------------------------------------------
# Cases that must be left alone
# ---------------------------------------------------------------------------------------------------------------------


def test_data_dependent_guard_untouched():
    sdfg = _make_sdfg('A[i] > 4.5')
    assert _apply(sdfg) is None
    assert len(_conditionals(sdfg)) == 1


def test_itervar_read_after_loop_untouched():
    sdfg = _make_sdfg('i >= 3', condition='i < 20')
    sdfg.add_symbol('i', dace.int64)
    loop = _loops(sdfg)[0]
    after = sdfg.add_state('after')
    sdfg.add_edge(loop, after, dace.InterstateEdge(assignments={'last': 'i'}))
    assert _apply(sdfg) is None
    assert len(_conditionals(sdfg)) == 1


def test_itervar_modified_in_body_untouched():
    sdfg = _make_sdfg('i >= 3', condition='i < 20')
    body = _conditionals(sdfg)[0].branches[0][1]
    state = body.start_block
    extra = body.add_state('extra')
    body.add_edge(state, extra, dace.InterstateEdge(assignments={'i': 'i + 1'}))
    assert _apply(sdfg) is None


def test_bound_symbol_modified_in_body_untouched():
    sdfg = _make_sdfg('i >= M', condition='i < 20')
    body = _conditionals(sdfg)[0].branches[0][1]
    state = body.start_block
    extra = body.add_state('extra')
    body.add_edge(state, extra, dace.InterstateEdge(assignments={'M': 'M + 1'}))
    assert _apply(sdfg) is None


def test_live_if_and_else_untouched():
    sdfg = _make_sdfg('i >= 3', condition='i < 20', with_else=True)
    els = _conditionals(sdfg)[0].branches[1][1]
    state = els.start_block
    tasklet = state.add_tasklet('t', {}, {'b'}, 'b = 0')
    state.add_edge(tasklet, 'b', state.add_write('B'), None, dace.Memlet('B[i]'))
    assert _apply(sdfg) is None


def test_overlapping_symbolic_disjunction_untouched():
    sdfg = _make_sdfg('i < M or i > 4')
    assert _apply(sdfg) is None


def test_break_with_multiple_ranges_untouched():
    sdfg = _make_sdfg('i != 7', condition='i < 20')
    body = _conditionals(sdfg)[0].branches[0][1]
    body_state = body.start_block
    brk = dace.sdfg.state.BreakBlock('brk', sdfg=sdfg, parent=body)
    body.add_node(brk)
    body.add_edge(body_state, brk, dace.InterstateEdge(condition='i == 12'))
    sdfg.validate()
    assert LoopRangeReduction().apply_pass(sdfg, {}) is None


def test_break_with_single_range_applied():
    """A ``break`` is fine when a single loop results: the guarded iterates simply stop at the same point."""
    sdfg = _make_sdfg('i >= 3', condition='i < 20')
    body = _conditionals(sdfg)[0].branches[0][1]
    body_state = body.start_block
    brk = dace.sdfg.state.BreakBlock('brk', sdfg=sdfg, parent=body)
    body.add_node(brk)
    body.add_edge(body_state, brk, dace.InterstateEdge(condition='i == 12'))
    assert _apply(sdfg) == _ONE_LOOP
    assert len(_conditionals(sdfg)) == 0
    assert np.allclose(_run(sdfg, N=20, M=0), _reference(np.arange(3, 13)))


def test_body_with_multiple_exits_untouched():
    """Folding the guard splices the branch body into the loop, which needs a unique exit block."""
    sdfg = _make_sdfg('i >= 3', condition='i < 20')
    body = _conditionals(sdfg)[0].branches[0][1]
    body_state = body.start_block
    brk = dace.sdfg.state.BreakBlock('brk', sdfg=sdfg, parent=body)
    body.add_node(brk)
    tail = body.add_state('tail')
    body.add_edge(body_state, brk, dace.InterstateEdge(condition='i == 12'))
    body.add_edge(body_state, tail, dace.InterstateEdge(condition='i != 12'))
    assert _apply(sdfg) is None
    assert len(_conditionals(sdfg)) == 1


def test_inverted_loop_untouched():
    sdfg = _make_sdfg('i >= 3', condition='i < 20')
    _loops(sdfg)[0].inverted = True
    assert _apply(sdfg) is None


def test_no_op_returns_none_and_is_idempotent():
    sdfg = _make_sdfg('i >= 3 and i < M')
    assert _apply(sdfg) == _ONE_LOOP
    assert _apply(sdfg) is None


# ---------------------------------------------------------------------------------------------------------------------
# Guard prologues (edge assignments feeding the condition) and arrays promoted to constants
# ---------------------------------------------------------------------------------------------------------------------


def _assignments_in(loop: LoopRegion) -> List[Dict[str, str]]:
    return [dict(e.data.assignments) for e in loop.all_interstate_edges() if e.data.assignments]


def test_frontend_array_promoted_to_constant():
    """A regular array argument that is afterwards also registered as a constant: the frontend lowers the guard to
    ``cstarr_index = cstarr[i]`` on an edge followed by ``if cstarr_index > 0``, and the constant value decides the
    ranges. The array stays a program argument (its runtime content is assumed to match the constant)."""

    @dace.program
    def prog(A: dace.float64[8], B: dace.float64[8], cstarr: dace.int32[8]):
        for i in range(8):
            if cstarr[i] > 0:
                B[i] = A[i] + 1.0

    sdfg = prog.to_sdfg(simplify=True)
    assert 'cstarr' in sdfg.arrays
    sdfg.add_constant('cstarr', CSTARR)
    assert 'cstarr' in sdfg.constants
    assert _apply(sdfg) == _ONE_LOOP
    loops = _loops(sdfg)
    assert len(loops) == 2 and len(_conditionals(sdfg)) == 0
    assert [(l.init_statement.as_string, l.loop_condition.as_string) for l in loops] == [('i = 3', '(i < 5)'),
                                                                                         ('i = 7', '(i < 8)')]
    # The prologue assignment only fed the guard and is gone.
    assert all(not _assignments_in(l) for l in loops)
    a = np.arange(8, dtype=np.float64)
    b = np.zeros(8)
    sdfg(A=a, B=b, cstarr=CSTARR)
    assert np.allclose(b, _reference(np.array([3, 4, 7]), size=8))


def test_frontend_compiletime_constant_array():
    """A ``dace.compiletime`` array becomes a closure array; registering its value as a constant enables the pass."""

    @dace.program
    def prog(A: dace.float64[8], B: dace.float64[8], cstarr: dace.compiletime):
        for i in range(8):
            if cstarr[i] > 0 and i != 4:
                B[i] = A[i] + 1.0

    sdfg = prog.to_sdfg(cstarr=CSTARR, simplify=True)
    (closure_name, ) = [name for name in sdfg.arrays if name.endswith('cstarr')]
    sdfg.add_constant(closure_name, CSTARR)
    assert _apply(sdfg) == _ONE_LOOP
    assert len(_loops(sdfg)) == 2 and len(_conditionals(sdfg)) == 0
    a = np.arange(8, dtype=np.float64)
    b = np.zeros(8)
    sdfg(A=a, B=b, **{closure_name: CSTARR})
    assert np.allclose(b, _reference(np.array([3, 7]), size=8))


def _make_prologue_sdfg(guard: str, assignments: Dict[str, str], body_code: str = 'b = a + 1') -> dace.SDFG:
    """``for i in range(8): <assignments>; if guard: B[i] = <body_code>`` with ``cstarr`` both an array and a constant."""
    sdfg = _make_sdfg(guard, condition='i < 8', size=8, constants={'cstarr': CSTARR})
    sdfg.add_array('cstarr', [8], dace.int32)
    for name in assignments:
        sdfg.add_symbol(name, dace.int32)
    loop = _loops(sdfg)[0]
    guard_block = _conditionals(sdfg)[0]
    pre = loop.add_state('pre', is_start_block=True)
    loop.add_edge(pre, guard_block, dace.InterstateEdge(assignments=assignments))
    state = guard_block.branches[0][1].start_block
    tasklet = next(n for n in state.nodes() if isinstance(n, dace.nodes.Tasklet))
    tasklet.code = CodeBlock(body_code)
    return sdfg


def test_prologue_with_residual_drops_assignment():
    sdfg = _make_prologue_sdfg('t > 0 and A[i] > 3.5', {'t': 'cstarr[i]'})
    assert _apply(sdfg) == _ONE_LOOP
    loops = _loops(sdfg)
    assert len(loops) == 2
    assert all(not _assignments_in(l) for l in loops)
    for l in loops:
        assert len([n for n in l.nodes() if isinstance(n, ConditionalBlock)]) == 1
    a = np.arange(8, dtype=np.float64)
    b = np.zeros(8)
    sdfg(A=a, B=b, cstarr=CSTARR, N=8, M=0)
    assert np.allclose(b, _reference(np.array([4, 7]), size=8))


def test_prologue_symbol_read_in_body_keeps_assignment():
    sdfg = _make_prologue_sdfg('t > 0', {'t': 'cstarr[i]'}, body_code='b = a + t')
    assert _apply(sdfg) == _ONE_LOOP
    loops = _loops(sdfg)
    assert len(loops) == 2 and len(_conditionals(sdfg)) == 0
    assert all(_assignments_in(l) == [{'t': 'cstarr[i]'}] for l in loops)
    a = np.arange(8, dtype=np.float64)
    b = np.zeros(8)
    sdfg(A=a, B=b, cstarr=CSTARR, N=8, M=0)
    expected = np.zeros(8)
    for k in (3, 4, 7):
        expected[k] = a[k] + CSTARR[k]
    assert np.allclose(b, expected)


def test_prologue_chain_is_composed():
    sdfg = _make_prologue_sdfg('u == 2', {'t': 'cstarr[i]'})
    loop = _loops(sdfg)[0]
    guard_block = _conditionals(sdfg)[0]
    sdfg.add_symbol('u', dace.int32)
    (edge, ) = loop.edges()
    mid = loop.add_state('mid')
    loop.remove_edge(edge)
    loop.add_edge(edge.src, mid, dace.InterstateEdge(assignments={'t': 'cstarr[i]'}))
    loop.add_edge(mid, guard_block, dace.InterstateEdge(assignments={'u': 't + 1'}))
    assert _apply(sdfg) == _ONE_LOOP
    assert [_header(l) for l in _loops(sdfg)] == ['i = 3 ; (i < 5) ; i = (i + 1)']
    assert not _assignments_in(_loops(sdfg)[0])


def test_prologue_symbol_read_after_loop_untouched():
    sdfg = _make_prologue_sdfg('t > 0', {'t': 'cstarr[i]'})
    loop = _loops(sdfg)[0]
    after = sdfg.add_state('after')
    sdfg.add_edge(loop, after, dace.InterstateEdge(assignments={'last': 't'}))
    assert _apply(sdfg) is None


def test_prologue_runtime_array_stays_residual():
    """Without the constant, ``cstarr[i]`` is runtime data: the guard stays (now reading the array directly)."""
    sdfg = _make_prologue_sdfg('t > 0 and i >= 4', {'t': 'cstarr[i]'})
    del sdfg.constants_prop['cstarr']
    assert _apply(sdfg) == _ONE_LOOP
    (loop, ) = _loops(sdfg)
    (guard, ) = _conditionals(sdfg)
    assert loop.init_statement.as_string == 'i = 4'
    assert guard.branches[0][0].as_string == '(cstarr[i] > 0)'
    assert not _assignments_in(loop)
    a = np.arange(8, dtype=np.float64)
    b = np.zeros(8)
    sdfg(A=a, B=b, cstarr=CSTARR, N=8, M=0)
    assert np.allclose(b, _reference(np.array([4, 7]), size=8))


# ---------------------------------------------------------------------------------------------------------------------
# Branches inside map scopes (nested SDFG bodies)
# ---------------------------------------------------------------------------------------------------------------------


def _maps(sdfg: dace.SDFG) -> List[dace.nodes.MapEntry]:
    return [n for n, _ in sdfg.all_nodes_recursive() if isinstance(n, dace.nodes.MapEntry)]


def _map_ranges(sdfg: dace.SDFG) -> List[str]:
    return sorted(str(m.map.range) for m in _maps(sdfg))


def test_map_symbolic_guard():

    @dace.program
    def prog(A: dace.float64[N], B: dace.float64[N]):
        for i in dace.map[0:N]:
            if i >= 1 and i < M:
                B[i] = A[i] * 2.0

    sdfg = prog.to_sdfg(simplify=True)
    if 'M' not in sdfg.symbols:
        sdfg.add_symbol('M', dace.int64)
    assert _apply(sdfg) == _ONE_MAP
    assert len(_conditionals(sdfg)) == 0
    (entry, ) = _maps(sdfg)
    (begin, end, step), = entry.map.range.ranges
    assert str(begin) == '1' and 'Min' in str(end) and str(step) == '1'
    for n, m in ((16, 9), (16, 40), (16, 1)):
        a = np.random.rand(n)
        b = np.zeros(n)
        sdfg(A=a, B=b, N=n, M=m)
        expected = np.zeros(n)
        expected[1:min(n, m)] = a[1:min(n, m)] * 2.0
        assert np.allclose(b, expected)


def test_map_constant_array_guard():
    """An array argument promoted to a constant: the guard reads it through an input scalar of the nested SDFG fed
    by ``cstarr[i]``; the map splits into ``3:4`` and ``7:7``."""

    @dace.program
    def prog(A: dace.float64[8], B: dace.float64[8], cstarr: dace.int32[8]):
        for i in dace.map[0:8]:
            if cstarr[i] > 0:
                B[i] = A[i] + 1.0

    sdfg = prog.to_sdfg(simplify=True)
    sdfg.add_constant('cstarr', CSTARR)
    assert _apply(sdfg) == _ONE_MAP
    assert len(_conditionals(sdfg)) == 0
    assert _map_ranges(sdfg) == ['3:5', '7']  # A single-element range prints as its only index
    a = np.arange(8, dtype=np.float64)
    b = np.zeros(8)
    sdfg(A=a, B=b, cstarr=CSTARR)
    assert np.allclose(b, _reference(np.array([3, 4, 7]), size=8))


def test_map_constant_guard_unsimplified():
    """Before simplification the guard is a dataflow-computed scalar (``__tmp0``) rather than a symbol: not reachable."""

    @dace.program
    def prog(A: dace.float64[8], B: dace.float64[8], cstarr: dace.int32[8]):
        for i in dace.map[0:8]:
            if cstarr[i] > 0:
                B[i] = A[i] + 1.0

    sdfg = prog.to_sdfg(simplify=False)
    sdfg.add_constant('cstarr', CSTARR)
    assert _apply(sdfg) is None


def test_map_runtime_data_residual():

    @dace.program
    def prog(A: dace.float64[N], B: dace.float64[N]):
        for i in dace.map[0:N]:
            if i >= 2 and A[i] > 4.5:
                B[i] = A[i] + 1.0

    sdfg = prog.to_sdfg(simplify=True)
    assert _apply(sdfg) == _ONE_MAP
    (entry, ) = _maps(sdfg)
    assert str(entry.map.range) == '2:N'
    (guard, ) = _conditionals(sdfg)
    assert 'A' not in guard.branches[0][0].as_string  # The residual stays in the nested SDFG's own names
    assert '4.5' in guard.branches[0][0].as_string
    a = np.arange(20, dtype=np.float64)
    b = np.zeros(20)
    sdfg(A=a, B=b, N=20)
    assert np.allclose(b, _reference(np.arange(5, 20)))


def test_map_two_dimensional_guard_on_both_params():

    @dace.program
    def prog(A: dace.float64[N, N], B: dace.float64[N, N]):
        for i, j in dace.map[0:N, 0:N]:
            if i >= 2 and j < M:
                B[i, j] = A[i, j] + 1.0

    sdfg = prog.to_sdfg(simplify=True)
    if 'M' not in sdfg.symbols:
        sdfg.add_symbol('M', dace.int64)
    # Both parameters are reduced within a single application.
    assert _apply(sdfg) == {'reduced_loops': 0, 'reduced_maps': 2}
    assert len(_conditionals(sdfg)) == 0
    (entry, ) = _maps(sdfg)
    assert str(entry.map.range.ranges[0][0]) == '2' and 'Min' in str(entry.map.range.ranges[1][1])
    n, m = 6, 4
    a = np.random.rand(n, n)
    b = np.zeros((n, n))
    sdfg(A=a, B=b, N=n, M=m)
    expected = np.zeros((n, n))
    expected[2:, :m] = a[2:, :m] + 1.0
    assert np.allclose(b, expected)


def test_map_guard_relating_two_params_stays():

    @dace.program
    def prog(A: dace.float64[N, N], B: dace.float64[N, N]):
        for i, j in dace.map[0:N, 0:N]:
            if j < i:
                B[i, j] = A[i, j] + 1.0

    sdfg = prog.to_sdfg(simplify=True)
    assert _apply(sdfg) is None


def test_map_contradiction_removes_scope():

    @dace.program
    def prog(A: dace.float64[8], B: dace.float64[8]):
        for i in dace.map[0:8]:
            if i > 100:
                B[i] = A[i] + 1.0

    sdfg = prog.to_sdfg(simplify=True)
    assert _apply(sdfg) == _ONE_MAP
    assert len(_maps(sdfg)) == 0
    sdfg.validate()


def test_map_inside_loop_and_loop_inside_map():
    """Loops nested in map bodies are handled before the map, and maps nested in loop bodies leave the loop's own
    guard analysis untouched."""

    @dace.program
    def prog(A: dace.float64[N, N], B: dace.float64[N, N]):
        for i in dace.map[0:N]:
            if i >= 1:
                for j in range(N):
                    if j < M:
                        B[i, j] = A[i, j] + 1.0

    sdfg = prog.to_sdfg(simplify=True)
    if 'M' not in sdfg.symbols:
        sdfg.add_symbol('M', dace.int64)
    result = _apply(sdfg)
    assert result == {'reduced_loops': 1, 'reduced_maps': 1}
    assert len(_conditionals(sdfg)) == 0
    n, m = 6, 4
    a = np.random.rand(n, n)
    b = np.zeros((n, n))
    sdfg(A=a, B=b, N=n, M=m)
    expected = np.zeros((n, n))
    expected[1:, :m] = a[1:, :m] + 1.0
    assert np.allclose(b, expected)


def _guarded_map_body(sdfg: dace.SDFG) -> dace.SDFG:
    """Nested SDFG ``flag = (c_in > 0); if flag: u_out = t_in + 1`` -- the shape the frontend emits for a guarded map
    body, with the guard read through an input scalar."""
    inner = dace.SDFG('body')
    inner.add_scalar('t_in', dace.float64)
    inner.add_scalar('c_in', dace.int32)
    inner.add_scalar('u_out', dace.float64)
    inner.add_symbol('flag', dace.int32)
    pre = inner.add_state('pre', is_start_block=True)
    guard = ConditionalBlock('guard', sdfg=inner, parent=inner)
    inner.add_node(guard)
    inner.add_edge(pre, guard, dace.InterstateEdge(assignments={'flag': '(c_in > 0)'}))
    branch = ControlFlowRegion('branch', sdfg=inner, parent=guard)
    guard.add_branch(CodeBlock('flag'), branch)
    state = branch.add_state('compute', is_start_block=True)
    tasklet = state.add_tasklet('t', {'t'}, {'u'}, 'u = t + 1')
    state.add_edge(state.add_read('t_in'), None, tasklet, 't', dace.Memlet('t_in[0]'))
    state.add_edge(tasklet, 'u', state.add_write('u_out'), None, dace.Memlet('u_out[0]'))
    return inner


def test_map_split_inside_fused_dataflow():
    """The split map is in the middle of one state: it consumes a transient a producer map wrote and feeds a consumer
    map. Both replicas must read the producer's output and both must complete before the consumer runs. The state is
    built by hand so the layout does not depend on the frontend's state fusion."""
    sdfg = dace.SDFG('fused')
    sdfg.add_array('A', [8], dace.float64)
    sdfg.add_array('B', [8], dace.float64)
    sdfg.add_array('cstarr', [8], dace.int32)
    sdfg.add_constant('cstarr', CSTARR)
    sdfg.add_transient('T', [8], dace.float64)
    sdfg.add_transient('U', [8], dace.float64)
    state = sdfg.add_state('main', is_start_block=True)
    a_node, t_node, u_node, b_node = (state.add_access(name) for name in ('A', 'T', 'U', 'B'))
    c_node = state.add_access('cstarr')
    state.add_mapped_tasklet('producer', {'i': '0:8'}, {'a': dace.Memlet('A[i]')},
                             't = a * 3.0', {'t': dace.Memlet('T[i]')},
                             external_edges=True,
                             input_nodes={'A': a_node},
                             output_nodes={'T': t_node})
    entry, exit_node = state.add_map('guarded', {'i': '0:8'})
    nsdfg = state.add_nested_sdfg(_guarded_map_body(sdfg), {
        't_in': dace.float64,
        'c_in': dace.int32
    }, {'u_out': dace.float64}, {'i': 'i'})
    state.add_memlet_path(t_node, entry, nsdfg, dst_conn='t_in', memlet=dace.Memlet('T[i]'))
    state.add_memlet_path(c_node, entry, nsdfg, dst_conn='c_in', memlet=dace.Memlet('cstarr[i]'))
    state.add_memlet_path(nsdfg, exit_node, u_node, src_conn='u_out', memlet=dace.Memlet('U[i]'))
    state.add_mapped_tasklet('consumer', {'i': '0:8'}, {'u': dace.Memlet('U[i]')},
                             'b = u * 2.0', {'b': dace.Memlet('B[i]')},
                             external_edges=True,
                             input_nodes={'U': u_node},
                             output_nodes={'B': b_node})

    assert _apply(sdfg) == _ONE_MAP
    assert len(_conditionals(sdfg)) == 0
    maps = [n for n in state.nodes() if isinstance(n, dace.nodes.MapEntry)]
    assert sorted(str(m.map.range) for m in maps) == ['0:8', '0:8', '3:5', '7']
    replicas = [m for m in maps if m.map.label.startswith('guarded')]
    assert len(replicas) == 2
    for replica in replicas:
        # Every replica reads the producer's output and the constant array and writes the consumer's input.
        sources = {e.src for e in state.in_edges(replica)}
        assert sources == {t_node, c_node}
        assert {e.dst for e in state.out_edges(state.exit_node(replica))} == {u_node}
        # ... and its nested SDFG body has no conditional left and no dead prologue symbol.
        (body, ) = [n for n in state.scope_children()[replica] if isinstance(n, dace.nodes.NestedSDFG)]
        assert 'flag' not in body.sdfg.symbols
        assert all(not e.data.assignments for e in body.sdfg.edges())
    a = np.arange(8, dtype=np.float64)
    b = np.zeros(8)
    sdfg(A=a, B=b, cstarr=CSTARR)
    selected = [3, 4, 7]
    assert np.allclose(b[selected], (a[selected] * 3.0 + 1.0) * 2.0)  # Other entries of ``U`` are uninitialized


def test_map_split_with_inout_and_multiple_inputs():
    """Replicas of a map that reads and writes the same array (``B[i] = B[i] + ...``) and reads two inputs."""

    @dace.program
    def prog(A: dace.float64[8], B: dace.float64[8], C: dace.float64[8], cstarr: dace.int32[8]):
        for i in dace.map[0:8]:
            if cstarr[i] > 0:
                B[i] = B[i] + A[i] * C[i]

    sdfg = prog.to_sdfg(simplify=True)
    sdfg.add_constant('cstarr', CSTARR)
    assert _apply(sdfg) == _ONE_MAP
    assert len(_conditionals(sdfg)) == 0
    assert _map_ranges(sdfg) == ['3:5', '7']
    a = np.arange(8, dtype=np.float64)
    c = np.arange(8, dtype=np.float64) + 10.0
    b = np.ones(8)
    sdfg(A=a, B=b, C=c, cstarr=CSTARR)
    expected = np.ones(8)
    expected[[3, 4, 7]] += a[[3, 4, 7]] * c[[3, 4, 7]]
    assert np.allclose(b, expected)


def test_map_split_with_reduction_output():
    """Both replicas accumulate into the same write-conflict-resolved output."""

    @dace.program
    def prog(A: dace.float64[8], s: dace.float64[1], cstarr: dace.int32[8]):
        for i in dace.map[0:8]:
            if cstarr[i] > 0:
                s[0] += A[i]

    sdfg = prog.to_sdfg(simplify=True)
    sdfg.add_constant('cstarr', CSTARR)
    assert _apply(sdfg) == _ONE_MAP
    assert len(_conditionals(sdfg)) == 0
    a = np.arange(8, dtype=np.float64)
    s = np.array([100.0])
    sdfg(A=a, s=s, cstarr=CSTARR)
    assert np.allclose(s, 100.0 + a[[3, 4, 7]].sum())


def test_map_split_nested_in_outer_map():
    """The guarded map is the body of an outer map (i.e., inside a nested SDFG); it is split there and the replicas
    connect to the outer scope's connectors."""

    @dace.program
    def prog(A: dace.float64[N, 8], B: dace.float64[N, 8], cstarr: dace.int32[8]):
        for i in dace.map[0:N]:
            for j in dace.map[0:8]:
                if cstarr[j] > 0:
                    B[i, j] = A[i, j] + 1.0

    sdfg = prog.to_sdfg(simplify=True)
    sdfg.add_constant('cstarr', CSTARR)
    result = _apply(sdfg)
    assert result is not None and result['reduced_maps'] == 1
    assert len(_conditionals(sdfg)) == 0
    n = 5
    a = np.random.rand(n, 8)
    b = np.zeros((n, 8))
    sdfg(A=a, B=b, N=n, cstarr=CSTARR)
    expected = np.zeros((n, 8))
    expected[:, [3, 4, 7]] = a[:, [3, 4, 7]] + 1.0
    assert np.allclose(b, expected)


def test_map_split_inside_loop_body():
    """A split map inside a loop region's state, with the loop variable in the map's memlets."""

    @dace.program
    def prog(A: dace.float64[N, 8], B: dace.float64[N, 8], cstarr: dace.int32[8]):
        for i in range(N):
            for j in dace.map[0:8]:
                if cstarr[j] > 0:
                    B[i, j] = A[i, j] + i

    sdfg = prog.to_sdfg(simplify=True)
    sdfg.add_constant('cstarr', CSTARR)
    result = _apply(sdfg)
    assert result is not None and result['reduced_maps'] == 1
    assert len(_conditionals(sdfg)) == 0
    n = 5
    a = np.random.rand(n, 8)
    b = np.zeros((n, 8))
    sdfg(A=a, B=b, N=n, cstarr=CSTARR)
    expected = np.zeros((n, 8))
    for i in range(n):
        expected[i, [3, 4, 7]] = a[i, [3, 4, 7]] + i
    assert np.allclose(b, expected)


def test_map_split_three_ranges_symbolic_disjunction():
    """A symbolic guard with three disjoint clauses yields three replicas of the map in one state."""

    @dace.program
    def prog(A: dace.float64[20], B: dace.float64[20]):
        for i in dace.map[0:20]:
            if i < 2 or i == 8 or i > 15:
                B[i] = A[i] + 1.0

    sdfg = prog.to_sdfg(simplify=True)
    assert _apply(sdfg) == _ONE_MAP
    assert len(_conditionals(sdfg)) == 0
    assert _map_ranges(sdfg) == ['0:2', '16:20', '8']
    a = np.arange(20, dtype=np.float64)
    b = np.zeros(20)
    sdfg(A=a, B=b)
    assert np.allclose(b, _reference(np.array([0, 1, 8, 16, 17, 18, 19])))


if __name__ == '__main__':
    import sys
    sys.exit(pytest.main([__file__, '-x', '-q']))
