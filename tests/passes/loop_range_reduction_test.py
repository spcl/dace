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


def _loops(sdfg: dace.SDFG) -> List[LoopRegion]:
    return [n for n in sdfg.all_control_flow_blocks() if isinstance(n, LoopRegion)]


def _conditionals(sdfg: dace.SDFG) -> List[ConditionalBlock]:
    return [n for n in sdfg.all_control_flow_blocks() if isinstance(n, ConditionalBlock)]


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
    assert _apply(sdfg) == {'reduced_loops': 1}
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
    assert _apply(sdfg) == {'reduced_loops': 1}
    assert len(_loops(sdfg)) == 0 and len(_conditionals(sdfg)) == 0
    assert np.allclose(_run(sdfg, N=20, M=0), np.zeros(20))


def test_symbolic_descending_loop():
    sdfg = _make_sdfg('i < M and i >= 2', init='i = N - 1', condition='i >= 0', update='i = i - 1')
    assert _apply(sdfg) == {'reduced_loops': 1}
    (loop, ) = _loops(sdfg)
    assert 'Min' in loop.init_statement.as_string
    assert loop.loop_condition.as_string == '(i >= 2)'
    for n, m in ((20, 11), (20, 30), (20, 1)):
        assert np.allclose(_run(sdfg, N=n, M=m), _reference(np.arange(2, min(n, m)) if m > 2 else np.array([], int)))


def test_symbolic_strided_loop():
    sdfg = _make_sdfg('i >= 3 and i < M', init='i = 1', condition='i < N', update='i = i + 2')
    assert _apply(sdfg) == {'reduced_loops': 1}
    (loop, ) = _loops(sdfg)
    assert loop.init_statement.as_string == 'i = 3'
    for n, m in ((20, 11), (20, 30), (20, 12)):
        assert np.allclose(_run(sdfg, N=n, M=m), _reference(np.arange(3, min(n, m), 2)))


def test_symbolic_lower_bound_start_symbolic_stride():
    sdfg = _make_sdfg('i >= 6', init='i = M', condition='i < N', update='i = i + 3')
    assert _apply(sdfg) == {'reduced_loops': 1}
    (loop, ) = _loops(sdfg)
    assert 'int_ceil' in loop.init_statement.as_string
    for n, m in ((20, 1), (20, 2), (20, 7), (20, 19)):
        assert np.allclose(_run(sdfg, N=n, M=m), _reference(np.array([k for k in range(m, n, 3) if k >= 6], int)))


def test_residual_data_guard_is_kept():
    sdfg = _make_sdfg('i >= 2 and A[i] > 4.5 and i < M')
    assert _apply(sdfg) == {'reduced_loops': 1}
    (loop, ) = _loops(sdfg)
    (guard, ) = _conditionals(sdfg)
    assert loop.init_statement.as_string == 'i = 2'
    assert guard.branches[0][0].as_string == '(A[i] > 4.5)'
    assert np.allclose(_run(sdfg, N=20, M=15), _reference(np.arange(5, 15)))


def test_scalar_constant_bound():
    sdfg = _make_sdfg('i < K', condition='i < 20', constants={'K': np.int64(6)})
    assert _apply(sdfg) == {'reduced_loops': 1}
    (loop, ) = _loops(sdfg)
    assert loop.loop_condition.as_string == '(i < 6)'
    assert np.allclose(_run(sdfg, N=20, M=0), _reference(np.arange(6)))


# ---------------------------------------------------------------------------------------------------------------------
# Constant-array guards
# ---------------------------------------------------------------------------------------------------------------------

CSTARR = np.array([0, 0, 0, 1, 1, 0, 0, 2], dtype=np.int32)


def test_constant_array_guard_constant_loop():
    sdfg = _make_sdfg('cstarr[i] > 0', condition='i < 8', size=8, constants={'cstarr': CSTARR})
    assert _apply(sdfg) == {'reduced_loops': 1}
    loops = _loops(sdfg)
    assert len(loops) == 2 and len(_conditionals(sdfg)) == 0
    assert _header(loops[0]) == 'i = 3 ; (i < 5) ; i = (i + 1)'
    assert _header(loops[1]) == 'i = 7 ; (i < 8) ; i = (i + 1)'
    assert np.allclose(_run(sdfg, size=8, N=8, M=0), _reference(np.array([3, 4, 7]), size=8))


def test_constant_array_guard_symbolic_loop():
    """With a symbolic trip count the domain comes from the array bounds and the loop range is clipped in."""
    sdfg = _make_sdfg('cstarr[i] > 0', size=8, constants={'cstarr': CSTARR})
    assert _apply(sdfg) == {'reduced_loops': 1}
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
    assert _apply(sdfg) == {'reduced_loops': 1}
    loops = _loops(sdfg)
    # Only visited iterates matter: 0, 2, 4 form one run although cst[1] == cst[3] == 0.
    assert [_header(l) for l in loops] == ['i = 0 ; (i < 5) ; i = (i + 2)']
    assert np.allclose(_run(sdfg, size=12, N=12, M=0), _reference(np.array([0, 2, 4]), size=12))


def test_constant_2d_array_guard():
    arr = np.zeros((3, 6), dtype=np.int32)
    arr[1, 2:4] = 1
    sdfg = _make_sdfg('cst[1, i] == 1', condition='i < 6', size=6, constants={'cst': arr})
    assert _apply(sdfg) == {'reduced_loops': 1}
    assert [_header(l) for l in _loops(sdfg)] == ['i = 2 ; (i < 4) ; i = (i + 1)']


def test_constant_array_all_false_removes_loop():
    sdfg = _make_sdfg('cstarr[i] > 5', condition='i < 8', size=8, constants={'cstarr': CSTARR})
    assert _apply(sdfg) == {'reduced_loops': 1}
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
    assert _apply(sdfg) == {'reduced_loops': 1}
    assert len(_conditionals(sdfg)) == 0
    assert np.allclose(_run(sdfg, N=20, M=0), _reference(np.arange(3, 20)))


def test_empty_else_branch():
    sdfg = _make_sdfg('i >= 3', condition='i < 20', with_else=True)
    assert _apply(sdfg) == {'reduced_loops': 1}
    assert len(_conditionals(sdfg)) == 0
    assert np.allclose(_run(sdfg, N=20, M=0), _reference(np.arange(3, 20)))


def test_live_else_branch():
    """``if i < 3: pass else: body`` runs the body iff ``not (i < 3)``."""
    sdfg = _make_sdfg('i < 3', condition='i < 20', with_else=True)
    guard = _conditionals(sdfg)[0]
    # Swap the bodies: the else arm now carries the computation and the if arm is empty.
    (cond, body), (_, els) = guard.branches
    guard._branches = [(cond, els), (None, body)]
    assert _apply(sdfg) == {'reduced_loops': 1}
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
    assert _apply(sdfg) == {'reduced_loops': 1}
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
    assert _apply(sdfg) == {'reduced_loops': 2}
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
    assert _apply(sdfg) == {'reduced_loops': 1}
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
    assert _apply(sdfg) == {'reduced_loops': 1}
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
    assert _apply(sdfg) == {'reduced_loops': 1}
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
    assert _apply(sdfg) == {'reduced_loops': 1}
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
    assert _apply(sdfg) == {'reduced_loops': 1}
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
    assert _apply(sdfg) == {'reduced_loops': 1}
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
    assert _apply(sdfg) == {'reduced_loops': 1}
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
    assert _apply(sdfg) == {'reduced_loops': 1}
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
    assert _apply(sdfg) == {'reduced_loops': 1}
    (loop, ) = _loops(sdfg)
    (guard, ) = _conditionals(sdfg)
    assert loop.init_statement.as_string == 'i = 4'
    assert guard.branches[0][0].as_string == '(cstarr[i] > 0)'
    assert not _assignments_in(loop)
    a = np.arange(8, dtype=np.float64)
    b = np.zeros(8)
    sdfg(A=a, B=b, cstarr=CSTARR, N=8, M=0)
    assert np.allclose(b, _reference(np.array([4, 7]), size=8))


if __name__ == '__main__':
    import sys
    sys.exit(pytest.main([__file__, '-x', '-q']))
