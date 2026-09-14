# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Tests for memlet schedules: ScheduleLoopCursors (analysis) + LowerMemletSchedules (codegen-window lowering of
schedules to loop-carried cursor symbols and flat references)."""
import json
import re
import warnings

import numpy as np
import pytest

import dace
from dace.sdfg.memlet_schedule import CopyOnAccess, LoopCursor
from dace.sdfg.state import LoopRegion
from dace.transformation.passes.analysis import loop_analysis
from dace.transformation.passes.memlet_schedules import LowerMemletSchedules, ScheduleLoopCursors

N = dace.symbol('N')


def _code(sdfg: dace.SDFG) -> str:
    return '\n'.join(c.clean_code for c in sdfg.generate_code())


def _scheduled_memlets(sdfg: dace.SDFG):
    return [
        e.data for s in sdfg.all_sdfgs_recursive() for st in s.states() for e in st.edges()
        if not e.data.schedule.is_default
    ]


def _loops(sdfg: dace.SDFG):
    return [
        r for s in sdfg.all_sdfgs_recursive() for r in s.all_control_flow_regions(recursive=False)
        if isinstance(r, LoopRegion)
    ]


def _cursors(loop: LoopRegion):
    """``{cursor symbol: (init expression, step expression)}`` of the cursors materialized on ``loop``."""
    inits = loop_analysis.get_assignments(loop.init_statement)
    updates = loop_analysis.get_assignments(loop.update_statement)
    result = {}
    for name, init in inits.items():
        if not name.startswith('__dace_cur_'):
            continue
        step = dace.symbolic.pystr_to_symbolic(updates[name]) - dace.symbolic.symbol(name)
        result[name] = (dace.symbolic.pystr_to_symbolic(init), step.expand())
    return result


def _cursor_of(loop: LoopRegion, array: str):
    """Name, init and step of the (single) cursor of ``array`` on ``loop``."""
    matches = [(n, v) for n, v in _cursors(loop).items() if n.startswith(f'__dace_cur_{array}_')]
    assert len(matches) == 1, matches
    return matches[0][0], matches[0][1][0], matches[0][1][1]


def _for_header(code: str) -> str:
    m = re.search(r'for \([^{]*__dace_cur[^{]*\{', code)
    assert m is not None, 'no loop header advancing a cursor in:\n' + code
    return m.group(0)


def _advance(cursor: str, step) -> str:
    return f'{cursor} = ({cursor} + {step})'


# ---------------------------------------------------------------------------------------------------------------
# Default schedule
# ---------------------------------------------------------------------------------------------------------------
def test_default_schedule_is_copy_on_access():
    m = dace.Memlet('A[0]')
    assert isinstance(m.schedule, CopyOnAccess) and m.schedule.is_default
    assert 'schedule' not in m.to_json()  # the default is not serialized
    assert isinstance(dace.Memlet.from_memlet(m).schedule, CopyOnAccess)
    sdfg = _strided_concrete.to_sdfg()
    sdfg2 = dace.SDFG.from_json(json.loads(json.dumps(sdfg.to_json())))
    assert all(isinstance(e.data.schedule, CopyOnAccess) for st in sdfg2.all_states() for e in st.edges())


# ---------------------------------------------------------------------------------------------------------------
# Shared cursor, per-access immediates, inner map parameter in the immediate
# ---------------------------------------------------------------------------------------------------------------
@dace.program
def _shared_and_inner(A: dace.float32[N, 16], B: dace.float32[N], C: dace.float32[N, 16]):
    for i in range(N):
        B[i] = A[i, 3] + 2 * A[i, 7]
        for j in dace.map[0:16]:
            C[i, j] = A[i, j] * 3


def test_schedule_shared_cursor_and_inner_immediate():
    sdfg = _shared_and_inner.to_sdfg()
    res = ScheduleLoopCursors(scope='all').apply_pass(sdfg, {})
    assert res is not None and res['scheduled'] >= 4
    # A[i,3], A[i,7] and A[i,j] share one class (same array, step 16, same non-constant base); B and C
    # have one class each.
    assert res['classes'] == 3
    for m in _scheduled_memlets(sdfg):
        assert isinstance(m.schedule, LoopCursor)
        assert m.schedule.loop and m.schedule.variable == 'i'
        assert not m.schedule.is_lowered
    steps = {m.data: str(m.schedule.step) for m in _scheduled_memlets(sdfg)}
    assert steps == {'A': '16', 'B': '1', 'C': '16'}

    low = LowerMemletSchedules().apply_pass(sdfg, {})
    assert low is not None and low['cursors'] == 3 and low['dropped'] == 0
    loop = _loops(sdfg)[0]
    cur_a, init_a, step_a = _cursor_of(loop, 'A')
    cur_c, _, step_c = _cursor_of(loop, 'C')
    assert str(step_a) == '16' and str(step_c) == '16' and str(_cursor_of(loop, 'B')[2]) == '1'
    assert str(init_a) == '0'  # anchored at A[i, 0], the lowest member (the map access A[i, j])
    # Lowered memlets address the flat references; the flat references are set once at SDFG entry.
    assert all(m.schedule.is_lowered and m.data == m.schedule.reference for m in _scheduled_memlets(sdfg))
    assert sdfg.start_block.label == '__dace_memlet_schedule_init'
    assert isinstance(sdfg.arrays['__dace_flat_A'], dace.data.Reference)
    sdfg.validate()

    code = _code(sdfg)
    header = _for_header(code)
    for name, (init, step) in _cursors(loop).items():  # three cursors, each initialized and advanced
        assert f'{name} = {init}' in header and _advance(name, step) in header
    assert f'__dace_flat_A[({cur_a} + 3)]' in code and f'__dace_flat_A[({cur_a} + 7)]' in code
    # Inner map parameter stays in the (loop-invariant) immediate.
    assert f'__dace_flat_A[({cur_a} + j)]' in code and f'__dace_flat_C[({cur_c} + j)]' in code
    # No full multiply-add offset left for the scheduled arrays inside the loop body.
    assert '16 * i' not in code and '16*i' not in code

    n = 33
    A = np.random.default_rng(0).random((n, 16)).astype(np.float32)
    B = np.zeros(n, np.float32)
    C = np.zeros((n, 16), np.float32)
    sdfg(A=A, B=B, C=C, N=n)
    np.testing.assert_allclose(B, A[:, 3] + 2 * A[:, 7])
    np.testing.assert_allclose(C, A * 3)


# ---------------------------------------------------------------------------------------------------------------
# Strided loop, concrete extent -> int32 cursor; symbolic extent -> int64 unless assume_int32
# ---------------------------------------------------------------------------------------------------------------
@dace.program
def _strided_concrete(A: dace.float32[64, 16], B: dace.float32[64]):
    for i in range(0, 64, 4):
        B[i] = A[i, 5]


def test_strided_loop_int32_cursor():
    sdfg = _strided_concrete.to_sdfg()
    assert ScheduleLoopCursors(scope='all').apply_pass(sdfg, {})['classes'] == 2
    LowerMemletSchedules().apply_pass(sdfg, {})
    loop = _loops(sdfg)[0]
    cur_a, init_a, step_a = _cursor_of(loop, 'A')
    cur_b, init_b, step_b = _cursor_of(loop, 'B')
    assert str(step_a) == '64' and str(step_b) == '4'
    assert sdfg.symbols[cur_a] == dace.int32 and sdfg.symbols[cur_b] == dace.int32
    # A[i, 5] is the only member of its class: the cursor is anchored at column 5 (added once, before the loop).
    assert str(init_a) == '5' and str(init_b) == '0'
    code = _code(sdfg)
    assert f'int {cur_a};' in code
    assert f'{cur_a} = 5' in _for_header(code) and f'__dace_flat_A[{cur_a}]' in code
    A = np.random.default_rng(1).random((64, 16)).astype(np.float32)
    B = np.zeros(64, np.float32)
    sdfg(A=A, B=B)
    np.testing.assert_allclose(B[::4], A[::4, 5])
    assert np.all(B[1::4] == 0)


def test_symbolic_extent_int64_and_assume_int32():
    sdfg = _shared_and_inner.to_sdfg()
    ScheduleLoopCursors(scope='all').apply_pass(sdfg, {})
    LowerMemletSchedules().apply_pass(sdfg, {})
    assert all(sdfg.symbols[c] == dace.int64 for c in _cursors(_loops(sdfg)[0]))

    sdfg = _shared_and_inner.to_sdfg()
    ScheduleLoopCursors(scope='all').apply_pass(sdfg, {})
    LowerMemletSchedules(assume_int32=True).apply_pass(sdfg, {})
    assert all(sdfg.symbols[c] == dace.int32 for c in _cursors(_loops(sdfg)[0]))

    # Explicit per-memlet request wins over auto.
    sdfg = _shared_and_inner.to_sdfg()
    ScheduleLoopCursors(scope='all').apply_pass(sdfg, {})
    for m in _scheduled_memlets(sdfg):
        m.schedule.cursor_type = dace.int64
    LowerMemletSchedules(assume_int32=True).apply_pass(sdfg, {})
    assert all(sdfg.symbols[c] == dace.int64 for c in _cursors(_loops(sdfg)[0]))

    # The analysis pass can pin the type for every schedule it creates.
    sdfg = _shared_and_inner.to_sdfg()
    ScheduleLoopCursors(scope='all', cursor_type=dace.int16).apply_pass(sdfg, {})
    assert all(m.schedule.cursor_type == dace.int16 for m in _scheduled_memlets(sdfg))
    LowerMemletSchedules().apply_pass(sdfg, {})
    cursors = _cursors(_loops(sdfg)[0])
    assert all(sdfg.symbols[c] == dace.int16 for c in cursors)
    code = _code(sdfg)
    assert all(f'{dace.int16.ctype} {c};' in code for c in cursors)


# ---------------------------------------------------------------------------------------------------------------
# Non-affine and loop-independent accesses are left alone
# ---------------------------------------------------------------------------------------------------------------
@dace.program
def _nonaffine(A: dace.float32[64], B: dace.float32[N], C: dace.float32[N]):
    for i in range(N):
        B[i] = A[i * i] + C[0]


def test_nonaffine_left_alone():
    sdfg = _nonaffine.to_sdfg()
    res = ScheduleLoopCursors(scope='all').apply_pass(sdfg, {})
    assert res is not None
    scheduled = {m.data for m in _scheduled_memlets(sdfg)}
    assert 'A' not in scheduled  # quadratic in i
    assert 'C' not in scheduled  # independent of i
    assert 'B' in scheduled
    assert res['skipped'] >= 2
    LowerMemletSchedules().apply_pass(sdfg, {})
    code = _code(sdfg)
    assert 'i * i' in code or 'i*i' in code
    n = 8
    A = np.arange(64, dtype=np.float32)
    B = np.zeros(n, np.float32)
    C = np.ones(n, np.float32)
    sdfg(A=A, B=B, C=C, N=n)
    np.testing.assert_allclose(B, np.arange(n, dtype=np.float32)**2 + 1)


# ---------------------------------------------------------------------------------------------------------------
# Loop inside a map -> nested SDFG: schedules live in the nested SDFG, cursors in its loop
# ---------------------------------------------------------------------------------------------------------------
@dace.program
def _loop_in_map(A: dace.float32[4, N, 16], B: dace.float32[4, N]):
    for k in dace.map[0:4]:
        for i in range(N):
            B[k, i] = A[k, i, 3] + A[k, i, 9]


def test_loop_inside_map_nested_sdfg():
    sdfg = _loop_in_map.to_sdfg()
    res = ScheduleLoopCursors(scope='all').apply_pass(sdfg, {})
    assert res is not None and res['classes'] >= 2
    low = LowerMemletSchedules().apply_pass(sdfg, {})
    assert low['cursors'] >= 2 and low['dropped'] == 0
    sdfg.validate()
    code = _code(sdfg)
    _for_header(code)
    n = 10
    A = np.random.default_rng(2).random((4, n, 16)).astype(np.float32)
    B = np.zeros((4, n), np.float32)
    sdfg(A=A, B=B, N=n)
    np.testing.assert_allclose(B, A[:, :, 3] + A[:, :, 9])


# ---------------------------------------------------------------------------------------------------------------
# Serialization, staleness, idempotence
# ---------------------------------------------------------------------------------------------------------------
def test_schedule_survives_serialization():
    sdfg = _shared_and_inner.to_sdfg()
    ScheduleLoopCursors(scope='all').apply_pass(sdfg, {})
    before = sorted(repr(sorted(m.schedule.to_json().items())) for m in _scheduled_memlets(sdfg))
    text = json.dumps(sdfg.to_json())
    assert 'LoopCursor' in text and 'CopyOnAccess' not in text
    sdfg2 = dace.SDFG.from_json(json.loads(text))
    after = sorted(repr(sorted(m.schedule.to_json().items())) for m in _scheduled_memlets(sdfg2))
    assert before == after
    # The lowered SDFG (symbols, references, loop statements) is an ordinary SDFG and round-trips too.
    LowerMemletSchedules().apply_pass(sdfg2, {})
    sdfg3 = dace.SDFG.from_json(json.loads(json.dumps(sdfg2.to_json())))
    assert _cursors(_loops(sdfg3)[0]) == _cursors(_loops(sdfg2)[0])
    assert all(m.schedule.is_lowered for m in _scheduled_memlets(sdfg3))
    assert _code(sdfg3) == _code(sdfg2)


def test_stale_schedule_is_dropped_not_miscompiled():
    sdfg = _strided_concrete.to_sdfg()
    ScheduleLoopCursors(scope='all').apply_pass(sdfg, {})
    # Simulate a transformation that changed the subset after scheduling (step would now be 2*64).
    for m in _scheduled_memlets(sdfg):
        if m.data == 'A':
            m.subset = dace.subsets.Range.from_string('2*i, 5')
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always')
        low = LowerMemletSchedules().apply_pass(sdfg, {})
    assert low['dropped'] == 1
    assert any('stale' in str(x.message) for x in w)
    assert all(m.data != 'A' for m in _scheduled_memlets(sdfg))
    A = np.random.default_rng(3).random((64, 16)).astype(np.float32)
    B = np.zeros(64, np.float32)
    sdfg(A=A, B=B)
    np.testing.assert_allclose(B[::4][:8], A[::8, 5][:8])


def test_lowering_is_idempotent():
    sdfg = _shared_and_inner.to_sdfg()
    ScheduleLoopCursors(scope='all').apply_pass(sdfg, {})
    first = LowerMemletSchedules().apply_pass(sdfg, {})
    code1 = _code(sdfg)
    nodes1 = sum(st.number_of_nodes() for st in sdfg.all_states())
    second = LowerMemletSchedules().apply_pass(sdfg, {})
    code2 = _code(sdfg)
    assert first['memlets'] == second['memlets'] and second['cursors'] == 0
    assert sum(st.number_of_nodes() for st in sdfg.all_states()) == nodes1
    assert len(_cursors(_loops(sdfg)[0])) == 3
    assert code1 == code2


def test_manual_schedule_lowering_matches_analysis():
    # A hand-written schedule (what a tuner would set) is honored as long as it re-derives consistently.
    from dace.transformation.passes.memlet_schedules import leaf_edges
    sdfg = _strided_concrete.to_sdfg()
    for st in sdfg.all_states():
        for e in leaf_edges(st):
            if e.data.data == 'A':
                e.data.schedule = LoopCursor(_loops(sdfg)[0].label, 'i', 64, cursor_type=dace.int32)
    low = LowerMemletSchedules().apply_pass(sdfg, {})
    assert low['cursors'] == 1 and low['dropped'] == 0
    cur, _, _ = _cursor_of(_loops(sdfg)[0], 'A')
    assert sdfg.symbols[cur] == dace.int32


# ---------------------------------------------------------------------------------------------------------------
# GPU scope filter (code generation only; no device needed)
# ---------------------------------------------------------------------------------------------------------------
def test_gpu_scope_filter_skips_host_loops():
    sdfg = _shared_and_inner.to_sdfg()
    assert ScheduleLoopCursors(scope='gpu').apply_pass(sdfg, {}) is None
    assert not _scheduled_memlets(sdfg)


@dace.program
def _gpu_loop(A: dace.float32[N, 64] @ dace.StorageType.GPU_Global,
              B: dace.float32[N, 64] @ dace.StorageType.GPU_Global):
    for t in dace.map[0:64] @ dace.ScheduleType.GPU_Device:
        acc = dace.float32(0)
        for i in range(N):
            acc = acc + A[i, t]
        B[0, t] = acc


@pytest.mark.parametrize('backend', ['cuda', 'hip'])
def test_gpu_kernel_loop_gets_cursors_in_kernel_code(backend):
    with dace.config.set_temporary('compiler', 'cuda', 'backend', value=backend):
        sdfg = _gpu_loop.to_sdfg()
        res = ScheduleLoopCursors(scope='gpu').apply_pass(sdfg, {})
        assert res is not None and res['scheduled'] >= 1
        # The frontend slices A[0:N, t] into a nested array bound at ``A + t`` (the lane part lives in that
        # binding); the loop-carried access inside the kernel loop is what gets the schedule.
        steps = [str(m.schedule.step) for m in _scheduled_memlets(sdfg)]
        assert '64' in steps
        low = LowerMemletSchedules().apply_pass(sdfg, {})
        assert low['cursors'] >= 1
        sdfg.validate()
        gpu_code = [c.clean_code for c in sdfg.generate_code() if c.title == 'CUDA'][0]
        assert '__global__' in gpu_code
        header = _for_header(gpu_code)
        loop = [l for l in _loops(sdfg) if _cursors(l)][0]
        cur = [c for c, (_, step) in _cursors(loop).items() if str(step) == '64'][0]
        assert _advance(cur, 64) in header
        assert f'[{cur}]' in gpu_code or f'[({cur}' in gpu_code


# ---------------------------------------------------------------------------------------------------------------
# Loop nests: inner cursors are chained to outer cursors (one add per level, no multiplies)
# ---------------------------------------------------------------------------------------------------------------
M = dace.symbol('M')


@dace.program
def _rect_nest(A: dace.float32[N, M], B: dace.float32[M], C: dace.float32[N, M]):
    for i in range(N):
        for j in range(M):
            C[i, j] = A[i, j] + B[j]


@dace.program
def _tri_nest(A: dace.float32[N, N], out: dace.float32[N]):
    for i in range(N):
        s = dace.float32(0)
        for j in range(i, N):
            s = s + A[i, j]
        out[i] = s


def test_nested_loops_chain_cursors():
    sdfg = _rect_nest.to_sdfg()
    ScheduleLoopCursors(scope='all').apply_pass(sdfg, {})
    low = LowerMemletSchedules().apply_pass(sdfg, {})
    # A and C: inner cursor (step 1) chained to an outer cursor (step M); B: inner cursor only.
    assert low['cursors'] == 5 and low['memlets'] == 3
    loops = {l.loop_variable: l for l in _loops(sdfg)}
    outer_a, outer_a_init, outer_a_step = _cursor_of(loops['i'], 'A')
    outer_c, _, _ = _cursor_of(loops['i'], 'C')
    inner_a, inner_a_init, inner_a_step = _cursor_of(loops['j'], 'A')
    inner_b, inner_b_init, _ = _cursor_of(loops['j'], 'B')
    inner_c, inner_c_init, _ = _cursor_of(loops['j'], 'C')
    assert {c for c in _cursors(loops['i'])} == {outer_a, outer_c}
    assert str(outer_a_step) == 'M' and str(inner_a_step) == '1' and str(outer_a_init) == '0'
    assert str(inner_a_init) == outer_a and str(inner_c_init) == outer_c and str(inner_b_init) == '0'
    code = _code(sdfg)
    assert f'{inner_a} = {outer_a}' in code
    assert 'M * i' not in code and 'M*i' not in code
    A = np.random.default_rng(0).random((7, 5)).astype(np.float32)
    B = np.random.default_rng(1).random(5).astype(np.float32)
    C = np.zeros((7, 5), np.float32)
    sdfg(A=A, B=B, C=C, N=7, M=5)
    np.testing.assert_allclose(C, A + B)


def test_nested_loops_chaining_can_be_disabled():
    sdfg = _rect_nest.to_sdfg()
    ScheduleLoopCursors(scope='all').apply_pass(sdfg, {})
    low = LowerMemletSchedules(chain_outer_loops=False).apply_pass(sdfg, {})
    assert low['cursors'] == 3
    loops = {l.loop_variable: l for l in _loops(sdfg)}
    assert not _cursors(loops['i'])
    assert str(_cursor_of(loops['j'], 'A')[1]) == 'M*i'


def test_triangular_nest_diagonal_step():
    sdfg = _tri_nest.to_sdfg()
    ScheduleLoopCursors(scope='all').apply_pass(sdfg, {})
    low = LowerMemletSchedules().apply_pass(sdfg, {})
    assert low['dropped'] == 0
    loops = {l.loop_variable: l for l in _loops(sdfg)}
    outer_a, _, outer_step = _cursor_of(loops['i'], 'A')
    _, inner_init, _ = _cursor_of(loops['j'], 'A')
    # Entry address of row i is A[i, i] = (N+1)*i: the outer cursor walks the diagonal.
    assert str(outer_step) == 'N + 1' and str(inner_init) == outer_a
    A = np.random.default_rng(2).random((9, 9)).astype(np.float32)
    out = np.zeros(9, np.float32)
    sdfg(A=A, out=out, N=9)
    np.testing.assert_allclose(out, np.triu(A).sum(axis=1), rtol=1e-5)


# ---------------------------------------------------------------------------------------------------------------
# Code generation lowers schedules by itself; the user's SDFG stays descriptive
# ---------------------------------------------------------------------------------------------------------------
def test_codegen_lowers_schedules_automatically():
    sdfg = _strided_concrete.to_sdfg()
    ScheduleLoopCursors(scope='all').apply_pass(sdfg, {})
    assert not _cursors(_loops(sdfg)[0])
    code = _code(sdfg)
    assert '__dace_flat_A[' in code and _advance('__dace_cur_A_' + _loops(sdfg)[0].label, 64) in _for_header(code)
    # Lowering happened on the code-generation copy only.
    assert not _cursors(_loops(sdfg)[0]) and '__dace_flat_A' not in sdfg.arrays
    assert all(not m.schedule.is_lowered for m in _scheduled_memlets(sdfg))
    A = np.random.default_rng(4).random((64, 16)).astype(np.float32)
    B = np.zeros(64, np.float32)
    sdfg(A=A, B=B)
    np.testing.assert_allclose(B[::4], A[::4, 5])


# ---------------------------------------------------------------------------------------------------------------
# Non-contiguous reads go through a per-iteration window reference; non-contiguous writes are not scheduled
# ---------------------------------------------------------------------------------------------------------------
def _window_sdfg(write: bool) -> dace.SDFG:
    """``for i in range(N - 1)``: copy the 2x2 block ``A[i:i+2, 0:2]`` (non-contiguous for M > 2) into a transient
    tile and reduce it into ``out[i]`` (``write=False``), or copy a 2x2 source into that block (``write=True``)."""
    sdfg = dace.SDFG('window_write' if write else 'window_read')
    sdfg.add_array('A', [N, M], dace.float32)
    sdfg.add_symbol('i', dace.int64)
    loop = LoopRegion('loop', 'i < N - 1', 'i', 'i = 0', 'i = i + 1')
    sdfg.add_node(loop, is_start_block=True)
    state = loop.add_state('body', is_start_block=True)
    block = dace.Memlet(data='A', subset='i:i + 2, 0:2', other_subset='0:2, 0:2')
    if write:
        sdfg.add_array('src', [2, 2], dace.float32)
        state.add_edge(state.add_read('src'), None, state.add_write('A'), None, block)
    else:
        sdfg.add_array('out', [N], dace.float32)
        sdfg.add_transient('tile', [2, 2], dace.float32)
        tile = state.add_access('tile')
        state.add_edge(state.add_read('A'), None, tile, None, block)
        tasklet = state.add_tasklet('corners', {'t'}, {'o'}, 'o = t[0, 0] + t[1, 1]')
        state.add_edge(tile, None, tasklet, 't', dace.Memlet('tile[0:2, 0:2]'))
        state.add_edge(tasklet, 'o', state.add_write('out'), None, dace.Memlet('out[i]'))
    return sdfg


def test_non_contiguous_read_uses_window_reference():
    sdfg = _window_sdfg(write=False)
    res = ScheduleLoopCursors(scope='all').apply_pass(sdfg, {})
    assert res is not None and {m.data for m in _scheduled_memlets(sdfg)} == {'A', 'out'}
    low = LowerMemletSchedules().apply_pass(sdfg, {})
    assert low == {'cursors': 2, 'memlets': 2, 'dropped': 0}
    sdfg.validate()
    windows = [n for n, d in sdfg.arrays.items() if n.startswith('__dace_win_A_')]
    assert len(windows) == 1
    win = sdfg.arrays[windows[0]]
    assert isinstance(win, dace.data.Reference) and tuple(win.shape) == (2, 2)
    assert tuple(win.strides) == tuple(sdfg.arrays['A'].strides)
    cur, init, step = _cursor_of(_loops(sdfg)[0], 'A')
    assert str(step) == 'M' and str(init) == '0'
    lowered = [m for m in _scheduled_memlets(sdfg) if m.schedule.window]
    assert len(lowered) == 1 and lowered[0].data == windows[0] and str(lowered[0].subset) == '0:2, 0:2'
    code = _code(sdfg)
    # The window is set once per iteration from the flat reference at the cursor, then read with its 2-D shape.
    assert f'{windows[0]} = (float*)(__dace_flat_A + {cur})' in code
    n, m = 6, 5
    A = np.random.default_rng(7).random((n, m)).astype(np.float32)
    out = np.zeros(n, np.float32)
    sdfg(A=A, out=out, N=n, M=m)
    np.testing.assert_allclose(out[:-1], A[:-1, 0] + A[1:, 1])


def test_non_contiguous_write_is_not_scheduled():
    sdfg = _window_sdfg(write=True)
    ScheduleLoopCursors(scope='all').apply_pass(sdfg, {})
    assert 'A' not in {m.data for m in _scheduled_memlets(sdfg)}


# ---------------------------------------------------------------------------------------------------------------
# Leaf memlets are the innermost edges of memlet paths: an access node inside a map fed through the map entry
# ---------------------------------------------------------------------------------------------------------------
def _access_map_access_sdfg() -> dace.SDFG:
    """``for i in range(N)``: a map over j copies ``A[i, j]`` into a scalar transient inside the map (access node
    -> map entry -> access node), doubles it and writes ``B[i, j]``."""
    sdfg = dace.SDFG('access_map_access')
    sdfg.add_array('A', [N, M], dace.float32)
    sdfg.add_array('B', [N, M], dace.float32)
    sdfg.add_scalar('tmp', dace.float32, transient=True)
    sdfg.add_symbol('i', dace.int64)
    loop = LoopRegion('loop', 'i < N', 'i', 'i = 0', 'i = i + 1')
    sdfg.add_node(loop, is_start_block=True)
    state = loop.add_state('body', is_start_block=True)
    entry, exit_ = state.add_map('cols', {'j': '0:M'})
    a, b, tmp = state.add_read('A'), state.add_write('B'), state.add_access('tmp')
    tasklet = state.add_tasklet('double', {'x'}, {'y'}, 'y = 2 * x')
    state.add_memlet_path(a, entry, tmp, memlet=dace.Memlet('A[i, j]'))
    state.add_edge(tmp, None, tasklet, 'x', dace.Memlet('tmp[0]'))
    state.add_memlet_path(tasklet, exit_, b, src_conn='y', memlet=dace.Memlet('B[i, j]'))
    return sdfg


def test_access_node_inside_map_is_a_leaf():
    sdfg = _access_map_access_sdfg()
    res = ScheduleLoopCursors(scope='all').apply_pass(sdfg, {})
    assert {m.data for m in _scheduled_memlets(sdfg)} == {'A', 'B'}
    assert res['scheduled'] == 2
    low = LowerMemletSchedules().apply_pass(sdfg, {})
    assert low == {'cursors': 2, 'memlets': 2, 'dropped': 0}
    sdfg.validate()
    state = next(iter(sdfg.all_states()))
    # The map-entry -> tmp edge now reads the flat reference; the outer edge was re-propagated over the map.
    inner = [e for e in state.edges() if isinstance(e.dst, dace.nodes.AccessNode) and e.dst.data == 'tmp'
             and not isinstance(e.src, dace.nodes.AccessNode)][0]
    cur, init, step = _cursor_of(_loops(sdfg)[0], 'A')
    assert inner.data.data == '__dace_flat_A' and str(inner.data.subset) == f'{cur} + j'
    outer = state.memlet_path(inner)[0]
    assert outer.src.data == '__dace_flat_A' and str(outer.data.subset) == f'{cur}:{cur} + M'
    assert str(step) == 'M' and str(init) == '0'
    n, m = 5, 7
    A = np.random.default_rng(8).random((n, m)).astype(np.float32)
    B = np.zeros_like(A)
    sdfg(A=A, B=B, N=n, M=m)
    np.testing.assert_allclose(B, 2 * A)


# ---------------------------------------------------------------------------------------------------------------
# Stress test 1: 3-D stencil on a window with padded strides and a nonzero base offset
# ---------------------------------------------------------------------------------------------------------------
K = dace.symbol('K')


@dace.program
def _stencil3d(A: dace.float32[N, M, K], B: dace.float32[N, M, K]):
    for i in range(1, N - 1):
        for j in range(1, M - 1):
            for k in range(1, K - 1):
                B[i, j, k] = (A[i - 1, j, k] + A[i + 1, j, k] + A[i, j - 1, k] + A[i, j + 1, k] + A[i, j, k - 1] +
                              A[i, j, k + 1] - 6 * A[i, j, k])


def _loop_text(code: str, var: str) -> str:
    """Text of the loop ``for (<var> = ...`` including its body (up to the matching closing brace)."""
    start = code.index(f'for ({var} = ')
    depth = 0
    for pos in range(code.index('{', start), len(code)):
        if code[pos] == '{':
            depth += 1
        elif code[pos] == '}':
            depth -= 1
            if depth == 0:
                return code[start:pos]
    raise AssertionError('unbalanced braces in generated code')


def _accesses(text: str, array: str):
    return re.findall(rf'\b{array}\[([^\]]*)\]', text)


_LOOPVAR_MULTIPLY = re.compile(r'\* \(?[ijk]\b|\b[ijk]\)? \*')


def test_stencil3d_window_padded_strides_and_offset():
    sdfg = _stencil3d.to_sdfg()
    # A and B are (N, M, K) windows starting at (1, 1, 1) inside padded (N+2, M+2, K+2) buffers: strides differ
    # from the shape, and the window base must be applied to the pointer before the loops are entered.
    pm, pk = M + 2, K + 2
    for name in ('A', 'B'):
        desc = sdfg.arrays[name]
        desc.strides = (pm * pk, pk, 1)
        desc.total_size = (N + 2) * pm * pk
        desc.offset = (1, 1, 1)

    res = ScheduleLoopCursors(scope='all').apply_pass(sdfg, {})
    assert res['scheduled'] == 8 and res['classes'] == 2  # the seven reads of A form one class
    low = LowerMemletSchedules().apply_pass(sdfg, {})
    assert low == {'cursors': 6, 'memlets': 8, 'dropped': 0}
    sdfg.validate()
    loops = {l.loop_variable: l for l in _loops(sdfg)}
    cur = {(v, arr): _cursor_of(loops[v], arr) for v in 'ijk' for arr in 'AB'}
    assert all(len(_cursors(loops[v])) == 2 for v in 'ijk')  # two live address registers per level
    # The flat references point at the buffers' physical element 0; the window base (descriptor offset (1,1,1) plus
    # loop start (1,1,1) in padded strides) is folded into the outermost cursor's init, and inner cursors start
    # exactly at their outer cursor. B's class has one member, so it is anchored at B[1,1,1] (physical (2,2,2));
    # A's class is anchored at its lowest-address member A[i-1,j,k] = A[0,1,1] (physical (1,2,2)), which makes all
    # seven immediates non-negative.
    physical = {'A': pm * pk + 2 * pk + 2, 'B': 2 * pm * pk + 2 * pk + 2}
    for arr in 'AB':
        assert (cur['i', arr][1] - physical[arr]).expand() == 0
        assert str(cur['j', arr][1]) == cur['i', arr][0] and str(cur['k', arr][1]) == cur['j', arr][0]
        assert (cur['i', arr][2] - pm * pk).expand() == 0
        assert (cur['j', arr][2] - pk).expand() == 0 and str(cur['k', arr][2]) == '1'

    code = _code(sdfg)
    body = _loop_text(code, 'k')
    reads = _accesses(body, '__dace_flat_A')
    assert len(reads) == 7 and len(set(reads)) == 7
    # All seven reads go through the single k-level cursor of A: the anchor access has no immediate at all, the
    # other six carry a positive stride combination that no longer mentions any loop variable.
    ka = cur['k', 'A'][0]
    assert reads.count(ka) == 1
    assert all(r == ka or (r.lstrip('(').startswith(f'{ka} + ') and '-' not in r) for r in reads)
    assert not any(re.search(r'\b[ijk]\b', r) for r in reads)
    assert _accesses(body, '__dace_flat_B') == [cur['k', 'B'][0]]
    assert _LOOPVAR_MULTIPLY.search(_loop_text(code, 'i')) is None

    n, m, k = 6, 5, 7
    big_a = np.random.default_rng(5).random((n + 2, m + 2, k + 2)).astype(np.float32)
    big_b = np.zeros_like(big_a)
    sdfg(A=big_a, B=big_b, N=n, M=m, K=k)
    a = big_a[1:-1, 1:-1, 1:-1]
    ref = np.zeros_like(a)
    ref[1:-1, 1:-1, 1:-1] = (a[:-2, 1:-1, 1:-1] + a[2:, 1:-1, 1:-1] + a[1:-1, :-2, 1:-1] + a[1:-1, 2:, 1:-1] +
                             a[1:-1, 1:-1, :-2] + a[1:-1, 1:-1, 2:] - 6 * a[1:-1, 1:-1, 1:-1])
    np.testing.assert_allclose(big_b[1:-1, 1:-1, 1:-1], ref, rtol=1e-5)
    halo = big_b.copy()
    halo[1:-1, 1:-1, 1:-1] = 0
    assert not halo.any()


# ---------------------------------------------------------------------------------------------------------------
# Stress test 2: a K loop with two successive I/J nests keeps advancing the K cursors instead of rebasing
# ---------------------------------------------------------------------------------------------------------------
@dace.program
def _kloop_two_nests(A: dace.float32[N, M, K], B: dace.float32[N, M, K], C: dace.float32[N, M, K]):
    for k in range(K):
        for i in range(N):
            for j in range(M):
                B[i, j, k] = A[i, j, k] * 2
        for i in range(N):
            for j in range(M):
                C[i, j, k] = B[i, j, k] + A[i, j, k]


def test_k_loop_with_successive_ij_nests_shares_k_cursors():
    sdfg = _kloop_two_nests.to_sdfg()
    ScheduleLoopCursors(scope='all').apply_pass(sdfg, {})
    low = LowerMemletSchedules().apply_pass(sdfg, {})
    # K level: A, B, C; first nest: A, B per level; second nest: A, B, C per level.
    assert low == {'cursors': 13, 'memlets': 5, 'dropped': 0}
    sdfg.validate()
    loops = _loops(sdfg)
    kloop = [l for l in loops if l.loop_variable == 'k'][0]
    iloops = [l for l in loops if l.loop_variable == 'i']
    jloops = [l for l in loops if l.loop_variable == 'j']
    assert len(iloops) == 2 and len(jloops) == 2
    kcur = {arr: _cursor_of(kloop, arr) for arr in 'ABC'}
    assert all(str(init) == '0' and str(step) == '1' for _, init, step in kcur.values())
    for iloop in iloops:
        jloop = [l for l in jloops if l.parent_graph is iloop][0]
        for name, (init, step) in _cursors(iloop).items():
            arr = name.split('_')[4]
            assert str(init) == kcur[arr][0] and str(step) == 'K*M'
        for name, (init, step) in _cursors(jloop).items():
            arr = name.split('_')[4]
            assert str(init) == _cursor_of(iloop, arr)[0] and str(step) == 'K'

    code = _code(sdfg)
    ktext = _loop_text(code, 'k')
    kheader = ktext[:ktext.index('{')]
    for name, _, _ in kcur.values():  # three cursors, each initialized and advanced by one
        assert f'{name} = 0' in kheader and _advance(name, 1) in kheader
    # Each nest re-enters with a plain assignment from the (still advancing) K cursor -- no per-nest rebase.
    for arr in 'AB':
        assert ktext.count(f'= {kcur[arr][0]},') + ktext.count(f'= {kcur[arr][0]};') == 2
    assert ktext.count(f'= {kcur["C"][0]},') + ktext.count(f'= {kcur["C"][0]};') == 1
    assert _LOOPVAR_MULTIPLY.search(ktext) is None
    for arr in 'ABC':
        for acc in _accesses(ktext, f'__dace_flat_{arr}'):
            assert acc.startswith('__dace_cur_') and '+' not in acc

    n, m, k = 4, 3, 5
    A = np.random.default_rng(6).random((n, m, k)).astype(np.float32)
    B = np.zeros_like(A)
    C = np.zeros_like(A)
    sdfg(A=A, B=B, C=C, N=n, M=m, K=k)
    np.testing.assert_allclose(B, 2 * A)
    np.testing.assert_allclose(C, 3 * A)


if __name__ == '__main__':
    test_default_schedule_is_copy_on_access()
    test_schedule_shared_cursor_and_inner_immediate()
    test_strided_loop_int32_cursor()
    test_symbolic_extent_int64_and_assume_int32()
    test_nonaffine_left_alone()
    test_loop_inside_map_nested_sdfg()
    test_schedule_survives_serialization()
    test_stale_schedule_is_dropped_not_miscompiled()
    test_lowering_is_idempotent()
    test_manual_schedule_lowering_matches_analysis()
    test_gpu_scope_filter_skips_host_loops()
    test_gpu_kernel_loop_gets_cursors_in_kernel_code('cuda')
    test_gpu_kernel_loop_gets_cursors_in_kernel_code('hip')
    test_nested_loops_chain_cursors()
    test_nested_loops_chaining_can_be_disabled()
    test_triangular_nest_diagonal_step()
    test_codegen_lowers_schedules_automatically()
    test_non_contiguous_read_uses_window_reference()
    test_non_contiguous_write_is_not_scheduled()
    test_access_node_inside_map_is_a_leaf()
    test_stencil3d_window_padded_strides_and_offset()
    test_k_loop_with_successive_ij_nests_shares_k_cursors()
