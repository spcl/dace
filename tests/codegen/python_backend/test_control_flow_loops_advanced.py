# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""
Tests for loop-related control flow code generation in the Python backend.

Covers all paths through ``_write_loop_region()`` (C-style for loops with
inverted/non-inverted variants, while-style loops, empty bodies) as well as
execution correctness for loops with break and continue blocks.
"""

import ast
import numpy as np
import pytest

import dace
from dace import dtypes
from dace.config import set_temporary
from dace.properties import CodeBlock
from dace.sdfg import SDFG, InterstateEdge
from dace.sdfg.state import (ControlFlowRegion, LoopRegion, BreakBlock, ContinueBlock, SDFGState)
from dace.codegen.py.control_flow import (
    control_flow_region_to_code,
    _write_loop_region,
    _unparse_py_expr,
    _unparse_codeblock,
)
from dace.codegen.py.prettycode import PythonCodeIOStream


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class _FakeCodegen:
    """Minimal stub for the codegen parameter."""
    pass


def _make_dispatch():
    """Returns a dispatch_state callable that records calls and returns dummy code."""
    dispatched = []

    def dispatch_state(state: SDFGState) -> str:
        dispatched.append(state.label)
        return f'# state: {state.label}\n'

    return dispatch_state, dispatched


def _gen(region, dispatch=None, codegen=None, symbols=None):
    """Helper: generate code for a region and return the string."""
    if dispatch is None:
        dispatch, _ = _make_dispatch()
    if codegen is None:
        codegen = _FakeCodegen()
    if symbols is None:
        symbols = {}
    stream = PythonCodeIOStream()
    with set_temporary('compiler', 'codegen_lineinfo', value=False):
        control_flow_region_to_code(region, dispatch, codegen, symbols, stream)
    return stream.getvalue()


def _gen_loop(loop, dispatch=None, codegen=None, symbols=None):
    """Helper: generate code for a loop region directly."""
    if dispatch is None:
        dispatch, _ = _make_dispatch()
    if codegen is None:
        codegen = _FakeCodegen()
    if symbols is None:
        symbols = {}
    stream = PythonCodeIOStream()
    with set_temporary('compiler', 'codegen_lineinfo', value=False):
        _write_loop_region(loop, dispatch, codegen, symbols, stream)
    return stream.getvalue()


# ---------------------------------------------------------------------------
# C-style for loop tests
# ---------------------------------------------------------------------------

class TestForLoops:
    """Tests for _write_loop_region with init/update/variable (C-style for)."""

    def test_for_loop_normal(self):
        """Non-inverted C-style for loop: init; while cond: body; update."""
        sdfg = SDFG('test_for_normal')
        loop = LoopRegion('forloop', condition_expr='i < 10', loop_var='i',
                          initialize_expr='i = 0', update_expr='i = i + 1', sdfg=sdfg)
        sdfg.add_node(loop, is_start_block=True)
        loop.add_state('loop_body', is_start_block=True)

        code = _gen_loop(loop)
        assert 'i = 0' in code
        assert 'while' in code
        assert 'i < 10' in code
        assert 'i = i + 1' in code or 'i = (i + 1)' in code
        # Should NOT have 'while True' (that's the inverted form)
        assert 'while True' not in code

    def test_for_loop_inverted_update_before_condition(self):
        """Inverted for loop with update_before_condition=True:
        init; while True: body; update; if not cond: break
        """
        sdfg = SDFG('test_for_inv_before')
        loop = LoopRegion('forloop', condition_expr='i < 10', loop_var='i',
                          initialize_expr='i = 0', update_expr='i = i + 1',
                          inverted=True, update_before_condition=True, sdfg=sdfg)
        sdfg.add_node(loop, is_start_block=True)
        loop.add_state('loop_body', is_start_block=True)

        code = _gen_loop(loop)
        assert 'i = 0' in code
        assert 'while True:' in code
        assert 'break' in code
        # update should appear before the condition check
        update_pos = code.find('i = i + 1') if 'i = i + 1' in code else code.find('i = (i + 1)')
        cond_pos = code.find('if not')
        assert update_pos < cond_pos, "update should come before the condition check"

    def test_for_loop_inverted_update_after_condition(self):
        """Inverted for loop with update_before_condition=False:
        init; while True: body; if not cond: break; update
        """
        sdfg = SDFG('test_for_inv_after')
        loop = LoopRegion('forloop', condition_expr='i < 10', loop_var='i',
                          initialize_expr='i = 0', update_expr='i = i + 1',
                          inverted=True, update_before_condition=False, sdfg=sdfg)
        sdfg.add_node(loop, is_start_block=True)
        loop.add_state('loop_body', is_start_block=True)

        code = _gen_loop(loop)
        assert 'i = 0' in code
        assert 'while True:' in code
        assert 'break' in code
        # condition check should appear before update
        cond_pos = code.find('if not')
        update_pos = code.find('i = i + 1') if 'i = i + 1' in code else code.find('i = (i + 1)')
        assert cond_pos < update_pos, "condition check should come before update"


# ---------------------------------------------------------------------------
# While-style loop tests
# ---------------------------------------------------------------------------

class TestWhileLoops:
    """Tests for _write_loop_region without init/update/variable (while-style)."""

    def test_while_loop_normal(self):
        """Non-inverted while loop: while cond: body."""
        sdfg = SDFG('test_while_normal')
        loop = LoopRegion('wloop', condition_expr='x > 0', sdfg=sdfg)
        sdfg.add_node(loop, is_start_block=True)
        loop.add_state('loop_body', is_start_block=True)

        code = _gen_loop(loop)
        assert 'while' in code
        assert 'x > 0' in code
        assert 'while True' not in code

    def test_while_loop_empty_body(self):
        """Non-inverted while loop with empty body: while cond: pass."""
        sdfg = SDFG('test_while_empty')
        loop = LoopRegion('wloop', condition_expr='flag', sdfg=sdfg)
        sdfg.add_node(loop, is_start_block=True)
        # Add a single state with nothing in it
        loop.add_state('empty_body', is_start_block=True)

        # dispatch returns empty string for the state
        def empty_dispatch(state):
            return ''

        stream = PythonCodeIOStream()
        with set_temporary('compiler', 'codegen_lineinfo', value=False):
            _write_loop_region(loop, empty_dispatch, _FakeCodegen(), {}, stream)
        code = stream.getvalue()
        assert 'while' in code
        assert 'pass' in code

    def test_while_loop_inverted(self):
        """Inverted while loop: while True: body; if not cond: break."""
        sdfg = SDFG('test_while_inv')
        loop = LoopRegion('wloop', condition_expr='x > 0', inverted=True, sdfg=sdfg)
        sdfg.add_node(loop, is_start_block=True)
        loop.add_state('loop_body', is_start_block=True)

        code = _gen_loop(loop)
        assert 'while True:' in code
        assert 'break' in code
        assert 'if not' in code


# ---------------------------------------------------------------------------
# Execution correctness tests
# ---------------------------------------------------------------------------

class TestLoopCorrectnessForLoop:
    """Compile-and-run tests for for-style loops."""

    def test_for_loop_sum_correctness(self):
        """For loop summing i=0..9, check result == 45."""
        sdfg = SDFG('test_for_sum')
        sdfg.backend = dace.dtypes.BackendLanguage.Python

        sdfg.add_array('result', [1], dace.int64)

        # Init state: result = 0
        init = sdfg.add_state('init', is_start_block=True)
        loop = LoopRegion('sumloop', condition_expr='i < 10', loop_var='i',
                          initialize_expr='i = 0', update_expr='i = i + 1', sdfg=sdfg)
        sdfg.add_node(loop)
        sdfg.add_edge(init, loop, InterstateEdge())

        init_t = init.add_tasklet('init_result', {}, {'out'}, 'out = 0')
        init_result = init.add_access('result')
        init.add_edge(init_t, 'out', init_result, None, dace.Memlet(data='result', subset='0'))

        # Loop body: result = result + i
        body = loop.add_state('body', is_start_block=True)
        t = body.add_tasklet('add', {'r'}, {'out'}, 'out = r + i')
        r_read = body.add_access('result')
        r_write = body.add_access('result')
        body.add_edge(r_read, None, t, 'r', dace.Memlet(data='result', subset='0'))
        body.add_edge(t, 'out', r_write, None, dace.Memlet(data='result', subset='0'))

        # After loop state to read result
        after = sdfg.add_state('after')
        sdfg.add_edge(loop, after, InterstateEdge())

        with set_temporary('compiler', 'codegen_lineinfo', value=False):
            csdfg = sdfg.compile()
            result = np.zeros(1, dtype=np.int64)
            csdfg(result=result)
            assert result[0] == 45

    def test_for_loop_array_fill_correctness(self):
        """For loop filling A[i] = i for i=0..4, check against numpy."""
        N = 5
        sdfg = SDFG('test_for_fill')
        sdfg.backend = dace.dtypes.BackendLanguage.Python

        sdfg.add_array('A', [N], dace.int64)

        loop = LoopRegion('fillloop', condition_expr=f'i < {N}', loop_var='i',
                          initialize_expr='i = 0', update_expr='i = i + 1', sdfg=sdfg)
        sdfg.add_node(loop, is_start_block=True)

        body = loop.add_state('body', is_start_block=True)
        t = body.add_tasklet('assign', {}, {'out'}, 'out = i')
        a_write = body.add_access('A')
        body.add_edge(t, 'out', a_write, None, dace.Memlet(data='A', subset='i'))

        with set_temporary('compiler', 'codegen_lineinfo', value=False):
            csdfg = sdfg.compile()
            A = np.zeros(N, dtype=np.int64)
            csdfg(A=A)
            np.testing.assert_array_equal(A, np.arange(N))


class TestLoopCorrectnessWhile:
    """Compile-and-run tests for while-style loops."""

    def test_while_loop_correctness(self):
        """While loop: result *= 2 while result < 100, starting at 1. Expect 128."""
        sdfg = SDFG('test_while_corr')
        sdfg.backend = dace.dtypes.BackendLanguage.Python

        sdfg.add_array('result', [1], dace.int64)

        # Init state: result = 1
        init = sdfg.add_state('init', is_start_block=True)
        loop = LoopRegion('wloop', condition_expr='result[0] < 100', sdfg=sdfg)
        sdfg.add_node(loop)
        sdfg.add_edge(init, loop, InterstateEdge())

        init_t = init.add_tasklet('init_result', {}, {'out'}, 'out = 1')
        init_result = init.add_access('result')
        init.add_edge(init_t, 'out', init_result, None, dace.Memlet(data='result', subset='0'))

        # Loop body: result = result * 2
        body = loop.add_state('body', is_start_block=True)
        t = body.add_tasklet('double', {'r'}, {'out'}, 'out = r * 2')
        r_read = body.add_access('result')
        r_write = body.add_access('result')
        body.add_edge(r_read, None, t, 'r', dace.Memlet(data='result', subset='0'))
        body.add_edge(t, 'out', r_write, None, dace.Memlet(data='result', subset='0'))

        after = sdfg.add_state('after')
        sdfg.add_edge(loop, after, InterstateEdge())

        with set_temporary('compiler', 'codegen_lineinfo', value=False):
            csdfg = sdfg.compile()
            result = np.zeros(1, dtype=np.int64)
            csdfg(result=result)
            assert result[0] == 128

    def test_inverted_loop_correctness(self):
        """Inverted loop (do-while): body executes at least once even if condition is false.
        result starts at 1, body doubles it, condition result < 1 → should still double once → 2.
        """
        sdfg = SDFG('test_inv_corr')
        sdfg.backend = dace.dtypes.BackendLanguage.Python

        sdfg.add_array('result', [1], dace.int64)

        init = sdfg.add_state('init', is_start_block=True)
        # condition: result < 1 → false after init, but inverted → body runs once first
        loop = LoopRegion('invloop', condition_expr='result[0] < 1', inverted=True, sdfg=sdfg)
        sdfg.add_node(loop)
        sdfg.add_edge(init, loop, InterstateEdge())

        init_t = init.add_tasklet('init_result', {}, {'out'}, 'out = 1')
        init_result = init.add_access('result')
        init.add_edge(init_t, 'out', init_result, None, dace.Memlet(data='result', subset='0'))

        body = loop.add_state('body', is_start_block=True)
        t = body.add_tasklet('double', {'r'}, {'out'}, 'out = r * 2')
        r_read = body.add_access('result')
        r_write = body.add_access('result')
        body.add_edge(r_read, None, t, 'r', dace.Memlet(data='result', subset='0'))
        body.add_edge(t, 'out', r_write, None, dace.Memlet(data='result', subset='0'))

        after = sdfg.add_state('after')
        sdfg.add_edge(loop, after, InterstateEdge())

        with set_temporary('compiler', 'codegen_lineinfo', value=False):
            csdfg = sdfg.compile()
            result = np.zeros(1, dtype=np.int64)
            csdfg(result=result)
            # Body runs once (doubling 1→2), then condition (2 < 1) is false → exit
            assert result[0] == 2


class TestLoopCorrectnessNested:
    """Compile-and-run tests for nested loops."""

    def test_nested_loops_correctness(self):
        """Nested loops: sum += i * 10 + j for i in 0..2 and j in 0..2.
        Expected: (0+1+2) + (10+11+12) + (20+21+22) = 3 + 33 + 63 = 99.
        Actually: sum of (i*10 + j) for i=0..2, j=0..2 = 0+1+2+10+11+12+20+21+22 = 99
        """
        sdfg = SDFG('test_nested')
        sdfg.backend = dace.dtypes.BackendLanguage.Python

        sdfg.add_array('result', [1], dace.int64)

        init = sdfg.add_state('init', is_start_block=True)

        # Outer loop over i
        outer = LoopRegion('outer', condition_expr='i < 3', loop_var='i',
                           initialize_expr='i = 0', update_expr='i = i + 1', sdfg=sdfg)
        sdfg.add_node(outer)
        sdfg.add_edge(init, outer, InterstateEdge())

        init_t = init.add_tasklet('init_result', {}, {'out'}, 'out = 0')
        init_result = init.add_access('result')
        init.add_edge(init_t, 'out', init_result, None, dace.Memlet(data='result', subset='0'))

        # Inner loop over j (child of outer)
        inner = LoopRegion('inner', condition_expr='j < 3', loop_var='j',
                           initialize_expr='j = 0', update_expr='j = j + 1', sdfg=sdfg)
        outer.add_node(inner, is_start_block=True)

        # Inner body: result = result + i * 10 + j
        body = inner.add_state('body', is_start_block=True)
        t = body.add_tasklet('add', {'r'}, {'out'}, 'out = r + i * 10 + j')
        r_read = body.add_access('result')
        r_write = body.add_access('result')
        body.add_edge(r_read, None, t, 'r', dace.Memlet(data='result', subset='0'))
        body.add_edge(t, 'out', r_write, None, dace.Memlet(data='result', subset='0'))

        after = sdfg.add_state('after')
        sdfg.add_edge(outer, after, InterstateEdge())

        with set_temporary('compiler', 'codegen_lineinfo', value=False):
            csdfg = sdfg.compile()
            result = np.zeros(1, dtype=np.int64)
            csdfg(result=result)
            assert result[0] == 99


class TestLoopCorrectnessBreakContinue:
    """Compile-and-run tests for loops with break and continue blocks."""

    def test_loop_with_break_correctness(self):
        """Loop with break: sum i=0..9 but break when i==5. Result = 0+1+2+3+4 = 10.

        Structure:
            loop (i=0..9):
                body_state → (if i==5) break_block
                           → (if i!=5) add_state
        """
        sdfg = SDFG('test_break')
        sdfg.backend = dace.dtypes.BackendLanguage.Python

        sdfg.add_array('result', [1], dace.int64)

        init = sdfg.add_state('init', is_start_block=True)

        loop = LoopRegion('sumloop', condition_expr='i < 10', loop_var='i',
                          initialize_expr='i = 0', update_expr='i = i + 1', sdfg=sdfg)
        sdfg.add_node(loop)
        sdfg.add_edge(init, loop, InterstateEdge())

        init_t = init.add_tasklet('init_result', {}, {'out'}, 'out = 0')
        init_result = init.add_access('result')
        init.add_edge(init_t, 'out', init_result, None, dace.Memlet(data='result', subset='0'))

        # Guard state (dispatch target)
        guard = loop.add_state('guard', is_start_block=True)
        # Break block
        brk = loop.add_break('do_break')
        # Add state for accumulation
        add_state = loop.add_state('add')
        t = add_state.add_tasklet('add', {'r'}, {'out'}, 'out = r + i')
        r_read = add_state.add_access('result')
        r_write = add_state.add_access('result')
        add_state.add_edge(r_read, None, t, 'r', dace.Memlet(data='result', subset='0'))
        add_state.add_edge(t, 'out', r_write, None, dace.Memlet(data='result', subset='0'))

        loop.add_edge(guard, brk, InterstateEdge(condition=CodeBlock('i == 5')))
        loop.add_edge(guard, add_state, InterstateEdge(condition=CodeBlock('i != 5')))

        after = sdfg.add_state('after')
        sdfg.add_edge(loop, after, InterstateEdge())

        with set_temporary('compiler', 'codegen_lineinfo', value=False):
            csdfg = sdfg.compile()
            result = np.zeros(1, dtype=np.int64)
            csdfg(result=result)
            assert result[0] == 10

    def test_loop_with_continue_correctness(self):
        """Loop with continue: take continue once, then accumulate i=0..9.

        Structure:
            loop (i=0..9):
                guard → (if do_continue==1) continue_block and reset symbol
                      → (if do_continue==0) add_state
        """
        sdfg = SDFG('test_continue')
        sdfg.backend = dace.dtypes.BackendLanguage.Python

        sdfg.add_array('result', [1], dace.int64)
        sdfg.add_symbol('do_continue', dace.int64)

        init = sdfg.add_state('init', is_start_block=True)

        loop = LoopRegion('sumloop', condition_expr='i < 10', loop_var='i',
                          initialize_expr='i = 0', update_expr='i = i + 1', sdfg=sdfg)
        sdfg.add_node(loop)
        sdfg.add_edge(init, loop, InterstateEdge(assignments={'do_continue': '1'}))

        init_t = init.add_tasklet('init_result', {}, {'out'}, 'out = 0')
        init_result = init.add_access('result')
        init.add_edge(init_t, 'out', init_result, None, dace.Memlet(data='result', subset='0'))

        guard = loop.add_state('guard', is_start_block=True)
        cont = loop.add_continue('do_continue')
        add_state = loop.add_state('add')
        t = add_state.add_tasklet('add', {'r'}, {'out'}, 'out = r + i')
        r_read = add_state.add_access('result')
        r_write = add_state.add_access('result')
        add_state.add_edge(r_read, None, t, 'r', dace.Memlet(data='result', subset='0'))
        add_state.add_edge(t, 'out', r_write, None, dace.Memlet(data='result', subset='0'))

        loop.add_edge(guard, cont, InterstateEdge(condition=CodeBlock('do_continue == 1'), assignments={'do_continue': '0'}))
        loop.add_edge(guard, add_state, InterstateEdge(condition=CodeBlock('do_continue == 0')))

        after = sdfg.add_state('after')
        sdfg.add_edge(loop, after, InterstateEdge())

        with set_temporary('compiler', 'codegen_lineinfo', value=False):
            csdfg = sdfg.compile()
            result = np.zeros(1, dtype=np.int64)
            csdfg(result=result)
            assert result[0] == 45


# ---------------------------------------------------------------------------
# Code structure tests (non-correctness)
# ---------------------------------------------------------------------------

class TestLoopCodeStructure:
    """Verify the shape of generated code, not execution."""

    def test_for_loop_has_all_parts(self):
        """Generated for loop should have init, condition, body dispatch, update."""
        sdfg = SDFG('test_parts')
        loop = LoopRegion('forloop', condition_expr='i < N', loop_var='i',
                          initialize_expr='i = 0', update_expr='i = i + 1', sdfg=sdfg)
        sdfg.add_node(loop, is_start_block=True)
        body = loop.add_state('body', is_start_block=True)

        dispatch, dispatched = _make_dispatch()
        code = _gen_loop(loop, dispatch=dispatch)

        assert 'body' in dispatched
        assert 'i = 0' in code
        assert 'while' in code
        assert 'i + 1' in code

    def test_while_loop_dispatches_body(self):
        """While loop should dispatch the body state."""
        sdfg = SDFG('test_dispatch')
        loop = LoopRegion('wloop', condition_expr='True', sdfg=sdfg)
        sdfg.add_node(loop, is_start_block=True)
        loop.add_state('mybod', is_start_block=True)

        dispatch, dispatched = _make_dispatch()
        _gen_loop(loop, dispatch=dispatch)
        assert 'mybod' in dispatched

    def test_inverted_for_loop_structure(self):
        """Inverted for loop has while True at top level."""
        sdfg = SDFG('test_inv_struct')
        loop = LoopRegion('forloop', condition_expr='i < 10', loop_var='i',
                          initialize_expr='i = 0', update_expr='i = i + 1',
                          inverted=True, sdfg=sdfg)
        sdfg.add_node(loop, is_start_block=True)
        loop.add_state('body', is_start_block=True)

        code = _gen_loop(loop)
        lines = [l.strip() for l in code.splitlines() if l.strip()]
        # First line: i = 0 (init)
        assert lines[0].startswith('i = 0') or 'i = 0' in lines[0]
        # Second line: while True:
        assert 'while True' in lines[1]

    def test_loop_with_multi_state_body(self):
        """Loop body with two sequential states — both dispatched."""
        sdfg = SDFG('test_multi_body')
        loop = LoopRegion('wloop', condition_expr='True', sdfg=sdfg)
        sdfg.add_node(loop, is_start_block=True)
        s1 = loop.add_state('s1', is_start_block=True)
        s2 = loop.add_state('s2')
        loop.add_edge(s1, s2, InterstateEdge())

        dispatch, dispatched = _make_dispatch()
        _gen_loop(loop, dispatch=dispatch)
        assert 's1' in dispatched
        assert 's2' in dispatched


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
