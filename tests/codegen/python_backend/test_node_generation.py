# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Tests for PythonCodeGen node generation (python_target.py).

Covers generate_node dispatch, _generate_AccessNode, and _generate_Tasklet.
"""
import pytest
import numpy as np

import dace
from dace import dtypes, data
from dace.codegen.py.prettycode import PythonCodeIOStream
from dace.codegen.py.python_target import PythonCodeGen
from dace.dtypes import ScheduleType, Language
from dace.sdfg import nodes, SDFG, NodeNotExpandedError
from dace.memlet import Memlet
def _MAP_XFAIL(func):
    return func


def _make_python_sdfg(name: str) -> SDFG:
    """Create an SDFG with backend set to Python."""
    sdfg = SDFG(name)
    sdfg.backend = dtypes.BackendLanguage.Python
    return sdfg


class _DummyDispatcher:
    def __init__(self):
        self.copies = []

    def register_node_dispatcher(self, _dispatcher):
        return None

    def register_map_dispatcher(self, _schedules, _dispatcher):
        return None

    def register_array_dispatcher(self, _storages, _dispatcher):
        return None

    def register_copy_dispatcher(self, _src_storage, _dst_storage, _wcr, _dispatcher):
        return None

    def dispatch_copy(self, src_node, dst_node, edge, sdfg, cfg, dfg, state_id, function_stream, callsite_stream):
        self.copies.append((src_node, dst_node, edge, state_id))

    def dispatch_subgraph(self,
                          sdfg,
                          cfg,
                          dfg_scope,
                          state_id,
                          function_stream,
                          callsite_stream,
                          skip_entry_node=False,
                          skip_exit_node=False):
        return None


class _DummyFrameCodegen:
    def __init__(self):
        self.dispatcher = _DummyDispatcher()


def _make_codegen(sdfg: SDFG):
    frame = _DummyFrameCodegen()
    codegen = PythonCodeGen(frame, sdfg)
    return codegen, frame.dispatcher


# =============================================================================
# generate_node dispatch
# =============================================================================


class TestGenerateNodeDispatch:
    """Test generate_node routing to the correct handler."""

    def test_generate_node_access_node(self):
        """AccessNode dispatched correctly -- generates code without error."""
        sdfg = _make_python_sdfg('test_access_dispatch')
        sdfg.add_array('A', [10], dace.float64)
        sdfg.add_array('B', [10], dace.float64)
        state = sdfg.add_state('s')
        a = state.add_read('A')
        b = state.add_write('B')
        state.add_edge(a, None, b, None, Memlet(data='A', subset='0:10', other_subset='0:10'))
        code_objs = sdfg.generate_code()
        assert len(code_objs) > 0

    def test_generate_node_tasklet(self):
        """Tasklet dispatched correctly -- generates code without error."""
        sdfg = _make_python_sdfg('test_tasklet_dispatch')
        sdfg.add_scalar('x', dace.float64)
        sdfg.add_scalar('y', dace.float64)
        state = sdfg.add_state('s')
        r = state.add_read('x')
        w = state.add_write('y')
        t = state.add_tasklet('inc', {'a'}, {'b'}, 'b = a + 1')
        state.add_edge(r, None, t, 'a', Memlet(data='x'))
        state.add_edge(t, 'b', w, None, Memlet(data='y'))
        code_objs = sdfg.generate_code()
        assert len(code_objs) > 0

    def test_generate_node_library_node_unexpanded(self):
        """Unexpanded LibraryNode raises error during code generation.

        The expansion step in expand_library_nodes raises KeyError when the
        library is not registered, preventing the node from reaching codegen.
        """

        class _DummyLib(nodes.LibraryNode):
            _dace_library_name = 'dummy_test_lib'
            implementations = {}
            default_implementation = None

        sdfg = _make_python_sdfg('test_lib_node')
        sdfg.add_scalar('x', dace.float64)
        sdfg.add_scalar('y', dace.float64)
        state = sdfg.add_state('s')
        lib = _DummyLib('mylib')
        lib.add_in_connector('inp')
        lib.add_out_connector('out')
        state.add_node(lib)
        r = state.add_read('x')
        w = state.add_write('y')
        state.add_edge(r, None, lib, 'inp', Memlet(data='x'))
        state.add_edge(lib, 'out', w, None, Memlet(data='y'))
        with pytest.raises(KeyError):
            sdfg.generate_code()

    def test_generate_node_unknown_type(self):
        """Mapped code generation succeeds without dispatching MapExit directly."""
        sdfg = _make_python_sdfg('test_unknown_node')
        sdfg.add_array('A', [10], dace.float64)
        sdfg.add_array('B', [10], dace.float64)
        state = sdfg.add_state('s')
        me, mx = state.add_map('m', {'i': '0:10'}, schedule=ScheduleType.Sequential)
        a = state.add_read('A')
        b = state.add_write('B')
        t = state.add_tasklet('t', {'inp'}, {'out'}, 'out = inp')
        state.add_memlet_path(a, me, t, dst_conn='inp', memlet=Memlet(data='A', subset='i'))
        state.add_memlet_path(t, mx, b, src_conn='out', memlet=Memlet(data='B', subset='i'))
        code = sdfg.generate_code()[0].code
        assert 'for i in range' in code


# =============================================================================
# _generate_AccessNode
# =============================================================================


class TestAccessNode:
    """Test _generate_AccessNode edge-handling logic."""

    def test_access_node_incoming_from_code_node(self):
        """Incoming edge from CodeNode (Tasklet) -- handled by tasklet, not access node."""
        sdfg = _make_python_sdfg('test_incoming_code')
        sdfg.add_scalar('x', dace.float64)
        sdfg.add_scalar('y', dace.float64)
        state = sdfg.add_state('s')
        r = state.add_read('x')
        w = state.add_write('y')
        t = state.add_tasklet('add', {'a'}, {'b'}, 'b = a + 1')
        state.add_edge(r, None, t, 'a', Memlet(data='x'))
        state.add_edge(t, 'b', w, None, Memlet(data='y'))
        code_objs = sdfg.generate_code()
        code = code_objs[0].code
        assert 'y[...] = b' in code

    def test_access_node_incoming_copy(self):
        """src is AccessNode, same scope -- dispatch_copy generates copy."""
        sdfg = _make_python_sdfg('test_copy_same_scope')
        sdfg.add_array('A', [10], dace.float64)
        sdfg.add_array('B', [10], dace.float64)
        state = sdfg.add_state('s')
        a = state.add_read('A')
        b = state.add_write('B')
        state.add_edge(a, None, b, None, Memlet(data='A', subset='0:10', other_subset='0:10'))
        code_objs = sdfg.generate_code()
        code = code_objs[0].code
        assert 'A[0:10]' in code and 'B[0:10]' in code

    def test_access_node_outgoing_to_code_node(self):
        """Outgoing edge to CodeNode (Tasklet) -- skip, handled by code node."""
        sdfg = _make_python_sdfg('test_outgoing_code')
        sdfg.add_scalar('x', dace.float64)
        sdfg.add_scalar('y', dace.float64)
        state = sdfg.add_state('s')
        r = state.add_read('x')
        w = state.add_write('y')
        t = state.add_tasklet('double', {'a'}, {'b'}, 'b = a * 2')
        state.add_edge(r, None, t, 'a', Memlet(data='x'))
        state.add_edge(t, 'b', w, None, Memlet(data='y'))
        code_objs = sdfg.generate_code()
        code = code_objs[0].code
        assert 'a = x' in code

    def test_access_node_outgoing_copy(self):
        """Outgoing to AccessNode, same scope -- dispatch_copy generates copy."""
        sdfg = _make_python_sdfg('test_outgoing_copy')
        sdfg.add_array('A', [5], dace.float64)
        sdfg.add_array('B', [5], dace.float64)
        state = sdfg.add_state('s')
        a = state.add_read('A')
        b = state.add_write('B')
        state.add_edge(a, None, b, None, Memlet(data='A', subset='0:5', other_subset='0:5'))
        code_objs = sdfg.generate_code()
        code = code_objs[0].code
        assert 'A' in code and 'B' in code

    def test_access_node_bidirectional(self):
        """AccessNode with both incoming and outgoing edges generates all copies."""
        sdfg = _make_python_sdfg('test_bidirectional')
        sdfg.add_array('A', [10], dace.float64)
        sdfg.add_array('B', [10], dace.float64, transient=True)
        sdfg.add_array('C', [10], dace.float64)
        state = sdfg.add_state('s')
        a = state.add_read('A')
        b = state.add_access('B')
        c = state.add_write('C')
        state.add_edge(a, None, b, None, Memlet(data='A', subset='0:10', other_subset='0:10'))
        state.add_edge(b, None, c, None, Memlet(data='B', subset='0:10', other_subset='0:10'))
        code_objs = sdfg.generate_code()
        code = code_objs[0].code
        assert 'A' in code and 'B' in code and 'C' in code

    @_MAP_XFAIL
    def test_access_node_incoming_different_scope(self):
        """AccessNode copy with scope_contains_scope and different scope."""
        sdfg = _make_python_sdfg('test_diff_scope')
        sdfg.add_array('A', [10], dace.float64)
        sdfg.add_array('B', [10], dace.float64)
        state = sdfg.add_state('s')
        me, mx = state.add_map('m', {'i': '0:10'}, schedule=ScheduleType.Sequential)
        a_read = state.add_read('A')
        b_write = state.add_write('B')
        t = state.add_tasklet('copy', {'inp'}, {'out'}, 'out = inp')
        state.add_memlet_path(a_read, me, t, dst_conn='inp', memlet=Memlet(data='A', subset='i'))
        state.add_memlet_path(t, mx, b_write, src_conn='out', memlet=Memlet(data='B', subset='i'))
        code_objs = sdfg.generate_code()
        code = code_objs[0].code
        assert 'for i in range' in code

    def test_access_node_outgoing_to_self(self):
        """AccessNode with a normal edge -- self-loop path is checked internally."""
        sdfg = _make_python_sdfg('test_self_loop')
        sdfg.add_array('A', [10], dace.float64)
        sdfg.add_array('B', [10], dace.float64)
        state = sdfg.add_state('s')
        a = state.add_read('A')
        b = state.add_write('B')
        state.add_edge(a, None, b, None, Memlet(data='A', subset='0:10', other_subset='0:10'))
        code_objs = sdfg.generate_code()
        assert len(code_objs) > 0


# =============================================================================
# _generate_Tasklet
# =============================================================================


class TestTasklet:
    """Test _generate_Tasklet for input/output/body generation."""

    def test_tasklet_scalar_input(self):
        """Scalar input -- connector = scalar_name."""
        sdfg = _make_python_sdfg('test_scalar_in')
        sdfg.add_scalar('s', dace.float64)
        sdfg.add_scalar('r', dace.float64)
        state = sdfg.add_state('s0')
        read_s = state.add_read('s')
        write_r = state.add_write('r')
        t = state.add_tasklet('t', {'a'}, {'b'}, 'b = a + 1')
        state.add_edge(read_s, None, t, 'a', Memlet(data='s'))
        state.add_edge(t, 'b', write_r, None, Memlet(data='r'))
        code = sdfg.generate_code()[0].code
        assert 'a = s' in code

    def test_tasklet_array_input(self):
        """Array input -- connector = array[subset]."""
        sdfg = _make_python_sdfg('test_array_in')
        sdfg.add_array('A', [10], dace.float64)
        sdfg.add_scalar('r', dace.float64)
        state = sdfg.add_state('s0')
        a = state.add_read('A')
        w = state.add_write('r')
        t = state.add_tasklet('t', {'inp'}, {'out'}, 'out = inp')
        state.add_edge(a, None, t, 'inp', Memlet(data='A', subset='5'))
        state.add_edge(t, 'out', w, None, Memlet(data='r'))
        code = sdfg.generate_code()[0].code
        assert 'inp = A[5]' in code

    def test_tasklet_scalar_output(self):
        """Scalar output -- scalar_name = connector."""
        sdfg = _make_python_sdfg('test_scalar_out')
        sdfg.add_scalar('s', dace.float64)
        sdfg.add_scalar('r', dace.float64)
        state = sdfg.add_state('s0')
        read_s = state.add_read('s')
        write_r = state.add_write('r')
        t = state.add_tasklet('t', {'a'}, {'b'}, 'b = a')
        state.add_edge(read_s, None, t, 'a', Memlet(data='s'))
        state.add_edge(t, 'b', write_r, None, Memlet(data='r'))
        code = sdfg.generate_code()[0].code
        assert 'r[...] = b' in code

    def test_tasklet_array_output(self):
        """Array output -- array[subset] = connector."""
        sdfg = _make_python_sdfg('test_array_out')
        sdfg.add_scalar('s', dace.float64)
        sdfg.add_array('B', [10], dace.float64)
        state = sdfg.add_state('s0')
        read_s = state.add_read('s')
        write_b = state.add_write('B')
        t = state.add_tasklet('t', {'a'}, {'b'}, 'b = a')
        state.add_edge(read_s, None, t, 'a', Memlet(data='s'))
        state.add_edge(t, 'b', write_b, None, Memlet(data='B', subset='3'))
        code = sdfg.generate_code()[0].code
        assert 'B[3] = b' in code

    def test_tasklet_no_input_connectors(self):
        """Edges without dst_conn are skipped."""
        sdfg = _make_python_sdfg('test_no_dst_conn')
        sdfg.add_scalar('x', dace.float64)
        sdfg.add_scalar('y', dace.float64)
        state = sdfg.add_state('s0')
        r = state.add_read('x')
        w = state.add_write('y')
        t = state.add_tasklet('t', set(), {'out'}, 'out = 42.0')
        state.add_edge(r, None, t, None, Memlet(data='x'))
        state.add_edge(t, 'out', w, None, Memlet(data='y'))
        code = sdfg.generate_code()[0].code
        assert 'y[...] = out' in code

    def test_tasklet_no_output_connectors(self):
        """Edges without src_conn are skipped."""
        sdfg = _make_python_sdfg('test_no_src_conn')
        sdfg.add_scalar('x', dace.float64)
        sdfg.add_scalar('y', dace.float64)
        state = sdfg.add_state('s0')
        r = state.add_read('x')
        w = state.add_write('y')
        t = state.add_tasklet('t', {'inp'}, set(), 'pass')
        state.add_edge(r, None, t, 'inp', Memlet(data='x'))
        state.add_edge(t, None, w, None, Memlet(data='y'))
        code = sdfg.generate_code()[0].code
        assert 'inp = x' in code

    def test_tasklet_multiple_inputs_outputs(self):
        """Multiple I/O connectors."""
        sdfg = _make_python_sdfg('test_multi_io')
        sdfg.add_scalar('a', dace.float64)
        sdfg.add_scalar('b', dace.float64)
        sdfg.add_scalar('c', dace.float64)
        sdfg.add_scalar('d', dace.float64)
        state = sdfg.add_state('s0')
        ra = state.add_read('a')
        rb = state.add_read('b')
        wc = state.add_write('c')
        wd = state.add_write('d')
        t = state.add_tasklet('t', {'x', 'y'}, {'p', 'q'}, 'p = x + y\nq = x - y')
        state.add_edge(ra, None, t, 'x', Memlet(data='a'))
        state.add_edge(rb, None, t, 'y', Memlet(data='b'))
        state.add_edge(t, 'p', wc, None, Memlet(data='c'))
        state.add_edge(t, 'q', wd, None, Memlet(data='d'))
        code = sdfg.generate_code()[0].code
        assert 'x = a' in code
        assert 'y = b' in code
        assert 'c[...] = p' in code
        assert 'd[...] = q' in code
        assert 'p = x + y' in code or 'p = (x + y)' in code

    def test_tasklet_python_body_multi_statement(self):
        """Multiple Python statements in tasklet body are all generated."""
        sdfg = _make_python_sdfg('test_multi_stmt')
        sdfg.add_scalar('x', dace.float64)
        sdfg.add_scalar('y', dace.float64)
        state = sdfg.add_state('s0')
        r = state.add_read('x')
        w = state.add_write('y')
        t = state.add_tasklet('t', {'a'}, {'b'}, 'temp = a * 2\nb = temp + 1')
        state.add_edge(r, None, t, 'a', Memlet(data='x'))
        state.add_edge(t, 'b', w, None, Memlet(data='y'))
        code = sdfg.generate_code()[0].code
        assert 'temp = a * 2' in code or 'temp = (a * 2)' in code
        assert 'b = temp + 1' in code or 'b = (temp + 1)' in code

    def test_tasklet_non_python_language(self):
        """Non-Python tasklet language raises NotImplementedError."""
        sdfg = _make_python_sdfg('test_non_python')
        sdfg.add_scalar('x', dace.float64)
        sdfg.add_scalar('y', dace.float64)
        state = sdfg.add_state('s0')
        r = state.add_read('x')
        w = state.add_write('y')
        t = state.add_tasklet('t', {'a'}, {'b'}, 'b = a;', language=Language.CPP)
        state.add_edge(r, None, t, 'a', Memlet(data='x'))
        state.add_edge(t, 'b', w, None, Memlet(data='y'))
        with pytest.raises(NotImplementedError, match="Python backend only supports Python tasklets"):
            sdfg.generate_code()

    def test_tasklet_code_to_code_input_memlet(self):
        """Input memlet with data=None (code-to-code) raises NotImplementedError."""
        sdfg = _make_python_sdfg('test_c2c_in')
        sdfg.add_scalar('y', dace.float64)
        state = sdfg.add_state('s0')

        t1 = state.add_tasklet('src', set(), {'out'}, 'out = 42.0')
        t2 = state.add_tasklet('dst', {'inp'}, {'res'}, 'res = inp')
        w = state.add_write('y')

        state.add_edge(t1, 'out', t2, 'inp', Memlet())
        state.add_edge(t2, 'res', w, None, Memlet(data='y'))

        with pytest.raises(dace.sdfg.validation.InvalidSDFGEdgeError, match="empty memlet"):
            sdfg.generate_code()

    def test_tasklet_code_to_code_output_memlet(self):
        """Output memlet with data=None (code-to-code) raises NotImplementedError."""
        sdfg = _make_python_sdfg('test_c2c_out')
        sdfg.add_scalar('x', dace.float64)
        state = sdfg.add_state('s0')

        r = state.add_read('x')
        t1 = state.add_tasklet('src', {'a'}, {'out'}, 'out = a')
        t2 = state.add_tasklet('dst', {'inp'}, set(), 'pass')

        state.add_edge(r, None, t1, 'a', Memlet(data='x'))
        state.add_edge(t1, 'out', t2, 'inp', Memlet())

        with pytest.raises(dace.sdfg.validation.InvalidSDFGEdgeError, match="empty memlet"):
            sdfg.generate_code()


# =============================================================================
# Direct branch tests for PythonCodeGen internals
# =============================================================================


class TestPythonTargetDirectBranches:
    def test_access_node_memlet_path_dst_mismatch_skips_copy(self, monkeypatch):
        sdfg = _make_python_sdfg('test_access_dst_mismatch')
        sdfg.add_array('A', [1], dace.float64)
        sdfg.add_array('B', [1], dace.float64)
        state = sdfg.add_state('s0')
        a = state.add_read('A')
        b = state.add_write('B')
        edge = state.add_edge(a, None, b, None, Memlet(data='A', subset='0', other_subset='0'))

        codegen, dispatcher = _make_codegen(sdfg)

        class _PathElem:
            def __init__(self, src, dst):
                self.src = src
                self.dst = dst

        fake_path = [_PathElem(a, a)]
        monkeypatch.setattr(state, 'memlet_path', lambda e: fake_path if e is edge else [_PathElem(a, b)])

        codegen._generate_AccessNode(sdfg, sdfg, state, 0, b, PythonCodeIOStream(), PythonCodeIOStream())

        assert dispatcher.copies == []

    def test_access_node_incoming_different_scope_dispatches_copy(self):
        sdfg = _make_python_sdfg('test_access_incoming_diff_scope')
        sdfg.add_array('A', [1], dace.float64)
        sdfg.add_array('B', [1], dace.float64)
        state = sdfg.add_state('s0')
        me, _ = state.add_map('m', {'i': '0:1'}, schedule=ScheduleType.Sequential)
        a = state.add_read('A')
        b = state.add_write('B')
        state.add_memlet_path(a, me, b, memlet=Memlet(data='A', subset='0', other_subset='i'))

        codegen, dispatcher = _make_codegen(sdfg)
        codegen._generate_AccessNode(sdfg, sdfg, state, 0, b, PythonCodeIOStream(), PythonCodeIOStream())

        assert len(dispatcher.copies) >= 1
        assert any(src is a and dst is b for src, dst, *_ in dispatcher.copies)

    def test_access_node_outgoing_self_loop_skips_copy(self, monkeypatch):
        sdfg = _make_python_sdfg('test_access_self_loop_skip')
        sdfg.add_array('A', [1], dace.float64)
        sdfg.add_array('B', [1], dace.float64)
        state = sdfg.add_state('s0')
        a = state.add_read('A')
        b = state.add_write('B')
        edge = state.add_edge(a, None, b, None, Memlet(data='A', subset='0', other_subset='0'))

        codegen, dispatcher = _make_codegen(sdfg)

        class _PathElem:
            def __init__(self, src, dst):
                self.src = src
                self.dst = dst

        monkeypatch.setattr(state, 'memlet_path', lambda e: [_PathElem(a, a)] if e is edge else [_PathElem(a, b)])

        codegen._generate_AccessNode(sdfg, sdfg, state, 0, a, PythonCodeIOStream(), PythonCodeIOStream())

        assert dispatcher.copies == []

    def test_access_node_outgoing_to_inner_scope_skips_copy(self):
        sdfg = _make_python_sdfg('test_access_outgoing_inner_scope_skip')
        sdfg.add_array('A', [1], dace.float64)
        sdfg.add_array('B', [1], dace.float64)
        state = sdfg.add_state('s0')
        me, _ = state.add_map('m', {'i': '0:1'}, schedule=ScheduleType.Sequential)
        a = state.add_read('A')
        b = state.add_write('B')
        state.add_memlet_path(a, me, b, memlet=Memlet(data='A', subset='0', other_subset='i'))

        codegen, dispatcher = _make_codegen(sdfg)
        codegen._generate_AccessNode(sdfg, sdfg, state, 0, a, PythonCodeIOStream(), PythonCodeIOStream())

        assert dispatcher.copies == []

    def test_generate_scope_empty_body_writes_pass(self):
        sdfg = _make_python_sdfg('test_generate_scope_pass')
        state = sdfg.add_state('s0')
        me, _ = state.add_map('m', {'i': '0:1'}, schedule=ScheduleType.Sequential)

        codegen, _ = _make_codegen(sdfg)
        scope = state.scope_subgraph(me)
        stream = PythonCodeIOStream()

        codegen.generate_scope(sdfg, sdfg, scope, 0, stream, stream)

        code = stream.getvalue()
        assert 'for i in range(' in code
        assert 'pass' in code

    def test_map_range_statement_with_symbolic_step_uses_runtime_stop_expression(self):
        sdfg = _make_python_sdfg('test_symbolic_range_statement')
        codegen, _ = _make_codegen(sdfg)

        statement = codegen._map_range_statement('i', 'begin', 'end', 'STEP')

        assert 'if (STEP) > 0 else' in statement

    def test_copy_memory_src_non_access_uses_src_connector(self):
        sdfg = _make_python_sdfg('test_copy_nonaccess_src')
        sdfg.add_scalar('B', dace.float64)
        state = sdfg.add_state('s0')
        src_tasklet = state.add_tasklet('src', set(), {'out'}, 'out = 1.0')
        b = state.add_write('B')
        edge = state.add_edge(src_tasklet, 'out', b, None, Memlet(data='B'))

        codegen, _ = _make_codegen(sdfg)
        stream = PythonCodeIOStream()
        codegen.copy_memory(sdfg, sdfg, state, 0, src_tasklet, b, edge, stream, stream)

        assert 'B[...] = out' in stream.getvalue()

    def test_copy_memory_dst_non_access_uses_dst_connector(self):
        sdfg = _make_python_sdfg('test_copy_nonaccess_dst')
        sdfg.add_scalar('A', dace.float64)
        state = sdfg.add_state('s0')
        a = state.add_read('A')
        dst_tasklet = state.add_tasklet('dst', {'inp'}, set(), 'pass')
        edge = state.add_edge(a, None, dst_tasklet, 'inp', Memlet(data='A'))

        codegen, _ = _make_codegen(sdfg)
        stream = PythonCodeIOStream()
        codegen.copy_memory(sdfg, sdfg, state, 0, a, dst_tasklet, edge, stream, stream)

        assert 'inp = A' in stream.getvalue()

    def test_nested_sdfg_helper_emits_single_header_and_import_block(self):
        sdfg = _make_python_sdfg('nested_header_once')
        sdfg.add_array('A', [1], dace.float64)
        sdfg.add_array('B', [1], dace.float64)

        nested_sdfg = SDFG('nested_header_child')
        nested_sdfg.add_array('X', [1], dace.float64)
        nested_sdfg.add_array('Y', [1], dace.float64)
        nested_state = nested_sdfg.add_state('nested_state', is_start_block=True)
        tasklet = nested_state.add_tasklet('copy', {'inp'}, {'out'}, 'out = inp')
        nested_state.add_edge(nested_state.add_read('X'), None, tasklet, 'inp', Memlet('X[0]'))
        nested_state.add_edge(tasklet, 'out', nested_state.add_write('Y'), None, Memlet('Y[0]'))

        state = sdfg.add_state('state')
        nested_node = state.add_nested_sdfg(nested_sdfg, {'X'}, {'Y'})
        state.add_edge(state.add_read('A'), None, nested_node, 'X', Memlet('A[0]'))
        state.add_edge(nested_node, 'Y', state.add_write('B'), None, Memlet('B[0]'))

        generated_code = sdfg.generate_code()[0].code

        assert generated_code.count('# DaCe AUTO-GENERATED FILE. DO NOT MODIFY') == 1
        assert generated_code.count('import numpy') == 1

    def test_nested_sdfg_helper_propagates_nested_environment_headers(self):
        sdfg = _make_python_sdfg('nested_environment_headers')
        sdfg.add_array('A', [1], dace.float64)
        sdfg.add_array('B', [1], dace.float64)

        nested_sdfg = SDFG('nested_environment_child')
        nested_sdfg.add_array('X', [1], dace.float64)
        nested_sdfg.add_array('Y', [1], dace.float64)
        nested_state = nested_sdfg.add_state('nested_state', is_start_block=True)
        tasklet = nested_state.add_tasklet('copy', {'inp'}, {'out'}, 'out = inp')

        class FakeEnv:
            headers = {'frame': ['import math']}
            dependencies = []
            state_fields = []

        environment_name = 'nested_environment_headers_fake_env'
        dace.library._DACE_REGISTERED_ENVIRONMENTS[environment_name] = FakeEnv
        tasklet.environments = {environment_name}
        nested_state.add_edge(nested_state.add_read('X'), None, tasklet, 'inp', Memlet('X[0]'))
        nested_state.add_edge(tasklet, 'out', nested_state.add_write('Y'), None, Memlet('Y[0]'))

        state = sdfg.add_state('state')
        nested_node = state.add_nested_sdfg(nested_sdfg, {'X'}, {'Y'})
        state.add_edge(state.add_read('A'), None, nested_node, 'X', Memlet('A[0]'))
        state.add_edge(nested_node, 'Y', state.add_write('B'), None, Memlet('B[0]'))

        try:
            generated_code = sdfg.generate_code()[0].code
        finally:
            dace.library._DACE_REGISTERED_ENVIRONMENTS.pop(environment_name, None)

        assert generated_code.count('import math') == 1

    def test_nested_scalar_bridge_rejects_multidimensional_singleton_arrays(self):
        sdfg = _make_python_sdfg('nested_singleton_shape_guard')
        codegen, _ = _make_codegen(sdfg)
        nested_sdfg = SDFG('nested_singleton_child')
        nested_sdfg.add_array('Y', [1, 1], dace.float64)

        with pytest.raises(NotImplementedError, match='size-1 buffers'):
            codegen._nested_buffer_initialization('nested_buffer', nested_sdfg.arrays['Y'])


# =============================================================================
# Correctness tests (compile, run, verify against numpy)
# =============================================================================


class TestNodeCorrectnessE2E:
    """End-to-end correctness tests for node generation."""

    def test_scalar_increment(self):
        """Scalar tasklet: y[0] = x[0] + 1, compile and verify."""
        sdfg = _make_python_sdfg('test_scalar_inc')
        sdfg.add_array('x', [1], dace.float64)
        sdfg.add_array('y', [1], dace.float64)
        state = sdfg.add_state('s')
        r = state.add_read('x')
        w = state.add_write('y')
        t = state.add_tasklet('inc', {'a'}, {'b'}, 'b = a + 1')
        state.add_edge(r, None, t, 'a', Memlet(data='x', subset='0'))
        state.add_edge(t, 'b', w, None, Memlet(data='y', subset='0'))
        csdfg = sdfg.compile()
        x = np.array([5.0], dtype=np.float64)
        y = np.array([0.0], dtype=np.float64)
        csdfg(x=x, y=y)
        assert y[0] == 6.0

    def test_array_copy_correctness(self):
        """Array copy via access nodes, compile and verify."""
        N = 10
        sdfg = _make_python_sdfg('test_arr_copy')
        sdfg.add_array('A', [N], dace.float64)
        sdfg.add_array('B', [N], dace.float64)
        state = sdfg.add_state('s')
        a = state.add_read('A')
        b = state.add_write('B')
        state.add_edge(a, None, b, None, Memlet(data='A', subset='0:10', other_subset='0:10'))
        csdfg = sdfg.compile()
        A = np.arange(N, dtype=np.float64)
        B = np.zeros(N, dtype=np.float64)
        csdfg(A=A, B=B)
        np.testing.assert_array_equal(B, A)

    def test_multi_tasklet_chain(self):
        """Chain of tasklets: x -> t1 -> tmp -> t2 -> y, verify correctness."""
        sdfg = _make_python_sdfg('test_chain')
        sdfg.add_array('x', [1], dace.float64)
        sdfg.add_array('y', [1], dace.float64)
        sdfg.add_scalar('tmp', dace.float64, transient=True)
        state = sdfg.add_state('s')
        rx = state.add_read('x')
        wy = state.add_write('y')
        tmp_access = state.add_access('tmp')
        t1 = state.add_tasklet('t1', {'a'}, {'b'}, 'b = a * 3')
        t2 = state.add_tasklet('t2', {'c'}, {'d'}, 'd = c + 10')
        state.add_edge(rx, None, t1, 'a', Memlet(data='x', subset='0'))
        state.add_edge(t1, 'b', tmp_access, None, Memlet(data='tmp'))
        state.add_edge(tmp_access, None, t2, 'c', Memlet(data='tmp'))
        state.add_edge(t2, 'd', wy, None, Memlet(data='y', subset='0'))
        csdfg = sdfg.compile()
        x = np.array([4.0], dtype=np.float64)
        y = np.array([0.0], dtype=np.float64)
        csdfg(x=x, y=y)
        assert y[0] == 22.0

    @_MAP_XFAIL
    def test_element_wise_with_map(self):
        """Element-wise array operation via map + tasklet, verify against numpy."""
        N = 20
        sdfg = _make_python_sdfg('test_ewise')
        sdfg.add_array('A', [N], dace.float64)
        sdfg.add_array('B', [N], dace.float64)
        state = sdfg.add_state('s')
        me, mx = state.add_map('m', {'i': '0:20'}, schedule=ScheduleType.Sequential)
        a = state.add_read('A')
        b = state.add_write('B')
        t = state.add_tasklet('double', {'inp'}, {'out'}, 'out = inp * 2')
        state.add_memlet_path(a, me, t, dst_conn='inp', memlet=Memlet(data='A', subset='i'))
        state.add_memlet_path(t, mx, b, src_conn='out', memlet=Memlet(data='B', subset='i'))
        csdfg = sdfg.compile()
        A = np.random.rand(N)
        B = np.zeros(N)
        csdfg(A=A, B=B)
        np.testing.assert_allclose(B, A * 2)

    @_MAP_XFAIL
    def test_multiple_io_correctness(self):
        """Tasklet with 2 inputs and 2 outputs in map, verify correctness."""
        N = 8
        sdfg = _make_python_sdfg('test_multi_io_corr')
        sdfg.add_array('A', [N], dace.float64)
        sdfg.add_array('B', [N], dace.float64)
        sdfg.add_array('C', [N], dace.float64)
        sdfg.add_array('D', [N], dace.float64)
        state = sdfg.add_state('s')
        me, mx = state.add_map('m', {'i': '0:8'}, schedule=ScheduleType.Sequential)
        ra = state.add_read('A')
        rb = state.add_read('B')
        wc = state.add_write('C')
        wd = state.add_write('D')
        t = state.add_tasklet('t', {'x', 'y'}, {'p', 'q'}, 'p = x + y\nq = x * y')
        state.add_memlet_path(ra, me, t, dst_conn='x', memlet=Memlet(data='A', subset='i'))
        state.add_memlet_path(rb, me, t, dst_conn='y', memlet=Memlet(data='B', subset='i'))
        state.add_memlet_path(t, mx, wc, src_conn='p', memlet=Memlet(data='C', subset='i'))
        state.add_memlet_path(t, mx, wd, src_conn='q', memlet=Memlet(data='D', subset='i'))
        csdfg = sdfg.compile()
        A = np.random.rand(N)
        B = np.random.rand(N)
        C = np.zeros(N)
        D = np.zeros(N)
        csdfg(A=A, B=B, C=C, D=D)
        np.testing.assert_allclose(C, A + B)
        np.testing.assert_allclose(D, A * B)

    def test_two_state_computation(self):
        """Two states: state1 writes tmp, state2 reads tmp and writes y."""
        sdfg = _make_python_sdfg('test_two_state')
        sdfg.add_array('x', [1], dace.float64)
        sdfg.add_array('y', [1], dace.float64)
        sdfg.add_scalar('tmp', dace.float64, transient=True)

        s1 = sdfg.add_state('s1')
        rx = s1.add_read('x')
        tw = s1.add_write('tmp')
        t1 = s1.add_tasklet('t1', {'a'}, {'b'}, 'b = a + 5')
        s1.add_edge(rx, None, t1, 'a', Memlet(data='x', subset='0'))
        s1.add_edge(t1, 'b', tw, None, Memlet(data='tmp'))

        s2 = sdfg.add_state('s2')
        tr = s2.add_read('tmp')
        wy = s2.add_write('y')
        t2 = s2.add_tasklet('t2', {'c'}, {'d'}, 'd = c * 2')
        s2.add_edge(tr, None, t2, 'c', Memlet(data='tmp'))
        s2.add_edge(t2, 'd', wy, None, Memlet(data='y', subset='0'))

        sdfg.add_edge(s1, s2, dace.InterstateEdge())
        csdfg = sdfg.compile()
        x = np.array([3.0], dtype=np.float64)
        y = np.array([0.0], dtype=np.float64)
        csdfg(x=x, y=y)
        assert y[0] == 16.0

    def test_interstate_variable(self):
        """Interstate variable assigned on edge and used in next state."""
        sdfg = _make_python_sdfg('test_isv')
        sdfg.add_array('y', [1], dace.float64)

        s1 = sdfg.add_state('s1')
        s2 = sdfg.add_state('s2')
        sdfg.add_edge(s1, s2, dace.InterstateEdge(assignments={'val': '42'}))

        wy = s2.add_write('y')
        t = s2.add_tasklet('t', set(), {'out'}, 'out = val')
        s2.add_edge(t, 'out', wy, None, Memlet(data='y', subset='0'))

        csdfg = sdfg.compile()
        y = np.array([0.0], dtype=np.float64)
        csdfg(y=y)
        assert y[0] == 42.0
