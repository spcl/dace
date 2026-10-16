# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.

import numpy as np
import pytest

import dace
from dace import data, dtypes, subsets, symbolic
from dace.codegen.py import utils as pyutils
from dace.codegen.py.framecode import DaCePythonCodeGenerator
from dace.codegen.py.prettycode import PythonCodeIOStream
from dace.codegen.py.python_target import PythonCodeGen
from dace.memlet import Memlet
from dace.sdfg import SDFG


def _make_python_sdfg(name: str) -> SDFG:
    sdfg = SDFG(name)
    sdfg.backend = dtypes.BackendLanguage.Python
    return sdfg


def _make_codegen(sdfg: SDFG):
    frame = DaCePythonCodeGenerator(sdfg)
    target = PythonCodeGen(frame, sdfg)
    return frame, target


def test_direct_accessnode_copy_remains_supported():
    sdfg = _make_python_sdfg('direct_copy_supported')
    sdfg.add_array('A', [4], dace.float64)
    sdfg.add_array('B', [4], dace.float64)
    state = sdfg.add_state('state')
    state.add_edge(state.add_read('A'), None, state.add_write('B'), None, Memlet('A[0:4] -> [0:4]'))

    generated = sdfg.generate_code()[0].code
    assert 'A' in generated
    assert 'B' in generated

    csdfg = sdfg.compile()
    a = np.arange(4, dtype=np.float64)
    b = np.zeros(4, dtype=np.float64)
    csdfg(A=a, B=b)

    np.testing.assert_array_equal(b, a)


def test_tasklet_mediated_array_copy_executes():
    sdfg = _make_python_sdfg('tasklet_copy_executes')
    sdfg.add_array('A', [6], dace.float64)
    sdfg.add_array('B', [6], dace.float64)
    state = sdfg.add_state('state')
    tasklet = state.add_tasklet('copy', {'inp'}, {'out'}, 'out = inp')
    state.add_edge(state.add_read('A'), None, tasklet, 'inp', Memlet('A[0:6]'))
    state.add_edge(tasklet, 'out', state.add_write('B'), None, Memlet('B[0:6]'))

    csdfg = sdfg.compile()
    a = np.arange(6, dtype=np.float64)
    b = np.zeros(6, dtype=np.float64)
    csdfg(A=a, B=b)

    np.testing.assert_array_equal(b, a)


def test_sequential_map_executes_with_python_range_loop():
    sdfg = _make_python_sdfg('sequential_map_executes')
    sdfg.add_array('A', [8], dace.float64)
    sdfg.add_array('B', [8], dace.float64)
    state = sdfg.add_state('state')
    map_entry, map_exit = state.add_map('m', {'i': '0:8'}, schedule=dtypes.ScheduleType.Sequential)
    tasklet = state.add_tasklet('double', {'inp'}, {'out'}, 'out = inp * 2')
    state.add_memlet_path(state.add_read('A'), map_entry, tasklet, dst_conn='inp', memlet=Memlet('A[i]'))
    state.add_memlet_path(tasklet, map_exit, state.add_write('B'), src_conn='out', memlet=Memlet('B[i]'))

    generated = sdfg.generate_code()[0].code
    assert 'for i in range(0, (7) + 1):' in generated

    csdfg = sdfg.compile()
    a = np.arange(8, dtype=np.float64)
    b = np.zeros(8, dtype=np.float64)
    csdfg(A=a, B=b)

    np.testing.assert_array_equal(b, a * 2)


def test_persistent_transient_survives_across_calls():
    sdfg = _make_python_sdfg('persistent_transient')
    sdfg.add_array('out', [1], dace.int64)
    sdfg.add_scalar('counter', dace.int64, transient=True, lifetime=dtypes.AllocationLifetime.Persistent)
    state = sdfg.add_state('state')
    counter_read = state.add_read('counter')
    counter_write = state.add_write('counter')
    out_write = state.add_write('out')
    tasklet = state.add_tasklet('inc', {'current'}, {'next_value', 'result'}, 'next_value = current + 1\nresult = next_value')
    state.add_edge(counter_read, None, tasklet, 'current', Memlet('counter'))
    state.add_edge(tasklet, 'next_value', counter_write, None, Memlet('counter'))
    state.add_edge(tasklet, 'result', out_write, None, Memlet('out[0]'))

    csdfg = sdfg.compile()
    out = np.zeros(1, dtype=np.int64)

    csdfg(out=out)
    assert out[0] == 1

    csdfg(out=out)
    assert out[0] == 2


def test_nested_sdfg_uses_python_helper_and_executes():
    outer = _make_python_sdfg('outer_nested_python')
    outer.add_array('A', [1], dace.float64)
    outer.add_array('B', [1], dace.float64)

    inner = SDFG('inner_nested_python')
    inner.backend = dtypes.BackendLanguage.Python
    inner.add_array('X', [1], dace.float64)
    inner.add_array('Y', [1], dace.float64)
    inner_state = inner.add_state('inner_state', is_start_block=True)
    tasklet = inner_state.add_tasklet('scale', {'inp'}, {'out'}, 'out = inp * 3')
    inner_state.add_edge(inner_state.add_read('X'), None, tasklet, 'inp', Memlet('X[0]'))
    inner_state.add_edge(tasklet, 'out', inner_state.add_write('Y'), None, Memlet('Y[0]'))

    outer_state = outer.add_state('outer_state')
    nested = outer_state.add_nested_sdfg(inner, {'X'}, {'Y'})
    outer_state.add_edge(outer_state.add_read('A'), None, nested, 'X', Memlet('A[0:1]'))
    outer_state.add_edge(nested, 'Y', outer_state.add_write('B'), None, Memlet('B[0:1]'))

    generated = outer.generate_code()[0].code
    assert 'def inner_nested_python_' in generated

    csdfg = outer.compile()
    a = np.array([4.0], dtype=np.float64)
    b = np.zeros(1, dtype=np.float64)
    csdfg(A=a, B=b)

    assert b[0] == 12.0


def test_nested_sdfg_multiple_scalar_connectors_preserve_order_and_bridge_scalars():
    outer = _make_python_sdfg('outer_nested_scalar_connectors')
    outer.add_array('A', [1], dace.int64)
    outer.add_array('B', [1], dace.int64)
    outer.add_array('C', [1], dace.int64)
    outer.add_array('D', [1], dace.int64)

    inner = SDFG('inner_nested_scalar_connectors')
    inner.backend = dtypes.BackendLanguage.Python
    inner.add_scalar('x', dace.int64)
    inner.add_scalar('y', dace.int64)
    inner.add_scalar('u', dace.int64)
    inner.add_scalar('v', dace.int64)
    inner_state = inner.add_state('inner_state', is_start_block=True)
    tasklet = inner_state.add_tasklet('mix', {'lhs', 'rhs'}, {'first', 'second'},
                                      'first = lhs + 10 * rhs\nsecond = 100 * lhs + rhs')
    inner_state.add_edge(inner_state.add_read('x'), None, tasklet, 'lhs', Memlet('x'))
    inner_state.add_edge(inner_state.add_read('y'), None, tasklet, 'rhs', Memlet('y'))
    inner_state.add_edge(tasklet, 'first', inner_state.add_write('u'), None, Memlet('u'))
    inner_state.add_edge(tasklet, 'second', inner_state.add_write('v'), None, Memlet('v'))

    outer_state = outer.add_state('outer_state', is_start_block=True)
    nested = outer_state.add_nested_sdfg(inner, {'x', 'y'}, {'u', 'v'})
    outer_state.add_edge(outer_state.add_read('A'), None, nested, 'x', Memlet('A[0]'))
    outer_state.add_edge(outer_state.add_read('B'), None, nested, 'y', Memlet('B[0]'))
    outer_state.add_edge(nested, 'u', outer_state.add_write('C'), None, Memlet('C[0]'))
    outer_state.add_edge(nested, 'v', outer_state.add_write('D'), None, Memlet('D[0]'))

    csdfg = outer.compile()
    a = np.array([2], dtype=np.int64)
    b = np.array([3], dtype=np.int64)
    c = np.zeros(1, dtype=np.int64)
    d = np.zeros(1, dtype=np.int64)
    csdfg(A=a, B=b, C=c, D=d)

    np.testing.assert_array_equal(c, np.array([32], dtype=np.int64))
    np.testing.assert_array_equal(d, np.array([203], dtype=np.int64))


def test_nested_sdfg_singleton_array_connector_bridges_to_outer_scalar():
    outer = _make_python_sdfg('outer_nested_singleton_array_bridge')
    outer.add_array('A', [1], dace.int64)
    outer.add_array('B', [1], dace.int64)
    outer.add_scalar('tmp', dace.int64, transient=True)

    inner = SDFG('inner_nested_singleton_array_bridge')
    inner.backend = dtypes.BackendLanguage.Python
    inner.add_array('X', [1], dace.int64)
    inner.add_array('Y', [1], dace.int64)
    inner_state = inner.add_state('inner_state', is_start_block=True)
    tasklet = inner_state.add_tasklet('inc', {'inp'}, {'out'}, 'out = inp + 1')
    inner_state.add_edge(inner_state.add_read('X'), None, tasklet, 'inp', Memlet('X[0]'))
    inner_state.add_edge(tasklet, 'out', inner_state.add_write('Y'), None, Memlet('Y[0]'))

    outer_state = outer.add_state('outer_state', is_start_block=True)
    nested = outer_state.add_nested_sdfg(inner, {'X'}, {'Y'})
    outer_state.add_edge(outer_state.add_read('A'), None, nested, 'X', Memlet('A[0:1]'))
    outer_state.add_edge(nested, 'Y', outer_state.add_write('tmp'), None, Memlet('tmp'))
    passthrough = outer_state.add_tasklet('passthrough', {'inp'}, {'out'}, 'out = inp * 2')
    outer_state.add_edge(outer_state.add_read('tmp'), None, passthrough, 'inp', Memlet('tmp'))
    outer_state.add_edge(passthrough, 'out', outer_state.add_write('B'), None, Memlet('B[0]'))

    generated = outer.generate_code()[0].code
    assert '__dace_nested_scalar_' in generated
    assert 'tmp[0:1]' not in generated

    csdfg = outer.compile()
    a = np.array([4], dtype=np.int64)
    b = np.zeros(1, dtype=np.int64)
    csdfg(A=a, B=b)

    np.testing.assert_array_equal(b, np.array([10], dtype=np.int64))


def test_structure_transient_allocation_uses_python_constructor():
    sdfg = _make_python_sdfg('structure_allocation')
    struct_desc = data.Structure({
        'value': data.Scalar(dace.int32),
        'buffer': data.Array(dace.float64, (2,)),
    }, name='Pair', transient=True)
    sdfg.add_datadesc('tmp', struct_desc)
    state = sdfg.add_state('state')
    access = state.add_access('tmp')
    _, target = _make_codegen(sdfg)
    allocation_stream = PythonCodeIOStream()

    target.allocate_array(
        sdfg,
        sdfg,
        state,
        state.block_id,
        access,
        sdfg.arrays['tmp'],
        PythonCodeIOStream(),
        PythonCodeIOStream(),
        allocation_stream,
    )

    generated = allocation_stream.getvalue()
    assert 'Pair(' in generated
    assert 'value=0' in generated
    assert 'buffer=numpy.empty' in generated


def test_advanced_indexing_helper_emits_python_index_expression():
    desc = data.Array(dace.float64, (10,))
    index_subset = subsets.Range.from_indices([symbolic.pystr_to_symbolic('idxs')])

    expression = pyutils.data_access_expression('A', desc, index_subset)

    assert expression == 'A[idxs]'
