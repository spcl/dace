import itertools

import numpy as np
import pytest

import dace
from dace.dtypes import BackendLanguage, ScheduleType
from dace.properties import CodeBlock
from dace.sdfg import InterstateEdge, SDFG
from dace.sdfg.state import ConditionalBlock, ControlFlowRegion, LoopRegion
from dace.transformation.interstate import InlineMultistateSDFG, InlineSDFG


_SDFG_COUNTER = itertools.count()


def _new_sdfg(prefix: str) -> SDFG:
    sdfg = SDFG(f"{prefix}_{next(_SDFG_COUNTER)}")
    sdfg.backend = BackendLanguage.Python
    return sdfg


def _run_sdfg(sdfg: SDFG, **kwargs):
    csdfg = sdfg.compile()
    csdfg(**kwargs)


def _assert_same(actual: np.ndarray, expected: np.ndarray) -> None:
    if np.issubdtype(actual.dtype, np.floating) or np.issubdtype(actual.dtype, np.complexfloating):
        np.testing.assert_allclose(actual, expected)
    else:
        np.testing.assert_array_equal(actual, expected)


def _add_branch_write_constant(region: ControlFlowRegion, array_name: str, value: int) -> None:
    state = region.add_state(f'{region.label}_state', is_start_block=True)
    tasklet = state.add_tasklet(f'{region.label}_write', {}, {'out'}, f'out = {value}')
    state.add_edge(tasklet, 'out', state.add_write(array_name), None, dace.Memlet(f'{array_name}[0]'))


def _add_branch_pick_index(region: ControlFlowRegion, src_name: str, dst_name: str, index_expr: str) -> None:
    state = region.add_state(f'{region.label}_pick', is_start_block=True)
    tasklet = state.add_tasklet(f'{region.label}_copy', {'inp'}, {'out'}, 'out = inp')
    state.add_edge(state.add_read(src_name), None, tasklet, 'inp', dace.Memlet(f'{src_name}[{index_expr}]'))
    state.add_edge(tasklet, 'out', state.add_write(dst_name), None, dace.Memlet(f'{dst_name}[0]'))


def _make_runtime_inlining_regression_sdfg(multistate: bool) -> tuple[SDFG, type]:
    """Build an outer/nested SDFG pair that relies on Python runtime code after inlining."""
    outer = _new_sdfg('runtime_inline_outer')
    outer.add_array('A', [1], dace.int64)
    outer.add_array('B', [1], dace.int64)
    outer.add_constant('OFFSET', 100)
    outer.set_global_code('OUTER_RUNTIME = 1', language=dace.dtypes.Language.Python)
    outer.append_init_code('OUTER_INIT = 2', language=dace.dtypes.Language.Python)
    outer.append_exit_code('OUTER_EXIT = 3', language=dace.dtypes.Language.Python)

    inner = SDFG('runtime_inline_inner')
    inner.backend = BackendLanguage.Python
    inner.add_constant('OFFSET', 4)
    inner.set_global_code('GLOBAL_RUNTIME = 10\nEXIT_RUNTIME = -1', language=dace.dtypes.Language.Python)
    inner.append_init_code('INIT_RUNTIME = OFFSET + GLOBAL_RUNTIME', language=dace.dtypes.Language.Python)
    inner.append_exit_code('EXIT_RUNTIME = INIT_RUNTIME + 1', language=dace.dtypes.Language.Python)
    inner.add_array('X', [1], dace.int64)
    inner.add_array('Y', [1], dace.int64)

    if multistate:
        entry_state = inner.add_state('entry', is_start_block=True)
        compute_state = inner.add_state('compute')
        inner.add_edge(entry_state, compute_state, InterstateEdge())
    else:
        compute_state = inner.add_state('compute', is_start_block=True)

    tasklet = compute_state.add_tasklet('shift', {'inp'}, {'out'}, 'out = inp + INIT_RUNTIME')
    compute_state.add_edge(compute_state.add_read('X'), None, tasklet, 'inp', dace.Memlet('X[0]'))
    compute_state.add_edge(tasklet, 'out', compute_state.add_write('Y'), None, dace.Memlet('Y[0]'))

    outer_state = outer.add_state('state', is_start_block=True)
    nested_node = outer_state.add_nested_sdfg(inner, {'X'}, {'Y'})
    outer_state.add_edge(outer_state.add_read('A'), None, nested_node, 'X', dace.Memlet('A[0]'))
    outer_state.add_edge(nested_node, 'Y', outer_state.add_write('B'), None, dace.Memlet('B[0]'))

    transformation = InlineMultistateSDFG if multistate else InlineSDFG
    return outer, transformation


def _make_runtime_inlining_symbol_mapping_sdfg(multistate: bool) -> tuple[SDFG, type]:
    """Build an inlining case where runtime code depends on both symbol and data remapping."""
    outer = _new_sdfg('runtime_inline_mapping_outer')
    outer.add_symbol('M', dace.int64)
    outer.add_array('A', [1], dace.int64)
    outer.add_array('B', [1], dace.int64)

    inner = SDFG('runtime_inline_mapping_inner')
    inner.backend = BackendLanguage.Python
    inner.add_symbol('N', dace.int64)
    inner.set_global_code('RUNTIME_BIAS = 1', language=dace.dtypes.Language.Python)
    inner.append_init_code('X[0] = N + X[0] + RUNTIME_BIAS', language=dace.dtypes.Language.Python)
    inner.add_array('X', [1], dace.int64)
    inner.add_array('Y', [1], dace.int64)

    if multistate:
        entry_state = inner.add_state('entry', is_start_block=True)
        compute_state = inner.add_state('compute')
        inner.add_edge(entry_state, compute_state, InterstateEdge())
    else:
        compute_state = inner.add_state('compute', is_start_block=True)

    tasklet = compute_state.add_tasklet('shift', {'inp'}, {'out'}, 'out = inp')
    compute_state.add_edge(compute_state.add_read('X'), None, tasklet, 'inp', dace.Memlet('X[0]'))
    compute_state.add_edge(tasklet, 'out', compute_state.add_write('Y'), None, dace.Memlet('Y[0]'))

    outer_state = outer.add_state('state', is_start_block=True)
    nested_node = outer_state.add_nested_sdfg(inner, {'X'}, {'Y'}, symbol_mapping={'N': 'M'})
    outer_state.add_edge(outer_state.add_read('A'), None, nested_node, 'X', dace.Memlet('A[0]'))
    outer_state.add_edge(nested_node, 'Y', outer_state.add_write('B'), None, dace.Memlet('B[0]'))

    transformation = InlineMultistateSDFG if multistate else InlineSDFG
    return outer, transformation


def _make_runtime_inlining_assignment_target_sdfg(multistate: bool) -> tuple[SDFG, type]:
    """Build an inlining case where a mapped symbol is assigned in runtime code and must stay local."""
    outer = _new_sdfg('runtime_inline_assignment_target_outer')
    outer.add_symbol('M', dace.int64)
    outer.add_array('A', [1], dace.int64)
    outer.add_transient('tmp', [1], dace.int64)
    outer.add_array('B', [1], dace.int64)

    inner = SDFG('runtime_inline_assignment_target_inner')
    inner.backend = BackendLanguage.Python
    inner.add_symbol('N', dace.int64)
    inner.append_init_code('N = X[0] + 1', language=dace.dtypes.Language.Python)
    inner.add_array('X', [1], dace.int64)
    inner.add_array('Y', [1], dace.int64)

    if multistate:
        entry_state = inner.add_state('entry', is_start_block=True)
        compute_state = inner.add_state('compute')
        inner.add_edge(entry_state, compute_state, InterstateEdge())
    else:
        compute_state = inner.add_state('compute', is_start_block=True)

    compute_state.add_edge(compute_state.add_read('X'), None, compute_state.add_write('Y'), None, dace.Memlet('X[0] -> [0]'))

    outer_state = outer.add_state('state', is_start_block=True)
    nested_node = outer_state.add_nested_sdfg(inner, {'X'}, {'Y'}, symbol_mapping={'N': 'M'})
    outer_state.add_edge(outer_state.add_read('A'), None, nested_node, 'X', dace.Memlet('A[0]'))
    outer_state.add_edge(nested_node, 'Y', outer_state.add_access('tmp'), None, dace.Memlet('tmp[0]'))

    tasklet = outer_state.add_tasklet('combine', {'inp'}, {'out'}, 'out = inp + M')
    outer_state.add_edge(outer_state.add_read('tmp'), None, tasklet, 'inp', dace.Memlet('tmp[0]'))
    outer_state.add_edge(tasklet, 'out', outer_state.add_write('B'), None, dace.Memlet('B[0]'))

    transformation = InlineMultistateSDFG if multistate else InlineSDFG
    return outer, transformation


def _make_runtime_inlining_local_data_name_sdfg(multistate: bool) -> tuple[SDFG, type]:
    """Build an inlining case where a local runtime variable matches a renamed transient data name."""
    outer = _new_sdfg('runtime_inline_local_data_outer')
    outer.add_array('A', [1], dace.int64)
    outer.add_array('B', [1], dace.int64)
    outer.add_array('tmp', [1], dace.int64)

    inner = SDFG('runtime_inline_local_data_inner')
    inner.backend = BackendLanguage.Python
    inner.append_init_code('tmp = X[0] + 1', language=dace.dtypes.Language.Python)
    inner.add_array('X', [1], dace.int64)
    inner.add_array('Y', [1], dace.int64)
    inner.add_transient('tmp', [1], dace.int64)

    if multistate:
        entry_state = inner.add_state('entry', is_start_block=True)
        compute_state = inner.add_state('compute')
        inner.add_edge(entry_state, compute_state, InterstateEdge())
    else:
        compute_state = inner.add_state('compute', is_start_block=True)

    compute_state.add_edge(compute_state.add_read('X'), None, compute_state.add_write('tmp'), None,
                           dace.Memlet('X[0] -> tmp[0]'))
    compute_state.add_edge(compute_state.add_read('tmp'), None, compute_state.add_write('Y'), None,
                           dace.Memlet('tmp[0] -> Y[0]'))

    outer_state = outer.add_state('state', is_start_block=True)
    nested_node = outer_state.add_nested_sdfg(inner, {'X'}, {'Y'})
    outer_state.add_edge(outer_state.add_read('A'), None, nested_node, 'X', dace.Memlet('A[0]'))
    outer_state.add_edge(nested_node, 'Y', outer_state.add_write('B'), None, dace.Memlet('B[0]'))

    transformation = InlineMultistateSDFG if multistate else InlineSDFG
    return outer, transformation


@pytest.mark.parametrize(('flag', 'expected'), [(True, 11), (False, 22)])
def test_conditional_block_singleton_output(flag, expected):
    sdfg = _new_sdfg('conditional_singleton')
    sdfg.add_array('A', [1], dace.int64)
    sdfg.add_scalar('flag', dace.bool_)

    conditional = ConditionalBlock('cond', sdfg=sdfg)
    sdfg.add_node(conditional, is_start_block=True)

    true_region = ControlFlowRegion('true_region', sdfg=sdfg, parent=conditional)
    false_region = ControlFlowRegion('false_region', sdfg=sdfg, parent=conditional)
    conditional.add_branch(CodeBlock('flag'), true_region)
    conditional.add_branch(None, false_region)

    _add_branch_write_constant(true_region, 'A', 11)
    _add_branch_write_constant(false_region, 'A', 22)

    a = np.zeros(1, dtype=np.int64)
    _run_sdfg(sdfg, A=a, flag=flag)

    np.testing.assert_array_equal(a, np.array([expected], dtype=np.int64))


@pytest.mark.parametrize(('dtype', 'np_dtype'), [(dace.float64, np.float64), (dace.int64, np.int64)])
def test_multistate_symbols_constants_pipeline(dtype, np_dtype):
    n_symbol = dace.symbol('N')
    sdfg = _new_sdfg('symbols_constants_pipeline')
    sdfg.add_symbol('N', dace.int64)
    sdfg.add_constant('SHIFT', 3)
    sdfg.add_array('A', [n_symbol], dtype)
    sdfg.add_array('B', [n_symbol], dtype)
    sdfg.add_array('C', [n_symbol], dtype)

    start = sdfg.add_state('start', is_start_block=True)
    sdfg.add_edge(start, sdfg.add_state('after_copy'), InterstateEdge(assignments={'k': '2'}))

    copy_state = list(sdfg.states())[1]
    copy_state.add_edge(copy_state.add_read('A'), None, copy_state.add_write('B'), None, dace.Memlet('A[0:N] -> [0:N]'))

    loop = LoopRegion('loop', condition_expr='i < N', loop_var='i', initialize_expr='i = 0', update_expr='i = i + 1', sdfg=sdfg)
    sdfg.add_node(loop)
    sdfg.add_edge(copy_state, loop, InterstateEdge())
    body = loop.add_state('body', is_start_block=True)
    tasklet = body.add_tasklet('compute', {'inp'}, {'out'}, 'out = inp + SHIFT + k')
    body.add_edge(body.add_read('B'), None, tasklet, 'inp', dace.Memlet('B[i]'))
    body.add_edge(tasklet, 'out', body.add_write('C'), None, dace.Memlet('C[i]'))

    n = 6
    a = np.arange(n, dtype=np_dtype)
    b = np.zeros(n, dtype=np_dtype)
    c = np.zeros(n, dtype=np_dtype)
    _run_sdfg(sdfg, A=a, B=b, C=c, N=n)

    _assert_same(b, a)
    _assert_same(c, a + np_dtype(5))


@pytest.mark.parametrize(('flag', 'n'), [(True, 4), (False, 1)])
def test_loop_then_conditional_kitchen_sink(flag, n):
    n_symbol = dace.symbol('N')
    sdfg = _new_sdfg('kitchen_sink')
    sdfg.add_symbol('N', dace.int64)
    sdfg.add_array('A', [n_symbol], dace.int64)
    sdfg.add_array('B', [n_symbol], dace.int64)
    sdfg.add_array('C', [n_symbol], dace.int64)
    sdfg.add_array('OUT', [1], dace.int64)
    sdfg.add_scalar('flag', dace.bool_)

    copy_state = sdfg.add_state('copy_state', is_start_block=True)
    copy_state.add_edge(copy_state.add_read('A'), None, copy_state.add_write('B'), None, dace.Memlet('A[0:N] -> [0:N]'))

    loop = LoopRegion('loop', condition_expr='i < N', loop_var='i', initialize_expr='i = 0', update_expr='i = i + 1', sdfg=sdfg)
    sdfg.add_node(loop)
    sdfg.add_edge(copy_state, loop, InterstateEdge(assignments={'bias': '1'}))
    body = loop.add_state('body', is_start_block=True)
    tasklet = body.add_tasklet('scale', {'inp'}, {'out'}, 'out = inp * 2 + bias')
    body.add_edge(body.add_read('B'), None, tasklet, 'inp', dace.Memlet('B[i]'))
    body.add_edge(tasklet, 'out', body.add_write('C'), None, dace.Memlet('C[i]'))

    conditional = ConditionalBlock('pick_output', sdfg=sdfg)
    sdfg.add_node(conditional)
    sdfg.add_edge(loop, conditional, InterstateEdge())

    true_region = ControlFlowRegion('pick_first', sdfg=sdfg, parent=conditional)
    false_region = ControlFlowRegion('pick_last', sdfg=sdfg, parent=conditional)
    conditional.add_branch(CodeBlock('flag'), true_region)
    conditional.add_branch(None, false_region)
    _add_branch_pick_index(true_region, 'C', 'OUT', '0')
    _add_branch_pick_index(false_region, 'C', 'OUT', 'N - 1')

    a = np.arange(n, dtype=np.int64) + 3
    b = np.zeros(n, dtype=np.int64)
    c = np.zeros(n, dtype=np.int64)
    out = np.zeros(1, dtype=np.int64)
    _run_sdfg(sdfg, A=a, B=b, C=c, OUT=out, N=n, flag=flag)

    expected_c = a * 2 + 1
    _assert_same(b, a)
    _assert_same(c, expected_c)
    np.testing.assert_array_equal(out, np.array([expected_c[0] if flag else expected_c[-1]], dtype=np.int64))


def test_array_constant_lookup_pipeline():
    n_symbol = dace.symbol('N')
    sdfg = _new_sdfg('array_constant_lookup')
    sdfg.add_symbol('N', dace.int64)
    sdfg.add_constant('LUT', np.array([3, 1, 4], dtype=np.int64))
    sdfg.add_array('A', [n_symbol], dace.int64)
    sdfg.add_array('B', [n_symbol], dace.int64)

    loop = LoopRegion('loop', condition_expr='i < N', loop_var='i', initialize_expr='i = 0', update_expr='i = i + 1', sdfg=sdfg)
    sdfg.add_node(loop, is_start_block=True)
    body = loop.add_state('body', is_start_block=True)
    tasklet = body.add_tasklet('lookup', {'inp'}, {'out'}, 'out = inp + LUT[i % 3]')
    body.add_edge(body.add_read('A'), None, tasklet, 'inp', dace.Memlet('A[i]'))
    body.add_edge(tasklet, 'out', body.add_write('B'), None, dace.Memlet('B[i]'))

    n = 7
    a = np.arange(n, dtype=np.int64)
    b = np.zeros(n, dtype=np.int64)
    _run_sdfg(sdfg, A=a, B=b, N=n)

    np.testing.assert_array_equal(b, a + np.array([3, 1, 4, 3, 1, 4, 3], dtype=np.int64))


def test_empty_state_in_chain():
    n_symbol = dace.symbol('N')
    sdfg = _new_sdfg('empty_state_chain')
    sdfg.add_symbol('N', dace.int64)
    sdfg.add_array('A', [n_symbol], dace.float64)
    sdfg.add_array('B', [n_symbol], dace.float64)

    empty_start = sdfg.add_state('empty_start', is_start_block=True)
    copy_state = sdfg.add_state('copy_state')
    empty_end = sdfg.add_state('empty_end')
    sdfg.add_edge(empty_start, copy_state, InterstateEdge())
    sdfg.add_edge(copy_state, empty_end, InterstateEdge())

    copy_state.add_edge(copy_state.add_read('A'), None, copy_state.add_write('B'), None, dace.Memlet('A[0:N] -> [0:N]'))

    n = 5
    a = np.linspace(-2.0, 2.0, n)
    b = np.zeros(n, dtype=np.float64)
    _run_sdfg(sdfg, A=a, B=b, N=n)

    np.testing.assert_allclose(b, a)


@pytest.mark.parametrize(('dtype', 'np_dtype'), [(dace.float64, np.float64), (dace.bool_, np.bool_)])
def test_zero_size_copy(dtype, np_dtype):
    sdfg = _new_sdfg('zero_size_copy')
    sdfg.add_array('A', [0], dtype)
    sdfg.add_array('B', [0], dtype)

    state = sdfg.add_state(is_start_block=True)
    state.add_edge(state.add_read('A'), None, state.add_write('B'), None, dace.Memlet('A[0:0] -> [0:0]'))

    a = np.zeros(0, dtype=np_dtype)
    b = np.zeros(0, dtype=np_dtype)
    _run_sdfg(sdfg, A=a, B=b)

    _assert_same(b, a)


@pytest.mark.parametrize(('shape', 'dtype', 'np_dtype'), [((1, 4), dace.float64, np.float64), ((3, 1), dace.int64, np.int64)])
def test_singleton_dimension_symbolic_2d_copy(shape, dtype, np_dtype):
    n_symbol = dace.symbol('N')
    m_symbol = dace.symbol('M')
    sdfg = _new_sdfg('singleton_dimension_2d')
    sdfg.add_symbol('N', dace.int64)
    sdfg.add_symbol('M', dace.int64)
    sdfg.add_array('A', [n_symbol, m_symbol], dtype)
    sdfg.add_array('B', [n_symbol, m_symbol], dtype)

    state = sdfg.add_state(is_start_block=True)
    state.add_edge(state.add_read('A'), None, state.add_write('B'), None, dace.Memlet('A[0:N, 0:M] -> [0:N, 0:M]'))

    a = np.arange(shape[0] * shape[1], dtype=np_dtype).reshape(shape)
    b = np.zeros_like(a)
    _run_sdfg(sdfg, A=a, B=b, N=shape[0], M=shape[1])

    _assert_same(b, a)


def test_loop_region_multistate_pipeline_with_scalar_transient():
    n_symbol = dace.symbol('N')
    sdfg = _new_sdfg('loop_scalar_transient_pipeline')
    sdfg.add_symbol('N', dace.int64)
    sdfg.add_array('A', [1], dace.int64)
    sdfg.add_array('B', [1], dace.int64)
    sdfg.add_array('C', [n_symbol], dace.int64)
    sdfg.add_scalar('tmp', dace.int64, transient=True)

    first = sdfg.add_state('first', is_start_block=True)
    second = sdfg.add_state('second')
    sdfg.add_edge(first, second, InterstateEdge())

    t1 = first.add_tasklet('t1', {'inp'}, {'out'}, 'out = inp + 5')
    first.add_edge(first.add_read('A'), None, t1, 'inp', dace.Memlet('A[0]'))
    first.add_edge(t1, 'out', first.add_write('tmp'), None, dace.Memlet('tmp'))

    t2 = second.add_tasklet('t2', {'inp'}, {'out'}, 'out = inp * 2')
    second.add_edge(second.add_read('tmp'), None, t2, 'inp', dace.Memlet('tmp'))
    second.add_edge(t2, 'out', second.add_write('B'), None, dace.Memlet('B[0]'))

    loop = LoopRegion('loop', condition_expr='i < N', loop_var='i', initialize_expr='i = 0', update_expr='i = i + 1', sdfg=sdfg)
    sdfg.add_node(loop)
    sdfg.add_edge(second, loop, InterstateEdge())
    body = loop.add_state('body', is_start_block=True)
    t3 = body.add_tasklet('t3', {'inp'}, {'out'}, 'out = inp + i')
    body.add_edge(body.add_read('B'), None, t3, 'inp', dace.Memlet('B[0]'))
    body.add_edge(t3, 'out', body.add_write('C'), None, dace.Memlet('C[i]'))

    a = np.array([3], dtype=np.int64)
    b = np.zeros(1, dtype=np.int64)
    c = np.zeros(4, dtype=np.int64)
    _run_sdfg(sdfg, A=a, B=b, C=c, N=4)

    np.testing.assert_array_equal(b, np.array([16], dtype=np.int64))
    np.testing.assert_array_equal(c, np.array([16, 17, 18, 19], dtype=np.int64))


def test_symbolic_slice_pipeline_three_states():
    n_symbol = dace.symbol('N')
    sdfg = _new_sdfg('symbolic_slice_pipeline')
    sdfg.add_symbol('N', dace.int64)
    sdfg.add_array('A', [n_symbol], dace.int64)
    sdfg.add_array('B', [n_symbol - 1], dace.int64)
    sdfg.add_array('C', [n_symbol - 1], dace.int64)

    state1 = sdfg.add_state('state1', is_start_block=True)
    state2 = sdfg.add_state('state2')
    state3 = sdfg.add_state('state3')
    sdfg.add_edge(state1, state2, InterstateEdge())
    sdfg.add_edge(state2, state3, InterstateEdge())

    state1.add_edge(state1.add_read('A'), None, state1.add_write('B'), None, dace.Memlet('A[1:N] -> [0:N-1]'))
    state2.add_edge(state2.add_read('B'), None, state2.add_write('C'), None, dace.Memlet('B[0:N-1] -> [0:N-1]'))

    tasklet = state3.add_tasklet('adjust', {'inp'}, {'out'}, 'out = inp + 10')
    state3.add_edge(state3.add_read('C'), None, tasklet, 'inp', dace.Memlet('C[0]'))
    state3.add_edge(tasklet, 'out', state3.add_write('C'), None, dace.Memlet('C[0]'))

    n = 6
    a = np.arange(n, dtype=np.int64)
    b = np.zeros(n - 1, dtype=np.int64)
    c = np.zeros(n - 1, dtype=np.int64)
    _run_sdfg(sdfg, A=a, B=b, C=c, N=n)

    expected = a[1:].copy()
    expected[0] += 10
    np.testing.assert_array_equal(b, a[1:])
    np.testing.assert_array_equal(c, expected)


@pytest.mark.parametrize(('flag', 'expected'), [(True, 6), (False, 7)])
def test_conditional_branch_uses_interstate_variable(flag, expected):
    sdfg = _new_sdfg('conditional_interstate_var')
    sdfg.add_array('A', [1], dace.int64)
    sdfg.add_scalar('flag', dace.bool_)

    start = sdfg.add_state('start', is_start_block=True)
    conditional = ConditionalBlock('cond', sdfg=sdfg)
    sdfg.add_node(conditional)
    sdfg.add_edge(start, conditional, InterstateEdge(assignments={'k': '5'}))

    true_region = ControlFlowRegion('true_region', sdfg=sdfg, parent=conditional)
    false_region = ControlFlowRegion('false_region', sdfg=sdfg, parent=conditional)
    conditional.add_branch(CodeBlock('flag'), true_region)
    conditional.add_branch(None, false_region)

    true_state = true_region.add_state('true_state', is_start_block=True)
    false_state = false_region.add_state('false_state', is_start_block=True)
    true_tasklet = true_state.add_tasklet('true_tasklet', {}, {'out'}, 'out = k + 1')
    false_tasklet = false_state.add_tasklet('false_tasklet', {}, {'out'}, 'out = k + 2')
    true_state.add_edge(true_tasklet, 'out', true_state.add_write('A'), None, dace.Memlet('A[0]'))
    false_state.add_edge(false_tasklet, 'out', false_state.add_write('A'), None, dace.Memlet('A[0]'))

    a = np.zeros(1, dtype=np.int64)
    _run_sdfg(sdfg, A=a, flag=flag)

    np.testing.assert_array_equal(a, np.array([expected], dtype=np.int64))


def test_loop_region_zero_iterations():
    n_symbol = dace.symbol('N')
    sdfg = _new_sdfg('loop_zero_iterations')
    sdfg.add_symbol('N', dace.int64)
    sdfg.add_array('A', [n_symbol], dace.int64)

    loop = LoopRegion('loop', condition_expr='i < N', loop_var='i', initialize_expr='i = 0', update_expr='i = i + 1', sdfg=sdfg)
    sdfg.add_node(loop, is_start_block=True)
    body = loop.add_state('body', is_start_block=True)
    tasklet = body.add_tasklet('fill', {}, {'out'}, 'out = 99')
    body.add_edge(tasklet, 'out', body.add_write('A'), None, dace.Memlet('A[i]'))

    a = np.zeros(0, dtype=np.int64)
    _run_sdfg(sdfg, A=a, N=0)

    np.testing.assert_array_equal(a, np.zeros(0, dtype=np.int64))


def test_nested_sdfg_simple():
    outer = _new_sdfg('outer_nested_simple')
    outer.add_array('A', [1], dace.float64)
    outer.add_array('B', [1], dace.float64)

    inner = SDFG('inner_nested_simple')
    inner.add_array('X', [1], dace.float64)
    inner.add_array('Y', [1], dace.float64)
    inner_state = inner.add_state(is_start_block=True)
    tasklet = inner_state.add_tasklet('compute', {'inp'}, {'out'}, 'out = inp + 1')
    inner_state.add_edge(inner_state.add_read('X'), None, tasklet, 'inp', dace.Memlet('X[0]'))
    inner_state.add_edge(tasklet, 'out', inner_state.add_write('Y'), None, dace.Memlet('Y[0]'))

    state = outer.add_state(is_start_block=True)
    nested = state.add_nested_sdfg(inner, {'X'}, {'Y'})
    state.add_edge(state.add_read('A'), None, nested, 'X', dace.Memlet('A[0]'))
    state.add_edge(nested, 'Y', state.add_write('B'), None, dace.Memlet('B[0]'))

    a = np.array([2.0], dtype=np.float64)
    b = np.zeros(1, dtype=np.float64)
    _run_sdfg(outer, A=a, B=b)
    np.testing.assert_allclose(b, np.array([3.0], dtype=np.float64))


def test_nested_sdfg_with_symbols():
    n_symbol = dace.symbol('N')
    outer = _new_sdfg('outer_nested_symbols')
    outer.add_symbol('N', dace.int64)
    outer.add_array('A', [n_symbol], dace.float64)
    outer.add_array('B', [n_symbol], dace.float64)

    inner = SDFG('inner_nested_symbols')
    inner.add_symbol('N', dace.int64)
    inner.add_array('X', [n_symbol], dace.float64)
    inner.add_array('Y', [n_symbol], dace.float64)
    inner_state = inner.add_state(is_start_block=True)
    inner_state.add_edge(inner_state.add_read('X'), None, inner_state.add_write('Y'), None, dace.Memlet('X[0:N] -> [0:N]'))

    state = outer.add_state(is_start_block=True)
    nested = state.add_nested_sdfg(inner, {'X'}, {'Y'}, symbol_mapping={'N': 'N'})
    state.add_edge(state.add_read('A'), None, nested, 'X', dace.Memlet('A[0:N]'))
    state.add_edge(nested, 'Y', state.add_write('B'), None, dace.Memlet('B[0:N]'))

    n = 4
    a = np.arange(n, dtype=np.float64)
    b = np.zeros(n, dtype=np.float64)
    _run_sdfg(outer, A=a, B=b, N=n)
    np.testing.assert_allclose(b, a)


@pytest.mark.xfail(reason='The inlining transformation do not account for variable names in init or global code')
@pytest.mark.parametrize('multistate', [False, True], ids=['single_state_inline', 'multistate_inline'])
def test_python_runtime_code_language_survives_inlining(multistate: bool):
    sdfg, transformation = _make_runtime_inlining_regression_sdfg(multistate)

    assert sdfg.apply_transformations(transformation) == 1
    assert sdfg.global_code['frame'].language == dace.dtypes.Language.Python
    assert sdfg.init_code['frame'].language == dace.dtypes.Language.Python
    assert sdfg.exit_code['frame'].language == dace.dtypes.Language.Python

    compiled = sdfg.compile()
    a = np.array([1], dtype=np.int64)
    b = np.zeros(1, dtype=np.int64)
    compiled(A=a, B=b)
    np.testing.assert_array_equal(b, np.array([15], dtype=np.int64))

    compiled.finalize()
    assert compiled._namespace['EXIT_RUNTIME'] == 15


@pytest.mark.xfail(reason='The inlining transformation do not account for variable names in init or global code')
@pytest.mark.parametrize('multistate', [False, True], ids=['single_state_inline', 'multistate_inline'])
def test_python_runtime_code_replacements_survive_inlining(multistate: bool):
    sdfg, transformation = _make_runtime_inlining_symbol_mapping_sdfg(multistate)

    assert sdfg.apply_transformations(transformation) == 1
    init_code = sdfg.init_code['frame'].as_string
    assert 'A[0]' in init_code
    assert 'M' in init_code
    assert 'X[0]' not in init_code
    assert 'N +' not in init_code

    compiled = sdfg.compile()
    a = np.array([3], dtype=np.int64)
    b = np.zeros(1, dtype=np.int64)
    compiled(A=a, B=b, M=4)

    np.testing.assert_array_equal(b, np.array([8], dtype=np.int64))


@pytest.mark.xfail(reason='The inlining transformation do not account for variable names in init or global code')
@pytest.mark.parametrize('multistate', [False, True], ids=['single_state_inline', 'multistate_inline'])
def test_python_runtime_assignment_targets_remain_local_during_inlining(multistate: bool):
    sdfg, transformation = _make_runtime_inlining_assignment_target_sdfg(multistate)

    assert sdfg.apply_transformations(transformation) == 1
    init_code = sdfg.init_code['frame'].as_string
    assert 'N = (A[0] + 1)' in init_code or 'N = A[0] + 1' in init_code
    assert 'M = A[0] + 1' not in init_code

    compiled = sdfg.compile()
    a = np.array([7], dtype=np.int64)
    b = np.zeros(1, dtype=np.int64)
    compiled(A=a, B=b, M=3)

    np.testing.assert_array_equal(b, np.array([10], dtype=np.int64))


@pytest.mark.xfail(reason='The inlining transformation do not account for variable names in init or global code')
@pytest.mark.parametrize('multistate', [False, True], ids=['single_state_inline', 'multistate_inline'])
def test_python_runtime_local_data_names_remain_local_during_inlining(multistate: bool):
    sdfg, transformation = _make_runtime_inlining_local_data_name_sdfg(multistate)

    assert sdfg.apply_transformations(transformation) == 1
    init_code = sdfg.init_code['frame'].as_string
    renamed_tmp = next(name for name in sdfg.arrays if name not in {'A', 'B', 'tmp'})
    assert 'tmp = (A[0] + 1)' in init_code or 'tmp = A[0] + 1' in init_code
    assert f'{renamed_tmp} =' not in init_code

    compiled = sdfg.compile()
    a = np.array([5], dtype=np.int64)
    b = np.zeros(1, dtype=np.int64)
    tmp = np.zeros(1, dtype=np.int64)
    compiled(A=a, B=b, tmp=tmp)

    np.testing.assert_array_equal(b, np.array([5], dtype=np.int64))


def test_nested_sdfg_helper_scope_isolated_from_parent_names():
    outer = _new_sdfg('outer_nested_same_connector_names')
    outer.add_array('X', [1], dace.float64)
    outer.add_array('Y', [1], dace.float64)

    inner = SDFG('inner_nested_same_connector_names')
    inner.backend = BackendLanguage.Python
    inner.add_array('X', [1], dace.float64)
    inner.add_array('Y', [1], dace.float64)
    inner_state = inner.add_state('inner_state', is_start_block=True)
    tasklet = inner_state.add_tasklet('shift', {'inp'}, {'out'}, 'out = inp + 2.0')
    inner_state.add_edge(inner_state.add_read('X'), None, tasklet, 'inp', dace.Memlet('X[0]'))
    inner_state.add_edge(tasklet, 'out', inner_state.add_write('Y'), None, dace.Memlet('Y[0]'))

    outer_state = outer.add_state('outer_state', is_start_block=True)
    nested_node = outer_state.add_nested_sdfg(inner, {'X'}, {'Y'})
    outer_state.add_edge(outer_state.add_read('X'), None, nested_node, 'X', dace.Memlet('X[0]'))
    outer_state.add_edge(nested_node, 'Y', outer_state.add_write('Y'), None, dace.Memlet('Y[0]'))

    generated_code = outer.generate_code()[0].code
    assert 'def inner_nested_same_connector_names_' in generated_code

    x = np.array([3.0], dtype=np.float64)
    y = np.zeros(1, dtype=np.float64)
    _run_sdfg(outer, X=x, Y=y)

    np.testing.assert_allclose(y, np.array([5.0], dtype=np.float64))


def test_nested_sdfg_local_constants_available_without_duplicate_headers():
    outer = _new_sdfg('outer_nested_constant_helper')
    outer.add_array('A', [1], dace.int64)
    outer.add_array('B', [1], dace.int64)

    inner = SDFG('inner_nested_constant_helper')
    inner.backend = BackendLanguage.Python
    inner.add_constant('LOOKUP', np.array([7], dtype=np.int64))
    inner.add_array('X', [1], dace.int64)
    inner.add_array('Y', [1], dace.int64)
    inner_state = inner.add_state('inner_state', is_start_block=True)
    tasklet = inner_state.add_tasklet('shift', {'inp'}, {'out'}, 'out = inp + LOOKUP[0]')
    inner_state.add_edge(inner_state.add_read('X'), None, tasklet, 'inp', dace.Memlet('X[0]'))
    inner_state.add_edge(tasklet, 'out', inner_state.add_write('Y'), None, dace.Memlet('Y[0]'))

    outer_state = outer.add_state('outer_state', is_start_block=True)
    nested_node = outer_state.add_nested_sdfg(inner, {'X'}, {'Y'})
    outer_state.add_edge(outer_state.add_read('A'), None, nested_node, 'X', dace.Memlet('A[0]'))
    outer_state.add_edge(nested_node, 'Y', outer_state.add_write('B'), None, dace.Memlet('B[0]'))

    generated_code = outer.generate_code()[0].code

    assert 'LOOKUP = numpy.array([7], dtype=numpy.int64)' in generated_code
    assert generated_code.count('# DaCe AUTO-GENERATED FILE. DO NOT MODIFY') == 1
    assert generated_code.count('import numpy') == 1

    a = np.array([2], dtype=np.int64)
    b = np.zeros(1, dtype=np.int64)
    _run_sdfg(outer, A=a, B=b)
    np.testing.assert_array_equal(b, np.array([9], dtype=np.int64))


def test_nested_sdfg_runtime_code_stays_helper_local_and_keeps_local_context():
    outer = _new_sdfg('outer_nested_runtime_local')
    outer.add_array('A', [1], dace.int64)
    outer.add_array('B', [1], dace.int64)

    inner = SDFG('inner_nested_runtime_local')
    inner.backend = BackendLanguage.Python
    inner.add_constant('LOOKUP', np.array([7], dtype=np.int64))
    inner.append_init_code('SHIFT = LOOKUP[0]', language=dace.dtypes.Language.Python)
    inner.append_exit_code('Y[0] = Y[0] + SHIFT', language=dace.dtypes.Language.Python)
    inner.add_array('X', [1], dace.int64)
    inner.add_array('Y', [1], dace.int64)
    inner_state = inner.add_state('inner_state', is_start_block=True)
    tasklet = inner_state.add_tasklet('shift', {'inp'}, {'out'}, 'out = inp + SHIFT')
    inner_state.add_edge(inner_state.add_read('X'), None, tasklet, 'inp', dace.Memlet('X[0]'))
    inner_state.add_edge(tasklet, 'out', inner_state.add_write('Y'), None, dace.Memlet('Y[0]'))

    outer_state = outer.add_state('outer_state', is_start_block=True)
    nested_node = outer_state.add_nested_sdfg(inner, {'X'}, {'Y'})
    outer_state.add_edge(outer_state.add_read('A'), None, nested_node, 'X', dace.Memlet('A[0]'))
    outer_state.add_edge(nested_node, 'Y', outer_state.add_write('B'), None, dace.Memlet('B[0]'))

    generated_code = outer.generate_code()[0].code

    assert 'SHIFT = LOOKUP[0]' in generated_code
    assert 'finally:' in generated_code
    assert generated_code.count('# DaCe AUTO-GENERATED FILE. DO NOT MODIFY') == 1
    assert generated_code.count('import numpy') == 1

    a = np.array([2], dtype=np.int64)
    b = np.zeros(1, dtype=np.int64)
    _run_sdfg(outer, A=a, B=b)

    np.testing.assert_array_equal(b, np.array([16], dtype=np.int64))


def test_map_inside_conditional():
    sdfg = _new_sdfg('map_inside_conditional')
    sdfg.add_array('A', [4], dace.float64)
    sdfg.add_array('B', [4], dace.float64)
    sdfg.add_scalar('flag', dace.bool_)

    conditional = ConditionalBlock('cond', sdfg=sdfg)
    sdfg.add_node(conditional, is_start_block=True)
    true_region = ControlFlowRegion('true_region', sdfg=sdfg, parent=conditional)
    false_region = ControlFlowRegion('false_region', sdfg=sdfg, parent=conditional)
    conditional.add_branch(CodeBlock('flag'), true_region)
    conditional.add_branch(None, false_region)

    state = true_region.add_state('mapped_state', is_start_block=True)
    map_entry, map_exit = state.add_map('m', {'i': '0:4'}, schedule=ScheduleType.Sequential)
    tasklet = state.add_tasklet('copy', {'inp'}, {'out'}, 'out = inp')
    state.add_memlet_path(state.add_read('A'), map_entry, tasklet, dst_conn='inp', memlet=dace.Memlet('A[i]'))
    state.add_memlet_path(tasklet, map_exit, state.add_write('B'), src_conn='out', memlet=dace.Memlet('B[i]'))
    _add_branch_write_constant(false_region, 'B', 0)

    a = np.arange(4, dtype=np.float64)
    b = np.zeros(4, dtype=np.float64)
    _run_sdfg(sdfg, A=a, B=b, flag=True)
    np.testing.assert_allclose(b, a)


def test_loop_region_with_map():
    sdfg = _new_sdfg('loop_with_map')
    sdfg.add_array('A', [4], dace.float64)
    sdfg.add_array('B', [4], dace.float64)

    loop = LoopRegion('loop', condition_expr='j < 1', loop_var='j', initialize_expr='j = 0', update_expr='j = j + 1', sdfg=sdfg)
    sdfg.add_node(loop, is_start_block=True)
    body = loop.add_state('body', is_start_block=True)
    map_entry, map_exit = body.add_map('m', {'i': '0:4'}, schedule=ScheduleType.Sequential)
    tasklet = body.add_tasklet('copy', {'inp'}, {'out'}, 'out = inp')
    body.add_memlet_path(body.add_read('A'), map_entry, tasklet, dst_conn='inp', memlet=dace.Memlet('A[i]'))
    body.add_memlet_path(tasklet, map_exit, body.add_write('B'), src_conn='out', memlet=dace.Memlet('B[i]'))

    a = np.arange(4, dtype=np.float64)
    b = np.zeros(4, dtype=np.float64)
    _run_sdfg(sdfg, A=a, B=b)
    np.testing.assert_allclose(b, a)


def test_vector_norm_reduction():
    sdfg = _new_sdfg('vector_norm')
    sdfg.add_array('A', [4], dace.float64)
    sdfg.add_array('out', [1], dace.float64)

    state = sdfg.add_state(is_start_block=True)
    init = state.add_tasklet('init', {}, {'result'}, 'result = 0.0')
    state.add_edge(init, 'result', state.add_write('out'), None, dace.Memlet('out[0]'))
    map_entry, map_exit = state.add_map('m', {'i': '0:4'}, schedule=ScheduleType.Sequential)
    tasklet = state.add_tasklet('square', {'inp'}, {'result'}, 'result = inp * inp')
    state.add_memlet_path(state.add_read('A'), map_entry, tasklet, dst_conn='inp', memlet=dace.Memlet('A[i]'))
    state.add_memlet_path(tasklet,
                          map_exit,
                          state.add_write('out'),
                          src_conn='result',
                          memlet=dace.Memlet('out[0]', wcr='lambda x, y: x + y'))

    a = np.arange(4, dtype=np.float64)
    out = np.zeros(1, dtype=np.float64)
    _run_sdfg(sdfg, A=a, out=out)
    np.testing.assert_allclose(out, np.array([14.0], dtype=np.float64))


def test_global_code_constant():
    sdfg = _new_sdfg('global_code_constant')
    sdfg.add_array('A', [1], dace.int64)
    sdfg.global_code[None] = CodeBlock('GLOBAL_CONSTANT = 13')

    state = sdfg.add_state(is_start_block=True)
    tasklet = state.add_tasklet('write', {}, {'out'}, 'out = GLOBAL_CONSTANT')
    state.add_edge(tasklet, 'out', state.add_write('A'), None, dace.Memlet('A[0]'))

    a = np.zeros(1, dtype=np.int64)
    _run_sdfg(sdfg, A=a)
    np.testing.assert_array_equal(a, np.array([13], dtype=np.int64))


def test_init_code_semantics():
    sdfg = _new_sdfg('init_code_semantics')
    sdfg.add_array('A', [1], dace.int64)
    sdfg.init_code[None] = CodeBlock('INIT_SENTINEL = 11')

    state = sdfg.add_state(is_start_block=True)
    tasklet = state.add_tasklet('write', {}, {'out'}, 'out = INIT_SENTINEL')
    state.add_edge(tasklet, 'out', state.add_write('A'), None, dace.Memlet('A[0]'))

    a = np.zeros(1, dtype=np.int64)
    _run_sdfg(sdfg, A=a)
    np.testing.assert_array_equal(a, np.array([11], dtype=np.int64))


def test_runtime_code_public_api_and_lifecycle_reinitialize():
    sdfg = _new_sdfg('runtime_code_helpers')
    sdfg.add_array('A', [1], dace.int64)
    sdfg.set_global_code('GLOBAL_SENTINEL = 7\nEXIT_LOG = []', language=dace.dtypes.Language.Python)
    sdfg.append_init_code('INIT_COUNTER = globals().get("INIT_COUNTER", 0) + 1',
                          language=dace.dtypes.Language.Python)
    sdfg.append_exit_code('EXIT_LOG.append(INIT_COUNTER)', language=dace.dtypes.Language.Python)

    state = sdfg.add_state(is_start_block=True)
    tasklet = state.add_tasklet('write', {}, {'out'}, 'out = GLOBAL_SENTINEL + INIT_COUNTER')
    state.add_edge(tasklet, 'out', state.add_write('A'), None, dace.Memlet('A[0]'))

    compiled = sdfg.compile()
    from dace.codegen.py.compiled_sdfg import PythonCompiledSDFG
    assert isinstance(compiled, PythonCompiledSDFG)
    a = np.zeros(1, dtype=np.int64)

    compiled(A=a)
    np.testing.assert_array_equal(a, np.array([8], dtype=np.int64))
    assert compiled._namespace['EXIT_LOG'] == []

    compiled.finalize()
    assert compiled._namespace['EXIT_LOG'] == [1]

    compiled.finalize()
    assert compiled._namespace['EXIT_LOG'] == [1] # Finalize should not run exit code again

    a.fill(0)
    compiled(A=a)
    np.testing.assert_array_equal(a, np.array([9], dtype=np.int64)) # Init code should run again because it was finalized
    
    a.fill(0)
    compiled(A=a)
    np.testing.assert_array_equal(a, np.array([9], dtype=np.int64)) # Init code should not run again

    compiled.finalize()
    assert compiled._namespace['EXIT_LOG'] == [1, 2]


def test_transient_array_pipeline():
    n_symbol = dace.symbol('N')
    sdfg = _new_sdfg('transient_array_pipeline')
    sdfg.add_symbol('N', dace.int64)
    sdfg.add_array('A', [n_symbol], dace.float64)
    sdfg.add_array('B', [n_symbol], dace.float64)
    sdfg.add_transient('tmp', [n_symbol], dace.float64)

    state1 = sdfg.add_state('state1', is_start_block=True)
    state2 = sdfg.add_state('state2')
    sdfg.add_edge(state1, state2, InterstateEdge())
    state1.add_edge(state1.add_read('A'), None, state1.add_access('tmp'), None, dace.Memlet('A[0:N] -> [0:N]'))
    state2.add_edge(state2.add_read('tmp'), None, state2.add_write('B'), None, dace.Memlet('tmp[0:N] -> [0:N]'))

    code = sdfg.generate_code()[0].code
    assert 'import numpy' in code

    n = 5
    a = np.arange(n, dtype=np.float64)
    b = np.zeros(n, dtype=np.float64)
    _run_sdfg(sdfg, A=a, B=b, N=n)
    np.testing.assert_allclose(b, a)