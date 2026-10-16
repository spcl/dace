"""View removal preserves slices and execution-order edges."""

import numpy as np
import pytest

import dace
from dace import Memlet, dtypes
from dace.transformation.passes.remove_views import RemoveViews

N = dace.symbol('N')


@pytest.mark.parametrize('backend', [dtypes.BackendLanguage.CPP, dtypes.BackendLanguage.Python])
@pytest.mark.parametrize('direction', ['read', 'write'])
def test_offset_view_preserves_ordering_and_empty_memlets(backend: dtypes.BackendLanguage, direction: str) -> None:
    sdfg = dace.SDFG(f'offset_view_ordering_{direction}_{backend.name}')
    sdfg.backend = backend
    sdfg.add_array('A', [N, 3], dace.float64)
    sdfg.add_array('B', [N], dace.float64)
    sdfg.add_array('X', [N], dace.float64)
    sdfg.add_view('V', [N], dace.float64, strides=[3])
    state = sdfg.add_state()
    a, b, x, view = [state.add_access(name) for name in ['A', 'B', 'X', 'V']]
    if direction == 'read':
        state.add_edge(a, None, view, 'views', Memlet(data='A', subset='0:N, 1', other_subset='0:N'))
        state.add_nedge(view, b, Memlet('B[0:N]'))
        state.add_nedge(a, x, Memlet(data='A', subset='0:N, 2', other_subset='0:N'))
        state.add_nedge(x, view, Memlet())
        expected_order = (x, b)
    else:
        state.add_nedge(b, view, Memlet('B[0:N]'))
        state.add_edge(view, 'views', a, None, Memlet(data='A', subset='0:N, 1', other_subset='0:N'))
        state.add_nedge(view, x, Memlet())
        consume = state.add_tasklet('consume', {'inp'}, {}, 'pass')
        state.add_edge(x, None, consume, 'inp', Memlet('X[0]'))
        expected_order = (b, x)
    sdfg.validate()

    assert RemoveViews().apply_pass(sdfg, {}) == {'V'}
    assert view not in state.nodes()
    ordering = [edge for edge in state.edges() if edge.data.is_empty()]
    assert [(edge.src, edge.dst) for edge in ordering] == [expected_order]
    assert all(edge.data.subset is None and edge.data.other_subset is None for edge in ordering)
    assert not state.has_cycles()
    sdfg.validate()

    n = 13
    a_value = np.arange(n * 3, dtype=np.float64).reshape(n, 3).copy()
    b_value = np.linspace(100, 200, n).copy()
    x_value = np.zeros(n, dtype=np.float64)
    expected = a_value.copy()
    if direction == 'write':
        expected[:, 1] = b_value
    sdfg.compile()(A=a_value, B=b_value, X=x_value, N=n)
    np.testing.assert_array_equal(a_value, expected)
    if direction == 'read':
        np.testing.assert_array_equal(b_value, expected[:, 1])
        np.testing.assert_array_equal(x_value, expected[:, 2])


def test_ordered_view_without_consumer_is_kept() -> None:
    sdfg = dace.SDFG('ordered_view_without_consumer')
    sdfg.add_array('A', [N], dace.float64)
    sdfg.add_array('X', [N], dace.float64)
    sdfg.add_view('V', [N], dace.float64)
    state = sdfg.add_state()
    a, x, view = [state.add_access(name) for name in ['A', 'X', 'V']]
    state.add_edge(a, None, view, 'views', Memlet('A[0:N]'))
    state.add_nedge(a, x, Memlet('A[0:N]'))
    ordering = state.add_nedge(x, view, Memlet())
    sdfg.validate()

    assert RemoveViews().apply_pass(sdfg, {}) is None
    assert view in state.nodes() and ordering in state.edges()
    sdfg.validate()


@pytest.mark.parametrize('backend', [dtypes.BackendLanguage.CPP, dtypes.BackendLanguage.Python])
@pytest.mark.parametrize('direction', ['read', 'write'])
def test_nested_connector_slice_runs(backend: dtypes.BackendLanguage, direction: str) -> None:
    sdfg = dace.SDFG(f'nested_connector_view_{direction}_{backend.name}')
    sdfg.backend = backend
    sdfg.add_array('A', [N, 3], dace.float64)
    sdfg.add_array('B', [N], dace.float64)
    sdfg.add_view('V', [N], dace.float64, strides=[3])
    state = sdfg.add_state()
    a, b, view = [state.add_access(name) for name in ['A', 'B', 'V']]
    inner = dace.SDFG('nested_scale')
    inner.add_array('inp', [N], dace.float64, strides=[3 if direction == 'read' else 1])
    inner.add_array('out', [N], dace.float64, strides=[3 if direction == 'write' else 1])
    body = inner.add_state()
    body.add_mapped_tasklet('scale', {'i': '0:N'}, {'x': Memlet('inp[i]')},
                            'y = x * 2', {'y': Memlet('out[i]')},
                            external_edges=True)
    nested = state.add_nested_sdfg(inner, {'inp'}, {'out'}, symbol_mapping={'N': 'N'})
    if direction == 'read':
        state.add_edge(a, None, view, 'views', Memlet(data='A', subset='0:N, 1', other_subset='0:N'))
        state.add_edge(view, None, nested, 'inp', Memlet('V[0:N]'))
        state.add_edge(nested, 'out', b, None, Memlet('B[0:N]'))
    else:
        state.add_edge(b, None, nested, 'inp', Memlet('B[0:N]'))
        state.add_edge(nested, 'out', view, None, Memlet('V[0:N]'))
        state.add_edge(view, 'views', a, None, Memlet(data='A', subset='0:N, 1', other_subset='0:N'))
    sdfg.validate()
    assert RemoveViews().apply_pass(sdfg, {}) == {'V'}
    assert view not in state.nodes()
    sdfg.validate()

    a_value = np.arange(39, dtype=np.float64).reshape(13, 3).copy()
    b_value = np.linspace(100, 200, 13).copy()
    expected_a = a_value.copy()
    expected_b = b_value.copy()
    if direction == 'read':
        expected_b = 2 * a_value[:, 1]
    else:
        expected_a[:, 1] = 2 * b_value
    sdfg.compile()(A=a_value, B=b_value, N=13)
    np.testing.assert_array_equal(a_value, expected_a)
    np.testing.assert_array_equal(b_value, expected_b)


@pytest.mark.parametrize('backend', [dtypes.BackendLanguage.CPP, dtypes.BackendLanguage.Python])
def test_read_view_outgoing_order_stays_empty(backend: dtypes.BackendLanguage) -> None:
    sdfg = dace.SDFG(f'read_view_empty_order_{backend.name}')
    sdfg.backend = backend
    sdfg.add_array('A', [N], dace.float64)
    sdfg.add_array('B', [N], dace.float64)
    sdfg.add_array('flag', [1], dace.int32)
    sdfg.add_view('V', [N], dace.float64)
    state = sdfg.add_state()
    a, view, b = [state.add_access(name) for name in ['A', 'V', 'B']]
    state.add_edge(a, None, view, 'views', Memlet('A[0:N]'))
    state.add_nedge(view, b, Memlet('B[0:N]'))
    tasklet = state.add_tasklet('mark', {}, {'out'}, 'out = 1')
    state.add_nedge(view, tasklet, Memlet())
    state.add_edge(tasklet, 'out', state.add_write('flag'), None, Memlet('flag[0]'))
    sdfg.validate()

    assert RemoveViews().apply_pass(sdfg, {}) == {'V'}
    ordering = state.edges_between(a, tasklet)
    assert len(ordering) == 1
    assert ordering[0].data.data is None
    assert ordering[0].data.subset is None and ordering[0].data.other_subset is None
    sdfg.validate()
    a_value = np.arange(13, dtype=np.float64)
    b_value = np.zeros_like(a_value)
    flag = np.zeros(1, dtype=np.int32)
    sdfg.compile()(A=a_value, B=b_value, flag=flag, N=13)
    np.testing.assert_array_equal(b_value, a_value)
    assert flag[0] == 1
