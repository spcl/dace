"""Runtime coverage for symbol mappings that collide with outer data names."""

import numpy as np
import pytest

import dace
from dace import Memlet, dtypes
from dace.transformation.interstate.multistate_inline import InlineMultistateSDFG

N = dace.symbol('N')


@pytest.mark.parametrize('backend', [dtypes.BackendLanguage.CPP, dtypes.BackendLanguage.Python])
@pytest.mark.parametrize('reserved_kind', ['array', 'constant'])
def test_colliding_mappings_keep_other_symbol_assignments(backend: dtypes.BackendLanguage, reserved_kind: str) -> None:
    sdfg = dace.SDFG(f'inline_collisions_{reserved_kind}_{backend.name}')
    sdfg.backend = backend
    sdfg.add_array('A', [N], dace.float64)
    sdfg.add_array('B', [N], dace.float64)
    sdfg.add_symbol('M', dace.int64)
    for name in ['K', 'L']:
        if reserved_kind == 'array':
            sdfg.add_array(name, [1], dace.int64)
        else:
            sdfg.add_constant(name, 100)
    state = sdfg.add_state()

    inner = dace.SDFG('inner_collisions')
    inner.add_array('a', [N], dace.float64)
    inner.add_array('b', [N], dace.float64)
    for name in ['K', 'L', 'R', 'Q']:
        inner.add_symbol(name, dace.int64)
    body = inner.add_state()
    inner.add_edge(body, inner.add_state(), dace.InterstateEdge())
    body.add_mapped_tasklet('add', {'i': '0:N'}, {'x': Memlet('a[i]')},
                            'y = x + K + 2 * L + R + Q', {'y': Memlet('b[i]')},
                            external_edges=True)
    nested = state.add_nested_sdfg(inner, {'a'}, {'b'},
                                   symbol_mapping={
                                       'N': 'N',
                                       'K': 'M',
                                       'L': 'M + 1',
                                       'R': 'N // 2',
                                       'Q': 3
                                   })
    state.add_edge(state.add_read('A'), None, nested, 'a', Memlet('A[0:N]'))
    state.add_edge(nested, 'b', state.add_write('B'), None, Memlet('B[0:N]'))

    assert sdfg.apply_transformations(InlineMultistateSDFG) == 1
    assignments = {key for edge in sdfg.all_interstate_edges() for key in edge.data.assignments}
    assert not assignments.intersection({'K', 'L'})
    assert 'R' in assignments
    sdfg.validate()

    n, m = 17, 4
    a = np.arange(n, dtype=np.float64)
    b = np.zeros_like(a)
    compiled = sdfg.compile()
    args = {'A': a, 'B': b, 'N': n, 'M': m}
    for name in ['K', 'L']:
        if name in sdfg.arglist():
            args[name] = np.array([100], dtype=np.int64)
    compiled(**args)
    np.testing.assert_allclose(b, a + m + 2 * (m + 1) + n // 2 + 3)


@pytest.mark.parametrize('backend', [dtypes.BackendLanguage.CPP, dtypes.BackendLanguage.Python])
def test_constant_mapping_seeds_reassigned_loop_counter(backend: dtypes.BackendLanguage) -> None:
    sdfg = dace.SDFG(f'inline_constant_counter_{backend.name}')
    sdfg.backend = backend
    sdfg.add_array('A', [N], dace.int32)
    state = sdfg.add_state()
    inner = dace.SDFG('counter')
    inner.add_array('B', [N], dace.int32)
    inner.add_symbol('i', dace.int64)
    guard = inner.add_state(is_start_block=True)
    body = inner.add_state()
    done = inner.add_state()
    inner.add_edge(guard, body, dace.InterstateEdge(condition='i <= N'))
    inner.add_edge(body, guard, dace.InterstateEdge(assignments={'i': 'i + 1'}))
    inner.add_edge(guard, done, dace.InterstateEdge(condition='i > N'))
    tasklet = body.add_tasklet('write', {}, {'out'}, 'out = i - 1')
    body.add_edge(tasklet, 'out', body.add_write('B'), None, Memlet('B[i - 1]'))
    nested = state.add_nested_sdfg(inner, {}, {'B'}, symbol_mapping={'N': 'N', 'i': 1})
    state.add_edge(nested, 'B', state.add_write('A'), None, Memlet('A[0:N]'))

    assert sdfg.apply_transformations(InlineMultistateSDFG) == 1
    sdfg.validate()
    output = np.full(17, -1, dtype=np.int32)
    sdfg.compile()(A=output, N=output.size)
    np.testing.assert_array_equal(output, np.arange(output.size, dtype=np.int32))
