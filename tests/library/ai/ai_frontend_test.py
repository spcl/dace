# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""
Tests of the ``dace.ai`` handle in the Python frontend.

These are offline: the model is replaced by :func:`ai_test_utils.stub_provider`, so what is checked
here is the plumbing between a Python call and an :class:`~dace.libraries.ai.nodes.ai_node.AINode`
-- which connectors the node gets, what it reads and writes, and that the description reaches the
prompt. The tests that compile and run assert that the tasklet ends up in the right place, not that
a model can write it.
"""

import os
import sys

import numpy as np
import pytest

import dace
from dace import dtypes, nodes
from dace.frontend.python.common import DaceSyntaxError
from dace.libraries.ai.nodes import AINode

sys.path.insert(0, os.path.dirname(__file__))
from ai_test_utils import prompt_of, stub_provider  # noqa: E402

from dace.libraries.ai.backend import TaskletSpec  # noqa: E402

N = dace.symbol('N')
M = 20

ADD_DESCRIPTION = 'Write _out[i] = _a[i] + _b[i] for every element of the vectors.'


def _ai_nodes(sdfg: dace.SDFG):
    """
    Returns every AI node of an SDFG, including the ones inside nested SDFGs.

    :param sdfg: The SDFG to search.
    :return: The AI nodes, in no particular order.
    """
    return [n for n, _ in sdfg.all_nodes_recursive() if isinstance(n, AINode)]


def _only_ai_node(sdfg: dace.SDFG) -> AINode:
    """
    Returns the single AI node of an SDFG.

    :param sdfg: The SDFG to search.
    :return: The node.
    """
    found = _ai_nodes(sdfg)
    assert len(found) == 1, f'expected one AI node, found {len(found)}'
    return found[0]


def test_call_becomes_an_ai_node():

    @dace.program
    def prog(A: dace.float64[M], B: dace.float64[M]):
        return dace.ai(ADD_DESCRIPTION, a=A, b=B)

    sdfg = prog.to_sdfg(simplify=False)
    node = _only_ai_node(sdfg)

    # The description is the specification, and is carried on the node rather than in the prompt only
    assert node.description == ADD_DESCRIPTION
    assert node.implementation is None
    assert node.default_implementation == 'ai'


def test_keyword_inputs_name_their_connectors():

    @dace.program
    def prog(A: dace.float64[M], B: dace.float64[M]):
        return dace.ai(ADD_DESCRIPTION, a=A, b=B)

    node = _only_ai_node(prog.to_sdfg(simplify=False))
    assert set(node.in_connectors) == {'_a', '_b'}
    assert set(node.out_connectors) == {'_out'}


def test_it_can_be_imported_by_name():
    from dace import ai

    @dace.program
    def prog(A: dace.float64[M], B: dace.float64[M]):
        return ai(ADD_DESCRIPTION, a=A, b=B)

    assert _only_ai_node(prog.to_sdfg(simplify=False)).description == ADD_DESCRIPTION


def test_connectors_keep_the_order_of_the_call():

    @dace.program
    def prog(A: dace.float64[M], B: dace.float64[M], C: dace.float64[M]):
        return dace.ai('Write _out[i] = _x[i] + _y[i] + _z[i].', x=A, y=B, z=C)

    # The connectors are listed to the model in this order, and the prompt is what the answer cache
    # is keyed on, so an arbitrary order would cost a request on every run
    node = _only_ai_node(prog.to_sdfg(simplify=False))
    assert list(node.in_connectors) == ['_x', '_y', '_z']


def test_positional_inputs_are_numbered():

    @dace.program
    def prog(A: dace.float64[M], B: dace.float64[M]):
        return dace.ai('Write _out[i] = _in0[i] + _in1[i].', A, B)

    node = _only_ai_node(prog.to_sdfg(simplify=False))
    assert set(node.in_connectors) == {'_in0', '_in1'}


def test_connectors_read_and_write_the_given_containers():

    @dace.program
    def prog(A: dace.float64[M], B: dace.float64[M], C: dace.float64[M]):
        C[:] = dace.ai(ADD_DESCRIPTION, a=A, b=B)

    sdfg = prog.to_sdfg(simplify=False)
    node = _only_ai_node(sdfg)
    state = next(s for s in sdfg.states() if node in s.nodes())

    reads = {e.dst_conn: e.data.data for e in state.in_edges(node)}
    writes = {e.src_conn: e.data.data for e in state.out_edges(node)}
    assert reads == {'_a': 'A', '_b': 'B'}
    assert list(writes) == ['_out']
    # The whole container is moved, so the generated code sees a pointer to all of it
    assert state.in_edges(node)[0].data.subset.num_elements() == M


def test_output_is_allocated_from_the_first_input():

    @dace.program
    def prog(A: dace.float32[M, M], B: dace.float32[M, M]):
        return dace.ai(ADD_DESCRIPTION, a=A, b=B)

    sdfg = prog.to_sdfg(simplify=False)
    node = _only_ai_node(sdfg)
    state = next(s for s in sdfg.states() if node in s.nodes())
    written = state.out_edges(node)[0].data.data

    assert list(sdfg.arrays[written].shape) == [M, M]
    assert sdfg.arrays[written].dtype == dace.float32


def test_output_shape_and_type_can_be_given():

    @dace.program
    def prog(A: dace.float32[M, M]):
        return dace.ai('Write the sum of every row of _a into _out.', a=A, shape=(M, ), dtype=dace.float64)

    sdfg = prog.to_sdfg(simplify=False)
    node = _only_ai_node(sdfg)
    state = next(s for s in sdfg.states() if node in s.nodes())
    written = state.out_edges(node)[0].data.data

    assert list(sdfg.arrays[written].shape) == [M]
    assert sdfg.arrays[written].dtype == dace.float64


def test_a_scalar_input_yields_a_scalar_output():

    @dace.program
    def prog(a: dace.float64):
        return dace.ai('Write twice _x into _out.', x=a)

    sdfg = prog.to_sdfg(simplify=False)
    node = _only_ai_node(sdfg)
    state = next(s for s in sdfg.states() if node in s.nodes())
    written = state.out_edges(node)[0].data.data

    assert isinstance(sdfg.arrays[written], dace.data.Scalar)


def test_writing_into_an_existing_container():

    @dace.program
    def prog(A: dace.float64[M], B: dace.float64[M], C: dace.float64[M]):
        dace.ai(ADD_DESCRIPTION, a=A, b=B, out=C)

    sdfg = prog.to_sdfg(simplify=False)
    node = _only_ai_node(sdfg)
    state = next(s for s in sdfg.states() if node in s.nodes())

    assert {e.src_conn: e.data.data for e in state.out_edges(node)} == {'_out': 'C'}
    # Nothing was allocated for an output that already exists
    assert not any(desc.transient for name, desc in sdfg.arrays.items() if name not in ('A', 'B', 'C'))


def test_several_outputs_are_numbered():

    @dace.program
    def prog(A: dace.float64[M], B: dace.float64[M], C: dace.float64[M]):
        dace.ai('Write the sum of _a into _out0 and its product into _out1.', a=A, out=(B, C))

    sdfg = prog.to_sdfg(simplify=False)
    node = _only_ai_node(sdfg)
    state = next(s for s in sdfg.states() if node in s.nodes())

    assert {e.src_conn: e.data.data for e in state.out_edges(node)} == {'_out0': 'B', '_out1': 'C'}


def test_nodes_of_one_program_get_distinct_names():

    @dace.program
    def prog(A: dace.float64[M]):
        B = dace.ai('Write twice _a into _out.', a=A)
        return dace.ai('Write twice _a into _out.', a=B)

    # The name identifies the slot's conversation, so two nodes must not share one
    names = sorted(node.name for node in _ai_nodes(prog.to_sdfg(simplify=False)))
    assert names == ['ai', 'ai_1']


def test_the_node_can_be_named():

    @dace.program
    def prog(A: dace.float64[M]):
        return dace.ai('Write twice _a into _out.', a=A, name='doubler')

    assert _only_ai_node(prog.to_sdfg(simplify=False)).name == 'doubler'


def test_the_description_reaches_the_model():

    @dace.program
    def prog(A: dace.float64[M], B: dace.float64[M]):
        return dace.ai(ADD_DESCRIPTION, a=A, b=B)

    sdfg = prog.to_sdfg(simplify=False)
    with stub_provider(TaskletSpec(code='for (int i = 0; i < 20; ++i) _out[i] = _a[i] + _b[i];')) as provider:
        sdfg.expand_library_nodes()

    prompt = prompt_of(provider)
    assert ADD_DESCRIPTION in prompt
    # The connectors named at the call site are the ones the model is told about
    assert '- _a (input)' in prompt
    assert '- _b (input)' in prompt
    assert '- _out (output)' in prompt


def test_it_compiles_and_runs():
    code = 'for (int i = 0; i < N; ++i) { _out[i] = _a[i] + _b[i]; }'

    @dace.program
    def prog(A: dace.float64[N], B: dace.float64[N]):
        return dace.ai(ADD_DESCRIPTION, a=A, b=B)

    rng = np.random.default_rng(0)
    a = rng.random(M)
    b = rng.random(M)
    with stub_provider(TaskletSpec(code=code)):
        result = prog(a, b)

    assert np.allclose(result, a + b)


def test_it_runs_inside_a_map():
    # A scalar connector inside a map: one element per iteration, assigned rather than written through
    code = '_out = 2.0 * _x;'

    @dace.program
    def prog(A: dace.float64[N], B: dace.float64[N]):
        for i in dace.map[0:N]:
            B[i] = dace.ai('Write twice the scalar _x into _out.', x=A[i])

    rng = np.random.default_rng(0)
    a = rng.random(M)
    b = np.zeros(M)
    with stub_provider(TaskletSpec(code=code)) as provider:
        prog(a, b)

    assert np.allclose(b, 2.0 * a)
    # The node sits inside the map, and the prompt says so
    assert 'map' in prompt_of(provider).lower()


def test_it_runs_writing_into_a_given_container():
    code = 'for (int i = 0; i < 20; ++i) { _out[i] = _a[i] + _b[i]; }'

    @dace.program
    def prog(A: dace.float64[M], B: dace.float64[M], C: dace.float64[M]):
        dace.ai(ADD_DESCRIPTION, a=A, b=B, out=C)

    rng = np.random.default_rng(0)
    a = rng.random(M)
    b = rng.random(M)
    c = np.zeros(M)
    with stub_provider(TaskletSpec(code=code)):
        prog(a, b, c)

    assert np.allclose(c, a + b)


def test_it_runs_a_tile_inside_a_map():
    # Slicing an operand hands the node a view rather than a copy, so the code must step through it
    # with the strides of the container behind it -- which the prompt states for every connector
    tile = 4
    code = f'''
    for (int i = 0; i < {tile}; ++i)
        for (int j = 0; j < {tile}; ++j) {{
            double acc = 0;
            for (int k = 0; k < N; ++k) acc += _a[i * N + k] * _b[k * N + j];
            _out[i * N + j] = acc;
        }}
    '''

    @dace.program
    def prog(A: dace.float64[N, N], B: dace.float64[N, N], C: dace.float64[N, N]):
        for ti, tj in dace.map[0:N:tile, 0:N:tile]:
            C[ti:ti + tile,
              tj:tj + tile] = dace.ai('Multiply the tiles: _out[i][j] = sum over k of '
                                      '_a[i][k] * _b[k][j].',
                                      a=A[ti:ti + tile, 0:N],
                                      b=B[0:N, tj:tj + tile],
                                      shape=(tile, tile),
                                      dtype=dace.float64)

    rng = np.random.default_rng(0)
    a = rng.random((M, M))
    b = rng.random((M, M))
    c = np.zeros((M, M))
    with stub_provider(TaskletSpec(code=code)) as provider:
        prog(a, b, c)

    assert np.allclose(c, a @ b)
    # The views the model was told to index through are the ones the tiles came from
    assert 'ArrayView' in prompt_of(provider)


def test_symbols_do_not_have_to_be_passed():
    # The description refers to N, which is a symbol of the program and therefore in scope
    code = 'for (int i = 0; i < N; ++i) { _out[i] = _a[i] * N; }'

    @dace.program
    def prog(A: dace.float64[N]):
        return dace.ai('Write _out[i] = _a[i] * N, where N is the length of the vector.', a=A)

    rng = np.random.default_rng(0)
    a = rng.random(M)
    with stub_provider(TaskletSpec(code=code)) as provider:
        result = prog(a)

    assert np.allclose(result, a * M)
    assert 'N: int' in prompt_of(provider)


def test_the_generated_tasklet_is_kept_in_the_sdfg():

    @dace.program
    def prog(A: dace.float64[M], B: dace.float64[M]):
        return dace.ai(ADD_DESCRIPTION, a=A, b=B)

    sdfg = prog.to_sdfg(simplify=False)
    with stub_provider(TaskletSpec(code='for (int i = 0; i < 20; ++i) _out[i] = _a[i] + _b[i];')) as provider:
        sdfg.expand_library_nodes()

    assert not _ai_nodes(sdfg)
    tasklet = next(n for n, _ in sdfg.all_nodes_recursive() if isinstance(n, nodes.Tasklet))
    assert '_out[i] = _a[i] + _b[i]' in tasklet.code.as_string

    # Compiling the expanded SDFG does not ask again
    calls = len(provider.calls)
    sdfg.compile()
    assert len(provider.calls) == calls


def test_the_schedule_can_be_given():

    @dace.program
    def prog(A: dace.float64[M]):
        return dace.ai('Write twice _a into _out.', a=A, schedule=dace.ScheduleType.Sequential)

    assert _only_ai_node(prog.to_sdfg(simplify=False)).schedule == dtypes.ScheduleType.Sequential


def test_a_missing_description_is_rejected():

    @dace.program
    def prog(A: dace.float64[M]):
        return dace.ai('   ', a=A)

    with pytest.raises(DaceSyntaxError, match='empty'):
        prog.to_sdfg(simplify=False)


def test_a_non_string_description_is_rejected():

    @dace.program
    def prog(A: dace.float64[M]):
        return dace.ai(A, a=A)

    with pytest.raises(DaceSyntaxError, match='must be a string'):
        prog.to_sdfg(simplify=False)


def test_a_symbol_cannot_be_an_input():

    @dace.program
    def prog(A: dace.float64[N]):
        return dace.ai('Write _out[i] = _a[i] * _n.', a=A, n=N)

    with pytest.raises(DaceSyntaxError, match='symbol'):
        prog.to_sdfg(simplify=False)


def test_the_output_cannot_be_described_twice():

    @dace.program
    def prog(A: dace.float64[M], B: dace.float64[M]):
        dace.ai('Write twice _a into _out.', a=A, out=B, dtype=dace.float64)

    with pytest.raises(DaceSyntaxError, match='cannot be combined with "out"'):
        prog.to_sdfg(simplify=False)


def test_the_same_container_cannot_be_written_twice():

    @dace.program
    def prog(A: dace.float64[M], B: dace.float64[M]):
        dace.ai('Write twice _a into _out0 and _out1.', a=A, out=(B, B))

    with pytest.raises(DaceSyntaxError, match='same container'):
        prog.to_sdfg(simplify=False)


def test_a_container_cannot_name_the_node():

    @dace.program
    def prog(A: dace.float64[M]):
        return dace.ai('Write twice _a into _out.', a=A, name=A)

    with pytest.raises(DaceSyntaxError, match='must be a string'):
        prog.to_sdfg(simplify=False)


def test_an_output_cannot_be_inferred_without_inputs():

    @dace.program
    def prog(A: dace.float64[M]):
        A[:] = dace.ai('Fill _out with ones.')

    with pytest.raises(DaceSyntaxError, match='shape'):
        prog.to_sdfg(simplify=False)


def test_calling_outside_a_program_explains_itself():
    with pytest.raises(NotImplementedError, match='inside a DaCe program'):
        dace.ai('Write twice _a into _out.', a=np.zeros(M))


if __name__ == '__main__':
    pytest.main([__file__])
