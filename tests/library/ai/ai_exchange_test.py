# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""
Tests for exchanging prompts and responses by hand, one node or a whole SDFG at a time.

No provider is configured that could answer: every response is supplied by the test, which is
the point of this API.
"""

import io
import json
import os

import numpy as np
import pytest

import dace
from dace import nodes
from dace.libraries import ai
from dace.libraries.ai import backend, exchange
from dace.libraries.ai.exceptions import AIExpansionError
from dace.libraries.ai.nodes import AINode, AITasklet


def _reply(code: str) -> str:
    """
    Builds a response as a model would write it.

    :param code: The tasklet body.
    :return: The JSON reply.
    """
    return json.dumps({
        'notes': '',
        'language': 'CPP',
        'code': code,
        'code_global': '',
        'code_init': '',
        'code_exit': '',
        'state_fields': [],
        'side_effects': False,
        'ignored_symbols': [],
        'use_environments': [],
        'environments': [],
    })


DOUBLE = _reply('_out = 2.0 * _in;')
NEGATE = _reply('_out = -_in;')


@pytest.fixture
def prompt_dir(tmp_path):
    """
    Points the exchange at a temporary directory, with a provider that cannot answer.

    :return: The directory.
    """
    with dace.config.set_temporary('ai', 'provider', value='nonexistent'):
        with dace.config.set_temporary('ai', 'manual_dir', value=str(tmp_path)):
            yield tmp_path


def _two_node_sdfg():
    """
    Builds an SDFG with one :class:`AINode` at the top level and one in a nested SDFG.

    :return: A tuple of (SDFG, top-level state, top-level node, nested state, nested node).
    """
    inner = dace.SDFG('inner')
    inner.add_array('x', [1], dace.float64)
    inner.add_array('y', [1], dace.float64)
    istate = inner.add_state()
    negate = AINode('negate', 'Negate the input.', inputs={'_in'}, outputs={'_out'})
    istate.add_node(negate)
    istate.add_edge(istate.add_read('x'), None, negate, '_in', dace.Memlet('x[0]'))
    istate.add_edge(negate, '_out', istate.add_write('y'), None, dace.Memlet('y[0]'))

    sdfg = dace.SDFG('ai_exchange')
    sdfg.add_array('A', [1], dace.float64)
    sdfg.add_array('B', [1], dace.float64)
    sdfg.add_array('C', [1], dace.float64)
    state = sdfg.add_state()
    double = AINode('double', 'Multiply the input by two.', inputs={'_in'}, outputs={'_out'})
    state.add_node(double)
    state.add_edge(state.add_read('A'), None, double, '_in', dace.Memlet('A[0]'))
    b = state.add_access('B')
    state.add_edge(double, '_out', b, None, dace.Memlet('B[0]'))
    nsdfg = state.add_nested_sdfg(inner, {'x'}, {'y'})
    state.add_edge(b, None, nsdfg, 'x', dace.Memlet('B[0]'))
    state.add_edge(nsdfg, 'y', state.add_write('C'), None, dace.Memlet('C[0]'))
    return sdfg, state, double, istate, negate


def _run(sdfg: dace.SDFG) -> np.ndarray:
    """
    Runs the two-node SDFG on 21.

    :return: The output, which is -42 if both nodes were expanded correctly.
    """
    a, b, c = np.array([21.0]), np.zeros([1]), np.zeros([1])
    sdfg(A=a, B=b, C=c)
    return c


def test_node_prompt_is_self_contained(prompt_dir):
    sdfg, state, double, _, _ = _two_node_sdfg()
    prompt = double.generate_prompt(state)

    assert 'Multiply the input by two.' in prompt
    assert 'Reply with a single JSON object' in prompt
    assert json.dumps(backend.RESPONSE_SCHEMA, indent=2) in prompt
    # Nothing is expanded by asking for the prompt
    assert double in state.nodes()


def test_node_response_replaces_the_node(prompt_dir):
    sdfg, state, double, inner_state, negate = _two_node_sdfg()

    tasklet = double.read_prompt_response(state, DOUBLE)
    assert isinstance(tasklet, AITasklet)
    assert tasklet in state.nodes() and double not in state.nodes()
    assert tasklet.code.as_string.strip() == '_out = 2.0 * _in;'

    negate.read_prompt_response(inner_state, '\n' + NEGATE + '\n')
    assert np.allclose(_run(sdfg), -42.0)


def test_invalid_response_is_reported(prompt_dir):
    sdfg, state, double, _, _ = _two_node_sdfg()

    with pytest.raises(AIExpansionError, match='not valid JSON'):
        double.read_prompt_response(state, 'Sure! ```json {} ```')
    assert double in state.nodes()


def test_response_that_does_not_compile_is_not_sent_for_repair(prompt_dir):
    sdfg, state, double, _, _ = _two_node_sdfg()

    # The configured provider does not exist, so a repair attempt would fail differently
    with pytest.raises(AIExpansionError, match='supplied response .* does not compile'):
        double.read_prompt_response(state, _reply('_out = does_not_exist(_in);'))
    assert double in state.nodes()


def test_sdfg_prompts_are_written_and_responses_read_back(prompt_dir):
    sdfg, _, _, _, _ = _two_node_sdfg()

    written = ai.generate_prompts(sdfg)
    assert len(written) == 2
    assert {os.path.basename(p).split('_')[0] for p in written} == {'double', 'negate'}

    for path in written:
        reply = DOUBLE if os.path.basename(path).startswith('double') else NEGATE
        with open(path.replace('_prompt.md', '_response.json'), 'w') as fp:
            fp.write(reply)

    tasklets = ai.read_prompt_responses(sdfg)
    assert sorted(t.label for t in tasklets) == ['double', 'negate']
    assert not any(isinstance(n, AINode) for n, _ in sdfg.all_nodes_recursive())
    assert np.allclose(_run(sdfg), -42.0)


def test_nodes_without_a_response_are_left_alone(prompt_dir):
    sdfg, state, double, _, _ = _two_node_sdfg()

    written = ai.generate_prompts(sdfg, str(prompt_dir / 'explicit'))
    double_prompt = next(p for p in written if os.path.basename(p).startswith('double'))
    with open(double_prompt.replace('_prompt.md', '_response.json'), 'w') as fp:
        fp.write(DOUBLE)

    tasklets = ai.read_prompt_responses(sdfg, str(prompt_dir / 'explicit'))
    assert [t.label for t in tasklets] == ['double']
    assert [n.name for n, _ in sdfg.all_nodes_recursive() if isinstance(n, AINode)] == ['negate']


def test_manual_provider_reuses_a_response_saved_for_the_batch(prompt_dir, monkeypatch):
    sdfg, state, double, _, _ = _two_node_sdfg()
    path = next(p for p in ai.generate_prompts(sdfg) if os.path.basename(p).startswith('double'))
    with open(path.replace('_prompt.md', '_response.json'), 'w') as fp:
        fp.write(DOUBLE)

    # Nothing typed: the answer can only come from the saved file
    monkeypatch.setattr('sys.stdin', io.StringIO(''))
    with dace.config.set_temporary('ai', 'provider', value='manual'):
        double.expand(state, 'ai')
    tasklet = next(n for n in state.nodes() if isinstance(n, nodes.Tasklet))
    assert tasklet.code.as_string.strip() == '_out = 2.0 * _in;'


def test_exchange_paths_are_keyed_by_prompt_content(tmp_path):
    first, _ = exchange.exchange_paths(str(tmp_path), 'prompt A', 'node')
    again, _ = exchange.exchange_paths(str(tmp_path), 'prompt A', 'node')
    other, _ = exchange.exchange_paths(str(tmp_path), 'prompt B', 'node')

    # Stable across processes, so an answer survives a re-run; distinct per question, so a repair
    # round does not overwrite the prompt it is repairing
    assert first == again
    assert first != other
    assert os.path.basename(first).startswith('node_')


if __name__ == '__main__':
    pytest.main([__file__])
