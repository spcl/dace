# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""
Tests for the ``manual`` provider, which relays the prompt through the user instead of an API.

Standard input is replaced with the text a user would paste, so the whole expansion path -- prompt
construction, reply parsing, verification, tasklet construction and code generation -- runs without
a key or an SDK.
"""

import io
import json
import os

import numpy as np
import pytest

import dace
from dace import nodes
from dace.libraries.ai import backend
from dace.libraries.ai.exceptions import AIExpansionError
from dace.libraries.ai.nodes import AINode

REPLY = {
    'notes': 'doubling',
    'language': 'CPP',
    'code': '_out = 2.0 * _in;',
    'code_global': '',
    'code_init': '',
    'code_exit': '',
    'state_fields': [],
    'side_effects': False,
    'ignored_symbols': [],
    'environments': [],
}


@pytest.fixture
def manual(tmp_path, monkeypatch):
    """
    Selects the manual provider and points it at a temporary directory.

    :return: The directory prompts and replies are exchanged through.
    """
    with dace.config.set_temporary('ai', 'provider', value='manual'):
        with dace.config.set_temporary('ai', 'manual_dir', value=str(tmp_path)):
            yield tmp_path


def _sdfg_and_node():
    """
    Builds a one-element SDFG around an :class:`AINode`.

    :return: A tuple of (SDFG, state, node).
    """
    sdfg = dace.SDFG('ai_manual')
    sdfg.add_array('A', [1], dace.float64)
    sdfg.add_array('B', [1], dace.float64)
    state = sdfg.add_state()
    node = AINode('double', 'Multiply the input by two.', inputs={'_in'}, outputs={'_out'})
    state.add_node(node)
    state.add_edge(state.add_read('A'), None, node, '_in', dace.Memlet('A[0]'))
    state.add_edge(node, '_out', state.add_write('B'), None, dace.Memlet('B[0]'))
    return sdfg, state, node


def test_prompt_is_written_and_reply_is_read_from_stdin(manual, monkeypatch, capsys):
    monkeypatch.setattr('sys.stdin', io.StringIO(json.dumps(REPLY)))
    sdfg, state, node = _sdfg_and_node()

    node.expand(state, 'ai')

    prompts = [f for f in os.listdir(manual) if f.endswith('_prompt.md')]
    assert len(prompts) == 1, 'the prompt was not written out for the user to copy'
    written = (manual / prompts[0]).read_text()
    # The prompt is self-contained: the contract, the node's context, and the required reply shape
    assert 'DaCe *tasklet*' in written
    assert 'Multiply the input by two.' in written
    assert 'Reply with a single JSON object' in written
    assert json.dumps(backend.RESPONSE_SCHEMA, indent=2) in written

    # Instructions go to stderr, so piping stdout stays usable
    assert str(manual) in capsys.readouterr().err

    tasklet = next(n for n in state.nodes() if isinstance(n, nodes.Tasklet))
    assert tasklet.code.as_string.strip() == '_out = 2.0 * _in;'

    a = np.array([21.0])
    b = np.zeros([1])
    sdfg(A=a, B=b)
    assert np.allclose(b, 42.0)


def test_reply_can_be_written_to_a_file(manual, monkeypatch):
    # First run: nothing is typed, so it fails, but the prompt is left on disk
    monkeypatch.setattr('sys.stdin', io.StringIO(''))
    sdfg, state, node = _sdfg_and_node()
    with pytest.raises(AIExpansionError):
        node.expand(state, 'ai')

    prompt = next(f for f in os.listdir(manual) if f.endswith('_prompt.md'))
    response_path = manual / prompt.replace('_prompt.md', '_response.json')
    response_path.write_text(json.dumps(REPLY))

    # Second run of the same program picks the saved answer up without asking again
    monkeypatch.setattr('sys.stdin', io.StringIO(''))
    sdfg, state, node = _sdfg_and_node()
    node.expand(state, 'ai')
    tasklet = next(n for n in state.nodes() if isinstance(n, nodes.Tasklet))
    assert tasklet.code.as_string.strip() == '_out = 2.0 * _in;'


def test_surrounding_whitespace_is_ignored(manual, monkeypatch):
    # What a terminal paste typically adds: leading blank lines and a trailing newline
    monkeypatch.setattr('sys.stdin', io.StringIO(f'\n\n  {json.dumps(REPLY)}  \n\n'))
    sdfg, state, node = _sdfg_and_node()

    node.expand(state, 'ai')
    assert any(isinstance(n, nodes.Tasklet) for n in state.nodes())


def test_missing_reply_is_reported_clearly(manual, monkeypatch):
    monkeypatch.setattr('sys.stdin', io.StringIO(''))
    sdfg, state, node = _sdfg_and_node()

    with pytest.raises(AIExpansionError) as info:
        node.expand(state, 'ai')

    message = str(info.value)
    assert 'No response was provided' in message
    assert 'interactive terminal' in message
    # The library node survives, so the prompt can be answered on a second attempt
    assert any(isinstance(n, AINode) for n in state.nodes())


def test_invalid_json_is_reported_clearly(manual, monkeypatch):
    monkeypatch.setattr('sys.stdin', io.StringIO('sure, here is a tasklet: _out = _in;'))
    sdfg, state, node = _sdfg_and_node()

    with pytest.raises(AIExpansionError, match='not valid JSON'):
        node.expand(state, 'ai')


def test_repair_round_asks_a_new_question(manual, monkeypatch):
    broken = dict(REPLY, code='_out = does_not_exist(_in);')
    monkeypatch.setattr('sys.stdin', io.StringIO(json.dumps(broken)))
    sdfg, state, node = _sdfg_and_node()

    # The broken answer is saved, so the first round is answered from disk on the next run; the
    # repair round is a different prompt and therefore asks again rather than reusing it.
    with pytest.raises(AIExpansionError):
        node.expand(state, 'ai')

    prompts = sorted(f for f in os.listdir(manual) if f.endswith('_prompt.md'))
    assert len(prompts) == 2, 'the repair round did not get its own prompt file'
    first, second = ((manual / p).read_text() for p in prompts)
    assert first != second
    assert 'does not compile' in first or 'does not compile' in second


if __name__ == '__main__':
    pytest.main([__file__])
