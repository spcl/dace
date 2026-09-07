# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""
Tests for the on-disk record of an expansion and for its progress printouts.

An expansion is a paid, non-reproducible call whose result is baked into the SDFG, so what the
model was told and what it answered has to survive the run -- especially when generated code passes
the probe and then fails the real build, which is only explicable by comparing the probe's
translation unit against what DaCe generates.
"""

import json
import os
import shutil
import sys

import pytest

import dace
from dace import nodes
from dace.libraries.ai import transcript
from dace.libraries.ai.backend import TaskletSpec
from dace.libraries.ai.exceptions import AIExpansionError
from dace.libraries.ai.nodes import AINode

sys.path.insert(0, os.path.dirname(__file__))
from ai_test_utils import stub_provider, stub_provider_sequence  # noqa: E402

GOOD = '_out = 2.0 * _in;'
BAD = '_out = this_symbol_does_not_exist(_in);'

needs_compiler = pytest.mark.skipif(shutil.which('c++') is None and shutil.which('g++') is None,
                                    reason='needs a host C++ compiler for the probe')


def _sdfg_and_node(name: str = 'double'):
    """
    Builds a one-element SDFG around an :class:`AINode`.

    :param name: Name of the node, which also names the transcript directory.
    :return: A tuple of (SDFG, state, node).
    """
    sdfg = dace.SDFG('ai_transcript')
    sdfg.add_array('A', [1], dace.float64)
    sdfg.add_array('B', [1], dace.float64)
    state = sdfg.add_state()
    node = AINode(name, 'Double the input.', inputs={'_in'}, outputs={'_out'})
    state.add_node(node)
    state.add_edge(state.add_read('A'), None, node, '_in', dace.Memlet('A[0]'))
    state.add_edge(node, '_out', state.add_write('B'), None, dace.Memlet('B[0]'))
    return sdfg, state, node


def _only_transcript(root):
    """
    Returns the single transcript directory written under a root.

    :param root: The configured transcript directory.
    :return: A tuple of (path, decoded transcript.json).
    """
    entries = sorted(os.listdir(root))
    assert len(entries) == 1, entries
    path = os.path.join(root, entries[0])
    with open(os.path.join(path, transcript.INDEX_NAME)) as fp:
        return path, json.load(fp)


def test_a_successful_expansion_is_recorded_in_full(tmp_path):
    sdfg, state, node = _sdfg_and_node()
    spec = TaskletSpec(code=GOOD, notes='because', raw_response='{"code": "verbatim from the model"}')

    with dace.config.set_temporary('ai', 'transcripts', value=True):
        with dace.config.set_temporary('ai', 'transcript_dir', value=str(tmp_path)):
            with dace.config.set_temporary('ai', 'verify', value=False):
                with stub_provider(spec) as provider:
                    node.expand(state, 'ai')

    path, index = _only_transcript(tmp_path)
    assert index['outcome'] == 'expanded'
    assert index['node_type'] == 'AINode' and index['node'] == 'double'
    assert index['sdfg'] == 'ai_transcript'

    # The prompts are kept verbatim, and separately from the index, so they can just be read
    with open(os.path.join(path, 'system_prompt.md')) as fp:
        assert fp.read() == provider.system
    with open(os.path.join(path, 'attempt_1_prompt.md')) as fp:
        assert fp.read() == provider.calls[0][0]['content']

    # The answer is stored exactly as the provider received it, not as it was re-serialized: a
    # response that parses into something unexpected is only explicable from the text that arrived
    with open(os.path.join(path, 'attempt_1_answer.json')) as fp:
        assert fp.read() == spec.raw_response
    assert index['attempts'][0]['answer']['code'] == GOOD
    assert index['attempts'][0]['answer']['notes'] == 'because'


@needs_compiler
def test_every_repair_round_is_recorded(tmp_path):
    sdfg, state, node = _sdfg_and_node()

    with dace.config.set_temporary('ai', 'transcripts', value=True):
        with dace.config.set_temporary('ai', 'transcript_dir', value=str(tmp_path)):
            with stub_provider_sequence([TaskletSpec(code=BAD), TaskletSpec(code=GOOD)]):
                node.expand(state, 'ai')

    path, index = _only_transcript(tmp_path)
    assert index['outcome'] == 'expanded'
    assert len(index['attempts']) == 2

    failed, repaired = index['attempts']
    assert failed['verification']['ok'] is False
    assert 'this_symbol_does_not_exist' in failed['verification']['diagnostics']
    assert repaired['verification']['ok'] is True
    # The repair round records the prompt that carried the diagnostics back to the model
    assert 'does not compile' in repaired['prompt']

    # The probe's translation unit is what a tasklet that compiles here and fails in the real
    # build has to be compared against, so it is written out rather than only summarized
    for attempt, code in ((1, BAD), (2, GOOD)):
        with open(os.path.join(path, f'attempt_{attempt}_probe.cpp')) as fp:
            assert code in fp.read()
    assert os.path.exists(os.path.join(path, 'attempt_1_probe.log'))


@needs_compiler
def test_a_failed_expansion_keeps_its_transcript_and_says_where(tmp_path):
    sdfg, state, node = _sdfg_and_node()

    with dace.config.set_temporary('ai', 'transcripts', value=True):
        with dace.config.set_temporary('ai', 'transcript_dir', value=str(tmp_path)):
            with dace.config.set_temporary('ai', 'max_repair_attempts', value=0):
                with stub_provider(TaskletSpec(code=BAD)):
                    with pytest.raises(AIExpansionError) as info:
                        node.expand(state, 'ai')

    path, index = _only_transcript(tmp_path)
    assert index['outcome'] == 'failed'
    assert 'this_symbol_does_not_exist' in index['error']
    # The run cost a model call, so the error must lead back to what it produced
    assert path in str(info.value)


def test_transcripts_can_be_turned_off(tmp_path):
    sdfg, state, node = _sdfg_and_node()

    with dace.config.set_temporary('ai', 'transcripts', value=False):
        with dace.config.set_temporary('ai', 'transcript_dir', value=str(tmp_path)):
            with dace.config.set_temporary('ai', 'verify', value=False):
                with stub_provider(TaskletSpec(code=GOOD)):
                    node.expand(state, 'ai')

    assert next((n for n in state.nodes() if isinstance(n, nodes.Tasklet)), None) is not None
    assert not tmp_path.exists() or not list(tmp_path.iterdir())


@needs_compiler
def test_debugprint_verbose_shows_the_prompts_and_the_attempts(capsys):
    sdfg, state, node = _sdfg_and_node()

    with dace.config.set_temporary('debugprint', value='verbose'):
        with stub_provider_sequence([TaskletSpec(code=BAD), TaskletSpec(code=GOOD)]):
            node.expand(state, 'ai')

    out = capsys.readouterr().out
    assert 'attempt 1/3' in out and 'attempt 2/3' in out
    assert 'asking for a repair' in out
    assert 'probe compilation failed' in out and 'probe compiled cleanly' in out
    # Verbose adds the material each step acted on
    assert '--- system prompt ---' in out
    assert '--- user prompt ---' in out and '--- repair prompt ---' in out
    assert '--- probe source ---' in out and '--- probe diagnostics ---' in out
    assert 'this_symbol_does_not_exist' in out


@needs_compiler
def test_plain_debugprint_reports_steps_without_the_text(capsys):
    sdfg, state, node = _sdfg_and_node()

    with dace.config.set_temporary('debugprint', value=True):
        with stub_provider(TaskletSpec(code=GOOD)):
            node.expand(state, 'ai')

    out = capsys.readouterr().out
    assert 'attempt 1/3' in out
    assert 'probe compiled cleanly' in out
    assert 'expanded into a tasklet' in out
    assert '--- user prompt ---' not in out, 'the full prompt should only appear in verbose mode'


def test_nothing_is_printed_by_default(capsys):
    sdfg, state, node = _sdfg_and_node()

    with dace.config.set_temporary('debugprint', value=False):
        with dace.config.set_temporary('ai', 'verify', value=False):
            with stub_provider(TaskletSpec(code=GOOD)):
                node.expand(state, 'ai')

    assert '[ai]' not in capsys.readouterr().out


if __name__ == '__main__':
    pytest.main([__file__])
