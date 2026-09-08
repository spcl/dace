# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""
Tests for iterating on a tasklet that has already been generated.

The point of refinement is that the model sees the code it wrote and the critique of it, so this is
a revision rather than a fresh generation -- and that it works on an SDFG loaded from disk, which is
when it is most often wanted.
"""

import contextlib
import os
import shutil
import sys

import numpy as np
import pytest

import dace
import dace.libraries.ai as ai
from dace import nodes
from dace.libraries.ai import iterate
from dace.libraries.ai.backend import TaskletSpec
from dace.libraries.ai.exceptions import AIExpansionError
from dace.libraries.ai.nodes import AINode, AITasklet

sys.path.insert(0, os.path.dirname(__file__))
from ai_test_utils import stub_provider  # noqa: E402

V1 = '_out = 2.0 * _in;'
V2 = '_out = _in + _in;'
BAD = '_out = this_symbol_does_not_exist(_in);'

needs_compiler = pytest.mark.skipif(shutil.which('c++') is None and shutil.which('g++') is None,
                                    reason='needs a host C++ compiler for the probe')


@contextlib.contextmanager
def iterating(tmp_path, verify=False):
    """
    Turns sessions on, pointed at a temporary directory.

    :param tmp_path: The pytest temporary directory.
    :param verify: Whether to run the probe compilation.
    """
    with dace.config.set_temporary('ai', 'sessions', value=True):
        with dace.config.set_temporary('ai', 'session_dir', value=str(tmp_path)):
            with dace.config.set_temporary('ai', 'verify', value=verify):
                yield


def _build(name: str = 'double', size: int = 16):
    """
    Builds a mapped SDFG around one :class:`AINode`.

    :param name: Name of the node, which also names the session.
    :param size: Length of the arrays.
    :return: A tuple of (SDFG, state, node).
    """
    sdfg = dace.SDFG('ai_refine')
    sdfg.add_array('A', [size], dace.float64)
    sdfg.add_array('B', [size], dace.float64)
    state = sdfg.add_state()
    node = AINode(name, 'Double the input.', inputs={'_in'}, outputs={'_out'})
    state.add_node(node)
    me, mx = state.add_map('m', {'i': f'0:{size}'})
    state.add_memlet_path(state.add_read('A'), me, node, dst_conn='_in', memlet=dace.Memlet('A[i]'))
    state.add_memlet_path(node, mx, state.add_write('B'), src_conn='_out', memlet=dace.Memlet('B[i]'))
    return sdfg, state, node


def _expand(sdfg, state, node, code=V1):
    """
    Expands a node with a stubbed answer.

    :param sdfg: The SDFG.
    :param state: The state holding the node.
    :param node: The library node.
    :param code: The tasklet body to pretend the model produced.
    """
    with stub_provider(TaskletSpec(code=code)):
        node.expand(state, 'ai')


def test_expansion_leaves_a_refinable_slot(tmp_path):
    sdfg, state, node = _build()
    with iterating(tmp_path):
        _expand(sdfg, state, node)

    slots = ai.sessions(sdfg)
    assert len(slots) == 1
    assert slots[0].name == 'double'
    assert slots[0].session == 'ai_refine.double'
    assert slots[0].round == 1
    assert not slots[0].pinned and not slots[0].edited
    assert isinstance(slots[0].tasklet, AITasklet)


def test_refine_continues_the_conversation(tmp_path):
    sdfg, state, node = _build()
    with iterating(tmp_path):
        _expand(sdfg, state, node)

        with stub_provider(TaskletSpec(code=V2)) as provider:
            ai.refine(sdfg, 'double', 'Too slow. Use an addition instead of a multiplication.')

    # The model was shown its own previous answer and then the critique, rather than being asked
    # the original question over again
    conversation = provider.calls[0]
    assert [m['role'] for m in conversation] == ['user', 'assistant', 'user']
    assert V1 in conversation[1]['content'], 'the previous answer was not replayed'
    assert 'Too slow' in conversation[-1]['content']
    assert 'accepted and used' in conversation[-1]['content']

    tasklet = ai.sessions(sdfg)[0].tasklet
    assert tasklet.code.as_string.strip() == V2
    assert ai.sessions(sdfg)[0].round == 2


def test_the_refined_tasklet_still_runs(tmp_path):
    sdfg, state, node = _build()
    with iterating(tmp_path):
        _expand(sdfg, state, node)
        with stub_provider(TaskletSpec(code=V2)):
            ai.refine(sdfg, 'double', 'Use addition.')

    A = np.random.rand(16)
    B = np.zeros(16)
    sdfg(A=A, B=B)
    assert np.allclose(B, 2 * A)


def test_refine_works_after_a_save_and_load(tmp_path):
    """ The case refinement exists for: an SDFG from disk, with no library node left in it. """
    sdfg, state, node = _build()
    with iterating(tmp_path):
        _expand(sdfg, state, node)

        path = str(tmp_path / 'saved.sdfg')
        sdfg.save(path)
        loaded = dace.SDFG.from_file(path)

        assert not [n for s in loaded.states() for n in s.nodes() if isinstance(n, nodes.LibraryNode)]
        assert type(ai.sessions(loaded)[0].tasklet) is AITasklet

        with stub_provider(TaskletSpec(code=V2)) as provider:
            ai.refine(loaded, 'double', 'Use addition.')

    assert V1 in provider.calls[0][1]['content'], 'the conversation did not survive the round trip'
    assert ai.sessions(loaded)[0].tasklet.code.as_string.strip() == V2

    A = np.random.rand(16)
    B = np.zeros(16)
    loaded(A=A, B=B)
    assert np.allclose(B, 2 * A)


@needs_compiler
def test_a_failed_refinement_leaves_the_working_tasklet_alone(tmp_path):
    sdfg, state, node = _build()
    with iterating(tmp_path, verify=True):
        _expand(sdfg, state, node)
        before = ai.sessions(sdfg)[0].tasklet

        with dace.config.set_temporary('ai', 'max_repair_attempts', value=0):
            with stub_provider(TaskletSpec(code=BAD)):
                with pytest.raises(AIExpansionError):
                    ai.refine(sdfg, 'double', 'Make it faster.')

    # The user had working code a moment ago and must still have it
    slots = ai.sessions(sdfg)
    assert len(slots) == 1
    assert slots[0].tasklet.code.as_string.strip() == V1
    assert slots[0].tasklet is before
    assert not [n for n in state.nodes() if isinstance(n, nodes.LibraryNode)], \
        'a failed refinement left the SDFG holding an unexpanded library node'

    A = np.random.rand(16)
    B = np.zeros(16)
    sdfg(A=A, B=B)
    assert np.allclose(B, 2 * A)


def test_history_and_rollback(tmp_path):
    sdfg, state, node = _build()
    with iterating(tmp_path):
        _expand(sdfg, state, node)
        with stub_provider(TaskletSpec(code=V2)):
            ai.refine(sdfg, 'double', 'Use addition.')

        rounds = ai.history(sdfg, 'double')
        assert [r.number for r in rounds] == [1, 2]
        assert [r.outcome for r in rounds] == ['expanded', 'expanded']
        assert 'Use addition' in rounds[1].feedback
        assert rounds[0].feedback == ''

        # Rollback asks the model nothing: every round's code is on disk
        ai.rollback(sdfg, 'double', round=1)

    slot = ai.sessions(sdfg)[0]
    assert slot.tasklet.code.as_string.strip() == V1
    assert slot.round == 1
    assert not slot.edited, 'rollback must leave a consistent fingerprint, not look like a hand edit'


def test_rollback_rejects_a_round_with_no_code(tmp_path):
    sdfg, state, node = _build()
    with iterating(tmp_path):
        _expand(sdfg, state, node)
        with pytest.raises(AIExpansionError, match='no code on record'):
            ai.rollback(sdfg, 'double', round=7)


def test_a_hand_edit_is_carried_into_the_next_round(tmp_path):
    sdfg, state, node = _build()
    with iterating(tmp_path):
        _expand(sdfg, state, node)

        edited = '_out = 3.0 * _in;  // by hand'
        tasklet = ai.sessions(sdfg)[0].tasklet
        tasklet.code = dace.properties.CodeBlock(edited, dace.dtypes.Language.CPP)
        assert ai.sessions(sdfg)[0].edited

        with stub_provider(TaskletSpec(code=V2)) as provider:
            ai.refine(sdfg, 'double', 'Now make it faster.')

    # Regenerating from the recorded round would silently discard the edit
    prompt = provider.calls[0][-1]['content']
    assert 'edited by hand' in prompt
    assert edited in prompt
    assert 'Now make it faster' in prompt


def test_pinning_refuses_regeneration(tmp_path):
    sdfg, state, node = _build()
    with iterating(tmp_path):
        _expand(sdfg, state, node)
        ai.pin(sdfg, 'double')
        assert ai.sessions(sdfg)[0].pinned

        with pytest.raises(AIExpansionError, match='pinned'):
            ai.refine(sdfg, 'double', 'Make it faster.')

        ai.unpin(sdfg, 'double')
        with stub_provider(TaskletSpec(code=V2)):
            ai.refine(sdfg, 'double', 'Make it faster.')
    assert ai.sessions(sdfg)[0].tasklet.code.as_string.strip() == V2


def test_pinning_survives_a_round_trip(tmp_path):
    sdfg, state, node = _build()
    with iterating(tmp_path):
        _expand(sdfg, state, node)
        ai.pin(sdfg, 'double')
        path = str(tmp_path / 'pinned.sdfg')
        sdfg.save(path)
        loaded = dace.SDFG.from_file(path)
    assert ai.sessions(loaded)[0].pinned


def test_selecting_among_several_slots(tmp_path):
    sdfg = dace.SDFG('two_slots')
    for name in 'ABCD':
        sdfg.add_array(name, [1], dace.float64)
    state = sdfg.add_state()
    with iterating(tmp_path):
        for src, dst in (('A', 'B'), ('C', 'D')):
            node = AINode(f'n_{src}', 'Copy.', inputs={'_in'}, outputs={'_out'})
            state.add_node(node)
            state.add_edge(state.add_read(src), None, node, '_in', dace.Memlet(f'{src}[0]'))
            state.add_edge(node, '_out', state.add_write(dst), None, dace.Memlet(f'{dst}[0]'))
            _expand(sdfg, state, node)

        assert len(ai.sessions(sdfg)) == 2

        # Ambiguity is an error that lists the candidates, never a silent pick
        with pytest.raises(AIExpansionError, match='more than one'):
            ai.refine(sdfg, None, 'faster')
        with pytest.raises(AIExpansionError, match='No AI-generated tasklet matches'):
            ai.refine(sdfg, 'nonexistent', 'faster')

        with stub_provider(TaskletSpec(code=V2)):
            ai.refine(sdfg, 'n_A', 'faster')

    by_name = {s.name: s for s in ai.sessions(sdfg)}
    assert by_name['n_A'].round == 2 and by_name['n_C'].round == 1
    assert by_name['n_C'].tasklet.code.as_string.strip() == V1


def test_refining_a_plain_sdfg_says_so():
    sdfg = dace.SDFG('nothing_here')
    sdfg.add_array('A', [1], dace.float64)
    sdfg.add_state()
    with pytest.raises(AIExpansionError, match='no AI-generated tasklets'):
        ai.refine(sdfg, None, 'faster')


def test_refine_without_feedback_does_not_return_the_cached_answer(tmp_path):
    """ "Try again" must not be served the very answer it is trying to replace. """
    sdfg, state, node = _build()
    with iterating(tmp_path):
        with dace.config.set_temporary('ai', 'cache', value=True):
            with dace.config.set_temporary('ai', 'cache_dir', value=str(tmp_path / 'cache')):
                _expand(sdfg, state, node)
                with stub_provider(TaskletSpec(code=V2)) as provider:
                    ai.refine(sdfg, 'double')
                assert len(provider.calls) == 1, 'the cached first answer was served again'

    assert ai.sessions(sdfg)[0].tasklet.code.as_string.strip() == V2


def test_show_renders_the_current_code(tmp_path, capsys):
    sdfg, state, node = _build()
    with iterating(tmp_path):
        with stub_provider(TaskletSpec(code=V1, code_global='#include <cmath>', state_fields=['int n;'])):
            node.expand(state, 'ai')
        rendered = ai.show(sdfg, 'double')

    assert V1 in rendered and '#include <cmath>' in rendered and 'int n;' in rendered
    assert rendered in capsys.readouterr().out


def test_provenance_of_a_plain_tasklet_is_absent():
    sdfg = dace.SDFG('plain')
    sdfg.add_array('A', [1], dace.float64)
    state = sdfg.add_state()
    t = state.add_tasklet('t', {}, {}, '')
    assert iterate.read(t) is None
    assert ai.sessions(sdfg) == []


if __name__ == '__main__':
    pytest.main([__file__])
