# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""
Tests for the reuse of model answers across runs.

The same library node in the same context produces the same prompt, and asking the same model the
same question again costs money for an answer already in hand.
"""

import contextlib
import os
import sys

import pytest

import dace
from dace import nodes
from dace.libraries.ai import backend
from dace.libraries.ai import cache as ai_cache
from dace.libraries.ai.backend import EnvironmentSpec, TaskletSpec
from dace.libraries.ai.exceptions import AIExpansionError
from dace.libraries.ai.nodes import AINode

sys.path.insert(0, os.path.dirname(__file__))
from ai_test_utils import stub_provider  # noqa: E402

CODE = '_out = 2.0 * _in;'


@contextlib.contextmanager
def caching(tmp_path):
    """
    Turns the cache on, pointed at a temporary directory.

    :param tmp_path: The pytest temporary directory.
    :return: The directory the cache writes to.
    """
    with dace.config.set_temporary('ai', 'cache', value=True):
        with dace.config.set_temporary('ai', 'cache_dir', value=str(tmp_path)):
            with dace.config.set_temporary('ai', 'verify', value=False):
                yield tmp_path


def _sdfg_and_node(name: str = 'double'):
    """
    Builds a one-element SDFG around an :class:`AINode`.

    :param name: Name of the node.
    :return: A tuple of (SDFG, state, node).
    """
    sdfg = dace.SDFG('ai_cache')
    sdfg.add_array('A', [1], dace.float64)
    sdfg.add_array('B', [1], dace.float64)
    state = sdfg.add_state()
    node = AINode(name, 'Double the input.', inputs={'_in'}, outputs={'_out'})
    state.add_node(node)
    state.add_edge(state.add_read('A'), None, node, '_in', dace.Memlet('A[0]'))
    state.add_edge(node, '_out', state.add_write('B'), None, dace.Memlet('B[0]'))
    return sdfg, state, node


def test_an_identical_prompt_is_not_asked_twice(tmp_path):
    spec = TaskletSpec(code=CODE, notes='why', raw_response='{"code": "as it arrived"}')

    with caching(tmp_path):
        _, first_state, first_node = _sdfg_and_node()
        with stub_provider(spec) as provider:
            first_node.expand(first_state, 'ai')
        assert len(provider.calls) == 1

        _, second_state, second_node = _sdfg_and_node()
        with stub_provider(spec) as provider:
            second_node.expand(second_state, 'ai')
        assert provider.calls == [], 'the model was asked again for an answer already on disk'

    tasklet = next(n for n in second_state.nodes() if isinstance(n, nodes.Tasklet))
    assert tasklet.code.as_string.strip() == CODE


def test_a_cached_answer_round_trips_every_field(tmp_path):
    spec = TaskletSpec(code=CODE,
                       code_global='#include <cmath>',
                       code_init='int x = 0;',
                       code_exit='// bye',
                       state_fields=['int counter;'],
                       side_effects=True,
                       ignored_symbols=['i'],
                       use_environments=['dace.libraries.blas.environments.openblas.OpenBLAS'],
                       environments=[EnvironmentSpec(name='FFTW', headers=['fftw3.h'])],
                       notes='why',
                       raw_response='{"code": "as it arrived"}')

    with caching(tmp_path):
        key = ai_cache.key('system', [{'role': 'user', 'content': 'prompt'}])
        ai_cache.store(key, spec, 'system', [{'role': 'user', 'content': 'prompt'}])
        restored = ai_cache.lookup(key)

    assert restored == spec


def test_the_key_covers_the_model_and_the_prompt(tmp_path):
    messages = [{'role': 'user', 'content': 'prompt'}]

    with caching(tmp_path):
        base = ai_cache.key('system', messages)
        assert ai_cache.key('system', messages) == base, 'the key is not stable'
        assert ai_cache.key('other system', messages) != base
        assert ai_cache.key('system', [{'role': 'user', 'content': 'other'}]) != base
        with dace.config.set_temporary('ai', 'model', value='some-other-model'):
            assert ai_cache.key('system', messages) != base
        with dace.config.set_temporary('ai', 'effort', value='low'):
            assert ai_cache.key('system', messages) != base


def test_a_different_context_is_asked_separately(tmp_path):
    """ Two nodes whose prompts differ must not share an entry, however similar they look. """
    with caching(tmp_path):
        sdfg = dace.SDFG('two_shapes')
        sdfg.add_array('A', [1], dace.float64)
        sdfg.add_array('B', [1], dace.float64)
        sdfg.add_array('C', [8, 8], dace.float64)
        sdfg.add_array('D', [8, 8], dace.float64)
        state = sdfg.add_state()

        for source, sink, subset in (('A', 'B', '0'), ('C', 'D', '0:8, 0:8')):
            node = AINode(f'n_{source}', 'Copy.', inputs={'_in'}, outputs={'_out'})
            state.add_node(node)
            state.add_edge(state.add_read(source), None, node, '_in', dace.Memlet(f'{source}[{subset}]'))
            state.add_edge(node, '_out', state.add_write(sink), None, dace.Memlet(f'{sink}[{subset}]'))
            with stub_provider(TaskletSpec(code='// generated')) as provider:
                node.expand(state, 'ai')
            assert len(provider.calls) == 1, 'a differently shaped node reused another one\'s answer'

        assert len(os.listdir(tmp_path)) == 2


def test_a_cache_hit_needs_no_provider(tmp_path):
    """
    A fully cached expansion must not require an SDK or a key.

    Re-expanding an SDFG on a machine that has neither is exactly when the cache earns its keep.
    """
    with caching(tmp_path):
        _, state, node = _sdfg_and_node()
        with stub_provider(TaskletSpec(code=CODE)):
            node.expand(state, 'ai')

        _, second_state, second_node = _sdfg_and_node()
        original = backend.get_provider

        def refuse():
            raise AIExpansionError('no credentials here')

        backend.get_provider = refuse
        try:
            second_node.expand(second_state, 'ai')
        finally:
            backend.get_provider = original

    assert any(isinstance(n, nodes.Tasklet) for n in second_state.nodes())


def test_caching_can_be_turned_off(tmp_path):
    with dace.config.set_temporary('ai', 'cache', value=False):
        with dace.config.set_temporary('ai', 'cache_dir', value=str(tmp_path)):
            with dace.config.set_temporary('ai', 'verify', value=False):
                for _ in range(2):
                    _, state, node = _sdfg_and_node()
                    with stub_provider(TaskletSpec(code=CODE)) as provider:
                        node.expand(state, 'ai')
                    assert len(provider.calls) == 1

    assert not tmp_path.exists() or not list(tmp_path.iterdir())


def test_an_unreadable_entry_is_ignored(tmp_path):
    with caching(tmp_path):
        key = ai_cache.key('system', [{'role': 'user', 'content': 'prompt'}])
        (tmp_path / f'{key}.json').write_text('{ this is not json')

        # A corrupt or hand-edited entry costs a request, never a failed expansion
        assert ai_cache.lookup(key) is None


if __name__ == '__main__':
    pytest.main([__file__])
