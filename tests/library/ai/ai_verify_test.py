# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
""" Tests for the probe compilation and the repair loop around it. """

import os
import shutil
import sys

import pytest

import dace
from dace import nodes
from dace.libraries.ai import verify
from dace.libraries.ai.backend import TaskletSpec
from dace.libraries.ai.context import collect_context
from dace.libraries.ai.exceptions import AIExpansionError
from dace.libraries.ai.nodes import AINode

sys.path.insert(0, os.path.dirname(__file__))
from ai_test_utils import stub_provider, stub_provider_sequence  # noqa: E402

pytestmark = pytest.mark.skipif(shutil.which('c++') is None and shutil.which('g++') is None,
                                reason='needs a host C++ compiler for the probe')

GOOD = '_out = 2.0 * _in;'
BAD = '_out = this_symbol_does_not_exist(_in);'


def _sdfg_and_node():
    """
    Builds a one-element SDFG around an :class:`AINode`.

    :return: A tuple of (SDFG, state, node).
    """
    sdfg = dace.SDFG('ai_verify')
    sdfg.add_array('A', [1], dace.float64)
    sdfg.add_array('B', [1], dace.float64)
    state = sdfg.add_state()
    node = AINode('double', 'Double the input.', inputs={'_in'}, outputs={'_out'})
    state.add_node(node)
    state.add_edge(state.add_read('A'), None, node, '_in', dace.Memlet('A[0]'))
    state.add_edge(node, '_out', state.add_write('B'), None, dace.Memlet('B[0]'))
    return sdfg, state, node


def test_probe_source_reconstructs_the_tasklet_environment():
    sdfg, state, node = _sdfg_and_node()
    ctx = collect_context(node, state, sdfg)
    spec = TaskletSpec(code=GOOD, code_global='#include <cmath>', state_fields=['int counter;'])

    source = verify.build_probe_source(spec, ctx, [])

    assert '#include <cmath>' in source
    assert 'int counter;' in source
    assert '__dace_ai_state' in source or '__dace_probe_state' in source
    assert GOOD in source
    # Connectors become parameters, with the types the generated code will actually see
    assert '_in' in source and '_out' in source


def test_probe_accepts_valid_code():
    sdfg, state, node = _sdfg_and_node()
    ctx = collect_context(node, state, sdfg)

    result = verify.probe_compile(TaskletSpec(code=GOOD), ctx, [])
    assert result.ok
    assert not result.inconclusive


def test_probe_rejects_invalid_code_with_diagnostics():
    sdfg, state, node = _sdfg_and_node()
    ctx = collect_context(node, state, sdfg)

    result = verify.probe_compile(TaskletSpec(code=BAD), ctx, [])
    assert not result.ok
    assert not result.inconclusive
    assert 'this_symbol_does_not_exist' in result.stderr
    assert result.command


def test_probe_skips_python_tasklets():
    sdfg, state, node = _sdfg_and_node()
    ctx = collect_context(node, state, sdfg)

    result = verify.probe_compile(TaskletSpec(code='_out = _in', language='Python'), ctx, [])
    assert result.ok and result.inconclusive


def test_repair_loop_recovers_from_a_compile_error():
    sdfg, state, node = _sdfg_and_node()

    with stub_provider_sequence([TaskletSpec(code=BAD), TaskletSpec(code=GOOD)]) as provider:
        node.expand(state, 'ai')

    assert len(provider.calls) == 2, 'the model was not asked to repair the broken code'
    repair_conversation = provider.calls[1]
    assert [m['role'] for m in repair_conversation] == ['user', 'assistant', 'user']
    assert 'this_symbol_does_not_exist' in repair_conversation[-1]['content']
    assert 'does not compile' in repair_conversation[-1]['content']

    tasklet = next(n for n in state.nodes() if isinstance(n, nodes.Tasklet))
    assert tasklet.code.as_string.strip() == GOOD


def test_repair_loop_gives_up_after_the_configured_attempts():
    sdfg, state, node = _sdfg_and_node()

    with dace.config.set_temporary('ai', 'max_repair_attempts', value=1):
        with stub_provider(TaskletSpec(code=BAD)) as provider:
            with pytest.raises(AIExpansionError, match='does not compile'):
                node.expand(state, 'ai')

    # One initial generation plus one repair
    assert len(provider.calls) == 2
    # The library node is left in place, so the failure can be diagnosed
    assert any(isinstance(n, AINode) for n in state.nodes())


def test_verification_can_be_disabled():
    sdfg, state, node = _sdfg_and_node()

    with dace.config.set_temporary('ai', 'verify', value=False):
        with stub_provider(TaskletSpec(code=BAD)) as provider:
            node.expand(state, 'ai')

    assert len(provider.calls) == 1, 'the code was verified even though verification is off'


if __name__ == '__main__':
    pytest.main([__file__])
