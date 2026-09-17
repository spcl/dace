# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
""" Tests for the probe compilation and the repair loop around it. """

import os
import shutil
import sys

import pytest

import dace
from dace import dtypes, nodes
from dace.libraries.ai import verify
from dace.libraries.ai.backend import TaskletSpec
from dace.libraries.ai.context import collect_context
from dace.libraries.ai.exceptions import AIExpansionError
from dace.libraries.ai.nodes import AINode
from dace.sdfg import infer_types

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


#: storage, memlet subset, dynamic, and the C++ types both the prompt and the generated code must
#: use for the input and the output. A GPU operand is a pointer even for a single element: taking
#: it by value would compile into a host-side load out of device memory. A dynamic output has no
#: single destination to write back to and so stays a pointer, while a dynamic input does not.
CONNECTOR_SHAPES = {
    'cpu_element': (dtypes.StorageType.CPU_Heap, '0, 0', False, 'float', 'float'),
    'cpu_tile': (dtypes.StorageType.CPU_Heap, '0:4, 0:4', False, 'float*', 'float*'),
    'gpu_element': (dtypes.StorageType.GPU_Global, '0, 0', False, 'float*', 'float*'),
    'gpu_tile': (dtypes.StorageType.GPU_Global, '0:4, 0:4', False, 'float*', 'float*'),
    'dynamic_element': (dtypes.StorageType.CPU_Heap, '0, 0', True, 'float', 'float*'),
}


@pytest.mark.parametrize('shape', sorted(CONNECTOR_SHAPES))
def test_the_probe_declares_connectors_the_way_codegen_will(shape):
    """
    The probe must predict connector types exactly as ``infer_types`` later resolves them.

    A pointer where the generated code will hold a value (or the reverse) makes the probe accept
    code the real build rejects -- the one failure the verification step exists to prevent, and one
    that no amount of prompting can recover from, since the model is told the wrong type too.
    """
    storage, subset, dynamic, expected_in, expected_out = CONNECTOR_SHAPES[shape]

    sdfg = dace.SDFG(f'ai_conn_{shape}')
    sdfg.add_array('A', [8, 8], dace.float32, storage=storage)
    sdfg.add_array('B', [8, 8], dace.float32, storage=storage)
    state = sdfg.add_state()
    node = AINode('n', 'Do something.', inputs={'_in'}, outputs={'_out'})
    state.add_node(node)
    read = dace.Memlet(f'A[{subset}]')
    write = dace.Memlet(f'B[{subset}]')
    read.dynamic = write.dynamic = dynamic
    state.add_edge(state.add_read('A'), None, node, '_in', read)
    state.add_edge(node, '_out', state.add_write('B'), None, write)

    predicted = {c.name: c.ctype for c in collect_context(node, state, sdfg).connectors}

    with dace.config.set_temporary('ai', 'verify', value=False):
        with stub_provider(TaskletSpec(code='// nothing')):
            node.expand(state, 'ai')
    infer_types.infer_connector_types(sdfg)

    tasklet = next(n for n in state.nodes() if isinstance(n, nodes.Tasklet))
    actual = {
        name: str(conntype.ctype)
        for name, conntype in list(tasklet.in_connectors.items()) + list(tasklet.out_connectors.items())
    }
    assert predicted == {'_in': expected_in, '_out': expected_out}
    assert actual == predicted


if __name__ == '__main__':
    pytest.main([__file__])
