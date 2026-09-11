# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""
Tests that one library node type is expanded differently depending on where it sits.

A ``Gemm`` inside a GPU kernel, a ``Gemm`` on the host over GPU arrays, and a ``Gemm`` on the host
over CPU arrays all need different code, and the difference is not visible from the node itself.
The context handed to the model must make it explicit.
"""

import os
import sys

import dace
import dace.libraries.ai as ai
from dace import dtypes, nodes
from dace.libraries.blas.nodes.gemm import Gemm
from dace.libraries.ai.context import collect_context
from dace.libraries.ai.prompts import build_user_prompt

sys.path.insert(0, os.path.dirname(__file__))
from ai_test_utils import stub_provider  # noqa: E402

from dace.libraries.ai.backend import TaskletSpec  # noqa: E402

M = 8


def _add_gemm(state: dace.SDFGState, name: str, arrays, entry=None, exit_node=None) -> Gemm:
    """
    Adds a ``Gemm`` node reading ``arrays[0]``, ``arrays[1]`` and writing ``arrays[2]``.

    :param state: The state to add the node to.
    :param name: Name of the node.
    :param arrays: The three container names to connect.
    :param entry: Enclosing map entry, if the node goes inside a map.
    :param exit_node: Enclosing map exit, if the node goes inside a map.
    :return: The added node.
    """
    a, b, c = arrays
    gemm = Gemm(name)
    state.add_node(gemm)
    subset = f'0:{M}, 0:{M}'
    if entry is None:
        state.add_edge(state.add_read(a), None, gemm, '_a', dace.Memlet(f'{a}[{subset}]'))
        state.add_edge(state.add_read(b), None, gemm, '_b', dace.Memlet(f'{b}[{subset}]'))
        state.add_edge(gemm, '_c', state.add_write(c), None, dace.Memlet(f'{c}[{subset}]'))
    else:
        state.add_memlet_path(state.add_read(a), entry, gemm, dst_conn='_a', memlet=dace.Memlet(f'{a}[{subset}]'))
        state.add_memlet_path(state.add_read(b), entry, gemm, dst_conn='_b', memlet=dace.Memlet(f'{b}[{subset}]'))
        state.add_memlet_path(gemm, exit_node, state.add_write(c), src_conn='_c', memlet=dace.Memlet(f'{c}[{subset}]'))
    return gemm


def _build_three_contexts():
    """
    Builds one SDFG holding the same node type in three different contexts.

    :return: A tuple of (SDFG, {context name: (node, state)}).
    """
    sdfg = dace.SDFG('three_contexts')
    for name in ('Ag', 'Bg', 'Cg'):
        sdfg.add_array(name, [M, M], dace.float32, storage=dtypes.StorageType.GPU_Global)
    for name in ('Ac', 'Bc', 'Cc'):
        sdfg.add_array(name, [M, M], dace.float32, storage=dtypes.StorageType.CPU_Heap)

    kernel_state = sdfg.add_state('in_kernel', is_start_block=True)
    entry, exit_node = kernel_state.add_map('grid', {'bi': '0:1'}, schedule=dtypes.ScheduleType.GPU_Device)
    device = _add_gemm(kernel_state, 'gemm_device', ('Ag', 'Bg', 'Cg'), entry, exit_node)

    host_gpu_state = sdfg.add_state_after(kernel_state, 'host_gpu')
    host_gpu = _add_gemm(host_gpu_state, 'gemm_host_gpu', ('Ag', 'Bg', 'Cg'))

    host_cpu_state = sdfg.add_state_after(host_gpu_state, 'host_cpu')
    host_cpu = _add_gemm(host_cpu_state, 'gemm_host_cpu', ('Ac', 'Bc', 'Cc'))

    return sdfg, {
        'device': (device, kernel_state),
        'host_gpu': (host_gpu, host_gpu_state),
        'host_cpu': (host_cpu, host_cpu_state),
    }


def test_capabilities_differ_by_context():
    sdfg, slots = _build_three_contexts()
    caps = {name: collect_context(node, state, sdfg).capabilities for name, (node, state) in slots.items()}

    assert [caps[k].device_level for k in ('device', 'host_gpu', 'host_cpu')] == [True, False, False]
    assert [caps[k].state_available for k in ('device', 'host_gpu', 'host_cpu')] == [False, True, True]
    assert [caps[k].current_stream_available for k in ('device', 'host_gpu', 'host_cpu')] == [False, True, False]
    assert [caps[k].environments_allowed
            for k in ('device', 'host_gpu', 'host_cpu')] == ['device-headers-only', 'full', 'full']

    # Reachability depends on where the code runs, not on the storage alone: GPU memory is
    # unreachable from the host and perfectly reachable from inside a kernel. Reporting it as
    # off limits inside the kernel would leave the model with nothing it is allowed to do.
    assert all(caps['device'].dereferenceable.values())
    assert not any(caps['host_gpu'].dereferenceable.values())
    assert all(caps['host_cpu'].dereferenceable.values())

    device_node, device_state = slots['device']
    device_prompt = build_user_prompt(collect_context(device_node, device_state, sdfg))
    assert 'Pointers you must NOT dereference' not in device_prompt


def test_storage_reaches_the_connectors():
    sdfg, slots = _build_three_contexts()

    for name, expected in (('host_gpu', 'GPU_Global'), ('host_cpu', 'CPU_Heap')):
        node, state = slots[name]
        ctx = collect_context(node, state, sdfg)
        assert {c.storage for c in ctx.connectors} == {expected}
        assert all(c.shape == (str(M), str(M)) for c in ctx.connectors)


def test_class_docstring_is_used_as_the_specification():
    from dace.libraries.ai.context import collect_class_docstring
    from dace.libraries.ai.nodes import AINode

    sdfg, slots = _build_three_contexts()
    node, state = slots['host_cpu']
    ctx = collect_context(node, state, sdfg)

    # For a library node shipped with DaCe, the class docstring is the only statement of what the
    # node computes, so it must reach the model
    assert 'alpha * (A @ B) + beta * C' in ctx.class_docstring
    assert 'alpha * (A @ B) + beta * C' in build_user_prompt(ctx)

    # The base class documents the IR, not a computation, and must not be picked up
    assert collect_class_docstring(nodes.LibraryNode('bare')) == ''

    # An explicit description supersedes the class documentation
    described = AINode('described', 'Compute the thing.', inputs={'_a'}, outputs={'_b'})
    described_state = sdfg.add_state('described')
    described_state.add_node(described)
    described_ctx = collect_context(described, described_state, sdfg)
    assert described_ctx.description == 'Compute the thing.'
    assert described_ctx.class_docstring == ''


def test_prompts_are_pairwise_distinct():
    sdfg, slots = _build_three_contexts()
    prompts = {name: build_user_prompt(collect_context(node, state, sdfg)) for name, (node, state) in slots.items()}

    assert len(set(prompts.values())) == 3, 'the same node type produced identical prompts in different contexts'

    assert '__state available in the body: no' in prompts['device']
    assert '__state available in the body: yes' in prompts['host_gpu']
    assert '__dace_current_stream in scope: yes' in prompts['host_gpu']
    assert '__dace_current_stream in scope: no' in prompts['host_cpu']
    assert 'Pointers you must NOT dereference here' in prompts['host_gpu']
    assert 'Pointers you must NOT dereference here' not in prompts['host_cpu']


def test_each_context_is_generated_independently():
    sdfg, slots = _build_three_contexts()
    seen_prompts = []

    with dace.config.set_temporary('ai', 'verify', value=False):
        for index, (name, (node, state)) in enumerate(slots.items()):
            spec = TaskletSpec(code=f'// generated for {name}\n_c[0] = _a[0] * _b[0];')
            with stub_provider(spec) as provider:
                node.expand(state, 'ai')
            seen_prompts.append(provider.calls[0][0]['content'])

    # One tasklet per slot, each carrying its own generated code
    tasklets = [n for s in sdfg.states() for n in s.nodes() if isinstance(n, nodes.Tasklet)]
    assert len(tasklets) == 3
    assert len({t.code.as_string for t in tasklets}) == 3
    assert not [n for s in sdfg.states() for n in s.nodes() if isinstance(n, Gemm)]

    # The synthesized expansion class is shared, but nothing about the result is
    assert len(set(seen_prompts)) == 3
    assert ai.ExpandAI.for_node_class(Gemm) is ai.ExpandAI.for_node_class(Gemm)


if __name__ == '__main__':
    test_capabilities_differ_by_context()
    test_storage_reaches_the_connectors()
    test_prompts_are_pairwise_distinct()
    test_each_context_is_generated_independently()
