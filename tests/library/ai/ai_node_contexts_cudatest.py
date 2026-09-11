# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""
Live counterpart to ``ai_node_contexts_test.py``: the same computation is generated three times,
in three different slots, and each result must be correct *and* respect its slot's constraints.

This is the end-to-end evidence that the context actually steers the model, rather than the model
producing one generic implementation regardless of where the node sits.
"""

import numpy as np
import pytest

import dace
from dace import dtypes, nodes
from dace.libraries.ai.nodes import AINode

M = 32

DESCRIPTION = f"""
Compute a {M}x{M} single-precision matrix product _c = _a * _b (row-major, contiguous). All three
pointers cover the whole {M}x{M} matrix.
""".strip()


def _matmul_sdfg(name: str, storage: dtypes.StorageType, inside_kernel: bool) -> dace.SDFG:
    """
    Builds a matrix multiplication around an :class:`AINode` in one specific slot.

    :param name: Name of the SDFG.
    :param storage: Storage type of the three matrices.
    :param inside_kernel: If True, the node is placed inside a ``GPU_Device`` map.
    :return: The SDFG.
    """
    sdfg = dace.SDFG(name)
    for array in ('A', 'B', 'C'):
        sdfg.add_array(array, [M, M], dace.float32, storage=storage)

    state = sdfg.add_state()
    node = AINode('matmul', DESCRIPTION, inputs={'_a', '_b'}, outputs={'_c'})
    state.add_node(node)
    subset = f'0:{M}, 0:{M}'

    if inside_kernel:
        entry, exit_node = state.add_map('grid', {'b': '0:1'}, schedule=dtypes.ScheduleType.GPU_Device)
        state.add_memlet_path(state.add_read('A'), entry, node, dst_conn='_a', memlet=dace.Memlet(f'A[{subset}]'))
        state.add_memlet_path(state.add_read('B'), entry, node, dst_conn='_b', memlet=dace.Memlet(f'B[{subset}]'))
        state.add_memlet_path(node, exit_node, state.add_write('C'), src_conn='_c', memlet=dace.Memlet(f'C[{subset}]'))
    else:
        state.add_edge(state.add_read('A'), None, node, '_a', dace.Memlet(f'A[{subset}]'))
        state.add_edge(state.add_read('B'), None, node, '_b', dace.Memlet(f'B[{subset}]'))
        state.add_edge(node, '_c', state.add_write('C'), None, dace.Memlet(f'C[{subset}]'))
    return sdfg


def _expand(sdfg: dace.SDFG) -> nodes.Tasklet:
    """
    Expands the SDFG's single AI node.

    :param sdfg: The SDFG to expand.
    :return: The generated tasklet.
    """
    state = sdfg.states()[0]
    node = next(n for n in state.nodes() if isinstance(n, AINode))
    node.expand(state, 'ai')
    return next(n for n in state.nodes() if isinstance(n, nodes.Tasklet))


def _run(sdfg: dace.SDFG, on_gpu: bool):
    """
    Runs the SDFG on random inputs and returns the result.

    :param sdfg: The compiled-and-run SDFG.
    :param on_gpu: If True, arguments are passed as GPU arrays.
    :return: A tuple of (a, b, c) as host arrays.
    """
    rng = np.random.default_rng(0)
    a = rng.random((M, M), dtype=np.float32)
    b = rng.random((M, M), dtype=np.float32)
    c = np.zeros((M, M), dtype=np.float32)
    if not on_gpu:
        sdfg(A=a, B=b, C=c)
        return a, b, c

    import cupy
    ga, gb, gc = cupy.asarray(a), cupy.asarray(b), cupy.zeros((M, M), dtype=np.float32)
    sdfg(A=ga, B=gb, C=gc)
    return a, b, cupy.asnumpy(gc)


@pytest.mark.ai
def test_host_cpu_context():
    sdfg = _matmul_sdfg('ai_ctx_host_cpu', dtypes.StorageType.CPU_Heap, inside_kernel=False)
    tasklet = _expand(sdfg)

    # Nothing GPU-related belongs in a pure host slot
    assert '__dace_current_stream' not in tasklet.code.as_string
    assert not tasklet.environments or all('cuda' not in e.lower() for e in tasklet.environments)

    a, b, c = _run(sdfg, on_gpu=False)
    assert np.allclose(c, a @ b, rtol=1e-4, atol=1e-4)


@pytest.mark.gpu
@pytest.mark.ai
def test_host_slot_over_gpu_arrays():
    sdfg = _matmul_sdfg('ai_ctx_host_gpu', dtypes.StorageType.GPU_Global, inside_kernel=False)
    tasklet = _expand(sdfg)
    body = tasklet.code.as_string

    # Host code over device pointers: it must hand them to a library or a kernel launch, and
    # order the work on the SDFG's stream, rather than dereferencing them directly
    assert '__dace_current_stream' in body or '<<<' in body or 'Launch' in body or 'blas' in body.lower()

    a, b, c = _run(sdfg, on_gpu=True)
    assert np.allclose(c, a @ b, rtol=1e-4, atol=1e-4)


@pytest.mark.gpu
@pytest.mark.ai
def test_device_slot_inside_a_kernel():
    sdfg = _matmul_sdfg('ai_ctx_device', dtypes.StorageType.GPU_Global, inside_kernel=True)
    tasklet = _expand(sdfg)
    body = tasklet.code.as_string

    # Inside a kernel there is no state struct, no stream variable, and no host library to call
    assert '__state' not in body
    assert '__dace_current_stream' not in body
    assert '<<<' not in body
    assert not tasklet.state_fields

    a, b, c = _run(sdfg, on_gpu=True)
    assert np.allclose(c, a @ b, rtol=1e-4, atol=1e-4)


if __name__ == '__main__':
    pytest.main([__file__, '-m', 'ai'])
