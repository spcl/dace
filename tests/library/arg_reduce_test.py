# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Standalone tests for the :class:`~dace.libraries.standard.nodes.ArgReduce`
libnode (argmax / argmin -> value + index, two scalar outputs)."""
import numpy as np
import pytest

import dace
from dace.libraries.standard.nodes import ArgReduce

N = dace.symbol('N')


def _build(op: str):
    """SDFG with a single ArgReduce over ``a[0:N]`` -> ``val`` (float64) +
    ``idx`` (int64)."""
    sdfg = dace.SDFG(f'argreduce_{op}')
    sdfg.add_array('a', [N], dace.float64)
    sdfg.add_array('val', [1], dace.float64)
    sdfg.add_array('idx', [1], dace.int64)
    state = sdfg.add_state()
    r = state.add_read('a')
    wv = state.add_write('val')
    wi = state.add_write('idx')
    node = ArgReduce('argreduce', op=op)
    state.add_node(node)
    state.add_edge(r, None, node, '_in', dace.Memlet('a[0:N]'))
    state.add_edge(node, '_out_val', wv, None, dace.Memlet('val[0]'))
    state.add_edge(node, '_out_idx', wi, None, dace.Memlet('idx[0]'))
    return sdfg


@pytest.mark.parametrize('op', ['max', 'min'])
def test_arg_reduce_value_and_index(op):
    sdfg = _build(op)
    sdfg.validate()
    sdfg.expand_library_nodes()
    n = 64
    rng = np.random.default_rng(0xA76 + (op == 'min'))
    a = rng.standard_normal(n)
    val = np.zeros(1)
    idx = np.zeros(1, dtype=np.int64)
    sdfg(a=a, val=val, idx=idx, N=n)
    if op == 'max':
        assert np.isclose(val[0], a.max())
        assert idx[0] == int(np.argmax(a))
    else:
        assert np.isclose(val[0], a.min())
        assert idx[0] == int(np.argmin(a))


@pytest.mark.parametrize('op', ['max', 'min'])
@pytest.mark.parametrize('stride', [2, 3])
def test_arg_reduce_strided_input(op, stride):
    """Strided input slice ``a[0:N*stride:stride]`` -- the expansion reads
    element ``j`` at ``_in[j*stride]`` (non-unit-stride code path) and returns
    the SLICE-LOCAL index ``j`` of the extreme strided element."""
    sdfg = dace.SDFG(f'argreduce_{op}_s{stride}')
    sdfg.add_array('a', [N], dace.float64)
    sdfg.add_array('val', [1], dace.float64)
    sdfg.add_array('idx', [1], dace.int64)
    state = sdfg.add_state()
    r = state.add_read('a')
    wv = state.add_write('val')
    wi = state.add_write('idx')
    node = ArgReduce('argreduce', op=op)
    state.add_node(node)
    # Reduce over the strided slice a[0 : N : stride].
    state.add_edge(r, None, node, '_in', dace.Memlet(f'a[0:N:{stride}]'))
    state.add_edge(node, '_out_val', wv, None, dace.Memlet('val[0]'))
    state.add_edge(node, '_out_idx', wi, None, dace.Memlet('idx[0]'))
    sdfg.validate()
    sdfg.expand_library_nodes()

    m = 16
    n = m * stride
    rng = np.random.default_rng(700 + stride + (op == 'min'))
    a = rng.standard_normal(n)
    val = np.zeros(1)
    idx = np.zeros(1, dtype=np.int64)
    sdfg(a=a, val=val, idx=idx, N=n)
    strided = a[0:n:stride]
    expected_j = int(np.argmax(strided)) if op == 'max' else int(np.argmin(strided))
    assert np.isclose(val[0], strided[expected_j]), f"value: got {val[0]}, expected {strided[expected_j]}"
    assert idx[0] == expected_j, f"slice-local index: got {idx[0]}, expected {expected_j}"


@pytest.mark.parametrize('op', ['max', 'min'])
def test_arg_reduce_tie_breaks_to_first(op):
    """Strict comparison -> the FIRST occurrence of the extreme wins (matches
    ``np.argmax``/``np.argmin``, which also return the first)."""
    sdfg = _build(op)
    sdfg.expand_library_nodes()
    # Two equal extremes; the earlier index must win.
    a = np.array([1.0, 5.0, 2.0, 5.0, 0.0, 0.0]) if op == 'max' else np.array([3.0, 0.0, 1.0, 0.0, 2.0])
    val = np.zeros(1)
    idx = np.zeros(1, dtype=np.int64)
    sdfg(a=a, val=val, idx=idx, N=a.shape[0])
    expected = int(np.argmax(a)) if op == 'max' else int(np.argmin(a))
    assert idx[0] == expected, f"{op}: got {idx[0]}, expected first extreme at {expected}"


def gpu_argreduce(op: str) -> dace.SDFG:
    """The same node with device operands, so the CUDA expansion is the one that applies."""
    sdfg = _build(op)
    sdfg.name = f'argreduce_gpu_{op}'
    # Input on the device, answers on the host: that is what ``host_connectors`` declares, and the
    # expansion writes both from the host code that issues the launch.
    sdfg.arrays['a'].storage = dace.StorageType.GPU_Global
    node = next(n for n, _ in sdfg.all_nodes_recursive() if isinstance(n, ArgReduce))
    node.implementation = 'CUDA'
    return sdfg


@pytest.mark.parametrize('op', ['max', 'min'])
def test_the_cuda_expansion_calls_cub_and_brings_the_pair_back(op):
    """CUB answers in DEVICE memory; both outputs are host scalars, so the pair has to be copied."""
    sdfg = gpu_argreduce(op)
    sdfg.expand_library_nodes()
    code = '\n'.join(c.clean_code for c in sdfg.generate_code())

    kind = 'ArgMax' if op == 'max' else 'ArgMin'
    # The gpu* spellings are the backend-neutral aliases, so one expansion serves CUDA and HIP.
    assert f'gpucub::DeviceReduce::{kind}' in code, f'the CUDA expansion did not call gpucub::DeviceReduce::{kind}'
    assert 'get_scratch<::dace::cub::ReduceTag>' in code, 'the workspace does not come from the scratch pool'
    assert 'gpuMemcpyAsync' in code and 'gpuMemcpyDeviceToHost' in code, \
        'the KeyValuePair is never copied back, so the outputs read device memory'
    assert 'gpuStreamSynchronize' in code, \
        'the copy back is asynchronous, so without a sync the host reads the pair before it lands'
    assert ArgReduce.host_connectors == frozenset(
        {'_out_val',
         '_out_idx'}), ('both answers are written by host code, so an offloader must be told to leave them there')


def test_the_cuda_expansion_refuses_a_strided_slice():
    """``cub::DeviceReduce`` takes a contiguous pointer; a strided slice needs an input iterator."""
    sdfg = dace.SDFG('argreduce_gpu_strided')
    sdfg.add_array('a', [64], dace.float64, storage=dace.StorageType.GPU_Global)
    sdfg.add_array('idx', [1], dace.int64)
    state = sdfg.add_state()
    node = ArgReduce('argreduce', op='max')
    node.implementation = 'CUDA'
    state.add_node(node)
    state.add_edge(state.add_read('a'), None, node, '_in', dace.Memlet('a[0:64:2]'))
    state.add_edge(node, '_out_idx', state.add_write('idx'), None, dace.Memlet('idx[0]'))

    with pytest.raises(NotImplementedError, match='contiguous pointer'):
        sdfg.expand_library_nodes()


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
