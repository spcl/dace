# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""hipTensor reads C in ``D = alpha * A * B + beta * C`` even when beta is 0.

A NaN left in the output buffer therefore comes out in every element (``0 * NaN``). A second
compiled program whose output buffers reuse the freed memory of the first reads that memory.
``ls3df_scf`` on ``dace_gpu_canonicalize`` failed this way when the harness ran optimize a second time.
"""
import numpy as np
import pytest

import dace
from dace import dtypes
from dace.codegen.common import get_gpu_backend
from dace.libraries.linalg.nodes.tensordot import TensorDot

L = 5


def contraction(implementation: str, out_shape=(L, L, L, 1), out_subset=None) -> dace.SDFG:
    """``C[a, q, r, s] = sum_j A[a, j] * B[j, q, r, s]``, written into ``C[out_subset]``."""
    sdfg = dace.SDFG(f'tensordot_out_init_{implementation}')
    for name, shape in (('A', [L, L]), ('B', [L, L, L, 1]), ('C', list(out_shape))):
        sdfg.add_array(name, shape, dace.float64, storage=dtypes.StorageType.GPU_Global)
    state = sdfg.add_state()
    node = TensorDot('contract', left_axes=[1], right_axes=[0])
    node.implementation = implementation
    state.add_node(node)
    for conn, name in (('_left_tensor', 'A'), ('_right_tensor', 'B')):
        state.add_edge(state.add_read(name), None, node, conn, dace.Memlet.from_array(name, sdfg.arrays[name]))
    out = dace.Memlet(f'C[{out_subset}]') if out_subset else dace.Memlet.from_array('C', sdfg.arrays['C'])
    state.add_edge(node, '_out_tensor', state.add_write('C'), None, out)
    sdfg.validate()
    return sdfg


def tasklet_code(sdfg: dace.SDFG) -> str:
    sdfg.expand_library_nodes()
    return '\n'.join(n.code.as_string for n, _ in sdfg.all_nodes_recursive() if isinstance(n, dace.nodes.Tasklet))


def test_hiptensor_zeroes_the_output_before_contracting():
    """The hipTensor expansion clears the output, on the contraction's stream, before the call."""
    code = tasklet_code(contraction('hipTENSOR'))
    assert 'gpuMemsetAsync(_out_tensor, 0' in code
    assert code.index('gpuMemsetAsync(_out_tensor') < code.index('hiptensorContract(')


def test_cutensor_output_is_not_zeroed():
    """cuTENSOR does not read C at beta 0, so its expansion issues no memset."""
    assert 'gpuMemsetAsync' not in tasklet_code(contraction('cuTENSOR'))


def test_hiptensor_strided_output_takes_the_pure_expansion():
    """An output that is not one run of memory cannot be zeroed by a memset; it is not handed to hipTensor."""
    code = tasklet_code(contraction('hipTENSOR', out_shape=(L, L, L, 2), out_subset=f'0:{L}, 0:{L}, 0:{L}, 0'))
    assert 'hiptensorContract' not in code


@pytest.mark.gpu
def test_contraction_ignores_a_nan_left_in_the_output():
    """The vendor contraction into a NaN-filled output equals ``np.tensordot``."""
    import cupy
    rng = np.random.default_rng(0)
    a, b = rng.random((L, L)), rng.random((L, L, L, 1))
    c = cupy.full((L, L, L, 1), np.nan)
    implementation = 'hipTENSOR' if get_gpu_backend() == 'hip' else 'cuTENSOR'
    contraction(implementation)(A=cupy.asarray(a), B=cupy.asarray(b), C=c)
    assert np.allclose(cupy.asnumpy(c), np.tensordot(a, b, axes=([1], [0])))


if __name__ == '__main__':
    test_hiptensor_zeroes_the_output_before_contracting()
    test_cutensor_output_is_not_zeroed()
    test_hiptensor_strided_output_takes_the_pure_expansion()
    test_contraction_ignores_a_nan_left_in_the_output()
