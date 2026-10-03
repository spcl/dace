# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""A device GEMM of a matrix that starts at an offset inside a bigger array stays inside it.

The cuBLAS/rocBLAS expansion wraps its call in a nested SDFG whose arrays were copies of the WHOLE
caller arrays, while the connector points at the first element of the box the memlet names. Once
the wrapper was inlined, the whole-array read shifted by the box's offset ran past the end of the
array: ``Memlet subset out-of-bounds`` for ``A[2:11, 0:3]`` of a ``(9, 3)`` array.
"""
import numpy as np
import pytest

import dace
from dace import dtypes
from dace.codegen import common
from dace.libraries.blas.nodes.gemm import Gemm

M, K, N = 5, 7, 3
PAD = 2


def offset_gemm(implementation: str, beta: float = 0.0) -> dace.SDFG:
    """``C[PAD:, :] = A[PAD:, PAD:] @ B[PAD:, :] + beta * C[PAD:, :]`` on device arrays."""
    sdfg = dace.SDFG(f'offset_gemm_{implementation}_{int(beta)}')
    gpu = dtypes.StorageType.GPU_Global
    sdfg.add_array('A', [M + PAD, K + PAD], dace.float64, storage=gpu)
    sdfg.add_array('B', [K + PAD, N], dace.float64, storage=gpu)
    sdfg.add_array('C', [M + PAD, N], dace.float64, storage=gpu)
    state = sdfg.add_state()
    node = Gemm('gemm', beta=beta)
    node.implementation = implementation
    state.add_node(node)
    state.add_edge(state.add_read('A'), None, node, '_a', dace.Memlet(f'A[{PAD}:{M + PAD}, {PAD}:{K + PAD}]'))
    state.add_edge(state.add_read('B'), None, node, '_b', dace.Memlet(f'B[{PAD}:{K + PAD}, 0:{N}]'))
    if beta:
        state.add_edge(state.add_read('C'), None, node, '_c', dace.Memlet(f'C[{PAD}:{M + PAD}, 0:{N}]'))
    state.add_edge(node, '_c', state.add_write('C'), None, dace.Memlet(f'C[{PAD}:{M + PAD}, 0:{N}]'))
    return sdfg


@pytest.mark.parametrize('beta', [0.0, 1.0], ids=['write', 'accumulate'])
@pytest.mark.parametrize('implementation', ['cuBLAS', 'rocBLAS'])
def test_the_expanded_gemm_reads_only_its_boxes(implementation, beta):
    """The wrapper is inlined on the way to codegen, which is where the whole-array memlet broke."""
    sdfg = offset_gemm(implementation, beta)
    sdfg.expand_library_nodes()
    sdfg.simplify()
    sdfg.validate()


@pytest.mark.gpu
@pytest.mark.parametrize('beta', [0.0, 1.0], ids=['write', 'accumulate'])
def test_an_offset_device_gemm_matches_numpy(beta):
    import cupy
    sdfg = offset_gemm('rocBLAS' if common.get_gpu_backend() == 'hip' else 'cuBLAS', beta)
    rng = np.random.default_rng(7)
    a = rng.standard_normal((M + PAD, K + PAD))
    b = rng.standard_normal((K + PAD, N))
    c = rng.standard_normal((M + PAD, N))
    ref = c.copy()
    ref[PAD:] = a[PAD:, PAD:] @ b[PAD:] + beta * c[PAD:]
    dc = cupy.asarray(c)
    sdfg(A=cupy.asarray(a), B=cupy.asarray(b), C=dc)
    np.testing.assert_allclose(cupy.asnumpy(dc), ref, rtol=1e-12, atol=1e-12)
