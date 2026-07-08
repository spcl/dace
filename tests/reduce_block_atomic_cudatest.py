# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Codegen coverage for ``ExpandReduceCUDABlockAtomic``.

The block-atomic reduce folds each thread's partial across the thread block with
``cub::BlockReduce`` and commits ONE atomic per block into a length-1 global
output. We assert the emitted CUDA (the cub call + the thread-0 ``reduce_atomic``)
without a GPU; when ``nvcc`` is present we also compile the generated TU.
"""
import shutil

import numpy as np
import pytest

import dace
from dace.memlet import Memlet
from dace.transformation.interstate import GPUTransformSDFG

_HAS_NVCC = shutil.which("nvcc") is not None


def _build_block_atomic_sum_sdfg():
    """Sum a 128-vector into a length-1 output: 2 blocks x 64 threads, each block
    reduces its 64 partials and atomically adds the block sum into ``B[0]``."""
    sdfg = dace.SDFG('block_atomic_reduction')
    sdfg.add_array('A', (128, ), dace.float32)
    sdfg.add_array('B', (1, ), dace.float32)
    sdfg.add_transient('tA', (1, ), dace.float32)
    state = sdfg.add_state('a')

    A = state.add_access('A')
    B = state.add_access('B')
    me, mx = state.add_map('grid', dict(bi='0:2'))
    mei, mxi = state.add_map('block', dict(i='0:64'))
    red = state.add_reduce('lambda a, b: a + b', None, 0)
    red.implementation = 'CUDA (block atomic)'
    tA = state.add_access('tA')

    state.add_edge(A, None, me, None, Memlet.simple(A, '0:128'))
    state.add_edge(me, None, mei, None, Memlet.simple(A, '(64*bi):(64*bi+64)'))
    state.add_edge(mei, None, tA, None, Memlet.simple('A', '(64*bi+i)'))
    state.add_edge(tA, None, red, None, Memlet.simple(tA, '0'))
    # The atomic lives inside the expansion; the block sum drains to the single
    # global element B[0] (dynamic: exactly one atomic per block, from thread 0).
    e_out = state.add_edge(red, None, mxi, None, Memlet.simple('B', '0', num_accesses=-1))
    state.add_edge(mxi, None, mx, None, Memlet.simple('B', '0', num_accesses=-1))
    state.add_edge(mx, None, B, None, Memlet.simple(B, '0'))
    sdfg.fill_scope_connectors()

    sdfg.apply_transformations(GPUTransformSDFG, options={'sequential_innermaps': False})
    return sdfg


def _generated_cuda(sdfg):
    code_objects = sdfg.generate_code()
    return "\n".join(co.code for co in code_objects if co.language in ("cu", "cuda")) or \
        "\n".join(co.code for co in code_objects)


def test_block_atomic_emits_cub_and_atomic():
    sdfg = _build_block_atomic_sum_sdfg()
    code = _generated_cuda(sdfg)
    assert "cub::BlockReduce<float, 64>" in code, "block reduce not typed to the 64-thread block"
    assert ".Reduce(" in code, "cub block Reduce call missing"
    assert "reduce_atomic" in code, "thread-0 atomic to the global output missing"
    assert "threadIdx.x == 0" in code, "atomic not guarded to a single thread per block"
    assert "__shared__" in code, "block-reduce temp storage not in shared memory"


@pytest.mark.skipif(not _HAS_NVCC, reason="nvcc not available; compile check skipped")
def test_block_atomic_compiles():
    sdfg = _build_block_atomic_sum_sdfg()
    sdfg.compile()


@pytest.mark.gpu
def test_block_atomic_runs():
    sdfg = _build_block_atomic_sum_sdfg()
    A = np.random.rand(128).astype(np.float32)
    B = np.zeros(1, dtype=np.float32)
    sdfg(A=A, B=B)
    assert abs(B[0] - np.sum(A)) / 128.0 <= 1e-4


if __name__ == '__main__':
    test_block_atomic_emits_cub_and_atomic()
    print('codegen ok')
