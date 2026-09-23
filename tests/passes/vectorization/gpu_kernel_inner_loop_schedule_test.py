# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""A loop inside a kernel that the GPU vectorizer turns into a map stays a loop of one thread.

The vectorizer's ``ParallelizeLoops`` makes a map of a ``for`` loop in a ``GPU_Device`` map's body,
and inferring its ``Default`` schedule there gave ``GPU_ThreadBlock``: the kernel then carried
thread-block maps sized by the loop next to the ``gpu_block_size`` the offload had chosen, which the
GPU code generator refuses. The fp64 x 2 vectorized CloudSC failed exactly that way.
"""
import numpy as np
import pytest

import dace
from dace.dtypes import DeviceType
from dace.transformation.passes.vectorization.config import VectorizeConfig
from dace.transformation.passes.vectorization.vectorize_multi_dim import VectorizeMultiDim

N, M = dace.symbol('N'), dace.symbol('M')


@dace.program
def scale_rows(a: dace.float64[N, M], b: dace.float64[N, M]):
    for i in dace.map[0:N]:
        for j in range(M):
            b[i, j] = 2.0 * a[i, j]


def vectorized_kernel():
    """``scale_rows`` offloaded as one kernel with a set block size, then vectorized fp64 x 2."""
    sdfg = scale_rows.to_sdfg(simplify=True)
    for name in ('a', 'b'):
        sdfg.arrays[name].storage = dace.StorageType.GPU_Global
    kernel = next(n for n, _ in sdfg.all_nodes_recursive() if isinstance(n, dace.nodes.MapEntry))
    kernel.map.schedule = dace.ScheduleType.GPU_Device
    kernel.map.gpu_block_size = [256, 1, 1]
    VectorizeMultiDim(VectorizeConfig(widths=(2, ), target_isa='SCALAR', device=DeviceType.GPU)).apply_pass(sdfg, {})
    return sdfg, kernel


def test_a_loop_the_vectorizer_maps_inside_a_kernel_is_sequential():
    sdfg, kernel = vectorized_kernel()
    inner = [n for n, _ in sdfg.all_nodes_recursive() if isinstance(n, dace.nodes.MapEntry) and n is not kernel]
    assert inner, 'the inner loop was not turned into a map'
    assert all(n.map.schedule == dace.ScheduleType.Sequential for n in inner), [(n.map.label, n.map.schedule)
                                                                                for n in inner]
    sdfg.generate_code()


@pytest.mark.gpu
def test_a_vectorized_kernel_with_an_inner_loop_computes_the_values():
    import cupy
    sdfg, _ = vectorized_kernel()
    a = np.random.default_rng(0).random((64, 37))
    b = cupy.zeros((64, 37))
    sdfg(a=cupy.asarray(a), b=b, N=64, M=37)
    assert np.allclose(b.get(), 2.0 * a)
