# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""GPU memory-pool (``cudaMallocAsync`` / ``cudaFreeAsync``) test for the experimental codegen."""
import glob
import os

import numpy as np
import pytest

import dace as dc
from dace import dtypes

N = dc.symbol('_MP_N', dtype=dc.int64)


@dc.program
def _pooled_kernel(A: dc.float64[N], B: dc.float64[N]):
    tmp = dc.define_local([N], dtype=dc.float64)
    for i in dc.map[0:N]:
        tmp[i] = A[i] * 2.0
    for i in dc.map[0:N]:
        B[i] = tmp[i] + 1.0


def _build_pooled_sdfg():
    sdfg = _pooled_kernel.to_sdfg(simplify=True)
    sdfg.apply_gpu_transformations()
    pooled = []
    for name, desc in sdfg.arrays.items():
        if desc.transient and desc.storage == dtypes.StorageType.GPU_Global:
            desc.pool = True
            pooled.append(name)
    assert pooled, "Expected at least one pooled GPU_Global transient after GPU transforms."
    return sdfg, pooled


@pytest.mark.gpu
@pytest.mark.new_gpu_codegen_only
def test_mempool_runs_correctly_and_emits_expected_calls():
    sdfg, pooled = _build_pooled_sdfg()
    compiled = sdfg.compile()

    n = 256
    A = np.arange(n, dtype=np.float64)
    B = np.zeros(n, dtype=np.float64)
    compiled(A=A, B=B, _MP_N=n)
    np.testing.assert_allclose(B, A * 2.0 + 1.0)

    # Async alloc/free calls are emitted on the host side; scan every emitted source.
    build = sdfg.build_folder
    sources = (glob.glob(os.path.join(build, 'src', '**', '*.cu'), recursive=True) +
               glob.glob(os.path.join(build, 'src', '**', '*.cpp'), recursive=True))
    assert sources, f"No generated sources found under {build}"
    src = '\n'.join(open(s).read() for s in sources)

    assert src.count('cudaDeviceGetDefaultMemPool') >= 1, "Pool header missing (DeviceGetDefaultMemPool)."
    assert src.count('cudaMemPoolSetAttribute') >= 1, "Pool header missing (MemPoolSetAttribute)."

    malloc_async = src.count('cudaMallocAsync')
    free_async = src.count('cudaFreeAsync')
    assert malloc_async >= len(pooled), (f"Expected >= {len(pooled)} cudaMallocAsync calls "
                                         f"(one per pooled array), got {malloc_async}.")
    assert free_async >= len(pooled), (f"Expected >= {len(pooled)} cudaFreeAsync calls "
                                       f"(one per pooled array), got {free_async}.")
