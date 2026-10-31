# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Transient placement in ``finalize_for_target``: device transients come from the stream-ordered pool."""

import numpy as np
import pytest

import dace
from dace.transformation.passes.canonicalize.finalize import finalize_for_target, offload_to_gpu
from dace.transformation.passes.canonicalize.pipeline import canonicalize

N = dace.symbol("N", dace.int64)


@pytest.mark.gpu
def test_gpu_transients_are_allocated_from_the_memory_pool():
    @dace.program
    def finalize_pooled_gemms(A: dace.float64[N, N], B: dace.float64[N, N]):
        T = A @ A
        B[:] = T @ A + 1.0

    sdfg = canonicalize(finalize_pooled_gemms.to_sdfg(simplify=False), target="gpu")
    offload_to_gpu(sdfg)
    finalize_for_target(sdfg, "gpu")
    device = [
        desc
        for sd in sdfg.all_sdfgs_recursive()
        for desc in sd.arrays.values()
        if desc.transient and desc.storage == dace.StorageType.GPU_Global and type(desc) is dace.data.Array
    ]
    assert device and all(desc.pool for desc in device)
    assert "MallocAsync" in "".join(obj.clean_code for obj in sdfg.generate_code())

    import cupy  # Only present on GPU runners.

    A = np.random.default_rng(1).random((16, 16))
    B = cupy.zeros((16, 16))
    sdfg(A=cupy.asarray(A), B=B, N=16)
    assert np.allclose(cupy.asnumpy(B), A @ A @ A + 1.0)


if __name__ == "__main__":
    test_gpu_transients_are_allocated_from_the_memory_pool()
