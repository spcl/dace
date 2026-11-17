# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""A complex reduction inside a GPU kernel compiles: the warp shuffle splits a complex value into its parts.

The vendor lane-xor shuffles take arithmetic types only, so ``dace::warpReduce<Sum, complex128>`` did not compile
on HIP (fft_3d's checksum ``np.sum`` over complex values).
"""

import numpy as np
import pytest

import dace
from dace.transformation.passes.canonicalize.finalize import finalize_for_target, offload_to_gpu
from dace.transformation.passes.canonicalize.pipeline import canonicalize

N = dace.symbol("N", dace.int64)


@pytest.mark.gpu
def test_a_complex_sum_reduced_on_the_gpu_compiles_and_matches_numpy():
    @dace.program
    def gpu_complex_sum(a: dace.complex128[N], out: dace.complex128[1]):
        out[0] = np.sum(a)

    sdfg = canonicalize(gpu_complex_sum.to_sdfg(simplify=False), target="gpu")
    offload_to_gpu(sdfg)
    finalize_for_target(sdfg, "gpu")
    import cupy  # Only present on GPU runners.

    rng = np.random.default_rng(7)
    a = rng.random(1024) + 1j * rng.random(1024)
    out = cupy.zeros(1, dtype=cupy.complex128)
    sdfg(a=cupy.asarray(a), out=out, N=1024)
    assert np.allclose(cupy.asnumpy(out)[0], a.sum(), rtol=1e-12)


if __name__ == "__main__":
    test_a_complex_sum_reduced_on_the_gpu_compiles_and_matches_numpy()
