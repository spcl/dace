# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""A GPU kernel's scatter WCR into a small array is privatized per block in shared memory: threads fold with
shared-memory atomics and each block commits every element with one global atomic, instead of one global atomic
per thread. The emitted HIP/CUDA is what is under test; the GPU run checks the values."""

import numpy as np
import pytest

import dace
from dace.codegen.codegen import generate_code

N = dace.symbol("N")
K = 256


@dace.program
def histogram(idx: dace.int64[N], w: dace.float64[N], h: dace.float64[K]):
    for i in dace.map[0:N]:
        h[idx[i]] += w[i]


@dace.program
def column_sums(a: dace.float64[N, 64], s: dace.float64[64]):
    for i, j in dace.map[0:N, 0:64]:
        s[j] += a[i, j]


def _gpu_sdfg(prog):
    sdfg = prog.to_sdfg(simplify=True)
    sdfg.apply_gpu_transformations()
    return sdfg


def _device_code(sdfg) -> str:
    with dace.config.set_temporary("compiler", "cuda", "implementation", value="experimental"):
        return "\n".join(
            c.clean_code for c in generate_code(sdfg) if c.target.target_name in ("cuda", "experimental_cuda")
        )


@pytest.mark.parametrize("prog", [histogram, column_sums], ids=["scatter", "column"])
def test_a_small_scatter_target_is_privatized_in_shared_memory(prog):
    code = _device_code(_gpu_sdfg(prog))
    assert "__shared__ double __bpart_" in code, "the block's copy of the target is not in shared memory"
    assert "&__bpart_" in code, "threads must fold into the shared copy"
    assert "__syncthreads();" in code


@pytest.mark.gpu
@pytest.mark.parametrize("prog", [histogram, column_sums], ids=["scatter", "column"])
def test_a_privatized_scatter_matches_numpy(prog):
    with dace.config.set_temporary("compiler", "cuda", "implementation", value="experimental"):
        sdfg = _gpu_sdfg(prog)
        rng = np.random.default_rng(0)
        n = 100003
        if prog is histogram:
            idx, w, h = rng.integers(0, K, n), rng.random(n), np.zeros(K)
            sdfg(idx=idx, w=w, h=h, N=n)
            assert np.allclose(h, np.bincount(idx, w, K))
        else:
            a, s = rng.random((n, 64)), np.zeros(64)
            sdfg(a=a, s=s, N=n)
            assert np.allclose(s, a.sum(0))


if __name__ == "__main__":
    for p in (histogram, column_sums):
        test_a_small_scatter_target_is_privatized_in_shared_memory(p)
