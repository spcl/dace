# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""A scatter accumulator only *forwarded* to a nested-SDFG device function (npbench ``azimint_hist``'s
``histu`` / ``histw``) compiles and runs under the experimental GPU generator: the pointer stays non-const
across the call.
"""

import numpy as np
import pytest

import dace
from tests.codegen.readable.conftest import EXPERIMENTAL, use_implementation

# N = number of samples, B = number of histogram bins
N, B = 64, 8


@dace.program
def scatter_histogram(data: dace.float64[N], bin_edges: dace.float64[B + 1], hist: dace.int64[B]):
    """On the GPU the per-sample bin search + accumulate lowers to a nested SDFG that writes ``hist``
    through a forwarded pointer."""
    for i in dace.map[0:N]:
        for b in range(B):
            if bin_edges[b] <= data[i] and data[i] < bin_edges[b + 1]:
                hist[b] += 1


def reference(data, bin_edges):
    return np.histogram(data, bins=bin_edges)[0].astype(np.int64)


@pytest.mark.gpu
def test_gpu_scatter_accumulator_compiles(require_experimental):
    """A const-qualified forwarded pointer fails to compile with a ``const int64_t*`` / ``int64_t*`` mismatch."""
    rng = np.random.default_rng(0)
    data = np.sort(rng.random(N))
    bin_edges = np.linspace(0.0, 1.0, B + 1)
    ref = reference(data, bin_edges)

    def build_and_run(implementation):
        with use_implementation(implementation):
            sdfg = scatter_histogram.to_sdfg(simplify=True)
            sdfg.apply_gpu_transformations()
            sdfg.name = f"scatter_hist_{implementation}"
            hist = np.zeros(B, dtype=np.int64)
            sdfg(data=data.copy(), bin_edges=bin_edges.copy(), hist=hist)
            return hist

    experimental = build_and_run(EXPERIMENTAL)  # must not raise CompilationError
    assert np.array_equal(experimental, ref), (experimental, ref)


if __name__ == "__main__":
    from tests.codegen.readable.conftest import experimental_available

    if experimental_available():
        test_gpu_scatter_accumulator_compiles(None)
    print("ok")
