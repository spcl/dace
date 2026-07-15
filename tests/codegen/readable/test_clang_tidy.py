# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""
clang-tidy safety for the experimental "readable" code generator.

The experimental generator runs ``clang-tidy -fix-errors`` in place on every
generated ``.cpp`` / ``.cu`` file (see :func:`dace.codegen.compiler.apply_clang_tidy`).
Most readability / modernize fixes are safe, but ``readability-non-const-parameter``
is not: it rewrites a pointer parameter to ``const T*`` when it cannot prove the
pointee is written. A scatter accumulator that is only *forwarded* to a nested-SDFG
device function (a ``DACE_DFI`` that writes through it -- e.g. ``histu`` / ``histw``
in npbench ``azimint_hist``) looks locally unmodified, so clang-tidy would add a
``const`` that clashes with the callee's non-const parameter and nvcc rejects the
``const T*`` -> ``T*`` argument. The check is therefore excluded.

These tests guard that exclusion: a deterministic assertion on the configured
check list, and an end-to-end GPU compile of exactly that scatter pattern.
"""
import numpy as np
import pytest

import dace
from dace.codegen.compiler import CLANG_TIDY_CHECKS
from tests.codegen.readable.conftest import EXPERIMENTAL, gpu_available, use_implementation


def test_clang_tidy_excludes_non_const_parameter():
    """The unsound ``readability-non-const-parameter`` fix must stay disabled: it
    silently const-qualifies written pointer parameters forwarded to nested-SDFG
    functions, producing a const/non-const mismatch that fails to compile."""
    assert '-readability-non-const-parameter' in CLANG_TIDY_CHECKS, (
        'readability-non-const-parameter must be excluded from the readable code '
        'generator clang-tidy pass (it miscompiles forwarded scatter accumulators)')
    # The pass is still meaningfully enabled (broad readability/modernize globs).
    assert 'readability-*' in CLANG_TIDY_CHECKS and 'modernize-*' in CLANG_TIDY_CHECKS


# --- N = number of samples, B = number of histogram bins ---
N, B = 64, 8


@dace.program
def scatter_histogram(data: dace.float64[N], bin_edges: dace.float64[B + 1], hist: dace.int64[B]):
    """A histogram scatter: each sample searches the bin edges and accumulates into
    ``hist``. On the GPU the per-sample bin search + accumulate lowers to a nested
    SDFG that writes ``hist`` through a forwarded pointer -- the exact pattern that
    trips ``readability-non-const-parameter`` (see npbench ``azimint_hist``)."""
    for i in dace.map[0:N]:
        for b in range(B):
            if bin_edges[b] <= data[i] and data[i] < bin_edges[b + 1]:
                hist[b] += 1


def reference(data, bin_edges):
    return np.histogram(data, bins=bin_edges)[0].astype(np.int64)


@pytest.mark.gpu
def test_gpu_scatter_accumulator_compiles(require_experimental, require_gpu):
    """The scatter accumulator compiles and runs correctly under the experimental
    GPU pipeline (which tidies the ``.cu`` in place). Without the
    ``readability-non-const-parameter`` exclusion this fails to compile with a
    ``const int64_t*`` / ``int64_t*`` argument mismatch."""
    rng = np.random.default_rng(0)
    data = np.sort(rng.random(N))
    bin_edges = np.linspace(0.0, 1.0, B + 1)
    ref = reference(data, bin_edges)

    def build_and_run(implementation):
        with use_implementation(implementation):
            sdfg = scatter_histogram.to_sdfg(simplify=True)
            sdfg.apply_gpu_transformations()
            sdfg.name = f'scatter_hist_{implementation}'
            hist = np.zeros(B, dtype=np.int64)
            sdfg(data=data.copy(), bin_edges=bin_edges.copy(), hist=hist)
            return hist

    experimental = build_and_run(EXPERIMENTAL)  # must not raise CompilationError
    assert np.array_equal(experimental, ref), (experimental, ref)


if __name__ == '__main__':
    test_clang_tidy_excludes_non_const_parameter()
    from tests.codegen.readable.conftest import experimental_available
    if experimental_available() and gpu_available():
        test_gpu_scatter_accumulator_compiles(None, None)
    print('ok')
