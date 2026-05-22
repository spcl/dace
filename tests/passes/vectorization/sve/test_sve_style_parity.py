# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""SVE-style end-to-end coverage for ``sve_style='fixed'`` on axpy /
copy / triad kernels.

Design pivot 2026-05-20: ``sve_style='variable'`` (runtime VL via
``svwhilelt`` / ``svcntd``, one CPP tasklet per map) is *deferred* —
for SVE hardware use ``sve_style='fixed'`` with ``vector_width``
matched to the target SVE register width (W=8 for SVE-512, W=4 for
SVE-256, etc.), since ``cpu_vectorizable_math_arm_sve.h`` already
emits svwhilelt + svcntd per W-chunk internally. The cross-style
parity test previously here was removed when ``variable`` became
NotImpl; the per-kernel correctness gates below are retained for the
``fixed`` path.
"""
import numpy as np
import pytest

import dace
from dace.transformation.passes.vectorization.vectorize_cpu import VectorizeCPU

N = dace.symbol("N")


@dace.program
def axpy(a: dace.float64[N], b: dace.float64[N], c: dace.float64[N]):
    for i in dace.map[0:N]:
        c[i] = a[i] + b[i]


@dace.program
def _copy(a: dace.float64[N], c: dace.float64[N]):
    for i in dace.map[0:N]:
        c[i] = a[i]


@dace.program
def _triad(a: dace.float64[N], b: dace.float64[N], c: dace.float64[N], d: dace.float64[N]):
    for i in dace.map[0:N]:
        d[i] = a[i] + b[i] + c[i]


@pytest.fixture(params=["fixed"])
def sve_style(request) -> str:
    """Only ``'fixed'`` is exercised; ``'variable'`` is deferred (see
    module docstring)."""
    return request.param


def _vectorize(prog, NV, style):
    sdfg = prog.to_sdfg(simplify=True)
    sdfg.replace_dict({"N": NV})
    sdfg.name = f"{prog.name}_{style}_{NV}"
    VectorizeCPU(vector_width=8, num_cores=8, sve_style=style, fail_on_unvectorizable=True).apply_pass(sdfg, {})
    sdfg.validate()
    return sdfg


def test_axpy_parity_across_sve_styles(sve_style):
    NV = 64
    a, b = np.random.rand(NV), np.random.rand(NV)
    sdfg = _vectorize(axpy, NV, sve_style)
    c = np.zeros(NV)
    sdfg.compile()(a=a.copy(), b=b.copy(), c=c, N=NV)
    # Single mul-free add: bit-exact across both styles.
    assert np.allclose(c, a + b, rtol=0, atol=0), \
        f"{sve_style}: max|d|={float(np.max(np.abs(c - (a + b))))}"


def test_copy_parity_across_sve_styles(sve_style):
    NV = 64
    a = np.random.rand(NV)
    sdfg = _vectorize(_copy, NV, sve_style)
    c = np.zeros(NV)
    sdfg.compile()(a=a.copy(), c=c, N=NV)
    assert np.allclose(c, a, rtol=0, atol=0), f"{sve_style}: max|d|={float(np.max(np.abs(c - a)))}"


def test_triad_parity_across_sve_styles(sve_style):
    """Triad ``a + b + c`` — chained adds; atol/rtol=1e-12 accepts
    legit FP associativity reorder between the two emission paths."""
    NV = 64
    a, b, c = np.random.rand(NV), np.random.rand(NV), np.random.rand(NV)
    sdfg = _vectorize(_triad, NV, sve_style)
    d = np.zeros(NV)
    sdfg.compile()(a=a.copy(), b=b.copy(), c=c.copy(), d=d, N=NV)
    expected = a + b + c
    assert np.allclose(d, expected, rtol=1e-12, atol=1e-12), \
        f"{sve_style}: max|d|={float(np.max(np.abs(d - expected)))}"


