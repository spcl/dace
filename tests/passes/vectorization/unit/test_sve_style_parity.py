# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""SVE-style parity: both ``sve_style='fixed'`` (AVX-backend / x86 with
fixed-W ``_iter_mask``) and ``sve_style='variable'`` (SVE-backend
runtime VL via ``svwhilelt_b64`` / ``svcntd``, scalar fallback on x86)
produce the correct result on the same kernels.

Per the user's forward directive: once both styles are ready, add
end-to-end tests parametrized over both. ``fixed`` is fully exercised
in compile-and-execute mode on this x86 host; ``variable`` exercises
the x86 scalar-fallback branch (the SVE intrinsics path is syntax-
checked but not run — no SVE hardware available locally).

Kernels covered: axpy (``c[i] = a[i] + b[i]``), copy (``c[i] = a[i]``),
triad (``d[i] = a[i] + b[i] + c[i]`` — exercises the variable chain
recogniser's multi-step path). Bit-exact for single-op; atol/rtol
1e-12 for triad (legit FP associativity-reorder under chained adds).
"""
import numpy as np
import pytest

import dace
from dace.transformation.passes.vectorization.vectorize_cpu import VectorizeCPU

N = dace.symbol("N")


@dace.program
def _axpy(a: dace.float64[N], b: dace.float64[N], c: dace.float64[N]):
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


# Both styles share the (num_cores, vector_width) knob shape for these
# kernels. ``fixed`` requires num_cores>1; ``variable`` accepts any.
@pytest.fixture(params=["fixed", "variable"])
def sve_style(request) -> str:
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
    sdfg = _vectorize(_axpy, NV, sve_style)
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


def test_axpy_two_styles_produce_same_result():
    """Direct cross-style check: ``fixed`` and ``variable`` on the
    same axpy input produce numerically equivalent output. Useful
    correctness gate that they don't drift relative to each other."""
    NV = 64
    a, b = np.random.rand(NV), np.random.rand(NV)
    fixed_sdfg = _vectorize(_axpy, NV, "fixed")
    var_sdfg = _vectorize(_axpy, NV, "variable")
    c_fixed = np.zeros(NV)
    c_var = np.zeros(NV)
    fixed_sdfg.compile()(a=a.copy(), b=b.copy(), c=c_fixed, N=NV)
    var_sdfg.compile()(a=a.copy(), b=b.copy(), c=c_var, N=NV)
    assert np.allclose(c_fixed, c_var, rtol=0, atol=0), \
        f"styles disagree on axpy: max|fixed-var|={float(np.max(np.abs(c_fixed - c_var)))}"
