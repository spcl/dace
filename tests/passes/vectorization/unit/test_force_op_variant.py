# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""
Unit tests for ``VectorizeCPU(force_autovec_ops=..., force_pscalar_ops=...)``.

These knobs rewrite the templates-dict entry for the listed ops so the
emitter calls ``vector_<op>_av`` or ``vector_<op>_pscalar`` (Option F
overlay siblings from ``cpu_vectorizable_math_common.h``) instead of the
default unsuffixed name.

What this exercises:
- Override rewrite produces the right C++ function name in the templates
  dict.
- Validation: overlapping sets raise ``ValueError``; unknown op keys
  raise ``KeyError``.
- End-to-end numerical equivalence: the override changes the emitted
  function call but preserves the output (the ``_av`` / ``_pscalar``
  siblings are semantically identical to the unsuffixed version).
"""
import copy

import dace
import numpy as np
import pytest

from dace.transformation.passes.vectorization.vectorize_cpu import VectorizeCPU

N = dace.symbol("N")


@dace.program
def add_div_program(a: dace.float64[N], b: dace.float64[N], c: dace.float64[N], out: dace.float64[N]):
    for i, in dace.map[0:N:1]:
        out[i] = (a[i] + b[i]) / c[i]


def _templates_after_override(force_autovec=None, force_pscalar=None) -> dict:
    """Build a VectorizeCPU and pull the rewritten templates dict out."""
    v = VectorizeCPU(vector_width=8, force_autovec_ops=force_autovec, force_pscalar_ops=force_pscalar)
    for p in v._passes:
        if hasattr(p, "templates") and isinstance(p.templates, dict):
            return p.templates
    raise AssertionError("Vectorize pass not found in pipeline")


def test_force_autovec_rewrites_one_op():
    t = _templates_after_override(force_autovec={"/"})
    assert t["/"].startswith("vector_div_av<")
    # Untouched ops stay unsuffixed.
    assert t["+"].startswith("vector_add<")
    assert t["*"].startswith("vector_mult<")


def test_force_pscalar_rewrites_one_op():
    t = _templates_after_override(force_pscalar={"exp"})
    assert t["exp"].startswith("vector_exp_pscalar<")
    assert t["log"].startswith("vector_log<")


def test_force_both_sets_compose_disjointly():
    t = _templates_after_override(force_autovec={"/"}, force_pscalar={"+"})
    assert t["+"].startswith("vector_add_pscalar<")
    assert t["/"].startswith("vector_div_av<")
    # No cross-contamination.
    assert t["-"].startswith("vector_sub<")


def test_overlap_raises():
    with pytest.raises(ValueError, match="overlap"):
        VectorizeCPU(vector_width=8, force_autovec_ops={"+"}, force_pscalar_ops={"+"})


def test_unknown_op_raises():
    with pytest.raises(KeyError, match="bogus"):
        VectorizeCPU(vector_width=8, force_autovec_ops={"bogus"})


def test_end_to_end_pscalar_div():
    """Forcing div to pscalar mode preserves numerical output."""
    Nv = 64
    a = np.random.rand(Nv)
    b = np.random.rand(Nv)
    c = np.random.rand(Nv) + 0.1  # avoid div-by-zero
    out_ref = np.zeros(Nv)
    out_vec = np.zeros(Nv)

    sdfg = add_div_program.to_sdfg(simplify=True)
    sdfg.name = "force_pscalar_div_ref"
    vsdfg = copy.deepcopy(sdfg)
    vsdfg.name = "force_pscalar_div_v"

    VectorizeCPU(vector_width=8, fail_on_unvectorizable=True, force_pscalar_ops={"/"}).apply_pass(vsdfg, {})

    sdfg(a=a, b=b, c=c, out=out_ref, N=Nv)
    vsdfg(a=a, b=b, c=c, out=out_vec, N=Nv)
    assert np.max(np.abs(out_ref - out_vec)) < 1e-12


def test_end_to_end_av_div():
    """Forcing div to av mode preserves numerical output."""
    Nv = 64
    a = np.random.rand(Nv)
    b = np.random.rand(Nv)
    c = np.random.rand(Nv) + 0.1
    out_ref = np.zeros(Nv)
    out_vec = np.zeros(Nv)

    sdfg = add_div_program.to_sdfg(simplify=True)
    sdfg.name = "force_av_div_ref"
    vsdfg = copy.deepcopy(sdfg)
    vsdfg.name = "force_av_div_v"

    VectorizeCPU(vector_width=8, fail_on_unvectorizable=True, force_autovec_ops={"/"}).apply_pass(vsdfg, {})

    sdfg(a=a, b=b, c=c, out=out_ref, N=Nv)
    vsdfg(a=a, b=b, c=c, out=out_vec, N=Nv)
    assert np.max(np.abs(out_ref - out_vec)) < 1e-12
