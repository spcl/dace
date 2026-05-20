# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""End-to-end integration of ``sve_style='variable'`` through
``VectorizeCPU`` for simple element-wise float64 1D maps.

The ``SveStyleVariableFinalize`` pass (wired into ``VectorizeCPU`` when
``sve_style='variable'``) recognises the dace-frontend canonical shape
``c[i] = a[i] (op) b[i]`` for ``op in {+, -, *, /}`` (plus the
``c[i] = a[i]`` copy) and replaces the entire map with a single CPP
tasklet whose body is the SVE runtime-VL while-loop pattern
(``svwhilelt_b64`` + ``svcntd`` + ``svld1_f64`` + ``sv{add,sub,mul,
div}_f64_z`` + ``svst1_f64``), guarded with ``#if defined(
__ARM_FEATURE_SVE)`` + a scalar fallback. The fallback path runs on
this x86 host (validates numeric correctness end-to-end); the SVE
branch is taken on SVE hardware (syntax-checked via .cpp inspection).

This is the production integration the D2 probe in
``test_sve_variable_probe.py`` proved the emission *pattern* for —
now reached via the user-facing ``VectorizeCPU(sve_style='variable')``
knob rather than hand-built SDFGs.
"""
import numpy as np
import pytest

import dace
from dace.transformation.passes.vectorization.vectorize_cpu import VectorizeCPU

N = dace.symbol("N")


@dace.program
def _axpy_add(a: dace.float64[N], b: dace.float64[N], c: dace.float64[N]):
    for i in dace.map[0:N]:
        c[i] = a[i] + b[i]


@dace.program
def _axpy_sub(a: dace.float64[N], b: dace.float64[N], c: dace.float64[N]):
    for i in dace.map[0:N]:
        c[i] = a[i] - b[i]


@dace.program
def _axpy_mul(a: dace.float64[N], b: dace.float64[N], c: dace.float64[N]):
    for i in dace.map[0:N]:
        c[i] = a[i] * b[i]


@dace.program
def _axpy_div(a: dace.float64[N], b: dace.float64[N], c: dace.float64[N]):
    for i in dace.map[0:N]:
        c[i] = a[i] / b[i]


@dace.program
def _copy_kernel(a: dace.float64[N], c: dace.float64[N]):
    for i in dace.map[0:N]:
        c[i] = a[i]


def _compile_and_run(prog, NV, mk_inputs, output_name):
    sdfg = prog.to_sdfg(simplify=True)
    sdfg.replace_dict({"N": NV})
    sdfg.name = f"{prog.name}_var_{NV}"
    VectorizeCPU(vector_width=8, num_cores=8, sve_style="variable", fail_on_unvectorizable=True).apply_pass(sdfg, {})
    sdfg.validate()
    inputs = mk_inputs()
    sdfg.compile()(**inputs, N=NV)
    return sdfg, inputs[output_name]


@pytest.mark.parametrize("op,expected_intrinsic", [
    ("+", "svadd_f64_z"),
    ("-", "svsub_f64_z"),
    ("*", "svmul_f64_z"),
    ("/", "svdiv_f64_z"),
])
def test_sve_variable_emits_correct_intrinsic_per_op(op, expected_intrinsic):
    """The generated .cpp contains the SVE intrinsic matching the
    recognised op. Plus the common svwhilelt_b64 + svcntd + svld1/svst1
    surrounding."""
    prog = {"+": _axpy_add, "-": _axpy_sub, "*": _axpy_mul, "/": _axpy_div}[op]
    sdfg = prog.to_sdfg(simplify=True)
    sdfg.replace_dict({"N": 64})
    sdfg.name = f"emit_{prog.name}"
    VectorizeCPU(vector_width=8, num_cores=8, sve_style="variable", fail_on_unvectorizable=True).apply_pass(sdfg, {})
    cpp = sdfg.generate_code()[0].clean_code
    for intrinsic in ("svwhilelt_b64", "svcntd()", "svld1_f64", expected_intrinsic, "svst1_f64"):
        assert intrinsic in cpp, f"op={op!r}: .cpp missing {intrinsic!r}"
    # The fallback for non-SVE hosts is what executes on this x86 box.
    assert "__ARM_FEATURE_SVE" in cpp
    assert "#else" in cpp


@pytest.mark.parametrize("op,np_op", [
    ("+", lambda a, b: a + b),
    ("-", lambda a, b: a - b),
    ("*", lambda a, b: a * b),
    ("/", lambda a, b: a / b),
])
def test_sve_variable_axpy_runs_correctly_via_scalar_fallback(op, np_op):
    """End-to-end on x86 via the scalar fallback: each op produces
    bit-exact ``a (op) b``. Bit-exact because each op is a single
    elementwise operation (no reordering)."""
    prog = {"+": _axpy_add, "-": _axpy_sub, "*": _axpy_mul, "/": _axpy_div}[op]
    NV = 64
    a = np.random.rand(NV) + 0.5  # avoid div-by-zero
    b = np.random.rand(NV) + 0.5
    _, c = _compile_and_run(prog, NV, lambda: {"a": a.copy(), "b": b.copy(), "c": np.zeros(NV)}, "c")
    expected = np_op(a, b)
    assert np.allclose(c, expected, rtol=0, atol=0), \
        f"op={op!r} max|d|={float(np.max(np.abs(c - expected)))}"


def test_sve_variable_copy_runs_correctly():
    """Copy-only kernel ``c[i] = a[i]`` also recognised and handled."""
    NV = 64
    a = np.random.rand(NV)
    _, c = _compile_and_run(_copy_kernel, NV, lambda: {"a": a.copy(), "c": np.zeros(NV)}, "c")
    assert np.allclose(c, a, rtol=0, atol=0), f"copy max|d|={float(np.max(np.abs(c - a)))}"


@dace.program
def _triad(a: dace.float64[N], b: dace.float64[N], c: dace.float64[N], d: dace.float64[N]):
    for i in dace.map[0:N]:
        d[i] = a[i] + b[i] + c[i]


def test_sve_variable_triad_chain():
    """Triad ``d[i] = a[i] + b[i] + c[i]`` is a linear chain of two
    single-binop tasklets. The chain recogniser walks it and emits a
    nested SVE intrinsic sequence (svadd then svadd-with-acc); FP
    associativity-reordering allowed (atol=rtol=1e-12)."""
    NV = 64
    sdfg, d = _compile_and_run(_triad,
                               NV,
                               lambda: {
                                   "a": np.random.rand(NV),
                                   "b": np.random.rand(NV),
                                   "c": np.random.rand(NV),
                                   "d": np.zeros(NV),
                               },
                               output_name="d")
    cpp = sdfg.generate_code()[0].clean_code
    # The chain emits svadd_f64_z called twice: once with two outer
    # operands, then once with the running ``acc``.
    assert cpp.count("svadd_f64_z") >= 2, f"expected >=2 svadd_f64_z calls; got {cpp.count('svadd_f64_z')}"
    # Numeric re-check with fresh inputs (closed-form ref) — atol/rtol
    # 1e-12 accepts the legit FP associativity reorder a chain produces.
    a = np.random.rand(NV)
    b = np.random.rand(NV)
    c = np.random.rand(NV)
    sd2 = _triad.to_sdfg(simplify=True)
    sd2.replace_dict({"N": NV})
    sd2.name = f"_triad_ref_{NV}"
    VectorizeCPU(vector_width=8, num_cores=8, sve_style="variable", fail_on_unvectorizable=True).apply_pass(sd2, {})
    d_out = np.zeros(NV)
    sd2.compile()(a=a.copy(), b=b.copy(), c=c.copy(), d=d_out, N=NV)
    expected = a + b + c
    assert np.allclose(d_out, expected, rtol=1e-12, atol=1e-12), \
        f"triad max|d|={float(np.max(np.abs(d_out - expected)))}"


@dace.program
def _spmv_dot(a: dace.float64[N], b: dace.float64[N], idx: dace.int64[N], y: dace.float64[1]):
    y[0] = 0.0
    for i in dace.map[0:N]:
        y[0] += a[i] * b[idx[i]]


def test_sve_variable_spmv_reduction_recognised_and_runs():
    """SpMV ``y = sum a[i] * b[idx[i]]``: WCRToAugAssign (run at the
    start of the variable-VL chain per user directive) converts ``+=``
    into an explicit augassign tasklet inside the dace-frontend's
    reduction NSDFG. ``_classify_spmv_reduction_body`` then recognises
    the multiply + augassign-over-gather shape and emits the full SVE
    reduction stack: svwhilelt_b64 + svld1_f64 + native gather
    svld1_gather_s64index_f64 + svmla_f64_m FMA accumulator +
    svaddv_f64 horizontal reduce + svcntd VL stride. Scalar fallback
    runs on x86 (the SVE branch is syntax-checked via .cpp text)."""
    NV = 64
    a = np.random.rand(NV)
    b = np.random.rand(NV)
    idx = np.random.permutation(NV).astype(np.int64)
    y = np.zeros(1)
    sdfg = _spmv_dot.to_sdfg(simplify=True)
    sdfg.replace_dict({"N": NV})
    sdfg.name = f"_spmv_dot_var_{NV}"
    VectorizeCPU(vector_width=8, num_cores=8, sve_style="variable", fail_on_unvectorizable=True).apply_pass(sdfg, {})
    sdfg.validate()
    cpp = sdfg.generate_code()[0].clean_code
    for intrinsic in ("svwhilelt_b64", "svld1_gather_s64index_f64", "svmla_f64_m", "svaddv_f64", "svcntd()"):
        assert intrinsic in cpp, f"emitted .cpp missing SVE intrinsic {intrinsic!r}"
    sdfg.compile()(a=a.copy(), b=b.copy(), idx=idx.copy(), y=y, N=NV)
    expected = float(np.sum(a * b[idx]))
    assert np.isclose(y[0], expected, rtol=1e-12, atol=1e-12), \
        f"SpMV mismatch: got {y[0]}, expected {expected}, diff {y[0] - expected:.2e}"


def test_sve_variable_unsupported_shape_raises_loudly():
    """A non-chain / non-SpMV body (multi-output) must raise rather
    than silently miscompile."""
    N_sym = dace.symbol("N_bad")

    @dace.program
    def _multi_output(a: dace.float64[N_sym], b: dace.float64[N_sym], c: dace.float64[N_sym], d: dace.float64[N_sym]):
        for i in dace.map[0:N_sym]:
            # Two independent outputs — not a linear chain.
            c[i] = a[i] + b[i]
            d[i] = a[i] * b[i]

    sdfg = _multi_output.to_sdfg(simplify=True)
    sdfg.replace_dict({"N_bad": 64})
    sdfg.name = "multi_output_unsupported"
    try:
        VectorizeCPU(vector_width=8, num_cores=8, sve_style="variable",
                     fail_on_unvectorizable=True).apply_pass(sdfg, {})
        assert False, "expected NotImplementedError on unrecognised body shape"
    except NotImplementedError as e:
        assert "recogniser supports" in str(e)
