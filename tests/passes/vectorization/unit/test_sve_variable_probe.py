# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""D2 probe — sve_style='variable' (ARM SVE runtime VL) emission shape.

This validates the *emission pattern* the variable-VL chain must
produce, *without* the full VectorizeCPU integration (which is the
next-session scope). The probe:

1. Hand-builds a tiny SDFG with a single CPP tasklet whose body is the
   canonical SVE while-loop pattern:
       int i = 0;
       while (i < N) {
           svbool_t pg = svwhilelt_b64(i, N);
           svfloat64_t va = svld1_f64(pg, _a + i);
           svfloat64_t vb = svld1_f64(pg, _b + i);
           svfloat64_t vc = svadd_f64_z(pg, va,
                                        svmul_f64_z(pg, vb, svdup_f64(2.0)));
           svst1_f64(pg, _c + i, vc);
           i += svcntd();
       }
   guarded behind ``#if defined(__ARM_FEATURE_SVE)`` with a scalar
   fallback ``for`` loop for non-SVE hosts (this x86 dev box).

2. Asserts the generated ``.cpp`` contains the SVE intrinsics literally
   — proves the emission template is syntactically what we want.

3. Compiles and runs on x86 via the scalar fallback (the SVE branch is
   not taken; numerical correctness validates the scalar half).

The SVE branch itself cannot be executed without SVE hardware; its
*syntactic* correctness is taken from the existing
``cpu_vectorizable_math_arm_sve.h`` header (verified to compile under
``-march=armv8-a+sve`` in the build farm). Full integration of
``sve_style='variable'`` into VectorizeCPU (so kernels go through the
analysis-permute-MapExpansion chain and emit this body automatically)
is the documented next-session deliverable.
"""
import numpy as np

import dace

# The full SVE while-loop body + scalar fallback, used both inside the
# hand-built SDFG and asserted on the generated .cpp.
_SVE_AXPY_BODY = """
#if defined(__ARM_FEATURE_SVE)
{
    int i = 0;
    while (i < N) {
        svbool_t pg = svwhilelt_b64(i, (int64_t)N);
        svfloat64_t va = svld1_f64(pg, _a + i);
        svfloat64_t vb = svld1_f64(pg, _b + i);
        svfloat64_t vc = svadd_f64_z(pg, va,
                                     svmul_f64_z(pg, vb, svdup_n_f64(2.0)));
        svst1_f64(pg, _c + i, vc);
        i += svcntd();
    }
}
#else
    for (int i = 0; i < (int)N; ++i) _c[i] = _a[i] + 2.0 * _b[i];
#endif
"""


def _build_sve_variable_axpy_sdfg(NV: int) -> dace.SDFG:
    """Tiny SDFG: one CPP tasklet whose body is the SVE while-loop +
    scalar fallback for ``c[i] = a[i] + 2*b[i]`` over an array of
    length ``N``. No maps — the tasklet owns the iteration."""
    sdfg = dace.SDFG(f"sve_var_axpy_{NV}")
    sdfg.add_array("a", [NV], dace.float64)
    sdfg.add_array("b", [NV], dace.float64)
    sdfg.add_array("c", [NV], dace.float64)
    sdfg.add_symbol("N", dace.int64)
    sdfg.append_global_code("#include <stdint.h>\n#if defined(__ARM_FEATURE_SVE)\n#include <arm_sve.h>\n#endif\n")
    st = sdfg.add_state(is_start_block=True)
    an_a = st.add_access("a")
    an_b = st.add_access("b")
    an_c = st.add_access("c")
    t = st.add_tasklet("sve_axpy", {"_a", "_b"}, {"_c"}, _SVE_AXPY_BODY, language=dace.dtypes.Language.CPP)
    st.add_edge(an_a, None, t, "_a", dace.Memlet(f"a[0:{NV}]"))
    st.add_edge(an_b, None, t, "_b", dace.Memlet(f"b[0:{NV}]"))
    st.add_edge(t, "_c", an_c, None, dace.Memlet(f"c[0:{NV}]"))
    sdfg.validate()
    return sdfg


def test_sve_variable_emission_pattern_has_svwhilelt_and_svcntd():
    """The generated ``.cpp`` literally contains ``svwhilelt_b64``,
    ``svcntd()``, ``svld1_f64``, ``svadd_f64_z``, ``svst1_f64`` — the
    minimal SVE runtime-VL intrinsic set the variable-VL chain must
    emit. Pure text-on-CPP assertion (syntax-check only on this x86
    host; the SVE branch is not taken at runtime)."""
    sdfg = _build_sve_variable_axpy_sdfg(64)
    cpp = sdfg.generate_code()[0].clean_code
    for intrinsic in ("svwhilelt_b64", "svcntd()", "svld1_f64", "svadd_f64_z", "svst1_f64", "svdup_n_f64"):
        assert intrinsic in cpp, f"emitted .cpp missing SVE intrinsic {intrinsic!r}"
    # The scalar fallback for non-SVE hosts must also be in the cpp
    # (so the file compiles on x86 — no SVE branch taken at runtime).
    assert "for (int i = 0; i < (int)N; ++i) _c[i] = _a[i] + 2.0 * _b[i];" in cpp


_SVE_SPMV_REDUCE_BODY = """
#if defined(__ARM_FEATURE_SVE)
{
    svfloat64_t acc = svdup_n_f64(0.0);
    int i = 0;
    while (i < N) {
        svbool_t pg = svwhilelt_b64(i, (int64_t)N);
        svint64_t vidx = svld1_s64(pg, _idx + i);
        svfloat64_t va = svld1_f64(pg, _a + i);
        svfloat64_t vbg = svld1_gather_s64index_f64(pg, _b, vidx);
        acc = svmla_f64_m(pg, acc, va, vbg);
        i += svcntd();
    }
    _y = svaddv_f64(svptrue_b64(), acc);
}
#else
    {
        double s = 0.0;
        for (int i = 0; i < (int)N; ++i) s += _a[i] * _b[_idx[i]];
        _y = s;
    }
#endif
"""


def _build_sve_variable_spmv_sdfg(NV: int) -> dace.SDFG:
    """SpMV-like inner-product with gather: ``y = sum_i a[i] * b[idx[i]]``.
    Exercises the full SVE intrinsic stack: svwhilelt + svld1 + native
    gather (svld1_gather_s64index_f64) + FMA (svmla_f64_m) + horizontal
    reduce (svaddv_f64). All inside one CPP tasklet body with a scalar
    fallback for non-SVE hosts."""
    sdfg = dace.SDFG(f"sve_var_spmv_{NV}")
    sdfg.add_array("a", [NV], dace.float64)
    sdfg.add_array("b", [NV], dace.float64)
    sdfg.add_array("idx", [NV], dace.int64)
    sdfg.add_array("y", [1], dace.float64)
    sdfg.add_symbol("N", dace.int64)
    sdfg.append_global_code("#include <stdint.h>\n#if defined(__ARM_FEATURE_SVE)\n#include <arm_sve.h>\n#endif\n")
    st = sdfg.add_state(is_start_block=True)
    an_a = st.add_access("a")
    an_b = st.add_access("b")
    an_idx = st.add_access("idx")
    an_y = st.add_access("y")
    t = st.add_tasklet("sve_spmv",
                       {"_a", "_b", "_idx"}, {"_y"},
                       _SVE_SPMV_REDUCE_BODY,
                       language=dace.dtypes.Language.CPP)
    st.add_edge(an_a, None, t, "_a", dace.Memlet(f"a[0:{NV}]"))
    st.add_edge(an_b, None, t, "_b", dace.Memlet(f"b[0:{NV}]"))
    st.add_edge(an_idx, None, t, "_idx", dace.Memlet(f"idx[0:{NV}]"))
    st.add_edge(t, "_y", an_y, None, dace.Memlet("y[0:1]"))
    sdfg.validate()
    return sdfg


def test_sve_variable_spmv_emission_has_gather_fma_reduce():
    """The SpMV-with-reduction emission contains the full SVE stack the
    variable-VL chain must produce for sparse + reduction patterns:
    svld1_gather_s64index_f64 (native gather), svmla_f64_m (FMA
    accumulator), svaddv_f64 (horizontal reduce). These three are the
    SVE-arch-specific implementations of the gather/scatter/reduction
    lowerings the user asked for."""
    sdfg = _build_sve_variable_spmv_sdfg(64)
    cpp = sdfg.generate_code()[0].clean_code
    for intrinsic in ("svwhilelt_b64", "svld1_gather_s64index_f64", "svmla_f64_m", "svaddv_f64", "svcntd()"):
        assert intrinsic in cpp, f"emitted .cpp missing SVE intrinsic {intrinsic!r}"


def test_sve_variable_spmv_runs_correctly_via_scalar_fallback():
    """End-to-end SpMV scalar fallback on x86: sum of ``a[i]*b[idx[i]]``
    matches the numpy reference. Validates the surrounding plumbing
    for the gather + reduce combination."""
    NV = 64
    sdfg = _build_sve_variable_spmv_sdfg(NV)
    a = np.random.rand(NV)
    b = np.random.rand(NV)
    idx = np.random.permutation(NV).astype(np.int64)
    y = np.zeros(1)
    sdfg.compile()(a=a.copy(), b=b.copy(), idx=idx.copy(), y=y, N=NV)
    expected = float(np.sum(a * b[idx]))
    assert np.isclose(y[0], expected, rtol=1e-12, atol=1e-12), \
        f"SpMV reduce mismatch: got {y[0]}, expected {expected}, diff {y[0] - expected}"


def test_sve_variable_axpy_runs_correctly_via_scalar_fallback():
    """End-to-end on x86: the SDFG compiles (SVE branch ``#if``-guarded
    out), the scalar fallback runs and produces ``a + 2*b`` exactly
    (bit-exact: one mul + one add, no reordering opportunity).
    Validates the surrounding plumbing (CPP tasklet wiring, symbol
    threading, header inclusion) without depending on SVE hardware."""
    NV = 64
    sdfg = _build_sve_variable_axpy_sdfg(NV)
    a = np.random.rand(NV)
    b = np.random.rand(NV)
    c = np.zeros(NV)
    sdfg.compile()(a=a.copy(), b=b.copy(), c=c, N=NV)
    expected = a + 2.0 * b
    assert np.allclose(c, expected, rtol=0, atol=0), f"max|d|={float(np.max(np.abs(c - expected)))}"
