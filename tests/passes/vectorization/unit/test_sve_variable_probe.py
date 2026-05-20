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
