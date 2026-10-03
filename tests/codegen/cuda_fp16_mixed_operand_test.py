# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""A CUDA ``dace::float16`` mixed with a built-in number, at the header level.

Under CUDA ``dace::float16`` IS ``__half``, which converts implicitly to every built-in arithmetic
type and is constructible from each of them. Any expression pairing a half with a ``float``,
``double`` or ``int`` therefore offers nvcc two equally good candidates -- the built-in operator
(converting the half up) and ``operator<op>(__half, __half)`` (converting the other side down) --
and nvcc rejects the translation unit:

    error: more than one operator "<" matches these operands:
                built-in operator "arithmetic < arithmetic"
                function "operator<(const __half &, const __half &)"
                operand types are: dace::float16 < double

The same tie hits every libm call (``sin``, ``floor``, ``fmod``, ``std::pow``, ``abs``: the
float/double/long double overloads are each one conversion away), compound assignment into a
built-in (``float x; x += h``), and every helper whose result type is ``std::common_type`` of a
half and a built-in (``Min``/``Max``, ``ITE``, ``IfExpr``), which has no answer and so dropped the
helper from overload resolution.

``dace/math.h`` now answers each with an exact match that promotes the half to ``float`` and runs
the built-in operation, so the result follows C++ promotion (half op float -> float, half op
double -> double) and a half op half still uses CUDA's native half operators.

Every probe here is compiled with nvcc, so the module needs nvcc but no GPU.
"""
import os
import shutil
import subprocess

import pytest

import dace

#: The whole module is about float16, so it carries the marker the fp16 CI leg selects on.
pytestmark = [
    pytest.mark.fp16,
    pytest.mark.skipif(shutil.which("nvcc") is None, reason="nvcc not available; CUDA header compile check skipped"),
]

INCLUDE_DIR = os.path.join(os.path.dirname(dace.__file__), "runtime", "include")

#: One operand of each kind, read from memory so nothing folds at compile time.
KERNEL_TEMPLATE = """
#include <cuda_runtime.h>
#include <dace/dace.h>

__global__ void k(dace::float16* H, float* F, double* D, int* I, bool* B) {{
  dace::float16 h = H[0];
  float f = F[0];
  double d = D[0];
  int i = I[0];
  bool b = B[0];
  {body};
}}
"""


def _nvcc(tmp_path, body: str) -> subprocess.CompletedProcess:
    src = tmp_path / "probe.cu"
    src.write_text(KERNEL_TEMPLATE.format(body=body))
    return subprocess.run(
        [
            "nvcc", "-I", INCLUDE_DIR, "-std=c++20", "--expt-relaxed-constexpr", "-arch=sm_80", "-x", "cu", "-c",
            str(src), "-o",
            str(tmp_path / "probe.o")
        ],
        capture_output=True,
        text=True,
    )


#: Expressions that used to be rejected as ambiguous (or, for ``Min``/``Max``/``ITE``/``IfExpr``,
#: as having no viable candidate) and must now compile.
MIXED_EXPRESSIONS = {
    # Binary operators against each built-in kind, on both sides.
    "half_lt_double_literal": "B[0] = h < 1e-14",
    "float_div_half": "F[0] = f / h",
    "half_div_float": "F[0] = h / f",
    "double_div_half": "D[0] = d / h",
    "float_eq_half": "B[0] = f == h",
    "half_ge_double": "B[0] = h >= d",
    "half_add_float": "F[0] = h + f",
    "double_sub_half": "D[0] = d - h",
    "half_mul_int": "F[0] = h * i",
    "half_ne_bool": "B[0] = h != b",
    # Compound assignment, with a built-in and with a half on the left.
    "float_add_assign_half": "f += h; F[0] = f",
    "double_div_assign_half": "d /= h; D[0] = d",
    "int_sub_assign_half": "i -= h; I[0] = i",
    "half_add_assign_double": "h += d; H[0] = h",
    # Result-type helpers emitted by codegen.
    "Min_double_half": "D[0] = Min(d, h)",
    "Max_half_float_half": "F[0] = Max(h, f, h)",
    "ITE_half_double": "D[0] = ITE(b, h, d)",
    "ITE_half_int_literal": "F[0] = ITE(b, h, 0)",
    "IfExpr_half_float": "F[0] = IfExpr(b, h, f)",
    # libm, which codegen leaves unqualified.
    "sin_half": "H[0] = sin(h)",
    "floor_half": "H[0] = floor(h)",
    "round_half": "H[0] = round(h)",
    "erf_half": "H[0] = erf(h)",
    "sqrt_half_bare": "H[0] = sqrt(h)",
    "fmod_half_half": "H[0] = fmod(h, h)",
    "fmod_half_double": "D[0] = fmod(h, d)",
    "atan2_half_float": "F[0] = atan2(h, f)",
    "copysign_float_half": "F[0] = copysign(f, h)",
    "abs_half": "H[0] = abs(h)",
    "Abs_half": "H[0] = Abs(h)",
    # libm as codegen qualifies it (``math.sin(h)`` -> ``dace::math::sin``).
    "dace_math_sin_half": "H[0] = dace::math::sin(h)",
    "dace_math_round_half": "H[0] = dace::math::round(h)",
    "dace_math_erf_half": "H[0] = dace::math::erf(h)",
    "dace_math_fmod_half_double": "D[0] = dace::math::fmod(h, d)",
    "dace_math_fmod_half_half": "H[0] = dace::math::fmod(h, h)",
    "dace_math_atan2_float_half": "F[0] = dace::math::atan2(f, h)",
    # Lowercase min/max, as codegen emits a tasklet's ``min(a, b)``.
    "min_double_half": "D[0] = min(d, h)",
    "max_half_float": "F[0] = max(h, f)",
    "min_half_int": "H[0] = min(h, i)",
    # ``a ** b`` lowers to ``dace::math::pow``, which forwarded a half straight to ``std::pow``.
    "pow_half_double": "D[0] = dace::math::pow(h, d)",
    "pow_double_half": "D[0] = dace::math::pow(d, h)",
    "pow_half_half": "H[0] = dace::math::pow(h, h)",
    "pow_half_int": "H[0] = dace::math::pow(h, i)",
}


@pytest.mark.parametrize("body", MIXED_EXPRESSIONS.values(), ids=MIXED_EXPRESSIONS.keys())
def test_mixed_half_expression_compiles(tmp_path, body):
    result = _nvcc(tmp_path, body)
    assert "more than one" not in result.stderr, f"`{body}` is still ambiguous:\n{result.stderr}"
    assert result.returncode == 0, f"`{body}` failed to compile:\n{result.stderr}"


#: Half-free expressions through every name the fix touched: the new overloads must not capture
#: or change them.
UNTOUCHED_EXPRESSIONS = {
    "builtin_math": "F[0] = sin(f) + floor(d) + abs(f) + abs(i) + fmod(f, 2.0f)",
    "builtin_helpers": "D[0] = Min(f, d) + Max(i, 3) + ITE(b, f, d) + IfExpr(b, i, d)",
    "builtin_pow": "D[0] = dace::math::pow(f, 2.0) + dace::math::pow(i, 3u)",
    "half_half_native": "H[0] = h * h + h - h / h",
    "dace_math_half": "H[0] = dace::math::sqrt(h) + dace::math::exp(h) + dace::math::log(h)",
}


@pytest.mark.parametrize("body", UNTOUCHED_EXPRESSIONS.values(), ids=UNTOUCHED_EXPRESSIONS.keys())
def test_half_free_expression_still_compiles(tmp_path, body):
    result = _nvcc(tmp_path, body)
    assert result.returncode == 0, f"`{body}` stopped compiling:\n{result.stderr}"


def test_mixed_half_result_types(tmp_path):
    """The promotion rule, pinned per helper: the half counts as a ``float``, a half with a half
    stays a half, and an integral exponent keeps ``pow`` in half. Lowercase ``min``/``max`` keep
    a half beside an integer in half, as they always did."""
    asserts = [
        ("decltype(h / f)", "float"),
        ("decltype(d - h)", "double"),
        ("decltype(h < 1e-14)", "bool"),
        ("decltype(h * h)", "dace::float16"),
        ("std::common_type<dace::float16, double>::type", "double"),
        ("std::common_type<int, dace::float16>::type", "float"),
        ("std::common_type<dace::float16, dace::float16>::type", "dace::float16"),
        ("std::common_type<dace::float16, float, double>::type", "double"),
        ("decltype(Min(d, h))", "double"),
        ("decltype(Min(h, h))", "dace::float16"),
        ("decltype(min(d, h))", "double"),
        ("decltype(max(h, f))", "float"),
        ("decltype(min(h, i))", "dace::float16"),
        ("decltype(min(h, h))", "dace::float16"),
        ("decltype(dace::math::sin(h))", "dace::float16"),
        ("decltype(dace::math::fmod(h, d))", "double"),
        ("decltype(ITE(b, h, f))", "float"),
        ("decltype(sin(h))", "dace::float16"),
        ("decltype(fmod(h, d))", "double"),
        ("decltype(dace::math::pow(h, d))", "double"),
        ("decltype(dace::math::pow(h, h))", "dace::float16"),
        ("decltype(dace::math::pow(h, 3))", "dace::float16"),
    ]
    body = "; ".join(f'static_assert(std::is_same<{t}, {want}>::value, "{t} is not {want}")' for t, want in asserts)
    result = _nvcc(tmp_path, body)
    assert result.returncode == 0, f"a mixed-half result type changed:\n{result.stderr}"


#: Expressions that must keep failing. Each pins a limit the header cannot or should not lift.
STILL_REJECTED = {
    # A raw ``?:`` cannot be overloaded: codegen has to emit ``ITE`` or cast an arm.
    "raw_ternary_half_double": ("D[0] = b ? h : d", 'ambiguous "?" operation'),
    # Min/Max reject an int beside a floating-point argument; a half is floating-point.
    "Min_half_int": ("F[0] = Min(h, i)", "mixing floating-point and integer arguments"),
}


@pytest.mark.parametrize("body,message", STILL_REJECTED.values(), ids=STILL_REJECTED.keys())
def test_mixed_half_expression_still_rejected(tmp_path, body, message):
    result = _nvcc(tmp_path, body)
    assert result.returncode != 0, f"`{body}` unexpectedly compiles"
    assert message in result.stderr, f"`{body}` failed for a different reason:\n{result.stderr}"
