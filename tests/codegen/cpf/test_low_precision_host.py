# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""A C++ CPF unit spells ``bfloat16`` and ``float16`` as the compiler's own ``__bf16`` and ``_Float16``.

Both are two bytes in the bit layout of a ``uint16_t`` storage, so an entry taking ``__bf16 *`` takes what a
caller hands over as ``uint16_t *`` (HB's ``__npb_bf16``). GCC 13+ and Clang 18+ do the arithmetic, the
conversions and the OpenMP reductions on them; the standard math functions are ambiguous on them, so they run
in ``float`` through CPF's ``cpf_lowp_*`` wrappers. The C dialect, whose helpers are typed by name, refuses them.
"""

import re
import shutil
import subprocess

import numpy as np
import pytest

import dace
from dace.codegen import cpf
from dace.transformation.passes.canonicalize.finalize import finalize_for_target
from dace.transformation.passes.canonicalize.pipeline import canonicalize

N = dace.symbol("N")
NATIVE = {dace.bfloat16: "__bf16", dace.float16: "_Float16"}


def scaled_sum(dtype: dace.typeclass) -> dace.SDFG:
    @dace.program
    def low_precision_scaled_sum(x: dtype[N], y: dtype[N], a: dtype, out: dtype[N], total: dtype[1]):
        out[:] = x * a + y / (N * 2) + np.exp(x)
        total[0] = 0
        for i in dace.map[0:N]:
            total[0] += out[i]

    sdfg = low_precision_scaled_sum.to_sdfg(simplify=True)
    sdfg.name = f"cpf_host_{NATIVE[dtype].strip('_')}"
    canonicalize(sdfg, validate=True, target="cpu")
    finalize_for_target(sdfg, "cpu", validate=True)
    return sdfg


def native_compiler() -> str | None:
    """A C++ compiler on this machine that takes ``__bf16`` and ``_Float16`` arithmetic, or None."""
    compiler = shutil.which("g++")
    if compiler is None:
        return None
    probe = "__bf16 f(__bf16 a, _Float16 b) { return a * (__bf16)(float)b; }\n"
    result = subprocess.run([compiler, "-x", "c++", "-fsyntax-only", "-"], input=probe, text=True, capture_output=True)
    return compiler if result.returncode == 0 else None


@pytest.mark.parametrize("dtype", [dace.bfloat16, dace.float16], ids=["bf16", "fp16"])
def test_a_low_precision_kernel_renders_with_the_native_type(dtype):
    """CPF refused both types outright ("no portable C++ bfloat16 type")."""
    rendering = cpf.render(scaled_sum(dtype), language="c++", check_compiles=False)
    native = NATIVE[dtype]
    assert "dace::" not in rendering.code
    assert re.search(rf"\b{native}\s*\*\s*__restrict__\s+x\b", rendering.code), "the entry takes the native type"
    assert "cpf_lowp_exp(" in rendering.code and "std::exp(cpf_lowp_widen" in rendering.code, "exp runs in float"


@pytest.mark.parametrize("dtype", [dace.bfloat16, dace.float16], ids=["bf16", "fp16"])
def test_the_c_dialect_refuses_a_low_precision_kernel(dtype):
    """Refused by type and with the way out, before a printer reports a missing typed helper instead."""
    with pytest.raises(NotImplementedError, match=r"C\+\+ dialect"):
        cpf.render(scaled_sum(dtype), language="c", check_compiles=False)


@pytest.mark.parametrize("dtype", [dace.bfloat16, dace.float16], ids=["bf16", "fp16"])
def test_a_low_precision_unit_builds_where_the_compiler_has_the_type(dtype):
    if native_compiler() is None:
        pytest.skip("no C++ compiler here takes __bf16 / _Float16 (GCC 13+ or Clang 18+ do)")
    cpf.render(scaled_sum(dtype), language="c++", check_compiles=True)


if __name__ == "__main__":
    for dtype in (dace.bfloat16, dace.float16):
        test_a_low_precision_kernel_renders_with_the_native_type(dtype)
        test_the_c_dialect_refuses_a_low_precision_kernel(dtype)
        if native_compiler() is not None:
            test_a_low_precision_unit_builds_where_the_compiler_has_the_type(dtype)
