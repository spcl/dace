# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""dace/math.h's CPU-emulated fp16 ``exp`` fallback used to duplicate the native
``dace::math::exp(half)`` that dace/cuda/halfvec.cuh already defines under CUDA
(``dace::float16`` IS ``half`` there, not a distinct type). Both were unconditionally
defined, so nvcc rejected EVERY CUDA translation unit that pulls in ``dace/dace.h`` and
touches ``dace::float16`` -- not just ones that call ``exp`` -- with "more than one
instance of overloaded function ``dace::math::exp`` matches the argument list", since a
non-template function body is checked at definition, not at first use. ``sqrt``/``log``
have no such native GPU overload (kept as-is), and ``bfloat16`` has no native
``dace::math::exp`` either (kept as-is); only the ``exp(dace::float16)`` fallback needed
excluding under ``__CUDACC__``.
"""
import os
import re
import shutil
import subprocess

import numpy as np
import pytest

import dace

HAS_NVCC = shutil.which("nvcc") is not None
INCLUDE_DIR = os.path.join(os.path.dirname(dace.__file__), "runtime", "include")

# No call to exp/sqrt/log anywhere -- the bug fires merely from DEFINING both headers'
# overloads in the same TU, so a plain fp16 add is enough to reproduce it.
FP16_KERNEL_SRC = """
#include <cuda_runtime.h>
#include <dace/dace.h>

__global__ void k(dace::float16* o, const dace::float16* a, const dace::float16* b) {
  o[0] = a[0] + b[0];
}
"""


@pytest.mark.skipif(not HAS_NVCC, reason="nvcc not available; CUDA header compile check skipped")
def test_cuda_dace_header_compiles_with_float16(tmp_path):
    """``dace/dace.h`` must compile for a CUDA TU touching ``dace::float16``: the CPU-emulation
    fallback in dace/math.h must not also define ``dace::math::exp`` for the SAME type as the
    native GPU one in dace/cuda/halfvec.cuh, or nvcc refuses every fp16 CUDA kernel as ambiguous."""
    src = tmp_path / "probe.cu"
    src.write_text(FP16_KERNEL_SRC)
    result = subprocess.run(
        [
            "nvcc", "-I", INCLUDE_DIR, "-std=c++20", "--expt-relaxed-constexpr", "-arch=sm_80", "-x", "cu", "-c",
            str(src), "-o",
            str(tmp_path / "probe.o")
        ],
        capture_output=True,
        text=True,
    )
    assert "more than one instance of overloaded function" not in result.stderr, \
        f"dace::math::exp(half) / exp(dace::float16) ambiguity regressed:\n{result.stderr}"
    assert result.returncode == 0, f"dace/dace.h failed to compile for CUDA float16:\n{result.stderr}"


# ``dace::float16`` IS the native CUDA ``half``, which non-explicitly converts to float, every
# integer width, and char -- a dozen built-in targets. A BARE ``sqrt(half_value)`` -- what the
# numpy frontend's ``simple_call`` used to emit for ``np.sqrt``: ``__out = sqrt(__inp)``, no module
# qualifier -- resolves through ordinary unqualified lookup to ``std::sqrt``'s float/double/long
# double overloads, each reachable through a DIFFERENT one of those conversion operators; two
# candidates tie for best and nvcc refuses the call as ambiguous. ``dace::math::sqrt`` already
# carries a non-template ``dace::float16`` overload (dace/math.h, ``DACE_MATH_UNARY_LP``) that is
# an exact-type match and so unambiguously wins, but nothing reaches it unless the call is
# qualified -- which is what ``cppunparse.py``'s ``_renamed_funcs`` now does for ``sqrt``.
NAKED_SQRT_KERNEL_SRC = """
#include <cuda_runtime.h>
#include <dace/dace.h>

__global__ void k(dace::float16* o, const dace::float16* a) {
  o[0] = sqrt(a[0]);
}
"""


@pytest.mark.skipif(not HAS_NVCC, reason="nvcc not available; CUDA header compile check skipped")
def test_cuda_naked_sqrt_is_still_ambiguous_for_float16(tmp_path):
    """Documents *why* codegen must never emit a bare ``sqrt(dace::float16)`` on CUDA: unlike the
    ``dace::math::exp`` case above (a header-only fix), a naked call is inherently ambiguous
    because ``half`` converts implicitly to a dozen built-in types -- there is no way to make this
    particular spelling compile without either qualifying the call or making ``dace::float16``
    stop converting implicitly. This is the negative control for
    ``test_cuda_generated_sqrt_is_qualified_for_float16`` below: if this ever starts compiling,
    the class of bug changed and the codegen-side fix may no longer be required."""
    src = tmp_path / "probe_naked_sqrt.cu"
    src.write_text(NAKED_SQRT_KERNEL_SRC)
    result = subprocess.run(
        [
            "nvcc", "-I", INCLUDE_DIR, "-std=c++20", "--expt-relaxed-constexpr", "-arch=sm_80", "-x", "cu", "-c",
            str(src), "-o",
            str(tmp_path / "probe_naked_sqrt.o")
        ],
        capture_output=True,
        text=True,
    )
    assert "more than one instance of overloaded function" in result.stderr
    assert result.returncode != 0


@pytest.mark.skipif(not HAS_NVCC, reason="nvcc not available; CUDA header compile check skipped")
def test_cuda_generated_sqrt_is_qualified_for_float16(tmp_path):
    """End-to-end regression for the reported bug: ``np.sqrt`` on a ``dace.float16`` array, lowered
    for GPU (``apply_gpu_transformations``), used to generate a bare ``sqrt(...)`` tasklet call and
    fail to compile with nvcc's "more than one instance of overloaded function \\"sqrt\\" matches
    the argument list" (matching ``test_cuda_naked_sqrt_is_still_ambiguous_for_float16`` above).
    Asserts BOTH the generated source text (the call must be qualified, not bare -- a structural
    check, not just "it compiled") and that the generated CUDA translation unit actually builds."""
    N = dace.symbol('N')

    @dace.program
    def sqrt_kernel(x: dace.float16[N], y: dace.float16[N]):
        y[:] = np.sqrt(x)

    sdfg = sqrt_kernel.to_sdfg(simplify=True)
    sdfg.specialize({'N': 256})
    sdfg.apply_gpu_transformations()
    code_objects = sdfg.generate_code()
    cuda_objects = [co for co in code_objects if co.name.endswith('_cuda')]
    assert len(cuda_objects) == 1, f"expected exactly one CUDA code object, got: {[co.name for co in code_objects]}"
    cuda_code = cuda_objects[0].clean_code

    # Structural: the emitted call must be the qualified library function, and no bare/ADL-only
    # ``sqrt(`` call (which would re-hit the ambiguity) may remain.
    assert 'dace::math::sqrt(' in cuda_code, \
        f"expected a qualified dace::math::sqrt(...) call in generated CUDA:\n{cuda_code}"
    assert re.search(r'(?<!math::)\bsqrt\(', cuda_code) is None, \
        f"a bare, unqualified sqrt(...) call remains in generated CUDA:\n{cuda_code}"

    src = tmp_path / "sqrt_kernel_cuda.cu"
    src.write_text(cuda_code)
    result = subprocess.run(
        [
            "nvcc", "-I", INCLUDE_DIR, "-std=c++20", "--expt-relaxed-constexpr", "-arch=sm_80", "-x", "cu", "-c",
            str(src), "-o",
            str(tmp_path / "sqrt_kernel_cuda.o")
        ],
        capture_output=True,
        text=True,
    )
    assert "more than one instance of overloaded function" not in result.stderr, \
        f"generated np.sqrt(float16[N]) CUDA kernel regressed the sqrt ambiguity:\n{result.stderr}"
    assert result.returncode == 0, f"generated np.sqrt(float16[N]) CUDA kernel failed to compile:\n{result.stderr}"
