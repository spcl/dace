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
import shutil
import subprocess

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
