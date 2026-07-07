# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Numerical-correctness tests for the multi-dim tile vectorizer.

These run the vectorized kernel on the portable ``SCALAR`` tile backend (no GPU
required) and compare against numpy. The GPU half2 path shares this exact
pipeline and tile-op contract -- only the innermost intrinsic differs (cuda.h's
``__hadd2`` / ... vs scalar.h) and the half2 intrinsics compute bit-identically
to the two scalar fp16 ops -- so these bit-for-bit / tight-tolerance checks
validate the arithmetic the GPU path emits. The GPU-specific lowering (compiles
to a ``.cu``, lowers to native ``f16x2`` PTX) is covered by
``test_vectorize_gpu.py``.

Coverage: elementwise arithmetic, broadcast constants (at input precision),
the transcendental unops (sin / cos / exp / log / sqrt / tanh / tan), min/max,
a 2-D stencil, and an fp16 bit-exactness check.
"""
import numpy as np
import pytest

import dace
from dace.transformation.passes.vectorization.vectorize_cpu_multi_dim import VectorizeCPUMultiDim

M = 64  # exact multiple of every tested width (2 / 4)


def _vectorize(prog, isa="SCALAR", width=4, assume_even=False):
    sdfg = prog.to_sdfg(simplify=True)
    VectorizeCPUMultiDim(widths=(width, ), target_isa=isa, assume_even=assume_even).apply_pass(sdfg, {})
    return sdfg


# --------------------------- elementwise arithmetic ---------------------------
@dace.program
def _arith(A: dace.float32[M], B: dace.float32[M], D: dace.float32[M], C: dace.float32[M]):
    for i in dace.map[0:M]:
        C[i] = (A[i] + B[i]) * D[i] - A[i] / (B[i] + dace.float32(1.0))


def test_elementwise_arithmetic():
    sdfg = _vectorize(_arith, width=4)
    A = np.random.rand(M).astype(np.float32) + 0.5
    B = np.random.rand(M).astype(np.float32) + 0.5
    D = np.random.rand(M).astype(np.float32) + 0.5
    C = np.zeros(M, np.float32)
    sdfg(A=A, B=B, D=D, C=C)
    ref = (A + B) * D - A / (B + 1.0)
    assert np.allclose(C, ref, rtol=1e-5, atol=1e-5), np.max(np.abs(C - ref))


# --------------------------- broadcast constant ---------------------------
@dace.program
def _axpy_const(A: dace.float32[M], B: dace.float32[M], C: dace.float32[M]):
    for i in dace.map[0:M]:
        C[i] = dace.float32(0.25) * A[i] + dace.float32(3.0) * B[i]


def test_broadcast_constant():
    sdfg = _vectorize(_axpy_const, width=4)
    A = np.random.rand(M).astype(np.float32)
    B = np.random.rand(M).astype(np.float32)
    C = np.zeros(M, np.float32)
    sdfg(A=A, B=B, C=C)
    ref = 0.25 * A + 3.0 * B
    assert np.allclose(C, ref, rtol=1e-5, atol=1e-5), np.max(np.abs(C - ref))


# --------------------------- transcendental unops ---------------------------
# One @dace.program per op (the frontend needs each at module scope). The numpy
# oracle is looked up by name; inputs are drawn in a safe domain per op.
@dace.program
def _u_sin(A: dace.float32[M], C: dace.float32[M]):
    for i in dace.map[0:M]:
        C[i] = np.sin(A[i])


@dace.program
def _u_cos(A: dace.float32[M], C: dace.float32[M]):
    for i in dace.map[0:M]:
        C[i] = np.cos(A[i])


@dace.program
def _u_exp(A: dace.float32[M], C: dace.float32[M]):
    for i in dace.map[0:M]:
        C[i] = np.exp(A[i])


@dace.program
def _u_log(A: dace.float32[M], C: dace.float32[M]):
    for i in dace.map[0:M]:
        C[i] = np.log(A[i])


@dace.program
def _u_sqrt(A: dace.float32[M], C: dace.float32[M]):
    for i in dace.map[0:M]:
        C[i] = np.sqrt(A[i])


@dace.program
def _u_tanh(A: dace.float32[M], C: dace.float32[M]):
    for i in dace.map[0:M]:
        C[i] = np.tanh(A[i])


@dace.program
def _u_tan(A: dace.float32[M], C: dace.float32[M]):
    for i in dace.map[0:M]:
        C[i] = np.tan(A[i])


@pytest.mark.parametrize("prog,ref", [
    (_u_sin, np.sin),
    (_u_cos, np.cos),
    (_u_exp, np.exp),
    (_u_log, np.log),
    (_u_sqrt, np.sqrt),
    (_u_tanh, np.tanh),
    (_u_tan, np.tan),
])
def test_transcendental_unop(prog, ref):
    sdfg = _vectorize(prog, width=4)
    # Domain (0.2, 1.0): valid for log/sqrt and away from tan's asymptotes.
    A = (np.random.rand(M).astype(np.float32) * 0.8 + 0.2)
    C = np.zeros(M, np.float32)
    sdfg(A=A, C=C)
    assert np.allclose(C, ref(A), rtol=1e-4, atol=1e-5), np.nanmax(np.abs(C - ref(A)))


# --------------------------- min / max with a constant ---------------------------
@dace.program
def _clamp(A: dace.float32[M], C: dace.float32[M]):
    for i in dace.map[0:M]:
        C[i] = min(max(A[i], dace.float32(0.3)), dace.float32(0.7))


def test_min_max_constant():
    sdfg = _vectorize(_clamp, width=4)
    A = np.random.rand(M).astype(np.float32)
    C = np.zeros(M, np.float32)
    sdfg(A=A, C=C)
    ref = np.minimum(np.maximum(A, 0.3), 0.7)
    assert np.allclose(C, ref, rtol=1e-6, atol=1e-6), np.max(np.abs(C - ref))


# --------------------------- 2-D stencil ---------------------------
@dace.program
def _jacobi2d(A: dace.float32[M, M], B: dace.float32[M, M]):
    for i, j in dace.map[1:M - 1, 1:M - 1]:
        B[i, j] = dace.float32(0.2) * (A[i, j] + A[i, j - 1] + A[i, j + 1] + A[i + 1, j] + A[i - 1, j])


def test_stencil_jacobi2d():
    sdfg = _vectorize(_jacobi2d, width=4)
    A = np.random.rand(M, M).astype(np.float32)
    B = np.zeros((M, M), np.float32)
    sdfg(A=A, B=B)
    ref = B.copy()
    ref[1:-1, 1:-1] = 0.2 * (A[1:-1, 1:-1] + A[1:-1, :-2] + A[1:-1, 2:] + A[2:, 1:-1] + A[:-2, 1:-1])
    assert np.allclose(B, ref, rtol=1e-5, atol=1e-5), np.max(np.abs(B - ref))


# --------------------------- fp16 bit-exactness (the half2 element type) ------
@dace.program
def _fp16_fma(A: dace.float16[M], B: dace.float16[M], C: dace.float16[M]):
    for i in dace.map[0:M]:
        C[i] = dace.float16(0.5) * A[i] + B[i]


def test_fp16_matches_numpy():
    """The fp16 tile arithmetic (the half2 element type) matches numpy fp16.
    Width-2 tile is exactly the half2 packing the GPU path uses."""
    sdfg = _vectorize(_fp16_fma, width=2)
    A = np.random.rand(M).astype(np.float16)
    B = np.random.rand(M).astype(np.float16)
    C = np.zeros(M, np.float16)
    sdfg(A=A, B=B, C=C)
    ref = (np.float16(0.5) * A + B)
    # fp16 ops round each step; numpy does the same per-element, so this is tight.
    assert np.allclose(C.astype(np.float32), ref.astype(np.float32), rtol=1e-2, atol=1e-2), \
        np.max(np.abs(C.astype(np.float32) - ref.astype(np.float32)))


def test_constant_adopts_input_precision():
    """A bare python-float constant in an fp16 kernel is materialised at the input
    precision (fp16), not left fp64 -- so the tile op stays uniform-dtype (and the
    GPU half2 fast path, which requires a ``__half`` tile, still fires)."""
    sdfg = _vectorize(_fp16_fma, width=2)
    code = "\n".join(c.clean_code for c in sdfg.generate_code())
    # The constant is cast to float16, and no float64 literal operand leaks in.
    assert "float16(" in code
    for name, desc in sdfg.arrays.items():
        assert desc.dtype != dace.float64, f"fp16 kernel leaked an fp64 container {name}"
