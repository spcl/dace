# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Regression tests for cuTile KERNEL codegen error classes found in the
NPBench sweep (durbin/go_fast, gemver/floyd_warshall, hdiff, arc_distance,
nussinov). Each test is a minimal ``@dace.program`` repro of one class:

1. **Float scalar staged into tile arithmetic** (durbin, go_fast): a scalar
   computed on host from device data reaches the kernel; it must be passed
   through device memory (1-element array + 0-d tile load) — by-value floats
   are typed float32 by cuda.tile, so this also guards f64 precision.
2. **Array-element scalar bridge** (gemver, floyd_warshall): a staged element
   like ``u[i]`` must be bound with a constant-shape ``ct.load(u, (i,),
   shape=())`` — the propagated outer subset is a non-constant slice the
   cuda.tile compiler rejects, and arrays are not subscriptable in-kernel.
3. **Ternary over a tile** (hdiff): ``0 if cond_tile else x`` must become
   ``ct.where(cond_tile, 0, x)``.
4. **Bare math-function names in kernel tasklets** (arc_distance): host-side
   numpy aliases (``sinh``, ``atan2``, ...) cannot be captured by a
   ``@ct.kernel``; calls are rewritten to ``ct.*`` spellings.
5. **Constant-fill dtype** (nussinov): ``np.zeros((N, N), np.int32)`` fills
   with the literal ``0.0``; the cutile expansion must emit a dtype-correct
   ``ct.full(..., ct.int32)`` instead of a float32 ``ct.broadcast_to``.
"""
import numpy as np
import pytest

import dace
from dace.transformation.passes.vectorization.vectorize_cutile import VectorizeCuTile

N = dace.symbol("N", dtype=dace.int64)


def _lower(prog, widths=(32, )):
    """Lower a ``@dace.program`` through the cuTile front door.

    :param prog: The DaCe program.
    :param widths: Tile widths for :class:`VectorizeCuTile`.
    :returns: The lowered SDFG.
    """
    sdfg = prog.to_sdfg(simplify=False)
    VectorizeCuTile(widths=widths).apply_pass(sdfg, {})
    return sdfg


# ---------------------------------------------------------------------------
# Class 1: float scalar staged into tile arithmetic (durbin / go_fast)
# ---------------------------------------------------------------------------


@dace.program
def _gofast_like(a: dace.float64[N, N]):
    trace = 0.0
    for i in range(N):
        trace += np.tanh(a[i, i])
    return a + trace


@pytest.mark.gpu
def test_float_scalar_device_path_runtime():
    """A host-accumulated f64 scalar reaches the kernel at full precision."""
    csdfg = _lower(_gofast_like).compile()
    n = 40  # non-divisible by 32: exercises the masked tail
    a = np.random.default_rng(7).random((n, n))
    ref = a + np.tanh(np.diag(a)).sum()
    out = np.asarray(csdfg(a=a.copy(), N=n))
    # Tight tolerance: an f32-typed scalar argument would show ~1e-8 error.
    assert np.abs(ref - out).max() < 1e-12


def test_float_scalar_launch_normalization_codegen():
    """The launch site passes float scalars as 1-element device arrays and
    the kernel binds them as scalar tiles."""
    sdfg = _lower(_gofast_like)
    code = sdfg.generate_code()[0].code.replace(" ", "")
    assert "trace=cupy.empty((1,),dtype=numpy.float64)" in code
    assert "ct.atomic_add(trace,(0,)," in code
    assert "ct.load(trace,(0,),shape=()).item()" in code


# ---------------------------------------------------------------------------
# Class 2: array-element scalar bridge (gemver / floyd_warshall)
# ---------------------------------------------------------------------------


@dace.program
def _outer_add(u: dace.float64[N], v: dace.float64[N], A: dace.float64[N, N]):
    A += np.multiply.outer(u, v)


@pytest.mark.gpu
def test_element_bridge_runtime():
    """A staged array element (``u[i]`` per row) binds as a scalar tile load."""
    csdfg = _lower(_outer_add).compile()
    n = 40
    rng = np.random.default_rng(11)
    u, v, A = rng.random(n), rng.random(n), rng.random((n, n))
    ref = A + np.multiply.outer(u, v)
    Ad = A.copy()
    csdfg(u=u, v=v, A=Ad, N=n)
    assert np.abs(ref - Ad).max() < 1e-12


def test_element_bridge_codegen_no_slice():
    """The bridge is a constant-shape element load, never an in-kernel slice
    with symbolic bounds (rejected by the cuda.tile compiler)."""
    sdfg = _lower(_outer_add)
    code = sdfg.generate_code()[0].code
    assert ", shape=())" in code  # scalar tile load of the staged element
    assert "[0:(((N - 1))" not in code  # the old propagated-subset slice


# ---------------------------------------------------------------------------
# Class 3: ternary over a tile (hdiff)
# ---------------------------------------------------------------------------


@dace.program
def _where_like(a: dace.float64[N], b: dace.float64[N]):
    return np.where(a * b > 0.0, 0.0, a)


@pytest.mark.gpu
def test_tile_ternary_becomes_where_runtime():
    """``x if cond_tile else y`` runs as ``ct.where`` (tiles can't branch)."""
    csdfg = _lower(_where_like).compile()
    n = 40
    rng = np.random.default_rng(13)
    a, b = rng.standard_normal(n), rng.standard_normal(n)
    ref = np.where(a * b > 0.0, 0.0, a)
    out = np.asarray(csdfg(a=a, b=b, N=n))
    assert np.abs(ref - out).max() < 1e-12


# ---------------------------------------------------------------------------
# Class 4: bare math-function names in kernel tasklets (arc_distance)
# ---------------------------------------------------------------------------


@dace.program
def _sinh_prog(a: dace.float64[N]):
    # sinh has no TileUnop mapping, so it stays a plain tasklet inside the
    # kernel — exactly the arc_distance ``atan2`` shape (ufunc-in-kernel).
    return np.sinh(a)


@pytest.mark.gpu
def test_kernel_math_name_mapping_runtime():
    """A plain-tasklet math call inside the kernel uses the ct spelling."""
    csdfg = _lower(_sinh_prog).compile()
    n = 40
    a = np.random.default_rng(17).standard_normal(n)
    out = np.asarray(csdfg(a=a, N=n))
    assert np.abs(np.sinh(a) - out).max() < 1e-12


def test_kernel_math_name_mapping_codegen():
    """The kernel body spells the call ``ct.sinh``, not the host alias."""
    sdfg = _lower(_sinh_prog)
    code = sdfg.generate_code()[0].code
    assert "ct.sinh(" in code


# ---------------------------------------------------------------------------
# Class 5: constant-fill dtype (nussinov)
# ---------------------------------------------------------------------------


@dace.program
def _int_fill(a: dace.int32[N]):
    # np.zeros fills with the float literal 0.0 into an int32 array.
    return a + np.zeros((N, ), np.int32) + np.int32(3)


@pytest.mark.gpu
def test_const_fill_dtype_runtime():
    """A float-literal fill into an int32 tile scatters without a cast error."""
    csdfg = _lower(_int_fill).compile()
    n = 40
    a = np.random.default_rng(19).integers(0, 10, n).astype(np.int32)
    out = np.asarray(csdfg(a=a, N=n))
    assert (out == a + 3).all()


def test_const_fill_dtype_codegen():
    """The Symbol-kind fill is emitted as a dtype-typed ``ct.full``."""
    sdfg = _lower(_int_fill)
    code = sdfg.generate_code()[0].code
    assert "ct.full" in code and "ct.int32" in code


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
