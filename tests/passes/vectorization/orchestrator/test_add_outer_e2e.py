# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""End-to-end regression tests for ``np.add.outer`` (outer-product) lowering
through the CPU multi-dim tile-op vectorizer (bug 15).

Two distinct root causes made ``np.add.outer`` miscompile on the CPU path; both
are exercised here numerically against NumPy (compile -> run -> compare), for
the ``'pure'`` (K>=2) AND the K=1 ISA-intrinsic paths, at divisible and
non-divisible (masked-tail) sizes:

1. **Restrict-pointer broadcast hijack** (``dace/runtime/include/dace/tile_ops/
   scalar.h``). ``std::is_pointer_v<T* __restrict__>`` is ``false`` under
   GCC/Clang, so the SFINAE-guarded by-value broadcast ``tile_load`` overload
   won for a strided load whose ``_src`` connector is emitted with
   ``__restrict__`` -- splatting ``src[0]`` across every lane. Manifested as the
   K=1 outer product ``out[i, j] = a[i] + b[0]`` (a constant per row).

2. **Outer-product operand collapse** (``dace/transformation/interstate/
   expand_nested_sdfg_inputs.py``). When one NSDFG boundary connector bundles
   two differently-shaped accesses to the same array (``M[i, k]`` and
   ``M[k, j]``), the inner memlets are rebased against a ``Min(k, i)``
   bounding-box origin; the offset re-addition heuristic mis-classified the
   rebased begin as "already absolute" and dropped the re-add, folding both
   operands into a full 2-D tile of ``M`` (computing ``2*M[i, j]``).
"""

import numpy as np
import pytest

import dace
from dace.transformation.passes.vectorization.config import VectorizeConfig
from dace.transformation.passes.vectorization.enums import ISA
from dace.transformation.passes.vectorization.vectorize_cpu_multi_dim import VectorizeCPUMultiDim

N = dace.symbol("N")
P = dace.symbol("P")


@dace.program
def _add_outer_1d(a: dace.float64[N], b: dace.float64[N], out: dace.float64[N, N]):
    out[:] = np.add.outer(a, b)


@dace.program
def _outer_2d_slice(M: dace.float64[N, N], out: dace.float64[N, N]):
    # The floyd-warshall core: outer product of a column and a row of the SAME
    # array (partial-dimension / broadcast operands sharing one connector).
    out[:] = np.add.outer(M[:, P], M[P, :])


def _vectorized(program, widths, isa):
    """Canonicalize + vectorize a copy of ``program`` and return the compiled SDFG."""
    from dace.transformation.passes.canonicalize import canonicalize

    sdfg = program.to_sdfg(simplify=False)
    canonicalize(sdfg)
    VectorizeCPUMultiDim(VectorizeConfig(widths=widths, target_isa=isa)).apply_pass(sdfg, {})
    return sdfg.compile()


@pytest.mark.parametrize("isa", [ISA.SCALAR, ISA.AUTO], ids=["SCALAR", "AUTO"])
@pytest.mark.parametrize("widths", [(8,), (8, 8), (4, 4)])
@pytest.mark.parametrize("n", [8, 12, 17])
def test_add_outer_two_vectors(widths, isa, n):
    """``out[i, j] = a[i] + b[j]`` matches NumPy. K=1 (SCALAR/AUTO intrinsic)
    exercises the restrict-pointer overload fix; K=2 the ``pure`` path;
    ``n=12, 17`` the masked-tail remainder."""
    rng = np.random.default_rng(seed=n)
    a = rng.random(n)
    b = rng.random(n)
    expected = np.add.outer(a, b)
    got = np.zeros((n, n), dtype=np.float64)
    _vectorized(_add_outer_1d, widths, isa)(a=a.copy(), b=b.copy(), out=got, N=n)
    np.testing.assert_allclose(got, expected, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("isa", [ISA.SCALAR, ISA.AUTO], ids=["SCALAR", "AUTO"])
@pytest.mark.parametrize("widths", [(8,), (8, 8), (4, 4)])
@pytest.mark.parametrize("n", [8, 12, 17])
def test_outer_product_of_matrix_slices(widths, isa, n):
    """``out[i, j] = M[i, P] + M[P, j]`` (column + row of one matrix) matches
    NumPy -- the operand-collapse regression that miscompiled floyd-warshall."""
    rng = np.random.default_rng(seed=n + 1)
    M = rng.random((n, n))
    p = 2
    expected = np.add.outer(M[:, p], M[p, :])
    got = np.zeros((n, n), dtype=np.float64)
    _vectorized(_outer_2d_slice, widths, isa)(M=M.copy(), out=got, N=n, P=p)
    np.testing.assert_allclose(got, expected, rtol=1e-12, atol=1e-12)


if __name__ == "__main__":
    for _n in (8, 12, 17):
        for _w in ((8,), (8, 8), (4, 4)):
            test_add_outer_two_vectors(_w, ISA.SCALAR, _n)
            test_outer_product_of_matrix_slices(_w, ISA.SCALAR, _n)
    print("ok")
