# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""End-to-end numerical-equivalence smoke for the walker-primary orchestrator.

These are the load-bearing gates that the whole walker + converter + lib-node
+ expansion stack actually produces a correct vectorized SDFG, not just one
that runs without crashing. For each kernel:

1. Build the unvectorized SDFG, compile, run on random input -> reference.
2. Build the same SDFG, run ``VectorizeCPUMultiDim`` through it, compile,
   run on the same input -> vectorized output.
3. Assert numerical equivalence to ``rtol=1e-12`` (modulo FMA reorders).

If any kernel here fails, the walker-primary path silently corrupts numerics
somewhere. These tests are the canary.
"""
import numpy as np
import pytest

import dace
from dace.transformation.passes.vectorization.vectorize_cpu_multi_dim import (VectorizeCPUMultiDim)
from dace.transformation.passes.vectorization.config import VectorizeConfig
from dace.transformation.passes.vectorization.enums import ISA


def _build_k1_copy_sdfg(N):
    """Trivial 1-D copy kernel ``B[i] = A[i]``."""
    sdfg = dace.SDFG("copy_k1_numeric")
    sdfg.add_array("A", (N, ), dace.float64, transient=False)
    sdfg.add_array("B", (N, ), dace.float64, transient=False)
    state = sdfg.add_state("s")
    me, mx = state.add_map("k", {"ii": f"0:{N}"})
    a = state.add_access("A")
    b = state.add_access("B")
    t = state.add_tasklet("body", {"_a"}, {"_b"}, "_b = _a")
    state.add_memlet_path(a, me, t, dst_conn="_a", memlet=dace.Memlet("A[ii]"))
    state.add_memlet_path(t, mx, b, src_conn="_b", memlet=dace.Memlet("B[ii]"))
    return sdfg


def _build_k1_axpy_sdfg(N):
    """Axpy kernel ``C[i] = A[i] + B[i]``."""
    sdfg = dace.SDFG("axpy_k1_numeric")
    sdfg.add_array("A", (N, ), dace.float64, transient=False)
    sdfg.add_array("B", (N, ), dace.float64, transient=False)
    sdfg.add_array("C", (N, ), dace.float64, transient=False)
    state = sdfg.add_state("s")
    me, mx = state.add_map("k", {"ii": f"0:{N}"})
    a = state.add_access("A")
    b = state.add_access("B")
    c = state.add_access("C")
    t = state.add_tasklet("body", {"_a", "_b"}, {"_c"}, "_c = _a + _b")
    state.add_memlet_path(a, me, t, dst_conn="_a", memlet=dace.Memlet("A[ii]"))
    state.add_memlet_path(b, me, t, dst_conn="_b", memlet=dace.Memlet("B[ii]"))
    state.add_memlet_path(t, mx, c, src_conn="_c", memlet=dace.Memlet("C[ii]"))
    return sdfg


def _build_k1_unop_sdfg(N):
    """Unary kernel ``C[i] = abs(A[i])``."""
    sdfg = dace.SDFG("unop_k1_numeric")
    sdfg.add_array("A", (N, ), dace.float64, transient=False)
    sdfg.add_array("C", (N, ), dace.float64, transient=False)
    state = sdfg.add_state("s")
    me, mx = state.add_map("k", {"ii": f"0:{N}"})
    a = state.add_access("A")
    c = state.add_access("C")
    t = state.add_tasklet("body", {"_a"}, {"_c"}, "_c = abs(_a)")
    state.add_memlet_path(a, me, t, dst_conn="_a", memlet=dace.Memlet("A[ii]"))
    state.add_memlet_path(t, mx, c, src_conn="_c", memlet=dace.Memlet("C[ii]"))
    return sdfg


def _run(sdfg, **kwargs):
    """Compile + run an SDFG. Returns ``None`` (outputs live in ``kwargs`` buffers)."""
    return sdfg.compile()(**kwargs)


@pytest.mark.parametrize("N", [8, 16, 32])
def test_k1_copy_matches_reference(N):
    """K=1 trivial copy: walker-primary output bit-equal to unvectorized reference."""
    rng = np.random.default_rng(seed=N)
    a = rng.random(N)
    b_ref = np.zeros(N)
    b_vec = np.zeros(N)
    ref = _build_k1_copy_sdfg(N)
    ref.name = f"copy_ref_{N}"
    vec = _build_k1_copy_sdfg(N)
    vec.name = f"copy_vec_{N}"
    VectorizeCPUMultiDim(VectorizeConfig(widths=(8, ), target_isa=ISA.SCALAR)).apply_pass(vec, {})
    _run(ref, A=a.copy(), B=b_ref)
    _run(vec, A=a.copy(), B=b_vec)
    np.testing.assert_allclose(b_vec, b_ref, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("N", [8, 16, 32])
def test_k1_axpy_matches_reference(N):
    """K=1 axpy ``C = A + B``: walker-primary output equals unvectorized."""
    rng = np.random.default_rng(seed=N + 1)
    a = rng.random(N)
    b = rng.random(N)
    c_ref = np.zeros(N)
    c_vec = np.zeros(N)
    ref = _build_k1_axpy_sdfg(N)
    ref.name = f"axpy_ref_{N}"
    vec = _build_k1_axpy_sdfg(N)
    vec.name = f"axpy_vec_{N}"
    VectorizeCPUMultiDim(VectorizeConfig(widths=(8, ), target_isa=ISA.SCALAR)).apply_pass(vec, {})
    _run(ref, A=a.copy(), B=b.copy(), C=c_ref)
    _run(vec, A=a.copy(), B=b.copy(), C=c_vec)
    np.testing.assert_allclose(c_vec, c_ref, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("N", [8, 16, 32])
def test_k1_unop_matches_reference(N):
    """K=1 unop ``C = abs(A)``: walker-primary output equals unvectorized."""
    rng = np.random.default_rng(seed=N + 2)
    a = rng.random(N) - 0.5  # mix of negative + positive
    c_ref = np.zeros(N)
    c_vec = np.zeros(N)
    ref = _build_k1_unop_sdfg(N)
    ref.name = f"unop_ref_{N}"
    vec = _build_k1_unop_sdfg(N)
    vec.name = f"unop_vec_{N}"
    VectorizeCPUMultiDim(VectorizeConfig(widths=(8, ), target_isa=ISA.SCALAR)).apply_pass(vec, {})
    _run(ref, A=a.copy(), C=c_ref)
    _run(vec, A=a.copy(), C=c_vec)
    np.testing.assert_allclose(c_vec, c_ref, rtol=1e-12, atol=1e-12)
