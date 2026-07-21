# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Unit tests for the ``LoopToEinsum`` canonicalization pass.

The pass lifts a contraction loop nest (matvec / transposed matvec / matmul) to a
single ``Einsum`` library node, and a pure matrix-transpose nest to a
``Transpose`` node, by probing a disposable copy of the loop and mapping the
lifted node's spec back onto the original arrays. A non-contraction loop is left
untouched (opt-in, safe).

BIT-EXACTNESS: the transpose (a pure copy) and the non-contraction no-op are
bit-exact vs numpy. A contraction lift necessarily REORDERS its floating-point
reduction (that is what parallelising / BLAS-lowering a sum is), so it agrees
with numpy only to a tight tolerance (~1 ULP), matching the repo's existing
``lift_einsum_matmul_test`` convention.
"""
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MPI4PY_RC_INITIALIZE", "0")
os.environ.setdefault("OMPI_MCA_pml", "ob1")
os.environ.setdefault("OMPI_MCA_btl", "self,vader")
os.environ.setdefault("UCX_VFS_ENABLE", "n")

import numpy as np
import pytest

import dace
from dace.sdfg.state import LoopRegion
from dace.libraries.blas.nodes.einsum import Einsum
from dace.libraries.linalg.nodes.transpose import Transpose
from dace.transformation.passes.pattern_matching import PatternMatchAndApplyRepeated
from dace.transformation.dataflow.wcr_conversion import WCRToAugAssign
from dace.transformation.dataflow.map_for_loop import MapToForLoop
from dace.transformation.passes.canonicalize.loop_to_einsum import LoopToEinsum

N = dace.symbol('N')


@dace.program
def matvec(A: dace.float64[N, N], x: dace.float64[N], y: dace.float64[N]):
    for i in range(N):
        for j in range(N):
            y[i] += A[i, j] * x[j]


@dace.program
def transposed_matvec(A: dace.float64[N, N], x: dace.float64[N], y: dace.float64[N]):
    for i in range(N):
        for j in range(N):
            y[j] += A[i, j] * x[i]


@dace.program
def matmul(A: dace.float64[N, N], B: dace.float64[N, N], C: dace.float64[N, N]):
    for i in range(N):
        for j in range(N):
            for k in range(N):
                C[i, j] += A[i, k] * B[k, j]


@dace.program
def matrix_transpose(A: dace.float64[N, N], B: dace.float64[N, N]):
    for i in range(N):
        for j in range(N):
            B[i, j] = A[j, i]


@dace.program
def elementwise(x: dace.float64[N], y: dace.float64[N]):
    for i in range(N):
        y[i] = x[i] + 1.0


@dace.program
def reset_then_transposed_matvec(A: dace.float64[N, N], t: dace.float64[N], y: dace.float64[N]):
    """atax's compute_y shape at source level: reset y, then accumulate the
    transposed matvec (reduction over the OUTER i axis) onto the zeroed y."""
    for k in range(N):
        y[k] = 0.0
    for i in range(N):
        for j in range(N):
            y[j] += A[i, j] * t[i]


@dace.program
def outer_axis_accumulate(A: dace.float64[N, N], t: dace.float64[N], y: dace.float64[N]):
    """The harder no-prior-writer variant of atax's compute_y: ``y[j] += A[i,j]*t[i]``
    accumulates over the OUTER i axis as a ``@dace.map`` WCR, and y has NO in-SDFG
    reset/writer (the caller provides the prior value)."""
    for i in range(N):

        @dace.map
        def acc(j: _[0:N]):
            a << A[i, j]
            tt << t[i]
            o >> y(1, lambda p, q: p + q)[j]
            o = a * tt


@dace.program
def alpha_matmul(A: dace.float64[N, N], B: dace.float64[N, N], C: dace.float64[N, N]):
    """Matmul with a compile-time coefficient: must lift with ``alpha == 2.5``."""
    for i in range(N):
        for j in range(N):
            for k in range(N):
                C[i, j] += 2.5 * A[i, k] * B[k, j]


@dace.program
def tensor_contraction(A: dace.float64[N, N, N], B: dace.float64[N, N], C: dace.float64[N, N, N]):
    """A 4-D nest contracting one axis: ``ijl,lk->ijk``."""
    for i in range(N):
        for j in range(N):
            for k in range(N):
                for l in range(N):
                    C[i, j, k] += A[i, j, l] * B[l, k]


@dace.program
def map_matmul(A: dace.float64[N, N], B: dace.float64[N, N], C: dace.float64[N, N]):
    """MAP-form contraction: the iteration space is one MapEntry with three params and
    the accumulation is a WCR sum, not a read/add/write."""
    for i, j, k in dace.map[0:N, 0:N, 0:N]:
        with dace.tasklet:
            a << A[i, k]
            b << B[k, j]
            c >> C(1, lambda x, y: x + y)[i, j]
            c = a * b


@dace.program
def mixed_loop_map_matmul(A: dace.float64[N, N], B: dace.float64[N, N], C: dace.float64[N, N]):
    """MIXED nest: an outer ``LoopRegion`` wrapping a 2-param map scope."""
    for i in range(N):
        for j, k in dace.map[0:N, 0:N]:
            with dace.tasklet:
                a << A[i, k]
                b << B[k, j]
                c >> C(1, lambda x, y: x + y)[i, j]
                c = a * b


@dace.program
def triangular_matvec(A: dace.float64[N, N], x: dace.float64[N], y: dace.float64[N]):
    """A parameter-dependent (triangular) inner bound. Lifting it to a dense einsum would
    contract over the FULL rectangle, so it must be refused -- this is also what keeps the
    generic matcher off the triangular shapes ``LoopToSyrk``/``LoopToSyr2k`` own."""
    for i in range(N):
        for j in range(i):
            y[i] += A[i, j] * x[j]


@dace.program
def self_referential_matmul(C: dace.float64[N, N], B: dace.float64[N, N]):
    """The output feeds its own contraction -- a genuine loop-carried dependence, not a
    matmul. Must be refused."""
    for i in range(N):
        for j in range(N):
            for k in range(N):
                C[i, j] += C[i, k] * B[k, j]


@dace.program
def subtracting_matmul(A: dace.float64[N, N], B: dace.float64[N, N], C: dace.float64[N, N]):
    """``-=`` is not a ``Sum`` reduction, so it is not an einsum contraction."""
    for i in range(N):
        for j in range(N):
            for k in range(N):
                C[i, j] -= A[i, k] * B[k, j]


def _n_einsum(sdfg):
    return sum(1 for n, _ in sdfg.all_nodes_recursive() if isinstance(n, Einsum))


def _n_transpose(sdfg):
    return sum(1 for n, _ in sdfg.all_nodes_recursive() if isinstance(n, Transpose))


def _n_loops(sdfg):
    return sum(1 for r in sdfg.all_control_flow_regions(recursive=True)
               if isinstance(r, LoopRegion) and r.loop_variable)


def test_matvec_lifts():
    """``y[i] += A[i,j]*x[j]`` lifts to one Einsum (``ij,j->i`` = A@x); the loop
    nest is gone and the result matches numpy."""
    sdfg = matvec.to_sdfg(simplify=True)
    LoopToEinsum().apply_pass(sdfg, {})
    assert _n_einsum(sdfg) == 1, 'the matvec nest must lift to exactly one Einsum node'
    assert _n_loops(sdfg) == 0, 'no sequential loop should remain'
    sdfg.validate()

    n = 12
    rng = np.random.default_rng(0)
    A, x, y = rng.random((n, n)), rng.random(n), np.zeros(n)
    sdfg(A=A, x=x, y=y, N=n)
    assert np.allclose(y, A @ x, rtol=1e-9, atol=1e-12)


def test_transposed_matvec_lifts():
    """``y[j] += A[i,j]*x[i]`` lifts to one Einsum (``ij,i->j`` = A^T@x)."""
    sdfg = transposed_matvec.to_sdfg(simplify=True)
    LoopToEinsum().apply_pass(sdfg, {})
    assert _n_einsum(sdfg) == 1, 'the transposed-matvec nest must lift to exactly one Einsum node'
    assert _n_loops(sdfg) == 0, 'no sequential loop should remain'
    sdfg.validate()

    n = 12
    rng = np.random.default_rng(1)
    A, x, y = rng.random((n, n)), rng.random(n), np.zeros(n)
    sdfg(A=A, x=x, y=y, N=n)
    assert np.allclose(y, A.T @ x, rtol=1e-9, atol=1e-12)


def test_matmul_lifts():
    """``C[i,j] += A[i,k]*B[k,j]`` lifts to one Einsum (``ij,jk->ik`` = A@B)."""
    sdfg = matmul.to_sdfg(simplify=True)
    LoopToEinsum().apply_pass(sdfg, {})
    assert _n_einsum(sdfg) == 1, 'the matmul nest must lift to exactly one Einsum node'
    assert _n_loops(sdfg) == 0, 'no sequential loop should remain'
    sdfg.validate()

    n = 10
    rng = np.random.default_rng(2)
    A, B, C = rng.random((n, n)), rng.random((n, n)), np.zeros((n, n))
    sdfg(A=A, B=B, C=C, N=n)
    assert np.allclose(C, A @ B, rtol=1e-9, atol=1e-12)


def test_transpose_lifts():
    """``B[i,j] = A[j,i]`` lifts to one Transpose node; result is bit-exact vs A.T
    (a pure copy reorders nothing)."""
    sdfg = matrix_transpose.to_sdfg(simplify=True)
    LoopToEinsum().apply_pass(sdfg, {})
    assert _n_transpose(sdfg) == 1, 'the transpose nest must lift to exactly one Transpose node'
    assert _n_einsum(sdfg) == 0, 'a transpose is not a contraction -- no Einsum'
    assert _n_loops(sdfg) == 0, 'no sequential loop should remain'
    sdfg.validate()

    n = 11
    rng = np.random.default_rng(3)
    A, B = rng.random((n, n)), np.zeros((n, n))
    sdfg(A=A, B=B, N=n)
    assert np.array_equal(B, A.T)


def test_non_contraction_noop():
    """A plain elementwise ``y[i] = x[i] + 1`` loop is NOT a contraction: no Einsum
    / Transpose is emitted, the loop is left intact, and it still runs correctly."""
    sdfg = elementwise.to_sdfg(simplify=True)
    loops_before = _n_loops(sdfg)
    LoopToEinsum().apply_pass(sdfg, {})
    assert _n_einsum(sdfg) == 0, 'an elementwise loop must not lift to an Einsum'
    assert _n_transpose(sdfg) == 0, 'an elementwise loop must not lift to a Transpose'
    assert _n_loops(sdfg) == loops_before, 'the non-contraction loop must be left unchanged'
    sdfg.validate()

    n = 12
    rng = np.random.default_rng(4)
    x, y = rng.random(n), np.zeros(n)
    sdfg(x=x, y=y, N=n)
    assert np.array_equal(y, x + 1.0)


def test_reset_then_transposed_matvec_lifts():
    """atax's compute_y shape: y is reset by a separate loop (prior in-SDFG writer),
    then ``y[j] += A[i,j]*t[i]`` accumulates over the OUTER i axis. The matvec must
    lift to one Einsum that FOLDS onto the reset value (beta=1), giving A^T@t. The
    reset loop is not a contraction and is left alone."""
    sdfg = reset_then_transposed_matvec.to_sdfg(simplify=True)
    LoopToEinsum().apply_pass(sdfg, {})
    einsums = [n for n, _ in sdfg.all_nodes_recursive() if isinstance(n, Einsum)]
    assert len(einsums) == 1, 'the transposed matvec must lift to exactly one Einsum node'
    assert einsums[0].einsum_str == 'ij,i->j'
    assert float(einsums[0].beta) == 1.0, 'accumulate onto the reset-zeroed y (beta=1)'
    sdfg.validate()

    n = 13
    rng = np.random.default_rng(5)
    A, t = rng.random((n, n)), rng.random(n)
    y = rng.random(n)  # garbage on entry -- the reset loop must zero it before the fold
    sdfg(A=A, t=t, y=y, N=n)
    assert np.allclose(y, A.T @ t, rtol=1e-9, atol=1e-12)


def test_atax_style_outer_axis_accumulate_lifts():
    """The harder no-prior-writer case (the coordinator's flagged shape). Canonicalize
    DE-WCRs the ``@dace.map`` accumulation (``WCRToAugAssign``) and lowers maps to loops
    (``MapToForLoop``) before ``loop_to_x``, leaving the multiply feeding the add via a
    DIRECT tasklet->tasklet edge inside a ``for i: for j`` nest -- the shape that a naive
    fuse-after-WCR probe fails to lift. It must still lift to one Einsum (``ij,i->j``)
    that FOLDS onto y's caller-provided prior (beta=1) even though y has NO in-SDFG
    writer -- that is the correct ``+=`` semantics (beta=0 would drop a nonzero caller y)."""
    sdfg = outer_axis_accumulate.to_sdfg(simplify=True)
    # Reproduce the reduce-stage de-WCR (-> direct tasklet->tasklet edge) and the
    # map->loop lowering that precede the loop_to_x stage LoopToEinsum runs in.
    PatternMatchAndApplyRepeated([WCRToAugAssign()]).apply_pass(sdfg, {})
    sdfg.apply_transformations_repeated(MapToForLoop, validate=False, validate_all=False)
    LoopToEinsum().apply_pass(sdfg, {})
    einsums = [n for n, _ in sdfg.all_nodes_recursive() if isinstance(n, Einsum)]
    assert len(einsums) == 1, 'the outer-axis transposed matvec must lift to exactly one Einsum node'
    assert einsums[0].einsum_str == 'ij,i->j'
    assert float(einsums[0].beta) == 1.0, 'no in-SDFG writer -> fold onto the caller-provided prior (beta=1)'
    sdfg.validate()

    n = 11
    rng = np.random.default_rng(6)
    A, t = rng.random((n, n)), rng.random(n)
    y = rng.random(n)  # NONZERO caller prior, no in-SDFG writer: beta=1 must fold onto it
    y_prior = y.copy()
    sdfg(A=A, t=t, y=y, N=n)
    assert np.allclose(y, y_prior + A.T @ t, rtol=1e-9, atol=1e-12)


def test_alpha_scaled_matmul_lifts():
    """``C[i,j] += 2.5*A[i,k]*B[k,j]``: the numeric coefficient becomes the Einsum's
    ``alpha``, collected across the SPLIT product tasklets the frontend emits
    (``__tmp0 = 2.5*A``; ``__tmp1 = __tmp0*B``) rather than one fused tasklet."""
    sdfg = alpha_matmul.to_sdfg(simplify=True)
    LoopToEinsum().apply_pass(sdfg, {})
    einsums = [n for n, _ in sdfg.all_nodes_recursive() if isinstance(n, Einsum)]
    assert len(einsums) == 1, 'the alpha-scaled matmul must lift to exactly one Einsum node'
    assert einsums[0].einsum_str == 'ij,jk->ik'
    assert float(einsums[0].alpha) == 2.5, 'the 2.5 coefficient must land on the Einsum alpha'
    assert _n_loops(sdfg) == 0
    sdfg.validate()

    n = 10
    rng = np.random.default_rng(7)
    A, B, C = rng.random((n, n)), rng.random((n, n)), np.zeros((n, n))
    sdfg(A=A, B=B, C=C, N=n)
    assert np.allclose(C, 2.5 * (A @ B), rtol=1e-9, atol=1e-12)


def test_4d_tensor_contraction_lifts():
    """A 4-D nest whose contracted axis is the INNERMOST of four: ``ijl,lk->ijk``."""
    sdfg = tensor_contraction.to_sdfg(simplify=True)
    LoopToEinsum().apply_pass(sdfg, {})
    einsums = [n for n, _ in sdfg.all_nodes_recursive() if isinstance(n, Einsum)]
    assert len(einsums) == 1, 'the 4-D contraction must lift to exactly one Einsum node'
    assert einsums[0].einsum_str == 'ijk,kl->ijl'
    assert _n_loops(sdfg) == 0
    sdfg.validate()

    n = 7
    rng = np.random.default_rng(8)
    A, B, C = rng.random((n, n, n)), rng.random((n, n)), np.zeros((n, n, n))
    sdfg(A=A, B=B, C=C, N=n)
    assert np.allclose(C, np.einsum('ijl,lk->ijk', A, B), rtol=1e-9, atol=1e-12)


def test_map_form_contraction_lifts():
    """A contraction written as ONE map with three params and a WCR-sum accumulation --
    no LoopRegion anywhere. The nest is still recognised and the map scope is replaced
    in place."""
    sdfg = map_matmul.to_sdfg(simplify=True)
    LoopToEinsum().apply_pass(sdfg, {})
    einsums = [n for n, _ in sdfg.all_nodes_recursive() if isinstance(n, Einsum)]
    assert len(einsums) == 1, 'the map-form contraction must lift to exactly one Einsum node'
    assert einsums[0].einsum_str == 'ij,jk->ik'
    assert not [n for n, _ in sdfg.all_nodes_recursive() if isinstance(n, dace.sdfg.nodes.MapEntry)], \
        'the contraction map scope must be gone'
    sdfg.validate()

    n = 9
    rng = np.random.default_rng(9)
    A, B, C = rng.random((n, n)), rng.random((n, n)), np.zeros((n, n))
    sdfg(A=A, B=B, C=C, N=n)
    assert np.allclose(C, A @ B, rtol=1e-9, atol=1e-12)


def test_mixed_loop_and_map_contraction_lifts():
    """A loop wrapping a map: the axes come from BOTH constructs and form one nest."""
    sdfg = mixed_loop_map_matmul.to_sdfg(simplify=True)
    LoopToEinsum().apply_pass(sdfg, {})
    einsums = [n for n, _ in sdfg.all_nodes_recursive() if isinstance(n, Einsum)]
    assert len(einsums) == 1, 'the mixed loop+map nest must lift to exactly one Einsum node'
    assert einsums[0].einsum_str == 'ij,jk->ik'
    assert _n_loops(sdfg) == 0
    sdfg.validate()

    n = 8
    rng = np.random.default_rng(10)
    A, B, C = rng.random((n, n)), rng.random((n, n)), np.zeros((n, n))
    sdfg(A=A, B=B, C=C, N=n)
    assert np.allclose(C, A @ B, rtol=1e-9, atol=1e-12)


@pytest.mark.parametrize('program,reason', [
    (triangular_matvec, 'a parameter-dependent (triangular) bound is not a dense contraction'),
    (self_referential_matmul, 'the output feeding its own contraction is a loop-carried dependence'),
    (subtracting_matmul, '``-=`` is not a Sum reduction'),
])
def test_refused_shapes(program, reason):
    """Shapes that look like a contraction but must NOT be lifted. The triangular case
    doubles as the guard that keeps this generic lift off the shapes the dedicated
    ``LoopToSyrk`` / ``LoopToSyr2k`` BLAS lifts claim earlier in the pipeline."""
    sdfg = program.to_sdfg(simplify=True)
    loops_before = _n_loops(sdfg)
    LoopToEinsum().apply_pass(sdfg, {})
    assert _n_einsum(sdfg) == 0, reason
    assert _n_transpose(sdfg) == 0, reason
    assert _n_loops(sdfg) == loops_before, 'the refused nest must be left untouched'
    sdfg.validate()


if __name__ == '__main__':
    raise SystemExit(pytest.main([__file__, '-v', '-p', 'no:cacheprovider']))
