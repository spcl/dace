# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""LiftEinsum on the matmul family (gemm / 2mm / 3mm), end to end.

Two paths per kernel, both checked against a numpy oracle:

* ``direct``  -- ``LiftEinsum`` applied to the post-simplify SDFG, then expanded.
* ``canon``   -- the full ``canonicalize`` + ``finalize_for_target`` pipeline,
                 which now contains the ``lift`` stage (so matmuls auto-lift).

Programs are defined inline (small symbols) so they parse fast and the test is a
cheap pre-check for the heavier numerical corpus sweep. Exercises the runtime
scalar coefficient path (gemm/2mm ``alpha`` -> the Einsum ``_alpha`` connector)
and the fresh-accumulator beta path (3mm).
"""
import numpy as np
import pytest

import dace
import dace.libraries.blas as blas
from dace.transformation.dataflow.lift_einsum import LiftEinsum
from dace.transformation.passes.vectorization.config import VectorizeConfig
from dace.transformation.passes.vectorization.enums import ISA, RemainderStrategy
from dace.transformation.passes.canonicalize import canonicalize
from dace.transformation.passes.canonicalize.finalize import finalize_for_target

M = dace.symbol('M')
K = dace.symbol('K')
N = dace.symbol('N')
L = dace.symbol('L')
P = dace.symbol('P')


# C = alpha*A@B + beta*C  (alpha/beta are runtime data scalars)
@dace.program
def gemm(C: dace.float64[M, N], A: dace.float64[M, K], B: dace.float64[K, N], alpha: dace.float64[1],
         beta: dace.float64[1]):

    @dace.map
    def mult_c(i: _[0:M], j: _[0:N]):
        cin << C[i, j]
        bb << beta[0]
        co >> C[i, j]
        co = cin * bb

    @dace.map
    def comp(i: _[0:M], k: _[0:K], j: _[0:N]):
        a << A[i, k]
        b << B[k, j]
        al << alpha[0]
        c >> C(1, lambda x, y: x + y)[i, j]
        c = al * a * b


# D = alpha*(A@B)@C + beta*D
@dace.program
def k2mm(A: dace.float64[M, K], B: dace.float64[K, N], C: dace.float64[N, L], D: dace.float64[M, L],
         alpha: dace.float64[1], beta: dace.float64[1]):
    tmp = dace.define_local([M, N], dtype=dace.float64)

    @dace.map
    def zerotmp(i: _[0:M], j: _[0:N]):
        o >> tmp[i, j]
        o = 0.0

    @dace.map
    def mult_tmp(i: _[0:M], j: _[0:N], k: _[0:K]):
        a << A[i, k]
        b << B[k, j]
        al << alpha[0]
        o >> tmp(1, lambda x, y: x + y)[i, j]
        o = al * a * b

    @dace.map
    def mult_d(i: _[0:M], j: _[0:L]):
        din << D[i, j]
        bb << beta[0]
        dout >> D[i, j]
        dout = din * bb

    @dace.map
    def comp_d(i: _[0:M], j: _[0:L], k: _[0:N]):
        t << tmp[i, k]
        c << C[k, j]
        o >> D(1, lambda x, y: x + y)[i, j]
        o = t * c


# C += A@B  -- C is a pre-filled INPUT with NO in-SDFG initializer and a
# no-identity WCR: the caller's C must be ACCUMULATED onto (beta=1). Regression
# for the latent hazard where beta defaulted to 0 and discarded the caller value.
@dace.program
def gemm_acc(C: dace.float64[M, N], A: dace.float64[M, K], B: dace.float64[K, N]):

    @dace.map
    def comp(i: _[0:M], k: _[0:K], j: _[0:N]):
        a << A[i, k]
        b << B[k, j]
        c >> C(1, lambda x, y: x + y)[i, j]
        c = a * b


# C = A@B  -- OVERWRITE (beta=0): a zero-initialized (``setzero``) accumulator on a
# non-transient output must ignore C's incoming value. NOTE: the Python frontend
# DROPS a WCR identity 3rd-arg (the memlet parser reads only accesses + the WCR
# lambda), so ``C(1, +, 0)`` parses identically to ``C(1, +)``. The case-1 beta=0
# path is reached via ``setzero`` (which real passes set), so the direct test marks
# it explicitly; see ``_mark_setzero``.
@dace.program
def gemm_ovr(C: dace.float64[M, N], A: dace.float64[M, K], B: dace.float64[K, N]):

    @dace.map
    def comp(i: _[0:M], k: _[0:K], j: _[0:N]):
        a << A[i, k]
        b << B[k, j]
        c >> C(1, lambda x, y: x + y, 0)[i, j]
        c = a * b


# y = A@x  (matvec / GEMV, Level-2 BLAS; setzero -> beta=0 overwrite)
@dace.program
def gemv(A: dace.float64[M, N], x: dace.float64[N], y: dace.float64[M]):

    @dace.map
    def comp(i: _[0:M], j: _[0:N]):
        a << A[i, j]
        xx << x[j]
        yy >> y(1, lambda p, q: p + q, 0)[i]
        yy = a * xx


# y = y + A@x  (matvec accumulate onto a MEANINGFUL prior y -> beta=1, no setzero).
# Exercises the GEMV branch's ``beta != 0`` fold (mvt's shape).
@dace.program
def gemv_acc(A: dace.float64[M, N], x: dace.float64[N], y: dace.float64[M]):

    @dace.map
    def comp(i: _[0:M], j: _[0:N]):
        a << A[i, j]
        xx << x[j]
        yy >> y(1, lambda p, q: p + q)[i]
        yy = a * xx


# G = (A@B)@(C@D)  (no alpha/beta; all fresh accumulators -> beta=0)
@dace.program
def k3mm(A: dace.float64[M, K], B: dace.float64[K, N], C: dace.float64[N, P], D: dace.float64[P, L],
         G: dace.float64[M, L]):
    E = dace.define_local([M, N], dtype=dace.float64)
    F = dace.define_local([N, L], dtype=dace.float64)

    @dace.map
    def mult_E(i: _[0:M], j: _[0:N], k: _[0:K]):
        a << A[i, k]
        b << B[k, j]
        o >> E(1, lambda x, y: x + y, 0)[i, j]
        o = a * b

    @dace.map
    def mult_F(i: _[0:N], j: _[0:L], k: _[0:P]):
        c << C[i, k]
        d << D[k, j]
        o >> F(1, lambda x, y: x + y, 0)[i, j]
        o = c * d

    @dace.map
    def mult_G(i: _[0:M], j: _[0:L], k: _[0:N]):
        e << E[i, k]
        f << F[k, j]
        o >> G(1, lambda x, y: x + y, 0)[i, j]
        o = e * f


def _gemm_case():
    m, k, n = 20, 30, 25
    rng = np.random.default_rng(1)
    inp = dict(C=rng.random((m, n)),
               A=rng.random((m, k)),
               B=rng.random((k, n)),
               alpha=np.array([1.5]),
               beta=np.array([1.2]))
    exp = inp['alpha'][0] * (inp['A'] @ inp['B']) + inp['beta'][0] * inp['C']
    return gemm, inp, dict(M=m, K=k, N=n), 'C', exp, 1


def _k2mm_case():
    m, k, n, l = 16, 22, 18, 24
    rng = np.random.default_rng(2)
    inp = dict(A=rng.random((m, k)),
               B=rng.random((k, n)),
               C=rng.random((n, l)),
               D=rng.random((m, l)),
               alpha=np.array([1.5]),
               beta=np.array([1.2]))
    exp = inp['alpha'][0] * ((inp['A'] @ inp['B']) @ inp['C']) + inp['beta'][0] * inp['D']
    return k2mm, inp, dict(M=m, K=k, N=n, L=l), 'D', exp, 2


def _k3mm_case():
    m, k, n, p, l = 16, 20, 18, 24, 22
    rng = np.random.default_rng(3)
    # G is a non-transient output written by a WCR-sum. The frontend DROPS the WCR
    # identity 3rd-arg, so the SDFG accumulates onto G's incoming value (beta=1) --
    # zero it so accumulate == overwrite, isolating what k3mm tests: the 3-einsum
    # chain (A@B)@(C@D). (Overwrite-on-prefilled is covered by gemm_ovr's setzero.)
    inp = dict(A=rng.random((m, k)),
               B=rng.random((k, n)),
               C=rng.random((n, p)),
               D=rng.random((p, l)),
               G=np.zeros((m, l)))
    exp = (inp['A'] @ inp['B']) @ (inp['C'] @ inp['D'])
    return k3mm, inp, dict(M=m, K=k, N=n, P=p, L=l), 'G', exp, 3


def _gemm_acc_case():
    m, k, n = 20, 30, 25
    rng = np.random.default_rng(4)
    inp = dict(C=rng.random((m, n)), A=rng.random((m, k)), B=rng.random((k, n)))
    exp = inp['C'] + (inp['A'] @ inp['B'])  # beta=1: accumulate onto caller's C
    return gemm_acc, inp, dict(M=m, K=k, N=n), 'C', exp, 1


def _gemm_ovr_case():
    m, k, n = 20, 30, 25
    rng = np.random.default_rng(5)
    inp = dict(C=rng.random((m, n)), A=rng.random((m, k)), B=rng.random((k, n)))
    exp = inp['A'] @ inp['B']  # beta=0: overwrite, ignore incoming C
    return gemm_ovr, inp, dict(M=m, K=k, N=n), 'C', exp, 1


def _gemv_case():
    m, n = 16, 20
    rng = np.random.default_rng(6)
    inp = dict(A=rng.random((m, n)), x=rng.random(n), y=np.zeros(m))
    exp = inp['A'] @ inp['x']  # setzero -> beta=0 overwrite
    return gemv, inp, dict(M=m, N=n), 'y', exp, 1


def _gemv_acc_case():
    m, n = 16, 20
    rng = np.random.default_rng(7)
    inp = dict(A=rng.random((m, n)), x=rng.random(n), y=rng.random(m))
    exp = inp['y'] + inp['A'] @ inp['x']  # beta=1 accumulate onto the meaningful prior y
    return gemv_acc, inp, dict(M=m, N=n), 'y', exp, 1


_CASES = {
    'gemm': _gemm_case,
    'k2mm': _k2mm_case,
    'k3mm': _k3mm_case,
    'gemm_acc': _gemm_acc_case,
    'gemm_ovr': _gemm_ovr_case,
    'gemv': _gemv_case,
    'gemv_acc': _gemv_acc_case,
}


def _run(sdfg, inp, syms, out_name, expected):
    call = {**{n: np.copy(v) for n, v in inp.items()}, **syms}
    sdfg(**call)
    got = call[out_name]
    assert np.allclose(got, expected, rtol=1e-9, atol=1e-12), f"max|diff|={np.max(np.abs(got - expected)):.3e}"


def _mark_setzero(sdfg, data):
    """Mark every written ``data`` accumulator as zero-initialized.

    Models what a real pass does (the frontend drops the WCR identity arg), so the
    ``gemm_ovr`` accumulator hits the case-1 beta=0 (overwrite) path in LiftEinsum.
    """
    for st in sdfg.states():
        for nd in st.nodes():
            if isinstance(nd, dace.nodes.AccessNode) and nd.data == data and st.in_degree(nd) > 0:
                nd.setzero = True


@pytest.mark.parametrize("name", list(_CASES))
def test_lift_direct(name):
    prog, inp, syms, out_name, expected, n_contractions = _CASES[name]()
    sdfg = prog.to_sdfg(simplify=True)
    if name in ('gemm_ovr', 'gemv'):  # model the zero-init accumulator (setzero -> beta=0)
        _mark_setzero(sdfg, out_name)
    sdfg.apply_transformations_repeated(LiftEinsum)
    n_einsum = sum(1 for st in sdfg.states() for nd in st.nodes() if isinstance(nd, blas.Einsum))
    assert n_einsum == n_contractions, f"expected {n_contractions} Einsum nodes, got {n_einsum}"
    sdfg.expand_library_nodes()
    sdfg.validate()
    _run(sdfg, inp, syms, out_name, expected)


@pytest.mark.parametrize("name", list(_CASES))
def test_lift_idempotent(name):
    """Lifting is idempotent: a SECOND ``LiftEinsum`` on an already-lifted SDFG is a
    clean no-op (the contraction is now an ``Einsum`` node, no longer a matchable
    map), and the twice-lifted SDFG still expands + runs correctly.

    Vectorization may run standalone on an SDFG a prior pass (canonicalize, or an
    earlier vectorize) already lifted, so re-running the lift MUST NOT re-lift or
    corrupt the graph."""
    prog, inp, syms, out_name, expected, n_contractions = _CASES[name]()
    sdfg = prog.to_sdfg(simplify=True)
    if name == 'gemm_ovr':
        _mark_setzero(sdfg, out_name)
    first = sdfg.apply_transformations_repeated(LiftEinsum)
    n_after_first = sum(1 for st in sdfg.states() for nd in st.nodes() if isinstance(nd, blas.Einsum))
    assert n_after_first == n_contractions, f"first lift: expected {n_contractions} Einsum, got {n_after_first}"
    # Re-run to fixpoint: no contraction maps remain, so nothing is lifted and the
    # Einsum node count is unchanged.
    second = sdfg.apply_transformations_repeated(LiftEinsum)
    assert second == 0, f"second LiftEinsum re-lifted {second} time(s); must be a no-op"
    n_after_second = sum(1 for st in sdfg.states() for nd in st.nodes() if isinstance(nd, blas.Einsum))
    assert n_after_second == n_contractions, f"second lift changed Einsum count {n_after_first}->{n_after_second}"
    sdfg.expand_library_nodes()
    sdfg.validate()
    _run(sdfg, inp, syms, out_name, expected)


@pytest.mark.parametrize("name", list(_CASES))
def test_vectorize_on_prelifted(name):
    """The multi-dim vectorizer accepts an SDFG whose contractions a PRIOR pass
    already lifted to ``Einsum`` nodes: its own internal ``LiftEinsum`` is a no-op on
    the existing nodes, and the finalize/expand tail still lowers them correctly.

    Models the standalone-after-canonicalize path -- the vectorizer must not assume
    it is the one that lifts, nor break on an already-lifted graph."""
    from dace.transformation.passes.vectorization.vectorize_cpu_multi_dim import VectorizeCPUMultiDim
    prog, inp, syms, out_name, expected, n_contractions = _CASES[name]()
    if name == 'gemm_ovr':
        pytest.skip('gemm_ovr setzero injection is a direct-lift mechanism test')
    sdfg = prog.to_sdfg(simplify=True)
    # Pre-lift: the contractions are Einsum nodes BEFORE the vectorizer runs.
    sdfg.apply_transformations_repeated(LiftEinsum)
    assert sum(1 for st in sdfg.states() for nd in st.nodes() if isinstance(nd, blas.Einsum)) == n_contractions
    VectorizeCPUMultiDim(VectorizeConfig(widths=(8, ), target_isa=ISA.SCALAR,
                                         remainder_strategy=RemainderStrategy.SCALAR_POSTAMBLE)).apply_pass(sdfg, {})
    sdfg.validate()
    _run(sdfg, inp, syms, out_name, expected)


@pytest.mark.parametrize("name", list(_CASES))
def test_lift_via_canon(name):
    if name in ('gemm_ovr', 'gemv'):
        # The ``setzero`` beta=0 trigger is a direct-lift mechanism test; the canon-path
        # beta=0 (overwrite) case is covered by k3mm (fresh transient) / gemv_acc.
        pytest.skip('setzero injection is a direct-lift test')
    prog, inp, syms, out_name, expected, _ = _CASES[name]()
    sdfg = prog.to_sdfg(simplify=True)
    sdfg = finalize_for_target(canonicalize(sdfg, validate=True, target='cpu'), 'cpu')
    sdfg.validate()
    _run(sdfg, inp, syms, out_name, expected)


if __name__ == '__main__':
    raise SystemExit(pytest.main([__file__, '-q', '-p', 'no:cacheprovider']))
