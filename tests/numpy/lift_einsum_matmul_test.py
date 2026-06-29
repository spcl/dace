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
        do >> D[i, j]
        do = din * bb

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


# C = A@B  -- non-transient output with an identity-0 WCR: must OVERWRITE (beta=0)
# regardless of C's incoming value.
@dace.program
def gemm_ovr(C: dace.float64[M, N], A: dace.float64[M, K], B: dace.float64[K, N]):

    @dace.map
    def comp(i: _[0:M], k: _[0:K], j: _[0:N]):
        a << A[i, k]
        b << B[k, j]
        c >> C(1, lambda x, y: x + y, 0)[i, j]
        c = a * b


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
    inp = dict(A=rng.random((m, k)),
               B=rng.random((k, n)),
               C=rng.random((n, p)),
               D=rng.random((p, l)),
               G=rng.random((m, l)))
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


_CASES = {
    'gemm': _gemm_case,
    'k2mm': _k2mm_case,
    'k3mm': _k3mm_case,
    'gemm_acc': _gemm_acc_case,
    'gemm_ovr': _gemm_ovr_case
}


def _run(sdfg, inp, syms, out_name, expected):
    call = {**{n: np.copy(v) for n, v in inp.items()}, **syms}
    sdfg(**call)
    got = call[out_name]
    assert np.allclose(got, expected, rtol=1e-9, atol=1e-12), f"max|diff|={np.max(np.abs(got - expected)):.3e}"


@pytest.mark.parametrize("name", list(_CASES))
def test_lift_direct(name):
    prog, inp, syms, out_name, expected, n_contractions = _CASES[name]()
    sdfg = prog.to_sdfg(simplify=True)
    sdfg.apply_transformations_repeated(LiftEinsum)
    n_einsum = sum(1 for st in sdfg.states() for nd in st.nodes() if isinstance(nd, blas.Einsum))
    assert n_einsum == n_contractions, f"expected {n_contractions} Einsum nodes, got {n_einsum}"
    sdfg.expand_library_nodes()
    sdfg.validate()
    _run(sdfg, inp, syms, out_name, expected)


@pytest.mark.parametrize("name", list(_CASES))
def test_lift_via_canon(name):
    prog, inp, syms, out_name, expected, _ = _CASES[name]()
    sdfg = prog.to_sdfg(simplify=True)
    sdfg = finalize_for_target(canonicalize(sdfg, validate=True, target='cpu'), 'cpu')
    sdfg.validate()
    _run(sdfg, inp, syms, out_name, expected)


if __name__ == '__main__':
    raise SystemExit(pytest.main([__file__, '-q', '-p', 'no:cacheprovider']))
