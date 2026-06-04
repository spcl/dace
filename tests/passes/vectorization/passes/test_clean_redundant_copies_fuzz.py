# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Fuzz-style end-to-end correctness suite for
:class:`CleanRedundantCopiesAndAssignments`.

Each test compares the numerical output of an untransformed reference SDFG
against the same SDFG after the cleanup pass (optionally chained with
:class:`SameWriteSetIfElseToITECFG` and :class:`BranchNormalization`).
A failure here means the pass altered semantics on the given shape.

Coverage matrix (12 kernels):

* axpy-like simple maps (3): map+tasklet+store, RMW, broadcast scalar.
* gather/scatter (3): permutation-index gather, scatter (non-conflicting
  indices), gather of gather (composed).
* if-else with sub-expression reuse (3): simple sum reuse, OR-condition
  reuse, gather inside the condition.
* cloudsc-extracted loopnest (1).
* branch-normalised case (2): a kernel run through
  ``SameWriteSetIfElseToITECFG`` + ``BranchNormalization`` end-to-end.
"""
import numpy as np
import pytest

import dace
from dace.transformation.passes.vectorization.clean_redundant_copies_and_assignments import (
    CleanRedundantCopiesAndAssignments, )
from dace.transformation.passes.vectorization.same_write_set_if_else_to_ite_cfg import (
    SameWriteSetIfElseToITECFG, )
from dace.transformation.passes.vectorization.branch_normalization import BranchNormalization

N = dace.symbol("N")
M = dace.symbol("M")
_RLMIN = 1e-3
_RALV = 2.5e3
_RALS = 2.8e3
_ZQTMST = 0.02


# ---- Helper: run cleanup, validate, compare numerics ----


def _check_e2e(prog, args_factory, *, with_branch_passes=False, name=None):
    """Run ``prog`` twice: once unmodified to get the reference, once
    after the cleanup (optionally branch passes) and compare every
    output argument. ``args_factory`` returns two dicts (ref, test) with
    matching seeds + a third dict of symbol values.
    """
    ref_args, test_args, syms = args_factory()
    ref_sdfg = prog.to_sdfg(simplify=True)
    ref_sdfg(**ref_args, **syms)

    test_sdfg = prog.to_sdfg(simplify=True)
    if with_branch_passes:
        SameWriteSetIfElseToITECFG().apply_pass(test_sdfg, {})
        BranchNormalization().apply_pass(test_sdfg, {})
    CleanRedundantCopiesAndAssignments().apply_pass(test_sdfg, {})
    test_sdfg.validate()
    test_sdfg(**test_args, **syms)
    for arg_name, ref_val in ref_args.items():
        if not isinstance(ref_val, np.ndarray) or ref_val.dtype.kind not in "fc":
            continue
        test_val = test_args[arg_name]
        assert np.allclose(test_val, ref_val), (f"{name or prog.name}/{arg_name}: max diff "
                                                f"{np.abs(test_val - ref_val).max():.3e}")


# ---- 1. axpy-style map ----


@dace.program
def _k1_axpy(a: dace.float64[N], b: dace.float64[N], out: dace.float64[N]):
    for i in dace.map[0:N]:
        out[i] = a[i] * 2.5 + b[i]


def test_axpy():

    def factory():
        rng = np.random.default_rng(1)
        a = rng.standard_normal(64)
        b = rng.standard_normal(64)
        return (dict(a=a.copy(), b=b.copy(), out=np.zeros(64)), dict(a=a.copy(), b=b.copy(), out=np.zeros(64)),
                dict(N=64))

    _check_e2e(_k1_axpy, factory)


# ---- 2. RMW (read-modify-write) ----


@dace.program
def _k2_rmw(a: dace.float64[N]):
    for i in dace.map[0:N]:
        a[i] = a[i] * 0.5 + 1.0


def test_rmw():

    def factory():
        rng = np.random.default_rng(2)
        a = rng.standard_normal(64)
        return dict(a=a.copy()), dict(a=a.copy()), dict(N=64)

    _check_e2e(_k2_rmw, factory)


# ---- 3. scalar broadcast (loop-invariant computation) ----


@dace.program
def _k3_broadcast(a: dace.float64[N], scale: dace.float64[1], out: dace.float64[N]):
    for i in dace.map[0:N]:
        out[i] = a[i] * scale[0]


def test_broadcast_scalar():

    def factory():
        rng = np.random.default_rng(3)
        a = rng.standard_normal(64)
        s = np.array([2.7])
        return (dict(a=a.copy(), scale=s.copy(), out=np.zeros(64)),
                dict(a=a.copy(), scale=s.copy(), out=np.zeros(64)), dict(N=64))

    _check_e2e(_k3_broadcast, factory)


# ---- 4. gather: arr[idx[i]] ----


@dace.program
def _k4_gather(a: dace.float64[N], idx: dace.int32[N], out: dace.float64[N]):
    for i in dace.map[0:N]:
        out[i] = a[idx[i]] * 2.0


def test_gather():

    def factory():
        rng = np.random.default_rng(4)
        a = rng.standard_normal(64)
        idx = rng.permutation(64).astype(np.int32)
        return (dict(a=a.copy(), idx=idx.copy(), out=np.zeros(64)),
                dict(a=a.copy(), idx=idx.copy(), out=np.zeros(64)), dict(N=64))

    _check_e2e(_k4_gather, factory)


# ---- 5. scatter: out[idx[i]] = ... (permutation indices => no conflict) ----


@dace.program
def _k5_scatter(a: dace.float64[N], idx: dace.int32[N], out: dace.float64[N]):
    for i in dace.map[0:N]:
        out[idx[i]] = a[i] + 1.0


def test_scatter_no_conflict():

    def factory():
        rng = np.random.default_rng(5)
        a = rng.standard_normal(64)
        idx = rng.permutation(64).astype(np.int32)
        return (dict(a=a.copy(), idx=idx.copy(), out=np.zeros(64)),
                dict(a=a.copy(), idx=idx.copy(), out=np.zeros(64)), dict(N=64))

    _check_e2e(_k5_scatter, factory)


# ---- 6. composed gather: arr[idx1[idx2[i]]] ----


@dace.program
def _k6_double_gather(a: dace.float64[N], idx1: dace.int32[N], idx2: dace.int32[N], out: dace.float64[N]):
    for i in dace.map[0:N]:
        out[i] = a[idx1[idx2[i]]]


def test_double_gather():

    def factory():
        rng = np.random.default_rng(6)
        a = rng.standard_normal(64)
        idx1 = rng.permutation(64).astype(np.int32)
        idx2 = rng.permutation(64).astype(np.int32)
        return (dict(a=a.copy(), idx1=idx1.copy(), idx2=idx2.copy(), out=np.zeros(64)),
                dict(a=a.copy(), idx1=idx1.copy(), idx2=idx2.copy(), out=np.zeros(64)), dict(N=64))

    _check_e2e(_k6_double_gather, factory)


# ---- 7. if-else with simple sum reuse ----


@dace.program
def _k7_if_sum_reuse(a: dace.float64[N], b: dace.float64[N], out: dace.float64[N]):
    for i in dace.map[0:N]:
        if a[i] + b[i] > 0.0:
            out[i] = (a[i] + b[i]) * 2.0
        else:
            out[i] = (a[i] + b[i]) * 3.0


def test_if_sum_reuse():

    def factory():
        rng = np.random.default_rng(7)
        a = rng.standard_normal(64)
        b = rng.standard_normal(64)
        return (dict(a=a.copy(), b=b.copy(), out=np.zeros(64)), dict(a=a.copy(), b=b.copy(), out=np.zeros(64)),
                dict(N=64))

    _check_e2e(_k7_if_sum_reuse, factory)


# ---- 8. if-else OR condition (cloudsc shape) ----


@dace.program
def _k8_or_condition(a: dace.float64[N], b: dace.float64[N], c: dace.float64[N], out: dace.float64[N]):
    for i in dace.map[0:N]:
        if a[i] + b[i] < _RLMIN or c[i] < _RLMIN:
            out[i] = a[i] * 2.0 + b[i] * 3.0
        else:
            out[i] = a[i] - b[i]


def test_or_condition():

    def factory():
        rng = np.random.default_rng(8)
        a = rng.standard_normal(64)
        b = rng.standard_normal(64)
        c = rng.standard_normal(64)
        return (dict(a=a.copy(), b=b.copy(), c=c.copy(), out=np.zeros(64)),
                dict(a=a.copy(), b=b.copy(), c=c.copy(), out=np.zeros(64)), dict(N=64))

    _check_e2e(_k8_or_condition, factory)


# ---- 9. gather inside the if condition ----


@dace.program
def _k9_gather_in_condition(a: dace.float64[N], idx: dace.int32[N], out: dace.float64[N]):
    for i in dace.map[0:N]:
        if a[idx[i]] < 0.5:
            out[i] = a[idx[i]] * 2.0
        else:
            out[i] = a[idx[i]] * 3.0


def test_gather_in_condition():

    def factory():
        rng = np.random.default_rng(9)
        a = rng.standard_normal(64)
        idx = rng.permutation(64).astype(np.int32)
        return (dict(a=a.copy(), idx=idx.copy(), out=np.zeros(64)),
                dict(a=a.copy(), idx=idx.copy(), out=np.zeros(64)), dict(N=64))

    _check_e2e(_k9_gather_in_condition, factory)


# ---- 10. cloudsc-extracted loopnest (tidy-branch reduction) ----


@dace.program
def _k10_cloudsc_tidy(zqx_l: dace.float64[N, M], zqx_i: dace.float64[N, M], zqx_v: dace.float64[N, M],
                     za: dace.float64[N, M], ptend_q: dace.float64[N, M], ptend_t: dace.float64[N, M]):
    for jk in range(N):
        for jl in range(M):
            if zqx_l[jk, jl] + zqx_i[jk, jl] < _RLMIN or za[jk, jl] < _RLMIN:
                zqadj_l = zqx_l[jk, jl] * _ZQTMST
                ptend_q[jk, jl] = ptend_q[jk, jl] + zqadj_l
                ptend_t[jk, jl] = ptend_t[jk, jl] - _RALV * zqadj_l
                zqx_v[jk, jl] = zqx_v[jk, jl] + zqx_l[jk, jl]
                zqx_l[jk, jl] = 0.0


def test_cloudsc_tidy_loopnest():

    def factory():
        rng = np.random.default_rng(10)

        def make():
            return dict(zqx_l=rng.standard_normal((4, 8)),
                        zqx_i=rng.standard_normal((4, 8)),
                        zqx_v=rng.standard_normal((4, 8)),
                        za=rng.standard_normal((4, 8)),
                        ptend_q=rng.standard_normal((4, 8)),
                        ptend_t=rng.standard_normal((4, 8)))

        ref = make()
        test = {k: v.copy() for k, v in ref.items()}
        return ref, test, dict(N=4, M=8)

    _check_e2e(_k10_cloudsc_tidy, factory)


# ---- 11. branch-normalised: explicit ITECFG + BranchNormalization first ----


@dace.program
def _k11_branch_normalised(a: dace.float64[N], b: dace.float64[N], out: dace.float64[N]):
    for i in dace.map[0:N]:
        if a[i] > b[i]:
            out[i] = a[i] * b[i] + (a[i] - b[i])
        else:
            out[i] = a[i] * b[i] - (a[i] + b[i])


def test_branch_normalised_kernel():

    def factory():
        rng = np.random.default_rng(11)
        a = rng.standard_normal(64)
        b = rng.standard_normal(64)
        return (dict(a=a.copy(), b=b.copy(), out=np.zeros(64)), dict(a=a.copy(), b=b.copy(), out=np.zeros(64)),
                dict(N=64))

    _check_e2e(_k11_branch_normalised, factory, with_branch_passes=True)


# ---- 12. branch-normalised + gather operand ----


@dace.program
def _k12_branch_norm_with_gather(a: dace.float64[N], idx: dace.int32[N], out: dace.float64[N]):
    for i in dace.map[0:N]:
        if a[idx[i]] > 0.0:
            out[i] = a[idx[i]] * 2.0 + a[i]
        else:
            out[i] = a[idx[i]] * 3.0 - a[i]


def test_branch_normalised_with_gather():

    def factory():
        rng = np.random.default_rng(12)
        a = rng.standard_normal(64)
        idx = rng.permutation(64).astype(np.int32)
        return (dict(a=a.copy(), idx=idx.copy(), out=np.zeros(64)),
                dict(a=a.copy(), idx=idx.copy(), out=np.zeros(64)), dict(N=64))

    _check_e2e(_k12_branch_norm_with_gather, factory, with_branch_passes=True)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
