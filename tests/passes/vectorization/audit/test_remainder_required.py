# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""
Tests for kernels whose iteration count is NOT divisible by vector_width.

Kernels are written with plain Python ``for i in range(...)`` loops
(NOT ``dace.map[...]``). LoopToMap is applied before VectorizeCPU to
convert the loop into a Map; this matches the typical front-end flow
(TSVC tests follow the same pattern).

Each test runs under multiple ``remainder_strategy`` values to exercise
the full pipeline matrix. Today's strategies:

- ``"divides_evenly"`` (default): assumes the range is divisible by W;
  produces correct output on non-divisible cases only by accident (the
  trailing-tile OOB reads/writes happen to land on freshly-allocated
  zero memory, so ``np.allclose`` against the scalar reference passes).
- ``"scalar"`` (R1): splits each non-divisible innermost map into a
  main step-W map + a step-1 sequential scalar postamble. No mask.
  Robust for any N.
- ``"masked"`` / ``"full_loop_mask"``: queued (R2 / R3); raise
  ``NotImplementedError`` from ``VectorizeCPU`` for now.
"""
import copy

import dace
import numpy as np
import pytest

from dace.transformation.interstate import LoopToMap
from dace.transformation.passes.vectorization.vectorize_cpu import VectorizeCPU


N = dace.symbol("N")


@dace.program
def shift_plus_one(a: dace.float64[N], b: dace.float64[N]):
    for i in range(N - 1):
        a[i] = 2.0 * b[i + 1]


@dace.program
def shift_plus_two(a: dace.float64[N], b: dace.float64[N]):
    for i in range(N - 2):
        a[i] = 3.0 * b[i + 2]


@pytest.fixture(params=["divides_evenly", "scalar"])
def remainder_strategy(request):
    """Parametrise tests across all currently-wired remainder strategies."""
    return request.param


def _run(prog, Nv: int, remainder_strategy: str):
    """Build SDFG with LoopToMap, apply VectorizeCPU, compare to scalar reference."""
    a_ref = np.zeros(Nv)
    b = np.random.rand(Nv)
    a_vec = np.zeros(Nv)

    sdfg = prog.to_sdfg(simplify=True)
    sdfg.name = f"{prog.name}_{Nv}_{remainder_strategy}_ref"
    sdfg.apply_transformations_repeated(LoopToMap())
    sdfg.simplify()

    vsdfg = copy.deepcopy(sdfg)
    vsdfg.name = f"{prog.name}_{Nv}_{remainder_strategy}_v"

    VectorizeCPU(vector_width=8, fail_on_unvectorizable=True,
                 use_fp_factor=True, branch_normalization=False,
                 insert_copies=False,
                 remainder_strategy=remainder_strategy).apply_pass(vsdfg, {})

    sdfg(a=a_ref, b=b, N=Nv)
    vsdfg(a=a_vec, b=b, N=Nv)
    diff = np.max(np.abs(a_ref - a_vec))
    return a_ref, a_vec, diff


# Non-divisible iteration counts (force remainder/mask handling).


def test_shift_plus_one_n10_remainder(remainder_strategy):
    """N=10, range(N-1)=range(9) = 9 iters → 1 vector tile + 1 remainder."""
    a_ref, a_vec, diff = _run(shift_plus_one, Nv=10, remainder_strategy=remainder_strategy)
    assert diff < 1e-12, f"max abs diff = {diff}\nref={a_ref}\nvec={a_vec}"


def test_shift_plus_one_n15_remainder(remainder_strategy):
    """N=15, range(N-1)=range(14) = 14 iters → 1 vector + 6 remainder."""
    a_ref, a_vec, diff = _run(shift_plus_one, Nv=15, remainder_strategy=remainder_strategy)
    assert diff < 1e-12, f"max abs diff = {diff}\nref={a_ref}\nvec={a_vec}"


def test_shift_plus_two_n11_remainder(remainder_strategy):
    """+2 shift, N=11, range(N-2)=range(9) = 9 iters → 1 vector + 1 remainder."""
    a_ref, a_vec, diff = _run(shift_plus_two, Nv=11, remainder_strategy=remainder_strategy)
    assert diff < 1e-12, f"max abs diff = {diff}\nref={a_ref}\nvec={a_vec}"


def test_shift_plus_two_n15_remainder(remainder_strategy):
    """+2 shift, N=15, range(N-2)=range(13) = 13 iters → 1 vector + 5 remainder."""
    a_ref, a_vec, diff = _run(shift_plus_two, Nv=15, remainder_strategy=remainder_strategy)
    assert diff < 1e-12, f"max abs diff = {diff}\nref={a_ref}\nvec={a_vec}"


# Divides-evenly sanity tests (regression guards on the working path).


def test_shift_plus_one_n9_divides_evenly(remainder_strategy):
    """N=9, range(N-1)=range(8) = 8 iters, divisible by 8."""
    a_ref, a_vec, diff = _run(shift_plus_one, Nv=9, remainder_strategy=remainder_strategy)
    assert diff < 1e-12, f"max abs diff = {diff}\nref={a_ref}\nvec={a_vec}"


def test_shift_plus_two_n10_divides_evenly(remainder_strategy):
    """N=10, range(N-2)=range(8) = 8 iters, divisible by 8."""
    a_ref, a_vec, diff = _run(shift_plus_two, Nv=10, remainder_strategy=remainder_strategy)
    assert diff < 1e-12, f"max abs diff = {diff}\nref={a_ref}\nvec={a_vec}"
