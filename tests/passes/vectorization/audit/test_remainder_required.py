# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""
Tests for kernels whose iteration count is NOT divisible by vector_width.

Kernels are written with plain Python ``for i in range(...)`` loops
(NOT ``dace.map[...]``). LoopToMap is applied before VectorizeCPU to
convert the loop into a Map; this matches the typical front-end flow
(TSVC tests follow the same pattern).

For both shift patterns (``b[i+1]`` and ``b[i+2]``) and a mix of N
values (divisible and non-divisible by W=8), the vectorized output
should match the scalar reference. The non-divisible cases pass today
because the trailing tile reads/writes happen to be benign on
``np.zeros`` / ``np.random.rand`` allocations (OOB reads return zero
from the fresh memory page). Once true remainder / masked-tail
support lands, this becomes a load-bearing correctness guard rather
than an accidental coincidence.
"""
import copy

import dace
import numpy as np

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


def _run(prog, Nv: int):
    """Build SDFG with LoopToMap, apply VectorizeCPU, compare to scalar reference."""
    a_ref = np.zeros(Nv)
    b = np.random.rand(Nv)
    a_vec = np.zeros(Nv)

    sdfg = prog.to_sdfg(simplify=True)
    sdfg.name = f"{prog.name}_{Nv}_ref"
    sdfg.apply_transformations_repeated(LoopToMap())
    sdfg.simplify()

    vsdfg = copy.deepcopy(sdfg)
    vsdfg.name = f"{prog.name}_{Nv}_v"

    VectorizeCPU(vector_width=8, fail_on_unvectorizable=True,
                 use_fp_factor=True, branch_normalization=False,
                 insert_copies=False).apply_pass(vsdfg, {})

    sdfg(a=a_ref, b=b, N=Nv)
    vsdfg(a=a_vec, b=b, N=Nv)
    diff = np.max(np.abs(a_ref - a_vec))
    return a_ref, a_vec, diff


# Iteration counts: for vector_width=8, pick Nv such that the loop range
# (`range(N-1)` for +1 shift; `range(N-2)` for +2 shift) is NOT divisible
# by 8, forcing a remainder/tail iteration.

def test_shift_plus_one_n10_remainder():
    """N=10, range(N-1)=range(9) = 9 iters → 1 vector tile + 1 remainder."""
    a_ref, a_vec, diff = _run(shift_plus_one, Nv=10)
    assert diff < 1e-12, f"max abs diff = {diff}\nref={a_ref}\nvec={a_vec}"


def test_shift_plus_one_n15_remainder():
    """N=15, range(N-1)=range(14) = 14 iters → 1 vector + 6 remainder."""
    a_ref, a_vec, diff = _run(shift_plus_one, Nv=15)
    assert diff < 1e-12, f"max abs diff = {diff}\nref={a_ref}\nvec={a_vec}"


def test_shift_plus_two_n11_remainder():
    """+2 shift, N=11, range(N-2)=range(9) = 9 iters → 1 vector + 1 remainder."""
    a_ref, a_vec, diff = _run(shift_plus_two, Nv=11)
    assert diff < 1e-12, f"max abs diff = {diff}\nref={a_ref}\nvec={a_vec}"


def test_shift_plus_two_n15_remainder():
    """+2 shift, N=15, range(N-2)=range(13) = 13 iters → 1 vector + 5 remainder."""
    a_ref, a_vec, diff = _run(shift_plus_two, Nv=15)
    assert diff < 1e-12, f"max abs diff = {diff}\nref={a_ref}\nvec={a_vec}"


# Divides-evenly sanity tests — these pass on today's pipeline and pin
# the working path against accidental breakage during remainder work.

def test_shift_plus_one_n9_divides_evenly():
    """N=9, range(N-1)=range(8) = 8 iters, divisible by 8."""
    a_ref, a_vec, diff = _run(shift_plus_one, Nv=9)
    assert diff < 1e-12, f"max abs diff = {diff}\nref={a_ref}\nvec={a_vec}"


def test_shift_plus_two_n10_divides_evenly():
    """N=10, range(N-2)=range(8) = 8 iters, divisible by 8."""
    a_ref, a_vec, diff = _run(shift_plus_two, Nv=10)
    assert diff < 1e-12, f"max abs diff = {diff}\nref={a_ref}\nvec={a_vec}"
