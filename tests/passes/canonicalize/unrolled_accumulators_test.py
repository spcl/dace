# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""A hand-unrolled sum keeps one accumulator per lane; every one of them is a reduction.

``retarget_wcr_accumulators`` privatized only the first chain of a loop (``s0``), so ``s1 .. s3`` kept the loop
sequential and the ``s0`` reduction was reverted again. Every chain is retargeted now and the loop maps.
"""

import copy

import numpy as np

import dace
from dace.sdfg.state import LoopRegion
from dace.transformation.passes.canonicalize import canonicalize
from dace.transformation.passes.canonicalize.finalize import finalize_for_target
from tests.corpus.measure_parallelization import cpu_params

N = dace.symbol("N")
M = dace.symbol("M")


@dace.program
def unrolled_sum(a: dace.float64[N], out: dace.float64[1]):
    s0 = 0.0
    s1 = 0.0
    s2 = 0.0
    s3 = 0.0
    i = 0
    while i + 4 <= N:
        s0 = s0 + a[i + 0]
        s1 = s1 + a[i + 1]
        s2 = s2 + a[i + 2]
        s3 = s3 + a[i + 3]
        i = i + 4
    tail = 0.0
    while i < N:
        tail = tail + a[i]
        i = i + 1
    out[0] = s0 + s1 + s2 + s3 + tail


def test_every_lane_accumulator_is_a_reduction():
    sdfg = unrolled_sum.to_sdfg(simplify=True)
    sdfg.name = "unrolled_accumulators"
    canonicalize(sdfg, validate=True, validate_all=False, **cpu_params())
    loops = [region.label for region in sdfg.all_control_flow_regions(recursive=True) if isinstance(region, LoopRegion)]
    assert not loops, loops
    compiled = finalize_for_target(copy.deepcopy(sdfg), "cpu").compile()
    for n in (0, 3, 4, 101):
        a, out = np.random.default_rng(n).random(n), np.zeros(1)
        compiled(a=a, out=out, N=n)
        assert np.isclose(out[0], a.sum()), n


@dace.program
def nested_two_sums(
    a: dace.float64[N, M], r1: dace.int64, r2: dace.int64, c1: dace.int64, c2: dace.int64, out: dace.float64[2]
):
    total = 0.0
    total2 = 0.0
    for i in range(int(r1), int(r2) + 1):
        for j in range(int(c1), int(c2) + 1):
            tmp = a[i, j]
            total = total + tmp
            total2 = total2 + tmp * tmp
    out[0] = total
    out[1] = total2


def test_a_scalar_accumulator_of_a_nested_sum_reduces_over_every_loop():
    """Retargeting the inner loop's chain privatized the scalar with an init and a writeback copy per row: a plain
    write in the outer loop's body that kept the row loop sequential (Rodinia srad's ROI sum). Both loops map."""
    sdfg = nested_two_sums.to_sdfg(simplify=True)
    sdfg.name = "nested_scalar_accumulators"
    canonicalize(sdfg, validate=True, validate_all=False, **cpu_params())
    loops = [region.label for region in sdfg.all_control_flow_regions(recursive=True) if isinstance(region, LoopRegion)]
    assert not loops, loops
    compiled = finalize_for_target(copy.deepcopy(sdfg), "cpu").compile()
    for n, m, r1, r2, c1, c2 in ((4, 3, 2, 1, 0, 2), (1, 1, 0, 0, 0, 0), (9, 13, 2, 7, 1, 11)):
        a, out = np.random.default_rng(n).random((n, m)), np.zeros(2)
        compiled(a=a, r1=r1, r2=r2, c1=c1, c2=c2, out=out, N=n, M=m)
        roi = a[r1 : r2 + 1, c1 : c2 + 1]
        assert np.allclose(out, [roi.sum(), (roi * roi).sum()]), (n, m)


@dace.program
def nested_sum(a: dace.float64[N, M], out: dace.float64[1]):
    total = 0.0
    for i in range(N):
        for j in range(M):
            total = total + a[i, j]
    out[0] = total


def test_a_nested_sum_is_one_reduce_over_both_axes():
    """The inner loop became a ``Reduce`` of one row seeded from ``total``, and the row loop around it stayed a
    sequential loop opening one parallel reduction per row. The row loop widens that ``Reduce`` to the whole array."""
    sdfg = nested_sum.to_sdfg(simplify=True)
    sdfg.name = "nested_sum_reduce"
    canonicalize(sdfg, validate=True, validate_all=False, **cpu_params())
    loops = [region.label for region in sdfg.all_control_flow_regions(recursive=True) if isinstance(region, LoopRegion)]
    assert not loops, loops
    compiled = finalize_for_target(copy.deepcopy(sdfg), "cpu").compile()
    for n, m in ((0, 3), (1, 1), (7, 13)):
        a, out = np.random.default_rng(n).random((n, m)), np.zeros(1)
        compiled(a=a, out=out, N=n, M=m)
        assert np.isclose(out[0], a.sum()), (n, m)


if __name__ == "__main__":
    test_every_lane_accumulator_is_a_reduction()
    test_a_scalar_accumulator_of_a_nested_sum_reduces_over_every_loop()
    test_a_nested_sum_is_one_reduce_over_both_axes()
