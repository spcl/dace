# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""A hand-unrolled sum keeps one accumulator per lane; every one of them is a reduction.

``RetargetWCRAccumulator`` privatized only the first chain of a loop (``s0``), so ``s1 .. s3`` kept the loop
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


if __name__ == "__main__":
    test_every_lane_accumulator_is_a_reduction()
