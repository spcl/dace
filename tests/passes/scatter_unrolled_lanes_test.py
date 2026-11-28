# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""An unrolled scatter is guarded like the loop it was unrolled from.

``b[ip[i + k]] = ...`` for ``k`` in ``0..3`` with ``i`` stepping by 4 stores through four slots of ``b`` per
iteration. Each slot reads ``ip`` at its own lane offset, and the offsets differ by less than one trip's advance, so
no two (iteration, lane) pairs read one position of ``ip``: the guard that ``ip`` holds no duplicate keeps every
store apart, exactly as for the rolled ``b[ip[i]]``.
"""

import contextlib
import io

import numpy as np

import dace
from dace.sdfg.state import LoopRegion
from dace.transformation.passes.canonicalize import canonicalize

N = dace.symbol("N", dtype=dace.int64, positive=True)


@dace.program
def scatter_unrolled_by_four(ip: dace.int64[N], a: dace.float64[N], b: dace.float64[N]):
    i = 0
    while i + 4 <= N:
        b[ip[i + 0]] = a[i + 0] + 1.0
        b[ip[i + 1]] = a[i + 1] + 1.0
        b[ip[i + 2]] = a[i + 2] + 1.0
        b[ip[i + 3]] = a[i + 3] + 1.0
        i = i + 4
    while i < N:
        b[ip[i]] = a[i] + 1.0
        i = i + 1


def reference(ip, a, b):
    for i in range(a.shape[0]):
        b[ip[i]] = a[i] + 1.0


def canonical() -> dace.SDFG:
    sdfg = scatter_unrolled_by_four.to_sdfg(simplify=True)
    with contextlib.redirect_stdout(io.StringIO()):
        canonicalize(sdfg, target="cpu")
    return sdfg


# Reproducer: canonicalize left TSVC s353's 17-way unrolled scatter loop sequential: its lane slots were taken as
# possibly colliding (LLR s353_scatter_unroll_17, 269x behind the reference solution on GPU)
def test_every_loop_left_is_a_sequential_fallback():
    loops = [r for r in canonical().all_control_flow_regions(recursive=True) if isinstance(r, LoopRegion)]
    assert loops and all(r.label.endswith("_seq_fallback") for r in loops), [r.label for r in loops]


def test_a_permutation_and_a_repeated_index_both_store_what_the_loop_stores():
    sdfg = canonical()
    rng = np.random.default_rng(0)
    n = 23
    for ip in (rng.permutation(n), rng.integers(0, 5, n)):
        a, b = rng.standard_normal(n), np.zeros(n)
        expected = b.copy()
        reference(ip, a, expected)
        sdfg(ip=ip.astype(np.int64), a=a, b=b, N=n)
        assert np.allclose(b, expected)


if __name__ == "__main__":
    test_every_loop_left_is_a_sequential_fallback()
    test_a_permutation_and_a_repeated_index_both_store_what_the_loop_stores()
