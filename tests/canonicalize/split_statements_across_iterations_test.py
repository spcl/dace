# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Statement splitting orders the per-statement loops of a guarded body by a dependence across iterations.

TSVC s161 stores ``c[i + 1]`` in one arm and reads ``c[i]`` in the other: the read sees the store of the iteration
before, so the storing statement's loop runs whole first. TSVC s277 tests ``a[i]`` and then updates it, and reads
``b[i]`` that the iteration before stored: the loop storing ``b`` runs first, the one updating ``a[i]`` last, and each
repeats the guard it was under. Every statement runs with the inputs and in the order it had, so the split is exact.
"""

import contextlib
import io

import numpy as np

import dace
from dace.sdfg.state import LoopRegion
from dace.transformation.passes.canonicalize import canonicalize

N = dace.symbol("N")


@dace.program
def s161(a: dace.float64[N], b: dace.float64[N], c: dace.float64[N], d: dace.float64[N], e: dace.float64[N]):
    for i in range(N - 1):
        if b[i] < 0.0:
            c[i + 1] = a[i] + d[i] * d[i]
        else:
            a[i] = c[i] + d[i] * e[i]


def s161_reference(a, b, c, d, e):
    for i in range(a.shape[0] - 1):
        if b[i] < 0.0:
            c[i + 1] = a[i] + d[i] * d[i]
        else:
            a[i] = c[i] + d[i] * e[i]


@dace.program
def s277(a: dace.float64[N], b: dace.float64[N], c: dace.float64[N], d: dace.float64[N], e: dace.float64[N]):
    for i in range(N - 1):
        if a[i] < 0.0:
            if b[i] < 0.0:
                a[i] = a[i] + c[i] * d[i]
            b[i + 1] = c[i] + d[i] * e[i]


def s277_reference(a, b, c, d, e):
    for i in range(a.shape[0] - 1):
        if a[i] < 0.0:
            if b[i] < 0.0:
                a[i] = a[i] + c[i] * d[i]
            b[i + 1] = c[i] + d[i] * e[i]


def check(prog, reference, target: str) -> None:
    sdfg = prog.to_sdfg(simplify=True)
    with contextlib.redirect_stdout(io.StringIO()):
        canonicalize(sdfg, target=target)
    loops = [r.label for r in sdfg.all_control_flow_regions(recursive=True) if isinstance(r, LoopRegion)]
    assert not loops, loops
    rng = np.random.default_rng(0)
    arrays = {name: rng.standard_normal(33) for name in "abcde"}
    expected = {name: value.copy() for name, value in arrays.items()}
    reference(**expected)
    if target == "cpu":
        sdfg(**arrays, N=33)
        for name in arrays:
            assert np.allclose(arrays[name], expected[name]), name


# Reproducer: canonicalize kept TSVC s161 one sequential loop: the c[i + 1] store and the c[i] read in the other arm
# were taken as one statement (LLR tsvc_2_s161, 268x behind the reference solution on GPU)
def test_s161_splits_into_parallel_loops_store_first():
    check(s161, s161_reference, "cpu")
    check(s161, s161_reference, "gpu")


# Reproducer: canonicalize kept TSVC s277 one sequential loop: its guard tests a[i], which the body then updates, and
# b[i + 1] feeds the next iteration's test (LLR tsvc_2_s277, 327x behind the reference solution on GPU)
def test_s277_splits_into_parallel_loops_store_first():
    check(s277, s277_reference, "cpu")
    check(s277, s277_reference, "gpu")


if __name__ == "__main__":
    test_s161_splits_into_parallel_loops_store_first()
    test_s277_splits_into_parallel_loops_store_first()
