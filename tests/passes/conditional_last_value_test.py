# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""``ConditionalLastValue`` in canonicalization: a scalar stored under a condition and read later in the iteration
(TSVC s258) splits the loop around a max-scan of the storing iterations, and both halves lift to maps."""

import numpy as np

import dace
from dace.sdfg.state import LoopRegion
from dace.transformation.passes.canonicalize import canonicalize

N = dace.symbol("N")


def canonical_loops(program) -> tuple[dace.SDFG, int]:
    sdfg = program.to_sdfg()
    canonicalize(sdfg, validate=True, target="cpu")
    loops = [r for r in sdfg.all_control_flow_regions(recursive=True) if isinstance(r, LoopRegion)]
    return sdfg, len(loops)


@dace.program
def last_stored(a: dace.float64[N], b: dace.float64[N], c: dace.float64[N], d: dace.float64[N], out: dace.float64[1]):
    s = out[0]
    for i in range(N):
        if a[i] > 0.0:
            s = d[i] * d[i]
        b[i] = s * c[i] + d[i]
    out[0] = s


def last_stored_reference(a, b, c, d, out) -> None:
    s = out[0]
    for i in range(a.shape[0]):
        if a[i] > 0.0:
            s = d[i] * d[i]
        b[i] = s * c[i] + d[i]
    out[0] = s


@dace.program
def read_before_store(a: dace.float64[N], b: dace.float64[N], c: dace.float64[N], d: dace.float64[N]):
    s = 0.0
    for i in range(N):
        b[i] = s * c[i]
        if a[i] > 0.0:
            s = d[i] * d[i]


def read_before_store_reference(a, b, c, d) -> None:
    s = 0.0
    for i in range(a.shape[0]):
        b[i] = s * c[i]
        if a[i] > 0.0:
            s = d[i] * d[i]


def run_last_stored(n: int) -> int:
    sdfg, loops = canonical_loops(last_stored)
    rng = np.random.default_rng(0)
    args = [rng.random(n) - 0.5, rng.random(n), rng.random(n), rng.random(n), np.array([3.0])]
    ref = [x.copy() for x in args]
    last_stored_reference(*ref)
    sdfg(a=args[0], b=args[1], c=args[2], d=args[3], out=args[4], N=n)
    assert all(np.allclose(x, y) for x, y in zip(args, ref))
    return loops


# Reproducer: LoopToMap refused NON_UNIQUE_WRITE on s in TSVC s258 (s kept from the last iteration with a[i] > 0)
def test_conditional_last_value_read_later_and_live_after_maps():
    assert run_last_stored(97) == 0


# Reproducer: a zero-trip loop must leave the scalar at its value from before the loop
def test_zero_trip_loop_keeps_scalar():
    assert run_last_stored(0) == 0


# Reproducer: a read before the store in the iteration sees the previous iteration's value: stays a loop
def test_read_before_the_store_stays_sequential():
    sdfg, loops = canonical_loops(read_before_store)
    assert loops == 1
    rng = np.random.default_rng(1)
    args = [rng.random(41) - 0.5, rng.random(41), rng.random(41), rng.random(41)]
    ref = [x.copy() for x in args]
    read_before_store_reference(*ref)
    sdfg(a=args[0], b=args[1], c=args[2], d=args[3], N=41)
    assert np.allclose(args[1], ref[1])


if __name__ == "__main__":
    test_conditional_last_value_read_later_and_live_after_maps()
    test_zero_trip_loop_keeps_scalar()
    test_read_before_the_store_stays_sequential()
