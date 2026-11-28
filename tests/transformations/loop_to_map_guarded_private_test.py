# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""LoopToMap privatizes a loop-local value that an iteration reads only on paths where it wrote it first.

Three shapes from ICON graupel, each a loop the column loop holds:
- a flag reset at the top of the iteration and set together with the value, the value read only under the flag
  (``if_cond = .false.; IF (...) THEN sig = ...; if_cond = .true.`` then ``IF (if_cond) ... IF (sig)``);
- a value one arm stores under a condition and the other arm stores outright, read later under that condition
  (``ice_dep`` under ``IF (sig)`` in ``IF (t < tmelt)``, ``ice_dep = 0`` in its ``ELSE``);
- a value a nested loop stores on its first trip, read on every later trip (``IF (k == kstart) vt = 0``).
"""

import numpy as np

import dace
from dace.sdfg.state import LoopRegion
from dace.transformation.interstate import LoopToMap

N = dace.symbol("N")
M = dace.symbol("M")


def outer_loop(sdfg: dace.SDFG) -> LoopRegion:
    return next(r for r in sdfg.nodes() if isinstance(r, LoopRegion))


def lifts(sdfg: dace.SDFG) -> bool:
    xf = LoopToMap()
    xf.loop = outer_loop(sdfg)
    return xf.can_be_applied(sdfg, 0, sdfg)


def lift_and_run(prog, reference, **arrays) -> None:
    sdfg = prog.to_sdfg(simplify=True)
    assert lifts(sdfg)
    LoopToMap.apply_to(sdfg, loop=outer_loop(sdfg))
    expected = {name: value.copy() for name, value in arrays.items()}
    reference(**expected)
    sdfg(**arrays, **{str(s): arrays["a"].shape[i] for i, s in enumerate(prog.f.__annotations__["a"].shape)})
    for name in arrays:
        assert np.allclose(arrays[name], expected[name]), name


@dace.program
def flag_guarded(a: dace.float64[N], b: dace.float64[N]):
    for i in range(N):
        f = 0
        if a[i] > 0:
            s = a[i] * 2.0
            f = 1
        if f != 0:
            b[i] = s


def flag_guarded_reference(a, b):
    for i in range(a.shape[0]):
        if a[i] > 0:
            b[i] = a[i] * 2.0


# Reproducer: LoopToMap refused graupel's column loop on sig, set with if_cond and read under it (dace-fortran graupel)
def test_a_value_read_only_under_the_flag_set_with_it_is_private():
    rng = np.random.default_rng(0)
    lift_and_run(flag_guarded, flag_guarded_reference, a=rng.standard_normal(16), b=np.zeros(16))


@dace.program
def flag_set_before_value(a: dace.float64[N], b: dace.float64[N]):
    for i in range(N):
        f = 0
        if a[i] > 0:
            f = 1
            s = a[i] * 2.0
        if f != 0:
            b[i] = s


# Reproducer: graupel's arm set if_cond before sig, so the flag held the stores before sig (dace-fortran graupel,
# LoopToMap sweep)
def test_a_flag_set_ahead_of_the_value_in_its_branch_still_guards_it():
    rng = np.random.default_rng(3)
    lift_and_run(flag_set_before_value, flag_guarded_reference, a=rng.standard_normal(16), b=np.zeros(16))


@dace.program
def flag_set_before_the_loop(a: dace.float64[N], b: dace.float64[N]):
    f = 0
    for i in range(N):
        if a[i] > 0:
            s = a[i] * 2.0
            f = 1
        if f != 0:
            b[i] = s


def test_a_flag_the_iteration_does_not_reset_keeps_the_value_carried():
    # A later iteration with a[i] <= 0 still sees f set and reads the s an earlier iteration stored.
    assert not lifts(flag_set_before_the_loop.to_sdfg(simplify=True))


@dace.program
def arm_guarded(a: dace.float64[N], c: dace.float64[N], b: dace.float64[N]):
    for i in range(N):
        g = c[i] > 0
        if a[i] > 0:
            if g:
                t = a[i] + c[i]
        else:
            t = 0.0
        if g:
            b[i] = t


def arm_guarded_reference(a, c, b):
    for i in range(a.shape[0]):
        if c[i] > 0:
            b[i] = a[i] + c[i] if a[i] > 0 else 0.0


# Reproducer: LoopToMap refused graupel's column loop on ice_dep, stored under IF (sig) in one arm (dace-fortran graupel)
def test_a_value_both_arms_store_once_the_condition_holds_is_private():
    rng = np.random.default_rng(1)
    lift_and_run(
        arm_guarded, arm_guarded_reference, a=rng.standard_normal(16), c=rng.standard_normal(16), b=np.zeros(16)
    )


@dace.program
def first_trip_reset(a: dace.float64[N, M], b: dace.float64[N, M]):
    for i in range(N):
        for k in range(M):
            if k == 0:
                t = 0.0
            t = t + a[i, k]
            b[i, k] = t


def first_trip_reset_reference(a, b):
    b[:] = np.cumsum(a, axis=1)


# Reproducer: LoopToMap refused graupel's column loop on vt, reset by IF (k == kstart) in the inner loop (dace-fortran graupel)
def test_a_value_the_inner_loop_resets_on_its_first_trip_is_private():
    rng = np.random.default_rng(2)
    lift_and_run(first_trip_reset, first_trip_reset_reference, a=rng.standard_normal((8, 5)), b=np.zeros((8, 5)))


if __name__ == "__main__":
    test_a_value_read_only_under_the_flag_set_with_it_is_private()
    test_a_flag_set_ahead_of_the_value_in_its_branch_still_guards_it()
    test_a_flag_the_iteration_does_not_reset_keeps_the_value_carried()
    test_a_value_both_arms_store_once_the_condition_holds_is_private()
    test_a_value_the_inner_loop_resets_on_its_first_trip_is_private()
