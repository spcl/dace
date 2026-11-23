# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""decide_known_guards: inside ``if c: A else: B``, a guard testing ``c`` again is decided.

Unswitching copies a body once per arm; a guard nested deeper than the hoist matched stays in both copies, and in the
else copy it never runs (icon_velocity_tendencies kept ``if (lextra_diffu)`` around a gather loop there, SciComp review
of 11-23).
"""

import numpy as np

import dace
from dace.sdfg.state import ConditionalBlock
from dace.transformation.interstate.move_loop_invariant_if_up import decide_known_guards

C = dace.symbol("C", dace.int64)


@dace.program
def retested_guard(b: dace.float64[16]):
    if C > 0:
        for i in range(16):
            if C > 0:
                b[i] = 1.0
    else:
        for i in range(16):
            if C > 0:
                b[i] = 2.0
            b[i] = b[i] + 3.0


def conditionals(sdfg: dace.SDFG) -> int:
    return sum(isinstance(r, ConditionalBlock) for r in sdfg.all_control_flow_regions(recursive=True))


def test_a_retested_guard_is_decided_in_both_arms():
    sdfg = retested_guard.to_sdfg(simplify=False)
    before = conditionals(sdfg)
    assert decide_known_guards(sdfg) == 2
    assert conditionals(sdfg) == before - 2
    sdfg.validate()
    for c, expected in ((1, 1.0), (0, 3.0)):
        b = np.zeros(16)
        sdfg(b=b, C=c)
        assert np.all(b == expected)


@dace.program
def reassigned_guard(b: dace.float64[16]):
    if C > 0:
        for i in range(16):
            b[i] = 1.0
    else:
        for C in range(16):
            if C > 0:
                b[C] = 2.0


def test_a_guard_whose_symbol_the_arm_assigns_stays():
    sdfg = reassigned_guard.to_sdfg(simplify=False)
    assert decide_known_guards(sdfg) == 0


@dace.program
def scalar_guard_with_elif(a: dace.int64[1], b: dace.float64[16], n: dace.int64):
    s = a[0]
    if s == 0:
        for i in range(16):
            b[i] = 1.0
        if s == 0:
            b[0] = 5.0
        elif n == 3:
            b[1] = 7.0
    else:
        if s == 0:
            b[0] = 5.0
        elif n == 3:
            b[1] = 7.0


def test_a_scalar_guard_with_elif_is_decided_in_both_arms():
    """qe_cegterg retested a one-element container (``if (arr19 == 0) { ... if (arr19 == 0) break; else if (i == 40)
    break; }``): the retest is decided by its first branch, and a false one leaves the ``elif`` (render review 11-23)."""
    sdfg = scalar_guard_with_elif.to_sdfg(simplify=False)
    assert decide_known_guards(sdfg) == 2
    sdfg.validate()
    for a, n, expected in ((0, 3, {0: 5.0}), (1, 3, {1: 7.0}), (1, 2, {})):
        b = np.zeros(16)
        sdfg(a=np.array([a]), b=b, n=n)
        want = np.full(16, 1.0) if a == 0 else np.zeros(16)
        for index, value in expected.items():
            want[index] = value
        assert np.all(b == want)


@dace.program
def written_scalar_guard(a: dace.int64[1], b: dace.float64[16]):
    s = a[0]
    if s == 0:
        s = 1
        if s == 0:
            b[0] = 5.0


def test_a_scalar_guard_the_arm_writes_stays():
    sdfg = written_scalar_guard.to_sdfg(simplify=False)
    assert decide_known_guards(sdfg) == 0


if __name__ == "__main__":
    test_a_retested_guard_is_decided_in_both_arms()
    test_a_guard_whose_symbol_the_arm_assigns_stays()
    test_a_scalar_guard_with_elif_is_decided_in_both_arms()
    test_a_scalar_guard_the_arm_writes_stays()
