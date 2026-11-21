# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Loop unswitching in ``MoveLoopInvariantIfUp``: an invariant ``if``/``else`` (or ``elif``) leaves its loop with one
copy of the loop per branch."""

import numpy as np

import dace
from dace.sdfg.state import ConditionalBlock, LoopRegion
from dace.transformation.interstate.move_loop_invariant_if_up import MoveLoopInvariantIfUp

N = dace.symbol("N")
C = dace.symbol("C")


@dace.program
def if_else_in_loop(a: dace.float64[N], b: dace.float64[N]):
    for i in range(N):
        if C > 0:
            b[i] = a[i] + 1.0
        else:
            b[i] = a[i] - 1.0


@dace.program
def three_way_in_loop(a: dace.float64[N], b: dace.float64[N]):
    for i in range(N):
        if C > 1:
            b[i] = a[i] * 2.0
        elif C > 0:
            b[i] = a[i] * 3.0
        else:
            b[i] = a[i] * 4.0


@dace.program
def variant_if_else_in_loop(a: dace.float64[N], b: dace.float64[N]):
    for i in range(N):
        if a[i] > 0:
            b[i] = a[i] + 1.0
        else:
            b[i] = a[i] - 1.0


def top_level_blocks(sdfg: dace.SDFG) -> list:
    return [type(b).__name__ for b in sdfg.nodes()]


def loops_per_branch(sdfg: dace.SDFG) -> list[int]:
    guard = next(b for b in sdfg.nodes() if isinstance(b, ConditionalBlock))
    return [sum(isinstance(r, LoopRegion) for r in branch.all_control_flow_regions()) for _, branch in guard.branches]


def run(sdfg: dace.SDFG, c: int) -> np.ndarray:
    a = np.linspace(-2.0, 2.0, 9)
    b = np.zeros(9)
    sdfg(a=a, b=b, N=9, C=c)
    return b


def test_an_invariant_if_else_unswitches_into_one_loop_per_branch():
    sdfg = if_else_in_loop.to_sdfg(simplify=True)
    reference = {c: run(if_else_in_loop.to_sdfg(simplify=True), c) for c in (1, 0)}
    assert MoveLoopInvariantIfUp().apply_pass(sdfg, {}) == 1
    sdfg.validate()
    assert "ConditionalBlock" in top_level_blocks(sdfg) and "LoopRegion" not in top_level_blocks(sdfg)
    assert loops_per_branch(sdfg) == [1, 1]
    for c, want in reference.items():
        assert np.allclose(run(sdfg, c), want)


def test_an_invariant_elif_chain_unswitches_every_branch():
    sdfg = three_way_in_loop.to_sdfg(simplify=True)
    reference = {c: run(three_way_in_loop.to_sdfg(simplify=True), c) for c in (2, 1, 0)}
    assert MoveLoopInvariantIfUp().apply_pass(sdfg, {})
    sdfg.validate()
    for c, want in reference.items():
        assert np.allclose(run(sdfg, c), want)


def test_a_condition_reading_the_loop_data_stays_inside():
    sdfg = variant_if_else_in_loop.to_sdfg(simplify=True)
    assert MoveLoopInvariantIfUp().apply_pass(sdfg, {}) is None


if __name__ == "__main__":
    test_an_invariant_if_else_unswitches_into_one_loop_per_branch()
    test_an_invariant_elif_chain_unswitches_every_branch()
    test_a_condition_reading_the_loop_data_stays_inside()
