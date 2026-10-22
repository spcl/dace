# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""A ``break`` leaves only the loop it belongs to: an outer loop whose inner loop breaks is still a map, a loop with
a ``break`` of its own is not (ls3df_scf's fragment loop stayed sequential over its Lanczos loop's early exit)."""

import numpy as np

import dace
from dace.sdfg.state import LoopRegion
from dace.transformation.interstate.loop_to_map import LoopToMap

N = dace.symbol("N")
M = dace.symbol("M")


@dace.program
def prefix_until_negative(A: dace.float64[N, M], out: dace.float64[N]):
    for i in range(N):
        s = 0.0
        for j in range(M):
            if A[i, j] < 0.0:
                break
            s = s + A[i, j]
        out[i] = s


@dace.program
def stops_at_first_negative(A: dace.float64[N], out: dace.float64[N]):
    for i in range(N):
        if A[i] < 0.0:
            break
        out[i] = A[i] * 2.0


def outer_loop(sdfg: dace.SDFG) -> LoopRegion:
    return next(r for r in sdfg.nodes() if isinstance(r, LoopRegion))


def test_an_inner_break_leaves_the_outer_loop_mappable():
    sdfg = prefix_until_negative.to_sdfg(simplify=True)
    xform = LoopToMap()
    loop = outer_loop(sdfg)
    xform.setup_match(sdfg, sdfg.cfg_id, -1, {LoopToMap.loop: sdfg.node_id(loop)}, 0, override=True)
    assert xform.can_be_applied(sdfg, 0, sdfg), xform.last_refusal_reason
    xform.apply(sdfg, sdfg)
    rng = np.random.default_rng(0)
    a = rng.random((7, 9)) - 0.2
    out = np.zeros(7)
    sdfg(A=a, out=out, N=7, M=9)
    want = [sum(row[: np.argmax(row < 0)] if (row < 0).any() else row) for row in a]
    assert np.allclose(out, want)


def test_a_break_of_the_loop_itself_is_refused():
    sdfg = stops_at_first_negative.to_sdfg(simplify=True)
    xform = LoopToMap()
    loop = outer_loop(sdfg)
    xform.setup_match(sdfg, sdfg.cfg_id, -1, {LoopToMap.loop: sdfg.node_id(loop)}, 0, override=True)
    assert not xform.can_be_applied(sdfg, 0, sdfg)
    assert "BreakBlock" in xform.last_refusal_reason


if __name__ == "__main__":
    test_an_inner_break_leaves_the_outer_loop_mappable()
    test_a_break_of_the_loop_itself_is_refused()
