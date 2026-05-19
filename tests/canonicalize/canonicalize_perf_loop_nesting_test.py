# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
""" Perfect-loop-nesting of a guarded imperfect nest.

    ``if c: for i: { for j: b[i,j]=...; c[i]=... }`` -- the ``c[i]`` write
    lives in the ``i`` loop but not in its own innermost ``j``-style loop
    (an imperfect nest). The intended canonical form wraps every bare
    state/tasklet in a trivial single-iteration loop so the nest is
    perfectly nested, after which the guard can be pushed into -- and
    duplicated across -- every loop. Canonicalization is always
    value-preserving (guard taken and not-taken); the structural ideal
    (no surviving top-level guard -- it has been moved/duplicated inside)
    is tracked as a strict xfail until the trivial-loop-wrap + move-if
    duplication is implemented.
"""
import numpy as np
import pytest

import dace
from dace.sdfg.state import ConditionalBlock
from dace.transformation.passes.canonicalize import canonicalize

N = dace.symbol('N')
M = dace.symbol('M')


@dace.program
def guard_over_imperfect_nest(a: dace.float64[N, M], b: dace.float64[N, M], c: dace.float64[N], act: dace.int32[1]):
    if act[0] > 0:
        for i in dace.map[0:N]:
            for j in dace.map[0:M]:
                b[i, j] = a[i, j] * 2.0
            c[i] = a[i, 0] + 1.0  # bare: in loop i, not in a j-loop


def _top_level_conds(sdfg):
    return [x for x in sdfg.nodes() if isinstance(x, ConditionalBlock)]


def _oracle(a, n, m, av):
    eb, ec = np.full((n, m), 9.0), np.full(n, 9.0)
    if av > 0:
        for i in range(n):
            for j in range(m):
                eb[i, j] = a[i, j] * 2.0
            ec[i] = a[i, 0] + 1.0
    return eb, ec


@pytest.mark.parametrize('av', [1, 0])
def test_guard_over_imperfect_nest_is_value_preserving(av):
    """Canonicalization of a guarded imperfect nest is value-preserving for
    the guard taken and not-taken."""
    n, m = 6, 5
    a = np.random.rand(n, m)
    eb, ec = _oracle(a, n, m, av)

    sdfg = guard_over_imperfect_nest.to_sdfg(simplify=True)
    canonicalize(sdfg, validate=True)

    ob, oc = np.full((n, m), 9.0), np.full(n, 9.0)
    sdfg(a=a.copy(), b=ob, c=oc, act=np.array([av], np.int32), N=n, M=m)
    assert np.allclose(ob, eb) and np.allclose(oc, ec), f"mismatch act={av}"


@pytest.mark.xfail(strict=True,
                   reason="Bare states (the imperfect-nest c[i]=... write) are not wrapped in "
                   "trivial single-iteration loops, so the loop-invariant guard is not pushed "
                   "into / duplicated across every loop -- it survives as a top-level "
                   "ConditionalBlock. Needs the trivial-loop-wrap perfect-nesting + move-if "
                   "duplication.")
def test_guard_over_imperfect_nest_guard_moved_inside():
    """Ideal: after trivial-loop-wrap perfect-nesting the guard is moved
    into / duplicated across the loops, leaving no top-level guard."""
    n, m = 6, 5
    a = np.random.rand(n, m)
    sdfg = guard_over_imperfect_nest.to_sdfg(simplify=True)
    canonicalize(sdfg, validate=True)
    assert not _top_level_conds(sdfg)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
