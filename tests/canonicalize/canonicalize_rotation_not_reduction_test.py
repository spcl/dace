# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""A loop-carried ROTATION is not a reduction.

``LoopToReduce`` walks back from a write to a candidate accumulator, through transient
pass-throughs, looking for the tasklet that produced the stored value. The walk could arrive over
an EMPTY memlet -- a sequencing edge that carries no data and only orders two nodes -- and still
treat the tasklet as the producer. TSVC s255's two-deep rotation reaches it exactly that way::

    x = b[N-1]; y = b[N-2]
    for i in range(N):
        a[i] = (b[i] + x + y) * 0.333
        y = x          # OVERWRITE, not accumulation
        x = b[i]

``y`` is read by the sum and written in the same iteration, and ``_Add_ -[]-> x -[x -> y]-> y``
looks like ``y = y + <sum>``. Folding that into a WCR turns the copy into an accumulation AND drops
``y`` from the sum, silently computing ``(b[i] + b[i-1]) * 0.333`` -- every element wrong, with no
error anywhere.
"""
import numpy as np
import pytest

import dace
from dace.transformation.passes.canonicalize import canonicalize

N = dace.symbol('N')


@dace.program
def rotate_two_deep(a: dace.float64[N], b: dace.float64[N]):
    x = b[N - 1]
    y = b[N - 2]
    for i in range(N):
        a[i] = (b[i] + x + y) * 0.333
        y = x
        x = b[i]


def reference(b):
    """``x`` trails one iteration, ``y`` two, both seeded from the end of ``b``."""
    n = len(b)
    out = np.empty(n)
    x, y = b[n - 1], b[n - 2]
    for i in range(n):
        out[i] = (b[i] + x + y) * 0.333
        y = x
        x = b[i]
    return out


def test_rotation_keeps_both_carried_values():
    n = 64
    rng = np.random.default_rng(1234)
    b = rng.random(n)
    want = reference(b)

    sdfg = rotate_two_deep.to_sdfg(simplify=True)
    sdfg.name = 'rotate_two_deep_canon'
    canonicalize(sdfg, validate=True, peel_limit=4)

    got = np.zeros(n)
    sdfg.compile()(a=got, b=b.copy(), N=n)
    assert np.allclose(got, want, rtol=0, atol=0), 'a carried value was dropped from the sum'


def test_the_rotation_is_not_turned_into_an_accumulation():
    """Structural half: no WCR anywhere. A rotation has nothing to reduce, so any WCR the pipeline
    mints here is the copy being mistaken for an accumulator."""
    sdfg = rotate_two_deep.to_sdfg(simplify=True)
    sdfg.name = 'rotate_two_deep_struct'
    canonicalize(sdfg, validate=True, peel_limit=4)

    wcrs = [(state.label, e.data.data) for sd in sdfg.all_sdfgs_recursive() for state in sd.states()
            for e in state.edges() if e.data is not None and e.data.wcr is not None]
    assert not wcrs, f'rotation lifted to a reduction: {wcrs}'


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
