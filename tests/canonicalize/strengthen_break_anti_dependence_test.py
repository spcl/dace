# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Strengthening test for BreakAntiDependence: a same-index (offset-0) read that
consumes a value written by an EARLIER STATE of the same iteration must NOT be
redirected to the pre-loop snapshot.

The whole-array snapshot rename used to move EVERY pure-read node of the renamed
array to the snapshot. A loop body split across states -- e.g. a later branch-body
state reading the ``a[i]`` the loop just produced -- has that read as a pure-read
node (in-degree 0 within its own state) even though it must read the freshly
written value. Redirecting it to the pre-loop snapshot reads the stale original and
silently corrupts the result. Bit-exact vs a sequential numpy oracle."""
import numpy as np

import dace
from dace.transformation.passes import BreakAntiDependence

N = dace.symbol('N')


def test_break_anti_dependence_offset0_read_across_state_not_snapshotted():
    """``a[i] = a[i+1] + b[i]`` (read-ahead WAR, renamable) followed by a branch that
    reads the FRESH ``a[i]`` into ``c``. The read-ahead ``a[i+1]`` may move to the
    snapshot, but the same-index ``a[i]`` reads (in the branch states) must stay on
    the live array -- they consume this iteration's write, not the pre-loop value."""

    @dace.program
    def kern(a: dace.float64[N], b: dace.float64[N], c: dace.float64[N]):
        for i in range(N - 1):
            a[i] = a[i + 1] + b[i]      # read-ahead WAR on a -> array is renamable
            if b[i] > 0.5:
                c[i] = a[i] * 2.0       # reads FRESH a[i], a separate branch-body state
            else:
                c[i] = a[i] * 3.0

    sdfg = kern.to_sdfg(simplify=True)
    assert BreakAntiDependence().apply_pass(sdfg, {}) == 1
    sdfg.validate()

    n = 12
    rng = np.random.default_rng(3)
    a = rng.random(n)
    b = rng.random(n)

    ra = a.copy()
    rc = np.zeros(n)
    for i in range(n - 1):
        ra[i] = ra[i + 1] + b[i]                      # read-ahead: reads ORIGINAL a[i+1]
        rc[i] = ra[i] * 2.0 if b[i] > 0.5 else ra[i] * 3.0  # reads the FRESH a[i]

    ao = a.copy()
    co = np.zeros(n)
    sdfg(a=ao, b=b.copy(), c=co, N=n)
    assert np.allclose(ao, ra), f'a max-diff {np.abs(ao - ra).max()}'
    assert np.allclose(co, rc), f'c max-diff {np.abs(co - rc).max()}'


if __name__ == '__main__':
    test_break_anti_dependence_offset0_read_across_state_not_snapshotted()
