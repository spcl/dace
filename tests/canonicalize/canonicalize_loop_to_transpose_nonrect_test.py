# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""``LoopToTranspose`` must refuse a non-rectangular nest.

The lift reads each loop's ``(init, last, stride)`` and rewrites the nest into a strided View pair. That
is only valid over a rectangular iteration space: a triangular one (``for j in range(i+1, M)``) is not a
transposable box, and lifting it splices the outer iterator into the View subsets
(``B[0:M-1, _loop_it_0+1:M]``). ``add_view`` then auto-registers that iterator in ``sdfg.symbols``, so it
reaches ``arglist()`` and the call fails with ``Missing program argument "_loop_it_0"`` -- a stranded
symbol standing in for a wrong answer.
"""
import numpy as np
import pytest

import dace
from dace.transformation.passes.canonicalize import canonicalize

M = dace.symbol('M')


@dace.program
def triangular_copy(A: dace.float64[M, M], B: dace.float64[M, M]):
    for i in range(0, M - 1):
        for j in range(i + 1, M):
            A[j, i] = B[i, j]


def test_refuses_non_rectangular_nest():
    sdfg = triangular_copy.to_sdfg(simplify=False)
    canonicalize(sdfg, validate=True, validate_all=False)

    # A stranded iterator would show up as a free symbol, hence as a required argument.
    assert not [s for s in sdfg.symbols if str(s).startswith('_loop_it')], \
        f'loop iterator stranded into sdfg.symbols: {sorted(map(str, sdfg.symbols))}'

    m = 6
    b = np.arange(m * m, dtype=np.float64).reshape(m, m)
    got = np.zeros((m, m))
    sdfg(A=got, B=b.copy(), M=m)

    want = np.zeros((m, m))
    for i in range(0, m - 1):
        for j in range(i + 1, m):
            want[j, i] = b[i, j]
    assert np.array_equal(got, want), 'triangular copy was lifted to a transpose'


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
