# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""A strided slice over promoted scalar bounds broadcasts against the unit-stride slice of the same length.

``out[lo * S + k : hi * S + k + 1 : S]`` has ``ceiling((S*hi - S*lo + 1)/S)`` elements, which is ``hi - lo + 1`` for
integer ``hi``, ``lo`` and a positive integer ``S``. A registry symbol is a bare name, so that integrality and sign live
in the SDFG's facts, not on the symbol; comparing the two extents without them refused the broadcast
(``conv_transpose3d_avg_pool_clamp_softmax_multiply``: the transposed convolution's strided scatter).
"""

import numpy as np

import dace

N = dace.symbol("N", dace.int64, positive=True)
S = dace.symbol("S", dace.int64, positive=True)


@dace.program
def strided_accumulate(out: dace.float64[N], x: dace.float64[N]):
    for k in range(3):
        lo = max(0, k - 1)
        hi = min(N // S - 1, k + 2)
        if lo > hi:
            continue
        out[lo * S + k : hi * S + k + 1 : S] = out[lo * S + k : hi * S + k + 1 : S] + x[lo : hi + 1]


def test_strided_slice_of_promoted_bounds_broadcasts_against_its_unit_stride_twin():
    n, s = 20, 3
    x = np.arange(n, dtype=np.float64)
    ref = np.zeros(n)
    for k in range(3):
        lo, hi = max(0, k - 1), min(n // s - 1, k + 2)
        if lo > hi:
            continue
        ref[lo * s + k : hi * s + k + 1 : s] += x[lo : hi + 1]
    out = np.zeros(n)
    strided_accumulate(out=out, x=x.copy(), N=n, S=s)
    assert np.allclose(out, ref)


if __name__ == "__main__":
    test_strided_slice_of_promoted_bounds_broadcasts_against_its_unit_stride_twin()
