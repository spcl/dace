# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""A nested program's shape symbols are solved from the caller's argument shapes. After two stride-2 stages a
caller's tensor is ``floor(floor(h / 4 - 1 / 4) / 2) + 1`` wide; an extent comparison may fold that to
``floor(h / 8 - 1 / 8) + 1``, but folded in the equations the solver gets, it left sympy no
``floor(s / 2 - 1 / 2) + 1`` to match the callee's output against (resnet101's stride-2 convolutions)."""

import numpy as np

import dace

H = dace.symbol("H", dtype=dace.int64, positive=True)
W = dace.symbol("W", dtype=dace.int64, positive=True)
S = dace.symbol("S", dtype=dace.int64, positive=True)
T = dace.symbol("T", dtype=dace.int64, positive=True)


@dace.program
def subsample(x: dace.float64[2, S, T], out: dace.float64[2, (S - 1) // 2 + 1, (T - 1) // 2 + 1]):
    """A stride-2 stage sized by its input's extents ``S, T``: every output element is its input's at twice the index."""
    for i in range((S - 1) // 2 + 1):
        for j in range((T - 1) // 2 + 1):
            out[:, i, j] = x[:, 2 * i, 2 * j]


@dace.program
def second_stage(
    x: dace.float64[2, H, W], out: dace.float64[2, ((H - 1) // 4 + 1 - 1) // 2 + 1, ((W - 1) // 4 + 1 - 1) // 2 + 1]
):
    first = np.empty((2, (H - 1) // 4 + 1, (W - 1) // 4 + 1), dtype=np.float64)
    for i in range((H - 1) // 4 + 1):
        for j in range((W - 1) // 4 + 1):
            first[:, i, j] = x[:, 4 * i, 4 * j]
    subsample(first, out)


def test_a_call_after_two_strided_stages_infers_its_symbols():
    second_stage.to_sdfg().validate()
    x = np.arange(2.0 * 19.0 * 13.0).reshape(2, 19, 13).copy()
    out = np.zeros(x[:, ::8, ::8].shape)
    second_stage(x=x, out=out, H=19, W=13)
    assert np.array_equal(out, x[:, ::8, ::8]), out


if __name__ == "__main__":
    test_a_call_after_two_strided_stages_infers_its_symbols()
