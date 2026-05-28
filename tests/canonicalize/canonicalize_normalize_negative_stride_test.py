# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Tests for :class:`NormalizeNegativeStride`."""
import numpy as np
import pytest

import dace
from dace.sdfg.state import LoopRegion
from dace.transformation.passes.analysis import loop_analysis
from dace.transformation.passes.canonicalize.normalize_negative_stride import NormalizeNegativeStride

N = dace.symbol('N')


def _loops(sdfg):
    return [r for r in sdfg.all_control_flow_regions() if isinstance(r, LoopRegion) and r.loop_variable]


def test_reverse_loop_normalizes_to_positive_stride():
    """``for i in range(N-2, -1, -1): a[i+1] = a[i] + b[i]`` rewrites to a
    positive-stride loop driving ``_loop_pos_<N>``, with ``i`` rebound on
    each iteration so the body's memlets stay valid and iteration order
    (and therefore semantics) is preserved."""

    @dace.program
    def reverse_recurrence(a: dace.float64[N], b: dace.float64[N]):
        for i in range(N - 2, -1, -1):
            a[i + 1] = a[i] + b[i]

    sdfg = reverse_recurrence.to_sdfg(simplify=True)
    res = NormalizeNegativeStride().apply_pass(sdfg, {})
    sdfg.validate()
    assert res == 1
    loops = _loops(sdfg)
    assert len(loops) == 1
    loop = loops[0]
    assert loop.loop_variable.startswith('_loop_pos_')
    stride = loop_analysis.get_loop_stride(loop)
    assert stride == 1, f"expected positive stride, got {stride}"


def test_reverse_recurrence_value_preserving():
    """End-to-end: the normalized loop produces the same numeric result as the
    original Python reference (iteration order preserved by the rebinding)."""

    @dace.program
    def reverse_recurrence(a: dace.float64[N], b: dace.float64[N]):
        for i in range(N - 2, -1, -1):
            a[i + 1] = a[i] + b[i]

    n = 8
    rng = np.random.default_rng(112)
    a0 = rng.standard_normal(n)
    b0 = rng.standard_normal(n)

    ref = a0.copy()
    for i in range(n - 2, -1, -1):
        ref[i + 1] = ref[i] + b0[i]

    sdfg = reverse_recurrence.to_sdfg(simplify=True)
    NormalizeNegativeStride().apply_pass(sdfg, {})
    sdfg.validate()
    got = a0.copy()
    sdfg(a=got, b=b0.copy(), N=n)
    assert np.allclose(got, ref)


def test_positive_stride_loop_is_noop():
    """A loop with positive stride is left untouched."""

    @dace.program
    def forward(a: dace.float64[N], b: dace.float64[N]):
        for i in range(N - 1):
            a[i + 1] = a[i] + b[i]

    sdfg = forward.to_sdfg(simplify=True)
    res = NormalizeNegativeStride().apply_pass(sdfg, {})
    assert res is None


def test_strided_negative_loop_normalizes():
    """``for i in range(N-1, -1, -2)`` (stride -2) also rewrites."""

    @dace.program
    def reverse_stride2(a: dace.float64[N], b: dace.float64[N]):
        for i in range(N - 1, -1, -2):
            a[i] = b[i] + 1.0

    sdfg = reverse_stride2.to_sdfg(simplify=True)
    res = NormalizeNegativeStride().apply_pass(sdfg, {})
    sdfg.validate()
    assert res == 1
    loops = _loops(sdfg)
    assert len(loops) == 1
    assert loop_analysis.get_loop_stride(loops[0]) == 1


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
