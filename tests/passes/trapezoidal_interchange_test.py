# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""A triangular / trapezoidal nest must still be reordered for locality.

Two gates kept every LLR nest in whatever order it was written. ``MapInterchange`` refused any nest
whose inner range mentions the outer parameter -- which is every triangular and trapezoidal nest --
and ``MinimizeStridePermutation`` refused any nest whose strides are not concrete numbers, which on
symbolic shapes is all of them. TSVC ``s1232`` therefore kept a stride-``LEN_2D`` innermost loop.
"""
import numpy as np
import pytest

import dace
from dace.sdfg import nodes
from dace.transformation.dataflow.map_interchange import MapInterchange
from dace.transformation.passes.canonicalize import pipeline as canon
from dace.transformation.passes.minimize_stride_permutation import (UndecidableStride, stride_difference_sign)

N = dace.symbol('N', dtype=dace.int64)
V = dace.symbol('V', dtype=dace.int64)


def nest(sdfg: dace.SDFG):
    """``(outer_entry, inner_entry)`` of the single two-level map nest."""
    for state in sdfg.all_states():
        scope = state.scope_dict()
        for outer in [n for n in state.nodes() if isinstance(n, nodes.MapEntry) and scope.get(n) is None]:
            inners = [n for n in state.nodes() if isinstance(n, nodes.MapEntry) and scope.get(n) is outer]
            if inners:
                return outer, inners[0]
    return None, None


def test_the_contiguous_parameter_ends_up_innermost():
    """``aa[i, j]`` is row-major, so ``j`` (stride 1) belongs inside ``i`` (stride N)."""

    @dace.program
    def trapez(aa: dace.float64[N, N], bb: dace.float64[N, N]):
        for j in range(N):
            for i in range(j * V, N):
                aa[i, j] = bb[i, j] + 1.0

    sdfg = trapez.to_sdfg(simplify=False)
    canon.canonicalize(sdfg)
    outer, inner = nest(sdfg)
    assert outer is not None, 'the nest did not survive canonicalization'
    # The inner parameter must be the one indexing the LAST (contiguous) axis.
    subsets = [
        e.data.subset for state in sdfg.all_states() for e in state.edges()
        if e.data is not None and e.data.data == 'aa' and e.data.subset is not None and len(e.data.subset) == 2
    ]
    contiguous = {str(s) for sub in subsets for s in sub.ranges[1][0].free_symbols}
    assert inner.map.params[0] in contiguous, (
        f'innermost parameter {inner.map.params[0]} does not index the contiguous axis '
        f'(that axis is indexed by {contiguous})')


@pytest.mark.parametrize('n,v', [(33, 1), (33, 4), (64, 3), (17, 7), (12, 20)])
def test_the_iteration_set_is_preserved(n, v):
    """The bound rewrite is a re-derivation, not an approximation: same elements, same values."""

    @dace.program
    def trapez(aa: dace.float64[N, N], bb: dace.float64[N, N]):
        for j in range(N):
            for i in range(j * V, N):
                aa[i, j] = bb[i, j] + 1.0

    rng = np.random.default_rng(n * 100 + v)
    bb = rng.random((n, n))
    want = np.zeros((n, n))
    for j in range(n):
        for i in range(j * v, n):
            want[i, j] = bb[i, j] + 1.0
    got = np.zeros((n, n))
    sdfg = trapez.to_sdfg(simplify=False)
    canon.canonicalize(sdfg)
    sdfg(aa=got, bb=bb, N=n, V=v)
    assert np.allclose(got, want)


def test_a_plain_swap_still_refuses_a_trapezoid():
    """``transform_bounds`` is opt-in: the default contract is unchanged."""

    @dace.program
    def trapez(aa: dace.float64[N, N], bb: dace.float64[N, N]):
        for j in dace.map[0:N]:
            for i in dace.map[V * j:N]:
                aa[i, j] = bb[i, j] + 1.0

    sdfg = trapez.to_sdfg(simplify=True)
    outer, inner = nest(sdfg)
    assert not MapInterchange.can_be_applied_to(sdfg, outer_map_entry=outer, inner_map_entry=inner)
    assert MapInterchange.can_be_applied_to(sdfg,
                                            options={'transform_bounds': True},
                                            outer_map_entry=outer,
                                            inner_map_entry=inner)


def test_stride_comparison_uses_the_shape_contract():
    """An extent is at least one, so a symbolic stride is not an undecidable one."""
    assert stride_difference_sign('1', 'N') == -1
    assert stride_difference_sign('N', '1') == 1
    assert stride_difference_sign('M', 'N*M') == -1
    assert stride_difference_sign('N', 'N') == 0
    with pytest.raises(UndecidableStride):
        stride_difference_sign('N*M', 'M + K')


if __name__ == '__main__':
    test_the_contiguous_parameter_ends_up_innermost()
    test_the_iteration_set_is_preserved(33, 4)
    test_a_plain_swap_still_refuses_a_trapezoid()
    test_stride_comparison_uses_the_shape_contract()
