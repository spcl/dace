# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""A reduction into ONE element of a larger array must tile, not be refused.

The tile-vectorizer's "no loose WCR in the region to be tiled" precondition used to accept a
map-exit WCR only when the accumulator was a scalar / length-1 array at the TOP level. Every other
reduction shape the frontend emits -- ``mean[j] (+)= data[i, j]`` over a collapsed ``[j, i]`` map
(polybench ``correlation``), ``tmp (+)= A[k, i] * B[k, j]`` under a nested scope (polybench
``trmm``) -- was refused, and since the refusal is per-KERNEL those kernels stayed entirely scalar.

Both are the SAME lifted-reduction boundary: the slot does not vary with the map's INNERMOST
parameter, the dim the widener turns into lanes, so within one tile there is exactly one
accumulator. The body folds the lanes to a single partial (``TileReduce``) and the boundary
resolves one accumulation per tile -- nesting and the accumulator's rank change nothing about that.

Pinned here:

* the array-slot reduction TILES (a ``__tile_main`` map strided by the width, no refusal warning);
* it is VALUE-PRESERVING against NumPy;
* the shapes that must STAY refused keep their refusal -- a per-lane scatter (slot indexed BY the
  innermost param) and an accumulator array also READ inside the map (lu's
  ``A[i, j] (+)= -A[i, k] * A[k, j]``, a recurrence the tile staging cannot separate).
"""
import warnings

import numpy as np
import pytest

import dace
from dace.memlet import Memlet
from dace.transformation.passes.vectorization.config import VectorizeConfig
from dace.transformation.passes.vectorization.enums import ISA
from dace.transformation.passes.vectorization.utils.pass_invariants import (no_wcr_in_map_body,
                                                                            no_wcr_inside_nested_sdfgs)
from dace.transformation.passes.vectorization.vectorize_multi_dim import VectorizeCPUMultiDim

M = dace.symbol('M')
N = dace.symbol('N')

WIDTH = 8


def _row_sum_sdfg(name: str = 'row_sum') -> dace.SDFG:
    """``mean[j] (+)= data[i, j]`` as one collapsed ``[j, i]`` map with a map-exit WCR.

    The accumulator slot ``mean[j]`` is fixed for the whole innermost (``i``) dim, so it is a
    reduction per ``j`` -- exactly the shape the guard used to refuse because ``mean`` has more
    than one element.
    """
    sdfg = dace.SDFG(name)
    sdfg.add_array('data', (N, M), dace.float64)
    sdfg.add_array('mean', (M, ), dace.float64)
    state = sdfg.add_state()
    src = state.add_access('data')
    sink = state.add_access('mean')
    me, mx = state.add_map('comp_mean', dict(j='0:M', i='0:N'))
    tlet = state.add_tasklet('t', {'_a'}, {'_o'}, '_o = _a')
    state.add_memlet_path(src, me, tlet, dst_conn='_a', memlet=Memlet('data[i, j]'))
    mx.add_in_connector('IN_mean')
    mx.add_out_connector('OUT_mean')
    inner = Memlet('mean[j]')
    inner.wcr = 'lambda x, y: x + y'
    state.add_edge(tlet, '_o', mx, 'IN_mean', inner)
    outer = Memlet('mean[0:M]')
    outer.wcr = 'lambda x, y: x + y'
    state.add_edge(mx, 'OUT_mean', sink, None, outer)
    sdfg.validate()
    return sdfg


def _scatter_sdfg() -> dace.SDFG:
    """``out[i] (+)= a[i]`` -- the slot IS the innermost map param, so every lane targets a
    DIFFERENT element: a per-lane scatter with no single accumulator per tile."""
    sdfg = dace.SDFG('lane_scatter')
    sdfg.add_array('a', (64, ), dace.float64)
    sdfg.add_array('out', (64, ), dace.float64)
    state = sdfg.add_state()
    src = state.add_access('a')
    sink = state.add_access('out')
    me, mx = state.add_map('m', dict(i='0:64'))
    tlet = state.add_tasklet('t', {'_a'}, {'_o'}, '_o = _a')
    state.add_memlet_path(src, me, tlet, dst_conn='_a', memlet=Memlet('a[i]'))
    mx.add_in_connector('IN_out')
    mx.add_out_connector('OUT_out')
    inner = Memlet('out[i]')
    inner.wcr = 'lambda x, y: x + y'
    state.add_edge(tlet, '_o', mx, 'IN_out', inner)
    state.add_edge(mx, 'OUT_out', sink, None, Memlet('out[0:64]'))
    return sdfg


def _accumulator_also_read_sdfg() -> dace.SDFG:
    """``A[3] (+)= A[k]`` -- the accumulator array is also READ inside the map, so the map is a
    recurrence over ``A`` rather than a fold of independent addends (polybench ``lu``'s
    ``A[i, j] (+)= -A[i, k] * A[k, j]`` in miniature)."""
    sdfg = dace.SDFG('acc_read_back')
    sdfg.add_array('A', (64, ), dace.float64)
    state = sdfg.add_state()
    src = state.add_access('A')
    sink = state.add_access('A')
    me, mx = state.add_map('m', dict(k='0:64'))
    tlet = state.add_tasklet('t', {'_a'}, {'_o'}, '_o = _a')
    state.add_memlet_path(src, me, tlet, dst_conn='_a', memlet=Memlet('A[k]'))
    mx.add_in_connector('IN_A')
    mx.add_out_connector('OUT_A')
    inner = Memlet('A[3]')
    inner.wcr = 'lambda x, y: x + y'
    state.add_edge(tlet, '_o', mx, 'IN_A', inner)
    state.add_edge(mx, 'OUT_A', sink, None, Memlet('A[3]'))
    return sdfg


def _tile_main_maps(sdfg: dace.SDFG):
    """Every ``__tile_main`` map entry in ``sdfg`` (recursively)."""
    return [
        n for n, _ in sdfg.all_nodes_recursive()
        if isinstance(n, dace.nodes.MapEntry) and n.map.label.endswith('__tile_main')
    ]


def test_array_slot_reduction_passes_the_wcr_preconditions():
    """The guards must ACCEPT ``mean[j] (+)= data[i, j]``: the slot is invariant over the
    innermost (widened) dim, so the tile fold has one accumulator, not one per lane."""
    sdfg = _row_sum_sdfg('row_sum_guard')
    assert no_wcr_in_map_body(sdfg) is None
    assert no_wcr_inside_nested_sdfgs(sdfg) is None


def test_lane_scatter_and_read_back_accumulator_stay_refused():
    """The two look-alikes must still be flagged: a slot varying with the innermost param has no
    single accumulator per tile, and a read-back accumulator is a recurrence, not a fold."""
    assert no_wcr_in_map_body(_scatter_sdfg()) is not None
    assert no_wcr_in_map_body(_accumulator_also_read_sdfg()) is not None


def test_array_slot_reduction_tiles_and_preserves_values():
    """End to end: the reduction is tiled (a ``__tile_main`` map strided by the width, no refusal)
    and computes the same column sums as NumPy."""
    sdfg = _row_sum_sdfg('row_sum_e2e')
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        VectorizeCPUMultiDim(VectorizeConfig(widths=(WIDTH, ), target_isa=ISA.SCALAR)).apply_pass(sdfg, {})
    refusals = [str(w.message) for w in caught if 'refusing to vectorize' in str(w.message)]
    assert not refusals, f'the array-slot reduction was refused: {refusals}'

    tiled = _tile_main_maps(sdfg)
    assert tiled, 'no __tile_main map: the reduction map was left scalar'
    assert any(str(entry.map.range[-1][2]) == str(WIDTH) for entry in tiled), \
        f'no __tile_main map strided by {WIDTH}: {[str(e.map.range) for e in tiled]}'

    rows, cols = 37, 11  # both indivisible by WIDTH, so the remainder path runs too
    rng = np.random.default_rng(0)
    data = rng.random((rows, cols))
    mean = np.zeros(cols)
    sdfg(data=data, mean=mean, M=cols, N=rows)
    assert np.allclose(mean, data.sum(axis=0)), f'column sums differ: {mean} vs {data.sum(axis=0)}'


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
