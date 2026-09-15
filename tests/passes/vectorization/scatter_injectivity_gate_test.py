# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""What the tile-lowerability gate may admit as a scatter store, and what it must keep refusing.

W tile lanes run concurrently, so two lanes computing the same address is a silent wrong answer
rather than a crash. ``classify_tile_access`` re-marks every dimension one map param drives twice
(``A[i, i]``) as GATHER -- an emitter-dispatch decision, since a diagonal is not a per-dim
contiguous window -- and the gate used to read that mark as "cannot prove injective" and refuse.
At one tile dim the proof is available: one dimension whose index has a nonzero affine coefficient
in the lane variable separates every pair of lanes, so the lanes write disjoint boxes of a plain
``Array``. These tests pin both halves of that boundary -- the shapes the proof closes on, and the
shapes it must not.
"""
import pytest

import dace
from dace import data as dt, subsets
from dace.transformation.passes.canonicalize import canonicalize
from dace.transformation.passes.vectorization.utils.injectivity import scatter_write_is_injective, write_subset_is_injective
from dace.transformation.passes.vectorization.utils.map_predicates import (is_innermost_map, map_body_is_tile_lowerable)

N = dace.symbol('N', dtype=dace.int64, positive=True)


@dace.program
def diagonal_scatter_store(src: dace.float64[N], A: dace.float64[N, N], scale: dace.float64):
    for i, in dace.map[0:N:1]:
        A[i, i] = src[i] * scale


def array_desc() -> dt.Array:
    return dt.Array(dtype=dace.float64, shape=(16, 16))


@pytest.mark.parametrize('written', ['i, i', '2*i, i', 'i, 2*i', 'j, i', 'i, j:j+4'])
def test_a_write_with_one_lane_separated_dimension_is_injective(written):
    assert scatter_write_is_injective(subsets.Range.from_string(written), 'i', array_desc()) is True


@pytest.mark.parametrize('written', [
    'i % 8',
    'i**2',
    'i*i',
    'idx[i]',
    'idx[i], i',
    '5',
    'N*i',
    'i:i+4',
])
def test_a_write_the_lane_variable_cannot_separate_is_refused(written):
    assert scatter_write_is_injective(subsets.Range.from_string(written), 'i', array_desc()) is False


def test_a_view_destination_is_refused_however_the_index_reads():
    view = dt.ArrayView(dtype=dace.float64, shape=(16, 16))
    assert scatter_write_is_injective(subsets.Range.from_string('i, i'), 'i', view) is False


def test_the_diagonal_scatter_is_admitted_at_one_tile_dim_and_refused_above_it():
    sdfg = diagonal_scatter_store.to_sdfg(simplify=False)
    sdfg.simplify(validate=True, validate_all=True)
    canonicalize(sdfg, validate=True)
    maps = [(n, st) for st in sdfg.all_states() for n in st.nodes()
            if isinstance(n, dace.nodes.MapEntry) and is_innermost_map(st, n)]
    assert len(maps) == 1, f'expected the one innermost map; got {[n.map.label for n, _ in maps]}'
    entry, state = maps[0]
    assert map_body_is_tile_lowerable(state, entry, 1) is True
    assert map_body_is_tile_lowerable(state, entry, 2) is False
    assert map_body_is_tile_lowerable(state, entry) is False


@pytest.mark.parametrize('typed', [True, False])
@pytest.mark.parametrize('written,params,injective', [
    ('i, j', ['i', 'j'], True),
    ('j, i', ['i', 'j'], True),
    ('2*i + 1', ['i'], True),
    ('i', ['i', 'j'], False),
    ('8*i + j', ['i', 'j'], False),
])
def test_a_write_over_several_map_params_is_injective_when_each_param_owns_a_dim(written, params, injective, typed):
    """A write to ``aa[i, j]`` over a map on ``i, j`` was refused outright (only one param was ever decided), and
    an ``int64`` iterator never matched its untyped spelling: TSVC s2275 kept a WCR the tile vectorizer refuses."""
    symbols = {
        name: dace.symbol(name, dtype=dace.int64) if typed else dace.symbolic.pystr_to_symbolic(name)
        for name in ('i', 'j')
    }
    dims = [
        dace.symbolic.pystr_to_symbolic(dim.strip()).subs({
            dace.symbolic.pystr_to_symbolic(n): s
            for n, s in symbols.items()
        }) for dim in written.split(',')
    ]
    subset = subsets.Range([(dim, dim, 1) for dim in dims])
    assert write_subset_is_injective(subset, params) is injective


if __name__ == '__main__':
    test_a_view_destination_is_refused_however_the_index_reads()
    test_the_diagonal_scatter_is_admitted_at_one_tile_dim_and_refused_above_it()
