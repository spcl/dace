# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
import copy

import numpy as np
import pytest

import dace
from dace.sdfg import nodes, utils as sdutil
from dace.transformation.interstate import SubgraphFission

N = dace.symbol('N')


@dace.program
def long_body(a: dace.float64[N], b: dace.float64[N], c: dace.float64[N]):
    for i in dace.map[0:N]:
        t = a[i] * 2.0
        for k in range(3):
            t = t + a[i]
        b[i] = t
        c[i] = b[i] + t


def body_of(sdfg: dace.SDFG) -> tuple:
    (state, entry), = [(s, n) for s in sdfg.states() for n in s.nodes() if isinstance(n, nodes.MapEntry)]
    (nsdfg, ) = {e.dst for e in state.out_edges(entry)}
    return state, entry, nsdfg


def fission_at(sdfg: dace.SDFG, index: int) -> int:
    state, entry, nsdfg = body_of(sdfg)
    cut = list(sdutil.dfs_topological_sort(nsdfg.sdfg))[index]
    return sdfg.apply_transformations(SubgraphFission, options={'cut': cut.label})


def run(sdfg: dace.SDFG) -> dict:
    rng = np.random.default_rng(0)
    arrays = {name: rng.random(7) for name in 'abc'}
    sdfg(**arrays, N=7)
    return arrays


@pytest.mark.parametrize('index', [0, 1])
def test_the_map_splits_into_two_maps_at_the_named_block(index):
    """``t`` crosses either cut, so each map iteration needs its own element of it."""
    sdfg = long_body.to_sdfg(simplify=True)
    reference = copy.deepcopy(sdfg)

    assert fission_at(sdfg, index) == 1

    maps = [n for s in sdfg.states() for n in s.nodes() if isinstance(n, nodes.MapEntry) and s.entry_node(n) is None]
    assert len(maps) == 2, maps
    assert [str(extent) for extent in sdfg.arrays['t'].shape] == ['N']
    got, want = run(sdfg), run(reference)
    for name in 'bc':
        assert np.allclose(got[name], want[name], rtol=1e-14, atol=0), name


def test_a_cut_after_the_last_block_is_refused():
    sdfg = long_body.to_sdfg(simplify=True)

    assert fission_at(sdfg, -1) == 0


def test_a_cut_naming_no_block_is_refused():
    sdfg = long_body.to_sdfg(simplify=True)

    assert sdfg.apply_transformations(SubgraphFission, options={'cut': 'no_such_block'}) == 0
