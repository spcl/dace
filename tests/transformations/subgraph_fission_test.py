# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
import copy

import numpy as np
import pytest

import dace
from dace.sdfg import nodes
from dace.sdfg import utils as sdutil
from dace.transformation.interstate import SubgraphFission

N = dace.symbol("N")


@dace.program
def long_body(a: dace.float64[N], b: dace.float64[N], c: dace.float64[N]):
    for i in dace.map[0:N]:
        t = a[i] * 2.0
        for k in range(3):
            t = t + a[i]
        b[i] = t
        c[i] = b[i] + t


def body_of(sdfg: dace.SDFG) -> tuple:
    ((state, entry),) = [(s, n) for s in sdfg.states() for n in s.nodes() if isinstance(n, nodes.MapEntry)]
    (nsdfg,) = {e.dst for e in state.out_edges(entry)}
    return state, entry, nsdfg


def fission_at(sdfg: dace.SDFG, index: int) -> int:
    state, entry, nsdfg = body_of(sdfg)
    cut = list(sdutil.dfs_topological_sort(nsdfg.sdfg))[index]
    return sdfg.apply_transformations(SubgraphFission, options={"cut": cut.label})


def run(sdfg: dace.SDFG) -> dict:
    rng = np.random.default_rng(0)
    arrays = {name: rng.random(7) for name in "abc"}
    sdfg(**arrays, N=7)
    return arrays


@pytest.mark.parametrize("index", [0, 1])
def test_the_map_splits_into_two_maps_at_the_named_block(index):
    """``t`` crosses either cut, so each map iteration needs its own element of it."""
    sdfg = long_body.to_sdfg(simplify=True)
    reference = copy.deepcopy(sdfg)

    assert fission_at(sdfg, index) == 1

    maps = [n for s in sdfg.states() for n in s.nodes() if isinstance(n, nodes.MapEntry) and s.entry_node(n) is None]
    assert len(maps) == 2, maps
    assert [str(extent) for extent in sdfg.arrays["t"].shape] == ["N"]
    got, want = run(sdfg), run(reference)
    for name in "bc":
        assert np.allclose(got[name], want[name], rtol=1e-14, atol=0), name


def test_a_cut_after_the_last_block_is_refused():
    sdfg = long_body.to_sdfg(simplify=True)

    assert fission_at(sdfg, -1) == 0


def test_a_cut_naming_no_block_is_refused():
    sdfg = long_body.to_sdfg(simplify=True)

    assert sdfg.apply_transformations(SubgraphFission, options={"cut": "no_such_block"}) == 0


@dace.program
def accumulate_then_overwrite_input(a: dace.float64[N], b: dace.float64[N], c: dace.float64[N]):
    for i in dace.map[0:N]:
        t = a[i] * 2.0
        for k in range(3):
            t = t + a[i]
        b[i] = b[i] + t
    for i in dace.map[0:N]:
        c[i] = a[i] + 1.0
    for i in dace.map[0:N]:
        a[i] = c[i] * 2.0


def test_a_split_whose_state_cannot_be_cut_around_it_keeps_both_maps_nested():
    """A later map overwrites ``a`` the split maps read, so the state cannot be cut around the nested SDFG holding
    them: the cleanup inline refused after ``can_be_applied`` had already accepted (``ValueError`` from ``apply``)."""
    sdfg = accumulate_then_overwrite_input.to_sdfg(simplify=True)
    reference = copy.deepcopy(sdfg)
    reference.name = "accumulate_then_overwrite_input_reference"
    (state,) = sdfg.states()
    (entry,) = [
        n
        for n in state.nodes()
        if isinstance(n, nodes.MapEntry) and isinstance(next(iter(state.out_edges(n))).dst, nodes.NestedSDFG)
    ]
    nsdfg = next(iter(state.out_edges(entry))).dst
    cut = list(sdutil.dfs_topological_sort(nsdfg.sdfg))[0]

    SubgraphFission.apply_to(sdfg, options={"cut": cut.label}, map_entry=entry, nested_sdfg=nsdfg)

    split = [
        n
        for n, _ in nsdfg.sdfg.all_nodes_recursive()
        if isinstance(n, nodes.MapEntry) and n.label == f"{entry.label}_fission"
    ]
    assert len(split) == 2, split
    rng = np.random.default_rng(0)
    want = {name: rng.random(7) for name in "abc"}
    got = copy.deepcopy(want)
    reference(**want, N=7)
    sdfg(**got, N=7)
    for name in "abc":
        assert np.allclose(got[name], want[name], rtol=1e-14, atol=0), name
