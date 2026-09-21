# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""A refusal that names its map leaves only that map scalar; the rest of the SDFG still tiles.

The vectorizer used to restore the whole SDFG on any refusal, so one un-tileable map in CloudSC
(a fused riming + melting map) took its other ~500 maps down with it: 13,691 tile ops became 0.
The gate is patched here so the refused map is known; which gate fires is not the property.
"""
import warnings

import numpy as np

import dace
from dace.libraries.tileops._dispatch import detect_host_isa
from dace.sdfg import nodes
from dace.transformation.passes.canonicalize import canonicalize
from dace.transformation.passes.vectorization import vectorize_multi_dim
from dace.transformation.passes.vectorization.config import VectorizeConfig
from dace.transformation.passes.vectorization.split_map_for_tile_remainder import TILE_MAIN_MARKER
from dace.transformation.passes.vectorization.utils.map_predicates import (NO_VECTORIZE_MARKER,
                                                                           innermost_enclosing_map_label)
from dace.transformation.passes.vectorization.vectorize_cpu_multi_dim import VectorizeCPUMultiDim

N = dace.symbol('N')
M = dace.symbol('M')


@dace.program
def two_maps(a: dace.float64[N], b: dace.float64[N], c: dace.float64[M], d: dace.float64[M]):
    """Two extents, so canonicalize cannot fuse the maps into one."""
    for i in dace.map[0:N]:
        b[i] = a[i] * 2.0
    for j in dace.map[0:M]:
        d[j] = c[j] + 1.0


def label_of_map_writing(sdfg: dace.SDFG, name: str) -> str:
    for node, state in sdfg.all_nodes_recursive():
        if isinstance(node, nodes.MapExit) and any(e.data.data == name for e in state.out_edges(node)):
            return node.map.label
    raise LookupError(name)


def refuse_once(label: str):
    """A stand-in gate refusing the body of the map labelled ``label``, found as a real gate finds it."""

    def gate(sdfg: dace.SDFG, widths):
        for node, _ in sdfg.all_nodes_recursive():
            if isinstance(node, nodes.NestedSDFG) and innermost_enclosing_map_label(node.sdfg) == (label, ):
                return node.sdfg.start_block, f'test refusal of {label}'
        return None

    return gate


def labels_starting_with(sdfg: dace.SDFG, label: str) -> tuple[str, ...]:
    """Current labels of the maps descended from ``label`` -- marked, or renamed by tiling."""
    return tuple(
        sorted({
            n.map.label
            for n, _ in sdfg.all_nodes_recursive() if isinstance(n, nodes.MapEntry) and n.map.label.startswith(label)
        }))


def refuse_always(sdfg: dace.SDFG, widths):
    """A stand-in gate that refuses on every attempt; paired with :func:`labels_starting_with`."""
    return sdfg.start_block, 'test refusal on every attempt'


def vectorized(monkeypatch, always: bool):
    sdfg = two_maps.to_sdfg(simplify=True)
    canonicalize(sdfg, validate=True)
    label = label_of_map_writing(sdfg, 'b')
    if always:
        monkeypatch.setattr(vectorize_multi_dim, 'lane_varying_interstate_guard', refuse_always)
        monkeypatch.setattr(vectorize_multi_dim, 'innermost_enclosing_map_label',
                            lambda graph: labels_starting_with(graph, label))
    else:
        monkeypatch.setattr(vectorize_multi_dim, 'lane_varying_interstate_guard', refuse_once(label))
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        VectorizeCPUMultiDim(VectorizeConfig(widths=(8, ), target_isa=detect_host_isa())).apply_pass(sdfg, {})
    messages = [str(w.message) for w in caught if 'VectorizeMultiDim' in str(w.message)]
    return sdfg, label, messages


def tile_nodes(sdfg: dace.SDFG) -> list:
    return [n for n, _ in sdfg.all_nodes_recursive() if type(n).__name__.startswith('Tile')]


def check_numbers(sdfg: dace.SDFG) -> None:
    rng = np.random.default_rng(0)
    a, c = rng.random(37), rng.random(29)
    b, d = np.zeros(37), np.zeros(29)
    sdfg(a=a, b=b, c=c, d=d, N=37, M=29)
    np.testing.assert_array_equal(b, a * 2.0)
    np.testing.assert_array_equal(d, c + 1.0)


def test_a_refused_map_stays_scalar_while_the_other_map_tiles(monkeypatch):
    sdfg, label, messages = vectorized(monkeypatch, always=False)
    assert not any('refusing to vectorize' in m for m in messages), messages
    assert any('leaving map(s)' in m and label in m for m in messages), messages
    assert label_of_map_writing(sdfg, 'b') == label + NO_VECTORIZE_MARKER
    assert tile_nodes(sdfg), 'the map nobody refused was not tiled'
    check_numbers(sdfg)


def test_a_map_refused_again_after_marking_refuses_the_whole_sdfg(monkeypatch):
    """Each retry must mark a new map, or the orchestrator would retry forever."""
    sdfg, _, messages = vectorized(monkeypatch, always=True)
    assert any('refusing to vectorize' in m for m in messages), messages
    assert tile_nodes(sdfg) == [], 'a whole-SDFG refusal must hand back the untiled input'
    check_numbers(sdfg)


def test_a_refusal_naming_a_tiled_region_marks_the_map_it_was_split_from(monkeypatch):
    """The gate runs after the remainder split, so on CloudSC's GPU leg it named ``..._map__tile_main``,
    a label the pristine snapshot does not have: nothing was marked and all ~500 maps were refused."""
    sdfg = two_maps.to_sdfg(simplify=True)
    canonicalize(sdfg, validate=True)
    label = label_of_map_writing(sdfg, 'b')
    monkeypatch.setattr(vectorize_multi_dim, 'lane_varying_interstate_guard', refuse_once(label))
    real = vectorize_multi_dim.innermost_enclosing_map_label
    monkeypatch.setattr(vectorize_multi_dim, 'innermost_enclosing_map_label',
                        lambda graph: tuple(f'{one}{TILE_MAIN_MARKER}' for one in real(graph)))
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        VectorizeCPUMultiDim(VectorizeConfig(widths=(8, ), target_isa=detect_host_isa())).apply_pass(sdfg, {})
    messages = [str(w.message) for w in caught if 'VectorizeMultiDim' in str(w.message)]
    assert not any('refusing to vectorize' in m for m in messages), messages
    assert label_of_map_writing(sdfg, 'b') == label + NO_VECTORIZE_MARKER
    assert tile_nodes(sdfg), 'the map nobody refused was not tiled'
    check_numbers(sdfg)
