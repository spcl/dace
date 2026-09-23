# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""FuseMaps propagates the inside of a nested SDFG once per change, not once per fusion of its scope."""
import copy
import json
from typing import Any, List, Tuple

import numpy as np
import pytest

import dace
from dace.sdfg import nodes, propagation
from dace.transformation.dataflow import map_fusion_helper as mfhelper
from dace.transformation.passes.fuse_maps import FuseMaps

CHAIN_LENGTH = 5


def add_one_nested_sdfg(index: int, width: int) -> dace.SDFG:
    inner = dace.SDFG(f'add_one_{index}')
    state = inner.add_state(is_start_block=True)
    if width == 1:
        inner.add_scalar('x', dace.float64)
        inner.add_scalar('y', dace.float64)
        tasklet = state.add_tasklet('add', {'a': None}, {'b': None}, 'b = a + 1.0')
        state.add_edge(state.add_access('x'), None, tasklet, 'a', dace.Memlet('x[0]'))
        state.add_edge(tasklet, 'b', state.add_access('y'), None, dace.Memlet('y[0]'))
        return inner
    inner.add_array('x', (width, ), dace.float64)
    inner.add_array('y', (width, ), dace.float64)
    state.add_mapped_tasklet('add', {'j': f'0:{width}'}, {'a': dace.Memlet('x[j]')},
                             'b = a + 1.0', {'b': dace.Memlet('y[j]')},
                             external_edges=True)
    return inner


def map_chain_with_nested_sdfgs(width: int = 1) -> dace.SDFG:
    """``CHAIN_LENGTH`` Maps in a row, each computing ``arr_k[i] = arr_{k-1}[i] + 1`` in a nested SDFG.

    ``width`` 1 passes one scalar per iteration; wider passes a row, whose inner strides the fusion rewrites.
    """
    sdfg = dace.SDFG(f'map_chain_with_nested_sdfgs_{width}')
    names = [f'arr_{k}' for k in range(CHAIN_LENGTH + 1)]
    shape = (20, ) if width == 1 else (20, width)
    for k, name in enumerate(names):
        sdfg.add_array(name, shape, dace.float64, transient=0 < k < CHAIN_LENGTH)
    state = sdfg.add_state(is_start_block=True)
    source = state.add_access(names[0])
    for k in range(1, CHAIN_LENGTH + 1):
        row = f'i{k}' if width == 1 else f'i{k}, 0:{width}'
        entry, exit_node = state.add_map(f'map_{k}', {f'i{k}': '0:20'})
        nsdfg = state.add_nested_sdfg(add_one_nested_sdfg(k, width), {'x': None}, {'y': None})
        target = state.add_access(names[k])
        state.add_memlet_path(source, entry, nsdfg, dst_conn='x', memlet=dace.Memlet(f'{names[k - 1]}[{row}]'))
        state.add_memlet_path(nsdfg, exit_node, target, src_conn='y', memlet=dace.Memlet(f'{names[k]}[{row}]'))
        source = target
    sdfg.validate()
    return sdfg


def count_maps(sdfg: dace.SDFG) -> int:
    return sum(1 for node, _ in sdfg.all_nodes_recursive() if isinstance(node, nodes.MapEntry))


def without_identity(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: without_identity(v) for k, v in obj.items() if k not in ('guid', 'hash', 'debuginfo')}
    if isinstance(obj, list):
        return [without_identity(v) for v in obj]
    return obj


def serialized(sdfg: dace.SDFG) -> str:
    return json.dumps(without_identity(sdfg.to_json()), sort_keys=True, default=str)


def cfg_tree_snapshot(sdfg: dace.SDFG) -> List[Tuple[int, int, int, int, int]]:
    """Per region of ``sdfg.cfg_list``: its identity, cfg_id and every parent pointer, by identity."""
    snapshot = []
    for region in sdfg.cfg_list:
        parent_nsdfg = region.parent_nsdfg_node if isinstance(region, dace.SDFG) else None
        snapshot.append((id(region), region.cfg_id, id(region.parent_sdfg), id(parent_nsdfg), id(region.parent_graph)))
    return snapshot


def fuse_counting_propagations(sdfg: dace.SDFG, monkeypatch: pytest.MonkeyPatch, use_cache: bool) -> List[dace.SDFG]:
    propagated: List[dace.SDFG] = []
    original_sdfg_propagation = propagation.propagate_memlets_sdfg
    original_scope_propagation = mfhelper.propagate_fused_map_scope

    def counting_sdfg_propagation(nested: dace.SDFG) -> None:
        propagated.append(nested)
        original_sdfg_propagation(nested)

    def uncached_scope_propagation(outer, state, map_entry, propagated_nsdfgs=None):
        original_scope_propagation(outer, state, map_entry, None)

    with monkeypatch.context() as patch:
        patch.setattr(propagation, 'propagate_memlets_sdfg', counting_sdfg_propagation)
        if not use_cache:
            patch.setattr(mfhelper, 'propagate_fused_map_scope', uncached_scope_propagation)
        FuseMaps(validate=True, strict_dataflow=False).apply_pass(sdfg, {})
    return propagated


def test_fuse_maps_propagates_each_nested_sdfg_of_a_chain_once(monkeypatch: pytest.MonkeyPatch):
    sdfg = map_chain_with_nested_sdfgs()

    propagated = fuse_counting_propagations(sdfg, monkeypatch, use_cache=True)

    assert count_maps(sdfg) == 1
    assert len(propagated) == CHAIN_LENGTH
    assert len({id(nested) for nested in propagated}) == CHAIN_LENGTH


def test_fuse_maps_without_the_cache_repropagates_nested_sdfgs_per_fusion(monkeypatch: pytest.MonkeyPatch):
    sdfg = map_chain_with_nested_sdfgs()

    propagated = fuse_counting_propagations(sdfg, monkeypatch, use_cache=False)

    assert count_maps(sdfg) == 1
    assert len(propagated) > CHAIN_LENGTH


@pytest.mark.parametrize('width', [1, 4])
def test_fuse_maps_with_the_cache_builds_the_same_sdfg_as_without(monkeypatch: pytest.MonkeyPatch, width: int):
    cached = map_chain_with_nested_sdfgs(width)
    uncached = copy.deepcopy(cached)

    fuse_counting_propagations(cached, monkeypatch, use_cache=True)
    fuse_counting_propagations(uncached, monkeypatch, use_cache=False)

    assert count_maps(cached) == 1 + (width > 1) * CHAIN_LENGTH
    assert serialized(cached) == serialized(uncached)


def test_fuse_maps_repropagates_a_nested_sdfg_whose_strides_the_fusion_rewrote(monkeypatch: pytest.MonkeyPatch):
    sdfg = map_chain_with_nested_sdfgs(width=4)

    propagated = fuse_counting_propagations(sdfg, monkeypatch, use_cache=True)

    assert len({id(nested) for nested in propagated}) == CHAIN_LENGTH
    assert len(propagated) > CHAIN_LENGTH


def test_fuse_maps_leaves_the_cfg_list_a_reset_would_build():
    sdfg = map_chain_with_nested_sdfgs()

    FuseMaps(validate=True, strict_dataflow=False).apply_pass(sdfg, {})
    kept = cfg_tree_snapshot(sdfg)
    sdfg.reset_cfg_list()

    assert kept == cfg_tree_snapshot(sdfg)
    assert len(kept) == CHAIN_LENGTH + 1


@pytest.mark.parametrize('width', [1, 4])
def test_fused_chain_with_nested_sdfgs_computes_the_chain(width: int):
    sdfg = map_chain_with_nested_sdfgs(width)
    shape = (20, ) if width == 1 else (20, width)
    source = np.arange(np.prod(shape), dtype=np.float64).reshape(shape)
    result = np.zeros(shape)

    FuseMaps(validate=True, strict_dataflow=False).apply_pass(sdfg, {})
    sdfg(arr_0=source, **{f'arr_{CHAIN_LENGTH}': result})

    assert np.allclose(result, source + CHAIN_LENGTH)


def test_forget_propagated_drops_the_sdfg_and_every_enclosing_sdfg_only():
    sdfg = map_chain_with_nested_sdfgs()
    first, second = [node.sdfg for node in sdfg.start_block.nodes() if isinstance(node, nodes.NestedSDFG)][:2]
    cache = dict.fromkeys([sdfg, first, second])

    mfhelper.forget_propagated(cache, first)

    assert list(cache) == [second]


def test_forget_propagated_accepts_no_cache():
    mfhelper.forget_propagated(None, map_chain_with_nested_sdfgs())
