# Copyright 2019-2022 ETH Zurich and the DaCe authors. All rights reserved.
import pytest
import json
import dace
import numpy as np

import dace.optimization.cutout_tuning as ct
import dace.optimization.cutout_tuning.spaces as spaces

from dace import nodes
from dace.transformation import helpers as xfh
from dace.optimization.cutout_tuning.utils import measure_main

I = dace.symbol("I")
J = dace.symbol("J")


@dace.program
def inc(A: dace.float64[I, J], B: dace.float64[I, J]):
    for j, i in dace.map[0:J, 0:I]:
        with dace.tasklet:
            inp << A[i, j]
            out >> B[i, j]
            out = inp + 1


@pytest.fixture(scope="session")
def config_path(tmpdir_factory):
    config_path = tmpdir_factory.mktemp("cutout_tuner_test_temp").join("tuner_config.json")
    config = {"name": "test_config", "stages": [["MapPermutationSpace"]]}

    with open(config_path, "w") as handle:
        json.dump(config, handle)
    return config_path


@pytest.fixture
def sdfg():
    sdfg = inc.to_sdfg(simplify=True)
    sdfg.specialize({"I": 4096, "J": 4096})
    return sdfg


def test_dry_run(sdfg):
    A = np.zeros((4096, 4096), dtype=np.float64)
    B = np.empty((4096, 4096), dtype=np.float64)
    dreport = ct.CutoutTuner.dry_run(sdfg, A=A, B=B)

    A_ = dreport["A"]
    B_ = dreport["B"]

    assert (A_ == 0).all()
    assert (B_ == 1).all()


def test_tuner_config(config_path, sdfg):
    dreport = sdfg.get_instrumented_data()
    tuner = ct.CutoutTuner(sdfg, dreport=dreport, config_path=config_path)

    assert tuner._name == "test_config"
    assert len(tuner._stages) == 1

    stage = tuner._stages[0]
    assert len(stage) == 1
    assert isinstance(stage[0], spaces.MapPermutationSpace)


def test_apply_best_configs(config_path, sdfg):
    dreport = sdfg.get_instrumented_data()
    tuner = ct.CutoutTuner(sdfg, dreport=dreport, config_path=config_path)

    perm_space = spaces.MapPermutationSpace()
    tile_space = spaces.MapTilingSpace()

    node_id = None
    node_id_ = None
    cutout = list(perm_space.cutouts(sdfg))[0]
    nested_cutout = list(tile_space.cutouts(cutout))[0]
    for state in cutout.nodes():
        for node in state.nodes():
            if not isinstance(node, nodes.MapEntry):
                continue

            node_id = state.node_id(node)
            node_id_ = nested_cutout.start_state.node_id(node)
            break

    stage = [perm_space, tile_space]
    perm_config = f"((1, 0), (0, {node_id}))"
    til_config = f"((16, 32), (0, {node_id_}))"

    search_cache = {
        "0": {
            "base_runtime": 1.0,
            "best_runtime": 0.5,
            "best_config": perm_config,
            "configs": {
                perm_config: {
                    "runtime": 0.5,
                    "subspace": {
                        "MapTilingSpace": {
                            "0": {
                                "base_runtime": 1.0,
                                "best_runtime": 0.5,
                                "best_config": til_config,
                                "configs": {
                                    til_config: {
                                        "runtime": 0.5,
                                        "subspace": {}
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    tuner._apply_best_configs(stage, sdfg, search_cache=search_cache)

    outer_map = None
    inner_map = None
    for state in sdfg.nodes():
        for node in state.nodes():
            if not isinstance(node, nodes.MapEntry):
                continue

            if xfh.get_parent_map(state, node) is None:
                assert outer_map is None
                outer_map = node
            else:
                assert inner_map is None
                inner_map = node

    assert outer_map is not None and inner_map is not None

    assert inner_map.map.params == ["i", "j"]
    assert outer_map.range.ranges[0][2] == 16 and outer_map.range.ranges[1][2] == 32


def test_tune(config_path, sdfg):
    map_entry = None
    for state in sdfg.nodes():
        for node in state.nodes():
            if not isinstance(node, nodes.MapEntry) or xfh.get_parent_map(state, node) is not None:
                continue

            map_entry = node
            break

    assert map_entry.map.params == ["j", "i"]

    dreport = sdfg.get_instrumented_data()
    tuner = ct.CutoutTuner(sdfg, dreport=dreport, config_path=config_path)
    tuner.measure = measure_main

    tuned_sdfg = tuner.tune(in_place=False)
    tmap_entry = None
    for state in tuned_sdfg.nodes():
        for node in state.nodes():
            if not isinstance(node, nodes.MapEntry) or xfh.get_parent_map(state, node) is not None:
                continue

            tmap_entry = node
            break

    assert tmap_entry.map.params == ["i", "j"]
