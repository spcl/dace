# Copyright 2019-2022 ETH Zurich and the DaCe authors. All rights reserved.
import pytest
import dace
import json

import dace.optimization.cutout_tuning.spaces as spaces

from dace.transformation import helpers as xfh

I = dace.symbol("I")
J = dace.symbol("J")


@dace.program
def til(A: dace.float64[I, J], B: dace.float64[I, J]):
    @dace.map
    def til(i: _[0:I], j: _[0:J]):
        inp << A[i, j]
        out >> B[i, j]
        out = inp + 1


@pytest.fixture(scope="session")
def sdfg():
    sdfg = til.to_sdfg(simplify=True)
    sdfg.specialize({"I": 256, "J": 64})
    return sdfg


def test_cutouts(sdfg):
    space = spaces.MapTilingSpace()
    cutouts = list(space.cutouts(sdfg))

    assert len(cutouts) == 1

    cutout = cutouts[0]
    map_entry = None
    for node in cutout.start_state.nodes():
        if not isinstance(node, dace.nodes.MapEntry) or xfh.get_parent_map(cutout.start_state, node) is not None:
            continue

        map_entry = node
        break

    assert map_entry.map.label == "til"


def test_configurations(sdfg):
    space = spaces.MapTilingSpace()
    cutout = list(space.cutouts(sdfg))[0]
    map_entry = None
    for node in cutout.start_state.nodes():
        if not isinstance(node, dace.nodes.MapEntry) or xfh.get_parent_map(cutout.start_state, node) is not None:
            continue

        map_entry = node
        break
    node_id = cutout.start_state.node_id(map_entry)

    configs = list(space.configurations(cutout))
    assert len(configs) == 3

    for i, config in enumerate(configs):
        param, target = config
        assert target == (0, node_id)

        tile_0 = 2**(i + 5)
        tile_1 = min(2**(i + 5), 32)
        assert param == (tile_0, tile_1)


def test_encode_config():
    space = spaces.MapPermutationSpace()
    config = ((16, 16), (2, 5))
    key = space.encode_config(config)

    assert key == "((16, 16), (2, 5))"


def test_decode_config():
    space = spaces.MapPermutationSpace()
    key = "((16, 16), (2, 5))"
    config = space.decode_config(key)

    assert config == ((16, 16), (2, 5))


def test_apply_config(sdfg):
    space = spaces.MapTilingSpace()
    cutout = list(space.cutouts(sdfg))[0]
    config = list(space.configurations(cutout))[0]

    map_entry = None
    for node in cutout.start_state.nodes():
        if not isinstance(node, dace.nodes.MapEntry) or xfh.get_parent_map(cutout.start_state, node) is not None:
            continue

        map_entry = node
        break

    steps = list(map(lambda r: r[2], map_entry.range.ranges))
    assert len(steps) == 2
    assert steps[0] == 1 and steps[1] == 1

    cutout_ = space.apply_config(cutout, config, make_copy=True)

    map_entry = None
    for node in cutout_.start_state.nodes():
        if not isinstance(node, dace.nodes.MapEntry) or xfh.get_parent_map(cutout_.start_state, node) is not None:
            continue

        map_entry = node
        break

    steps = list(map(lambda r: r[2], map_entry.range.ranges))
    assert len(steps) == 2
    assert steps[0] == 32 and steps[1] == 32


def test_translate_config(sdfg):
    space = spaces.MapTilingSpace()
    cutout = list(space.cutouts(sdfg))[0]
    config = list(space.configurations(cutout))[0]
    cparam, ctarget = config

    sconfig = space.translate_config(cutout, sdfg, config)
    sparam, starget = sconfig

    snode_id = None
    sstate_id = None
    for state in sdfg.nodes():
        for node in state.nodes():
            if not isinstance(node, dace.nodes.MapEntry) or xfh.get_parent_map(state, node) is not None:
                continue

            sstate_id = sdfg.node_id(state)
            snode_id = state.node_id(node)
            break

    assert cparam == sparam
    assert starget == (sstate_id, snode_id)
