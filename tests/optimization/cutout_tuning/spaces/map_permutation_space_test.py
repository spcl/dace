# Copyright 2019-2022 ETH Zurich and the DaCe authors. All rights reserved.
import pytest
import dace

import dace.optimization.cutout_tuning.spaces as spaces

from dace.transformation import helpers as xfh

I = dace.symbol("I")
J = dace.symbol("J")


@dace.program
def perm(A: dace.float64[I, J], B: dace.float64[I, J]):
    @dace.map
    def perm(j: _[0:J], i: _[0:I]):
        inp << A[i, j]
        out >> B[i, j]
        out = inp + 1


@pytest.fixture(scope="session")
def sdfg():
    sdfg = perm.to_sdfg(simplify=True)
    sdfg.specialize({"I": 4096, "J": 4096})
    return sdfg


def test_cutouts(sdfg):
    space = spaces.MapPermutationSpace()
    cutouts = list(space.cutouts(sdfg))

    assert len(cutouts) == 1

    cutout = cutouts[0]
    map_entry = None
    for node in cutout.start_state.nodes():
        if not isinstance(node, dace.nodes.MapEntry) or xfh.get_parent_map(cutout.start_state, node) is not None:
            continue

        map_entry = node
        break

    assert map_entry.map.label == "perm"


def test_configurations(sdfg):
    space = spaces.MapPermutationSpace()
    cutout = list(space.cutouts(sdfg))[0]

    configs = list(space.configurations(cutout))
    assert len(configs) == 1

    param, target = configs[0]
    assert param == (1, 0)

    map_entry = None
    for node in cutout.start_state.nodes():
        if not isinstance(node, dace.nodes.MapEntry) or xfh.get_parent_map(cutout.start_state, node) is not None:
            continue

        map_entry = node
        break

    node_id = cutout.start_state.node_id(map_entry)
    assert target == (0, node_id)


def test_encode_config():
    space = spaces.MapPermutationSpace()
    config = ((0, 1), (2, 5))
    key = space.encode_config(config)

    assert key == "((0, 1), (2, 5))"


def test_decode_config():
    space = spaces.MapPermutationSpace()
    key = "((0, 1), (2, 5))"
    config = space.decode_config(key)

    assert config == ((0, 1), (2, 5))


def test_apply_config(sdfg):
    space = spaces.MapPermutationSpace()
    cutout = list(space.cutouts(sdfg))[0]
    config = list(space.configurations(cutout))[0]

    map_entry = None
    for node in cutout.start_state.nodes():
        if not isinstance(node, dace.nodes.MapEntry) or xfh.get_parent_map(cutout.start_state, node) is not None:
            continue

        map_entry = node
        break

    assert map_entry.map.params == ["j", "i"]

    cutout_ = space.apply_config(cutout, config, make_copy=True)

    map_entry = None
    for node in cutout_.start_state.nodes():
        if not isinstance(node, dace.nodes.MapEntry) or xfh.get_parent_map(cutout_.start_state, node) is not None:
            continue

        map_entry = node
        break

    assert map_entry.map.params == ["i", "j"]


def test_translate_config(sdfg):
    space = spaces.MapPermutationSpace()
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
