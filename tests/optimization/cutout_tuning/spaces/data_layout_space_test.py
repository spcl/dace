# Copyright 2019-2022 ETH Zurich and the DaCe authors. All rights reserved.
import pytest
import dace

import dace.optimization.cutout_tuning.spaces as spaces

from dace.transformation import helpers as xfh

I = dace.symbol("I")
J = dace.symbol("J")


@dace.program
def strid(A: dace.float64[I, J], B: dace.float64[I, J]):
    @dace.map
    def strid(j: _[0:J], i: _[0:I]):
        inp << A[j, i]
        out >> B[i, j]
        out = inp + 1


@pytest.fixture(scope="session")
def sdfg():
    sdfg = strid.to_sdfg(simplify=True)
    I.set(4096)
    J.set(4096)
    sdfg.specialize({I: 4096, J: 4096})
    return sdfg


def test_cutouts(sdfg):
    space = spaces.DataLayoutSpace()
    cutouts = list(space.cutouts(sdfg))

    assert len(cutouts) == 1
    assert cutouts[0] == sdfg


def test_configurations(sdfg):
    space = spaces.DataLayoutSpace()
    cutout = list(space.cutouts(sdfg))[0]
    configs = list(space.configurations(cutout))

    assert len(configs) == 4


def test_encode_config(sdfg):
    space = spaces.DataLayoutSpace()
    config = {}
    for name, array in sdfg.arrays.items():
        config[name] = array.strides

    key = space.encode_config(config)
    assert key == '{"A": "(J, 1)","B": "(J, 1)"}'


def test_decode_config(sdfg):
    space = spaces.DataLayoutSpace()
    key = '{"A": "(J, 1)","B": "(J, 1)"}'
    config = space.decode_config(key)

    assert len(config) == len(sdfg.arrays)

    for name, array in sdfg.arrays.items():
        assert config[name] == array.strides


def test_apply_config(sdfg):
    space = spaces.DataLayoutSpace()
    cutout = list(space.cutouts(sdfg))[0]
    configs = list(space.configurations(cutout))
    config = configs[1]

    cutout_ = space.apply_config(cutout, config, make_copy=True)

    assert str(cutout_.arrays["A"].strides) == "(1, I)"
    assert str(cutout_.arrays["B"].strides) == "(J, 1)"


def test_translate_config(sdfg):
    space = spaces.DataLayoutSpace()
    cutout = list(space.cutouts(sdfg))[0]
    config = list(space.configurations(cutout))[0]

    config_ = space.translate_config(cutout, sdfg, config)
    assert config == config_
