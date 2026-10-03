# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""FuseMaps validates the SDFG after fusing only when its ``validate`` knob asks for it."""
import copy

import numpy as np
import pytest

import dace
from dace.sdfg import nodes
from dace.sdfg.validation import InvalidSDFGError
from dace.transformation.passes.fuse_maps import FuseMaps


@dace.program
def two_elementwise_maps(a: dace.float64[20], b: dace.float64[20]):
    tmp = a + 1.0
    b[:] = tmp * 2.0


def count_maps(sdfg: dace.SDFG) -> int:
    return sum(1 for node, _ in sdfg.all_nodes_recursive() if isinstance(node, nodes.MapEntry))


def fusible_sdfg_with_an_unrelated_dangling_connector() -> dace.SDFG:
    sdfg = two_elementwise_maps.to_sdfg(simplify=True)
    stray = sdfg.add_state_after(sdfg.sink_nodes()[0], 'stray')
    stray.add_tasklet('dangling', {'x'}, {}, 'pass')
    return sdfg


def test_fuse_maps_without_validation_fuses_an_sdfg_validation_would_reject():
    sdfg = fusible_sdfg_with_an_unrelated_dangling_connector()

    FuseMaps(validate=False, validate_all=False).apply_pass(sdfg, {})

    assert count_maps(sdfg) == 1


def test_fuse_maps_with_validation_rejects_the_same_sdfg():
    sdfg = fusible_sdfg_with_an_unrelated_dangling_connector()

    with pytest.raises(InvalidSDFGError):
        FuseMaps(validate=True, validate_all=False).apply_pass(sdfg, {})


@pytest.mark.parametrize('validate', [True, False])
def test_fused_maps_compute_the_same_values_with_or_without_validation(validate: bool):
    sdfg = copy.deepcopy(two_elementwise_maps.to_sdfg(simplify=True))
    a = np.arange(20, dtype=np.float64)
    b = np.zeros(20)

    FuseMaps(validate=validate, validate_all=False).apply_pass(sdfg, {})

    assert count_maps(sdfg) == 1
    sdfg(a=a, b=b)
    assert np.allclose(b, (a + 1.0) * 2.0)
