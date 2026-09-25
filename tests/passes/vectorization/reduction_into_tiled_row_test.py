# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""A reduction into ``y[i]`` over ``(i, k)`` has one accumulator per row: only ``k`` is tiled, and its
lanes fold into ``y[i]``."""
import numpy as np
import pytest

import dace
from dace.libraries.tileops import TileReduce
from dace.libraries.tileops._dispatch import detect_host_isa
from dace.memlet import Memlet
from dace.transformation.passes.vectorization.config import VectorizeConfig
from dace.transformation.passes.vectorization.enums import ISA
from dace.transformation.passes.vectorization.utils.map_predicates import is_tile_eligible
from dace.transformation.passes.vectorization.vectorize_cpu_multi_dim import VectorizeCPUMultiDim


def row_sums() -> tuple[dace.SDFG, dace.SDFGState, dace.nodes.MapEntry]:
    sdfg = dace.SDFG('row_sums')
    sdfg.add_array('A', ('N', 'M'), dace.float64)
    sdfg.add_array('y', ('N', ), dace.float64)
    state = sdfg.add_state()
    _, entry, _ = state.add_mapped_tasklet('rows', {
        'i': '0:N',
        'k': '0:M'
    }, {'a': Memlet('A[i, k]')},
                                           'o = a', {'o': Memlet('y[i]', wcr='lambda x, y: x + y')},
                                           external_edges=True)
    return sdfg, state, entry


def test_reduction_into_a_row_is_tiled_on_the_inner_param_only():
    sdfg, state, entry = row_sums()
    assert not is_tile_eligible(state, entry, K=2)

    VectorizeCPUMultiDim(VectorizeConfig(widths=(4, 8), target_isa=ISA.SCALAR,
                                         expand_tile_nodes=False)).apply_pass(sdfg, {})

    steps = [
        tuple(str(step) for _, _, step in node.map.range) for node, _ in sdfg.all_nodes_recursive()
        if isinstance(node, dace.nodes.MapEntry)
    ]
    assert ('1', '8') in steps
    assert all(i_step == '1' for i_step, _ in steps)
    assert any(isinstance(node, TileReduce) for node, _ in sdfg.all_nodes_recursive())


@pytest.mark.parametrize('isa', sorted({ISA.SCALAR.value, detect_host_isa()}))
@pytest.mark.parametrize('n,m', [(8, 16), (12, 16), (12, 21), (5, 3)])
def test_reduction_into_a_tiled_row_sums_each_row(n, m, isa):
    rng = np.random.default_rng(seed=n)
    A = rng.random((n, m))
    y = np.zeros(n)
    sdfg, _, _ = row_sums()
    sdfg.name = f'row_sums_{n}_{m}_{isa}'
    VectorizeCPUMultiDim(VectorizeConfig(widths=(4, 8), target_isa=isa)).apply_pass(sdfg, {})

    sdfg(A=A.copy(), y=y, N=n, M=m)

    np.testing.assert_allclose(y, A.sum(axis=1), rtol=1e-12)
