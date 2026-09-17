# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""A reduction whose target is indexed by a tiled parameter (``y[i] += A[i, k]`` over a tiled
``(i, k)``) has one accumulator per lane of ``i``; a tile fold writes all lanes into ``y[i]``."""
import numpy as np
import pytest

import dace
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


def test_reduction_into_a_tiled_row_is_not_tile_eligible():
    sdfg, state, entry = row_sums()

    assert not is_tile_eligible(state, entry, K=2)
    assert is_tile_eligible(state, entry, K=1)


@pytest.mark.parametrize('n', [8, 12])
def test_reduction_into_a_tiled_row_sums_each_row(n):
    m = 16
    rng = np.random.default_rng(seed=n)
    A = rng.random((n, m))
    y = np.zeros(n)
    sdfg, _, _ = row_sums()
    VectorizeCPUMultiDim(VectorizeConfig(widths=(4, 8), target_isa=ISA.SCALAR)).apply_pass(sdfg, {})

    sdfg(A=A.copy(), y=y, N=n, M=m)

    np.testing.assert_allclose(y, A.sum(axis=1), rtol=1e-12)
