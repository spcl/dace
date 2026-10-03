# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""A lane-invariant ``v[c] = s`` in a vectorized map still writes element ``c``.

``ConvertTaskletsToTileOps`` folds the trivial assign tasklet into a direct AN -> AN copy and built
its memlet from the SOURCE side alone, so the copy lost the destination element and wrote ``v[0]``.
CloudSC's ``zvqx[ncldqi] = rvice`` (and the rain / snow slots) all landed in ``zvqx[0]``: every
fall speed but liquid was zero and no precipitation fell below the cloud top.
"""
import numpy as np
import pytest

import dace
from dace.transformation.passes.vectorization.config import VectorizeConfig
from dace.transformation.passes.vectorization.vectorize_cpu_multi_dim import VectorizeCPUMultiDim

N = dace.symbol('N')


@dace.program
def fill_slot_one(a: dace.float64[N], out: dace.float64[N], v: dace.float64[5], s: dace.float64):
    for i in dace.map[0:N]:
        v[1] = s
        out[i] = a[i] * 2.0


@dace.program
def fill_slot_four(a: dace.float64[N], out: dace.float64[N], v: dace.float64[5], s: dace.float64):
    for i in dace.map[0:N]:
        v[4] = s
        out[i] = a[i] * 2.0


@pytest.mark.parametrize('program,slot', [(fill_slot_one, 1), (fill_slot_four, 4)])
def test_a_vectorized_constant_slot_copy_writes_its_own_slot(program, slot):
    sdfg = program.to_sdfg(simplify=True)
    VectorizeCPUMultiDim(
        VectorizeConfig(widths=(2, ), target_isa='SCALAR', remainder_strategy='masked_tail',
                        validate_all=True)).apply_pass(sdfg, {})
    a = np.random.rand(7)
    out = np.zeros(7)
    v = np.zeros(5)
    sdfg(a=a, out=out, v=v, s=3.5, N=7)
    want = np.zeros(5)
    want[slot] = 3.5
    np.testing.assert_array_equal(v, want)
    np.testing.assert_array_equal(out, 2.0 * a)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
