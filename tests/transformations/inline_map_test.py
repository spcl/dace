# Copyright 2019-2024 ETH Zurich and the DaCe authors. All rights reserved.
import dace
import numpy as np

from dace.transformation.interstate import StateFusion, InlineMap

def test_inline_map_with_condition_fissioning():
    
    @dace.program
    def sdfg_with_condition_fissioning(a: dace.float32[32, 32], b: dace.float32[32, 32]):
        for i in dace.map[0:32]:
            for j in dace.map[0:32]:
                if j < 16:
                    b[i, j] = a[i, j]
                else:
                    b[i, j] = -a[i, j]


    sdfg = sdfg_with_condition_fissioning.to_sdfg(simplify=False)
    sdfg.apply_transformations_repeated(StateFusion)

    a = np.random.random((32, 32)).astype(np.float32)
    a_ = np.copy(a)
    b = np.zeros((32, 32), dtype=np.float32)
    b_ = np.copy(b)

    sdfg(a, b)

    applied = sdfg.apply_transformations(InlineMap)
    assert applied > 0

    sdfg(a_, b_)
    assert np.allclose(a, a_)
    assert np.allclose(b, b_)
