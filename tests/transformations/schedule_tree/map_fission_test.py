# Copyright 2019-2022 ETH Zurich and the DaCe authors. All rights reserved.
import dace
import numpy as np
from dace.sdfg.analysis.schedule_tree.sdfg_to_tree import as_schedule_tree, as_sdfg
from dace.sdfg.analysis.schedule_tree.transformations import map_fission


def test_map_with_tasklet_and_library():

    N = dace.symbol('N')
    @dace.program
    def map_with_tasklet_and_library(A: dace.float32[N, 5, 5], B: dace.float32[N, 5, 5], cst: dace.int32):
        out = np.ndarray((N, 5, 5), dtype=dace.float32)
        for i in dace.map[0:N]:
            out[i] = cst * (A[i] @ B[i])
        return out
    
    rng = np.random.default_rng(42)
    A = rng.random((10, 5, 5), dtype=np.float32)
    B = rng.random((10, 5, 5), dtype=np.float32)
    cst = rng.integers(0, 100, dtype=np.int32)
    ref = cst * (A @ B)

    val0 = map_with_tasklet_and_library(A, B, cst)
    sdfg0 = map_with_tasklet_and_library.to_sdfg()
    tree = as_schedule_tree(sdfg0)
    result = map_fission(tree.children[0], tree)
    assert result
    sdfg1 = as_sdfg(tree)
    val1 = sdfg1(A=A, B=B, cst=cst, N=A.shape[0])
    
    assert np.allclose(val0, ref)
    assert np.allclose(val1, ref)


if __name__ == "__main__":
    test_map_with_tasklet_and_library()
