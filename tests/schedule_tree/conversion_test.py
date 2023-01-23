# Copyright 2019-2022 ETH Zurich and the DaCe authors. All rights reserved.
import dace
import numpy as np
from dace.sdfg.analysis.schedule_tree.sdfg_to_tree import as_schedule_tree, as_sdfg


# TODO: The test fails because of the ambiguity when having access nodes inside a MapScope but on the same SDFG level.
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
    pcode, _ = tree.as_python()
    print(pcode)
    sdfg1 = as_sdfg(tree)
    val1 = sdfg1(A=A, B=B, cst=cst, N=A.shape[0])
    
    assert np.allclose(val0, ref)
    assert np.allclose(val1, ref)


def test_azimint_naive():

    N, npt = (dace.symbol(s) for s in ('N', 'npt'))
    @dace.program
    def dace_azimint_naive(data: dace.float64[N], radius: dace.float64[N]):
        rmax = np.amax(radius)
        res = np.zeros((npt, ), dtype=np.float64)
        for i in range(npt):
        # for i in dace.map[0:npt]:
            r1 = rmax * i / npt
            r2 = rmax * (i + 1) / npt
            mask_r12 = np.logical_and((r1 <= radius), (radius < r2))
            on_values = 0
            tmp = np.float64(0)
            for j in dace.map[0:N]:
                if mask_r12[j]:
                    tmp += data[j]
                    on_values += 1
            res[i] = tmp / on_values
        return res
    
    rng = np.random.default_rng(42)
    SN, Snpt = 1000, 10
    data, radius = rng.random((SN, )), rng.random((SN, ))
    ref = dace_azimint_naive(data, radius, npt=Snpt)

    sdfg0 = dace_azimint_naive.to_sdfg()
    tree = as_schedule_tree(sdfg0)
    sdfg1 = as_sdfg(tree)
    val = sdfg1(data=data, radius=radius, N=SN, npt=Snpt)
    
    assert np.allclose(val, ref)


if __name__ == "__main__":
    # test_map_with_tasklet_and_library()
    test_azimint_naive()
