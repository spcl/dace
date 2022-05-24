# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
import numpy as np
import dace

from scipy.signal import convolve2d

from dace.transformation.dataflow import OTFMapFusion


def count_maps(sdfg):
    maps = 0
    for state in sdfg.nodes():
        for node in state.nodes():
            if isinstance(node, dace.nodes.MapEntry):
                maps += 1

    return maps


@dace.program
def fusion_chain(A: dace.float64[10, 20], B: dace.float64[10, 20]):
    tmp1 = A * 2
    tmp2 = tmp1 * 4
    B[:] = tmp2 + 5


@dace.program
def fusion_permutation(A: dace.float64[10, 20], B: dace.float64[10, 20]):
    tmp = dace.define_local([10, 20], dtype=A.dtype)
    for i, j in dace.map[0:10, 0:20]:
        with dace.tasklet:
            a << A[i, j]
            b >> tmp[i, j]

            b = a * a

    for j, i in dace.map[0:20, 0:10]:
        with dace.tasklet:
            a << tmp[i, j]
            b >> B[i, j]

            b = a + 2


@dace.program
def fusion_recomputation(A: dace.float64[20, 20], B: dace.float64[16, 16]):
    tmp = dace.define_local([18, 18], dtype=A.dtype)
    for i, j in dace.map[1:19, 1:19]:
        with dace.tasklet:
            a0 << A[i - 1, j - 1]
            a1 << A[i - 1, j]
            a2 << A[i - 1, j + 1]
            a3 << A[i, j - 1]
            a4 << A[i, j]
            a5 << A[i, j + 1]
            a6 << A[i + 1, j - 1]
            a7 << A[i + 1, j]
            a8 << A[i + 1, j + 1]
            b >> tmp[i - 1, j - 1]

            b = (a0 + a1 + a2 + a3 + a4 + a5 + a6 + a7 + a8) / 9.0

    for i, j in dace.map[1:17, 1:17]:
        with dace.tasklet:
            a0 << tmp[i - 1, j - 1]
            a1 << tmp[i - 1, j]
            a2 << tmp[i - 1, j + 1]
            a3 << tmp[i, j - 1]
            a4 << tmp[i, j]
            a5 << tmp[i, j + 1]
            a6 << tmp[i + 1, j - 1]
            a7 << tmp[i + 1, j]
            a8 << tmp[i + 1, j + 1]
            b >> B[i - 1, j - 1]

            b = (a0 + a1 + a2 + a3 + a4 + a5 + a6 + a7 + a8) / 9.0


def test_fusion_chain():
    sdfg = fusion_chain.to_sdfg()
    sdfg.simplify()

    assert count_maps(sdfg) == 3

    sdfg.apply_transformations(OTFMapFusion)
    assert count_maps(sdfg) == 2

    sdfg.apply_transformations(OTFMapFusion)
    assert count_maps(sdfg) == 1

    # Validate output

    A = np.random.rand(10, 20)
    B = np.zeros_like(A)
    target = A * 2 * 4 + 5
    sdfg(A=A, B=B)

    diff = np.linalg.norm(target - B)
    assert diff <= 1e-4


def test_fusion_permutation():
    sdfg = fusion_permutation.to_sdfg()
    sdfg.simplify()

    assert count_maps(sdfg) == 2

    sdfg.apply_transformations(OTFMapFusion)
    assert count_maps(sdfg) == 1

    # Validate output

    A = np.random.rand(10, 20)
    B = np.zeros_like(A)
    target = A * A + 2
    sdfg(A=A, B=B)

    diff = np.linalg.norm(target - B)
    assert diff <= 1e-4


def test_fusion_recomputation():
    sdfg = fusion_recomputation.to_sdfg()
    sdfg.simplify()

    assert count_maps(sdfg) == 2

    sdfg.apply_transformations(OTFMapFusion)
    assert count_maps(sdfg) == 1

    # Validate output

    A = np.random.rand(20, 20)
    B = np.zeros((16, 16), dtype=np.float64)

    mask = np.ones((3, 3)) / 9.0
    tmp = np.zeros((18, 18), dtype=A.dtype)
    target = np.zeros((16, 16), dtype=A.dtype)

    tmp = convolve2d(A, mask, mode="valid")
    target = convolve2d(tmp, mask, mode="valid")

    sdfg(A=A, B=B)

    diff = np.linalg.norm(target - B)
    assert diff <= 1e-4


if __name__ == '__main__':
    test_fusion_chain()
