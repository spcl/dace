# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
import numpy as np
import dace

from dace.transformation.dataflow import MapFusion
from dace.transformation.dataflow import StencilDetection
from dace.transformation.dataflow import SimpleTaskletFusion


@dace.program
def stencil1d(A: dace.float32[12], B: dace.float32[12]):
    B[1:-1] = 0.3333 * (A[:-2] + A[1:-1] + A[2:])


def test_stencil1d():
    sdfg = stencil1d.to_sdfg(strict=True)
    sdfg.apply_transformations_repeated(MapFusion)
    sdfg.apply_transformations(StencilDetection)
    sdfg.apply_transformations_repeated(SimpleTaskletFusion)
    A = np.arange(12, dtype=np.float32)
    ref = np.zeros((12, ), dtype=np.float32)
    sdfg(A=A, B=ref)

    sdfg.apply_transformations(StencilDetection)
    B = np.zeros((12, ), dtype=np.float32)
    sdfg(A=A, B=B)
    assert (np.allclose(B, ref))


if __name__ == '__main__':
    test_stencil1d()
