# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
from dace.fpga_testing import fpga_test
from dace.transformation.dataflow import MapFusion
from dace.transformation.interstate import FPGATransformSDFG
from mapfusion_test import multiple_fusions, fusion_with_transient
import numpy as np


@fpga_test()
def test_multiple_fusions_fpga():
    sdfg = multiple_fusions.to_sdfg()
    sdfg.coarsen_dataflow()
    assert sdfg.apply_transformations_repeated(MapFusion) >= 2
    assert sdfg.apply_transformations_repeated(FPGATransformSDFG) == 1
    A = np.random.rand(10, 20).astype(np.float32)
    B = np.zeros_like(A)
    C = np.zeros_like(A)
    out = np.zeros(shape=1, dtype=np.float32)
    sdfg(A=A, B=B, C=C, out=out)
    diff1 = np.linalg.norm(A * A + 1 - B)
    diff2 = np.linalg.norm(A * A + 2 - C)
    assert diff1 <= 1e-4
    assert diff2 <= 1e-4
    return sdfg


@fpga_test()
def test_fusion_with_transient_fpga():
    A = np.random.rand(2, 20)
    expected = A * A * 2
    sdfg = fusion_with_transient.to_sdfg()
    sdfg.coarsen_dataflow()
    assert sdfg.apply_transformations_repeated(MapFusion) >= 2
    assert sdfg.apply_transformations_repeated(FPGATransformSDFG) == 1
    sdfg(A=A)
    assert np.allclose(A, expected)
    return sdfg


if __name__ == "__main__":
    multiple_fusions_fpga(None)
    fusion_with_transient_fpga(None)
