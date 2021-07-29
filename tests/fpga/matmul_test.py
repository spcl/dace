# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
from dace.fpga_testing import fpga_test, import_sample
from dace.transformation.interstate import FPGATransformSDFG
import numpy as np
from pathlib import Path


@fpga_test(assert_ii_1=False)
def test_naive_matmul_fpga():
    matmul = import_sample(Path("simple") / "matmul.py")
    sdfg = matmul.matmul.to_sdfg()
    sdfg.apply_transformations(FPGATransformSDFG)

    n, k, m = 64, 64, 64

    A = np.random.rand(m, k).astype(np.float64)
    B = np.random.rand(k, n).astype(np.float64)
    C = np.zeros((m, n), dtype=np.float64)

    sdfg(A=A, B=B, C=C, N=n, K=k, M=m)

    expected = A @ B
    diff = np.linalg.norm(C - expected) / (m * n)

    assert diff <= 1e-6

    return sdfg


if __name__ == "__main__":
    test_matmul_fpga(None)
