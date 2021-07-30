# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
from dace.fpga_testing import fpga_test, import_sample
from pathlib import Path


@fpga_test(assert_ii_1=False)
def test_jacobi_fpga():
    jacobi = import_sample(Path("fpga") / "jacobi_fpga_systolic.py")
    return jacobi.run_jacobi(64, 128, 16, 4)


if __name__ == "__main__":
    test_jacobi_fpga(None)
