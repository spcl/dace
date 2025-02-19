# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
from dace.fpga_testing import fpga_test, import_sample
from pathlib import Path


@fpga_test()
def test_gemv_fpga():
    gemv = import_sample(Path("fpga") / "gemv_fpga.py")
    return gemv.run_gemv(1024, 1024, False)


if __name__ == "__main__":
    test_gemv_fpga(None)
