#!/usr/bin/env python3
# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
from pathlib import Path

import dace
from dace.fpga_testing import fpga_test, import_sample


@fpga_test(assert_ii_1=False)
def test_spmv_fpga():
    spmv = import_sample(Path("fpga") / "spmv_fpga_stream.py")
    return spmv.run_spmv(64, 64, 640, False)


if __name__ == "__main__":
    test_spmv_fpga(None)
