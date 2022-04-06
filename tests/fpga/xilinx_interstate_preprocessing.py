# Copyright 2019-2022 ETH Zurich and the DaCe authors. All rights reserved.
"""
    Xilinx SDFG preprocessing replaces variable names in interstate edges with their
    qualified name (i.e.,  appending '_in'/'_out' to the container name).
    This behaviour is tested by `tests/npbench/nussinov_test.py`.

    This test (issue #972) tests that type inference for interstate edge variables (triggered
    after preprocessing) is done correctly.

"""

import dace.dtypes
import numpy as np
import dace as dc
import pytest
import argparse
from dace.fpga_testing import fpga_test
from dace.transformation.interstate import FPGATransformSDFG, InlineSDFG

N0 = dace.symbol('N0', dtype=dace.uint32)
N1 = dace.symbol('N1', dtype=dace.uint32)


@dace.program
def ComputeRestriction(Axf: dace.float64[N0], rc: dace.float64[N1], f2c: dace.uint32[N0], rf: dace.float64[N0]):
    for i_res in range(0, N1):
        problem_variable = f2c[i_res]
        rc[i_res] = rf[problem_variable] - Axf[problem_variable]


@dace.program
def program(f2cOperator0: dace.uint32[N0], rc1: dace.float64[N1], Axf0: dace.float64[N0], x: dace.float64[N0]):
    ComputeRestriction(Axf=Axf0, rc=rc1, f2c=f2cOperator0, rf=x)


@fpga_test(assert_ii_1=False, intel=False)
def test_type_inference():
    sdfg = program.to_sdfg()
    sdfg.apply_transformations(FPGATransformSDFG)
    f2cOperator = np.array([0, 1, 2, 3], dtype=np.uint32)
    rc = np.array([42, 42, 42, 42], dtype=np.float64)
    Axf = np.array([0, 2, 4, 6], dtype=np.float64)
    x = np.array([0, 1, 2, 3], dtype=np.float64)
    sdfg(f2cOperator0=f2cOperator, rc1=rc, Axf0=Axf, x=x, N0=4, N1=4)

    assert ((rc == np.array([0, -1, -2, -3])).all())


if __name__ == "__main__":
    test_type_inference(None)
