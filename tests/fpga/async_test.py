# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
import os
import multiprocessing as mp

from simple_systolic_array_test import P, N, make_sdfg
from dace.config import Config
from dace.fpga_testing import fpga_test
import dace.dtypes

import numpy as np


def run_test(do_async):

    N.set(128)
    P.set(4)
    A = np.empty((N.get()), dtype=np.int32)

    Config.set("compiler", "intel_fpga", "launch_async", value=do_async)

    name = "async_test"
    sdfg = make_sdfg(name)
    sdfg.specialize({"P": P.get(), "N": N.get()})
    # We don't care about the result, as long as it compiles and runs
    sdfg(A=A)

    return sdfg


@fpga_test()
def test_async_fpga_true():
    return run_test(True)


@fpga_test()
def test_async_fpga_false():
    return run_test(False)


if __name__ == "__main__":
    test_async_fpga_true(None)
    # test_async_fpga_false(None)
