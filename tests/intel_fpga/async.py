# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
import os
import multiprocessing as mp

from simple_systolic_array import P, N, make_sdfg
from dace.config import Config
import dace.dtypes

import numpy as np

def run_test(do_async):

    Config.set("compiler", "intel_fpga", "launch_async", value=do_async)

    name = "async_test"
    sdfg = make_sdfg(name)
    sdfg.specialize({"P": P.get(), "N": N.get()})
    # We don't care about the result, as long as it compiles and runs
    sdfg(A=A)


if __name__ == "__main__":

    N.set(128)
    P.set(4)

    Config.set("compiler", "fpga_vendor", value="intel_fpga")
    Config.set("compiler", "intel_fpga", "mode", value="emulator")

    A = np.empty((N.get()), dtype=np.int32)

    for v in [False, True]:
        # Has to be a separate process, as the Intel FPGA runtime cannot be
        # initialized twice in the same executable
        p = mp.Process(target=run_test, args=(v,))
        p.start()
        p.join()
