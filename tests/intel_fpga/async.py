import os
import subprocess as sp

from simple_systolic_array import P, N, make_sdfg
from dace.config import Config
import dace.dtypes

import numpy as np

if __name__ == "__main__":

    N.set(128)
    P.set(4)

    Config.set("compiler", "fpga_vendor", value="intel_fpga")
    Config.set("compiler", "intel_fpga", "mode", value="emulator")

    A = np.empty((N.get()), dtype=np.int32)

    for v in [False, True]:

        Config.set("compiler", "intel_fpga", "launch_async", value=v)

        name = "async_test"
        sdfg = make_sdfg(name)
        sdfg.specialize({"P": P.get(), "N": N.get()})
        # We don't care about the result, as long as it compiles and runs
        sdfg(A=A)

