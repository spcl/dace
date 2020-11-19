# Copyright 2019-2020 ETH Zurich and the DaCe authors. All rights reserved.
"""
Helper functions for memory movements
"""
import math
import numpy as np

from dace import vector
from dace.memlet import Memlet
from dace import dtypes


# ---------- ----------
# SDFG
# ---------- ----------
def fpga_copy_cpu_to_global(sdfg,
                            state,
                            sources,
                            sizes,
                            types,
                            bank=None,
                            veclen=1):
    """Copies memory from the CPU host to FPGA Global memory.
    """

    inputs = zip(sources, sizes, types, range(len(sources)))
    outputs = []
    names = []

    for src, size, dtype, i in inputs:

        dest = "f_" + src

        vec_type = vector(dtype, veclen)

        name, desc = sdfg.add_array(dest,
                                    shape=[size / veclen],
                                    dtype=vec_type,
                                    storage=dtypes.StorageType.FPGA_Global,
                                    transient=True)
        if bank is not None:
            if isinstance(bank, list):
                desc.location = {"bank": bank[i]}
            else:
                desc.location = {"bank": bank}
            # print("Set bank of:", name, " to ", desc.location)

        cpu_in = state.add_read(src)
        fpga_out = state.add_write(dest)

        state.add_memlet_path(cpu_in,
                              fpga_out,
                              memlet=Memlet.simple(
                                  cpu_in.data, "0:{}/{}".format(size, veclen)))

        outputs.append(fpga_out)
        names.append(dest)

    return (outputs, names)


def fpga_copy_global_to_cpu(sdfg,
                            state,
                            destinations,
                            sizes,
                            types,
                            bank=None,
                            veclen=1):
    """Copies memory from FPGA Global memory back to CPU host memory.
    """

    inputs = zip(destinations, sizes, types, range(len(sizes)))
    outputs = []
    names = []

    for dest, size, dtype, i in inputs:

        src = "fpga_" + dest

        vec_type = vector(dtype, veclen)

        name, desc = sdfg.add_array(src,
                                    shape=[size / veclen],
                                    dtype=vec_type,
                                    storage=dtypes.StorageType.FPGA_Global,
                                    transient=True)
        if bank is not None:
            if isinstance(bank, list):
                desc.location = {"bank": bank[i]}
            else:
                desc.location = {"bank": bank}
            # print("Set bank of:", name, " to ", desc.location)

        fpga_in = state.add_read(src)
        cpu_out = state.add_write(dest)

        state.add_memlet_path(fpga_in,
                              cpu_out,
                              memlet=Memlet.simple(
                                  cpu_out.data, "0:{}/{}".format(size, veclen)))

        outputs.append(fpga_in)
        names.append(src)

    return (outputs, names)
