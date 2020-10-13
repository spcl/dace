"""
Helper functions for memory movements
"""
import dace
from dace.memlet import Memlet
import math
from dace.libraries.blas.utility.initialization import *
from dace import dtypes
import numpy as np



# ---------- ----------
# NUMPY
# ---------- ----------
def aligned_ndarray(arr, alignment=64):
    """
    Allocates a and returns a copy of ``arr`` as an ``alignment``-byte aligned
    array. Useful for aligned vectorized access.
    
    Based on https://stackoverflow.com/a/20293172/6489142
    """
    if (arr.ctypes.data % alignment) == 0:
        return arr

    extra = alignment // arr.itemsize
    buf = np.empty(arr.size + extra, dtype=arr.dtype)
    ofs = (-buf.ctypes.data % alignment) // arr.itemsize
    result = buf[ofs:ofs + arr.size].reshape(arr.shape)
    np.copyto(result, arr)
    assert (result.ctypes.data % alignment) == 0
    return result


# ---------- ----------
# SDFG
# ---------- ----------
def fpga_copy_CPU_to_global(sdfg, state, sources, sizes, types, bank=None, veclen=1):

    inputs = zip(sources, sizes, types, range(len(sources)))
    outputs = []
    names = []

    for src, size, dtype, i in inputs:

        dest = "f_" + src

        vecType = dace.vector(dtype, veclen)

        name, desc = sdfg.add_array(
            dest,
            shape=[size/veclen],
            dtype=vecType,
            storage=dtypes.StorageType.FPGA_Global,
            transient=True
        )
        if bank is not None:
            if isinstance(bank, list):
                desc.location = {"bank": bank[i]}
            else:
                desc.location = {"bank": bank}
            # print("Set bank of:", name, " to ", desc.location)

        cpu_in = state.add_read(src)
        fpga_out = state.add_write(dest)

        
        state.add_memlet_path(
            cpu_in, fpga_out,
            memlet=Memlet.simple(cpu_in.data, "0:{}/{}".format(size, veclen))
        )

        outputs.append(fpga_out)
        names.append(dest)

    return (outputs, names)



def fpga_copy_global_to_CPU(sdfg, state, destinations, sizes, types, bank=None, veclen=1):

    inputs = zip(destinations, sizes, types, range(len(sizes)))
    outputs = []
    names = []

    for dest, size, dtype, i in inputs:

        src = "fpga_" + dest

        vecType = dace.vector(dtype, veclen)

        name, desc = sdfg.add_array(
            src,
            shape=[size/veclen],
            dtype=vecType,
            storage=dtypes.StorageType.FPGA_Global,
            transient=True
        )
        if bank is not None:
            if isinstance(bank, list):
                desc.location = {"bank": bank[i]}
            else:
                desc.location = {"bank": bank}
            # print("Set bank of:", name, " to ", desc.location)

        fpga_in = state.add_read(src)
        cpu_out = state.add_write(dest)

        state.add_memlet_path(
            fpga_in, cpu_out,
            memlet=Memlet.simple(cpu_out.data, "0:{}/{}".format(size, veclen))
        )

        outputs.append(fpga_in)
        names.append(src)

    return (outputs, names)
