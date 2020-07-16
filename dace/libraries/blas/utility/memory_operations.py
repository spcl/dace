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
def fpga_copy_global_to_local(sdfg, state, src, memSize, dtype):

    descriptor = src + "_local"
    sdfg.add_array(descriptor,
        shape=[memSize],
        dtype=dtype,
        storage=dtypes.StorageType.FPGA_Local,
        transient=True
    )

    data_in = state.add_read(src)
    buf_out = state.add_write(descriptor)

    copyMap_entry, copyMap_exit = state.add_map(
        'copyToLocal_map',
        dict(i='0:{}'.format(memSize)),
        schedule=dtypes.ScheduleType.FPGA_Device
    )

    copy_tasklet = state.add_tasklet(
        'copyToLocal_task',
        ['inCon'],
        ['outCon'],
        'outCon = inCon'
    )

    state.add_memlet_path(
        data_in, copyMap_entry, copy_tasklet,
        dst_conn='inCon',
        memlet=Memlet.simple(data_in.data, 'i')
    )

    state.add_memlet_path(
        copy_tasklet, copyMap_exit, buf_out,
        src_conn='outCon',
        memlet=Memlet.simple(buf_out.data, 'i')
    )

    return buf_out, descriptor

def fpga_copy_global_to_register(sdfg, state, src, memSize, dtype):

    descriptor = src + "_reg"
    sdfg.add_array(descriptor,
        shape=[memSize],
        dtype=dtype,
        storage=dtypes.StorageType.FPGA_Registers,
        transient=True
    )

    data_in = state.add_read(src)
    buf_out = state.add_write(descriptor)

    copyMap_entry, copyMap_exit = state.add_map(
        'copyToLocal_map',
        dict(i_copy='0:{}'.format(memSize)),
        schedule=dtypes.ScheduleType.FPGA_Device
    )

    copy_tasklet = state.add_tasklet(
        'copyToLocal_task',
        ['inCon'],
        ['outCon'],
        'outCon = inCon'
    )

    state.add_memlet_path(
        data_in, copyMap_entry, copy_tasklet,
        dst_conn='inCon',
        memlet=Memlet.simple(data_in.data, 'i_copy')
    )

    state.add_memlet_path(
        copy_tasklet, copyMap_exit, buf_out,
        src_conn='outCon',
        memlet=Memlet.simple(buf_out.data, 'i_copy')
    )

    return buf_out, descriptor


def fpga_copy_CPU_to_global(sdfg, state, sources, sizes, types, bank=None):

    inputs = zip(sources, sizes, types, range(len(sources)))
    outputs = []
    names = []

    for src, size, dtype, i in inputs:

        dest = "f_" + src

        name, desc = sdfg.add_array(
            dest,
            shape=[size],
            dtype=dtype,
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
            memlet=Memlet.simple(cpu_in.data, "0:{}".format(size))
        )

        outputs.append(fpga_out)
        names.append(dest)

    return (outputs, names)



def fpga_copy_global_to_CPU(sdfg, state, destinations, sizes, types, bank=None):

    inputs = zip(destinations, sizes, types, range(len(sizes)))
    outputs = []
    names = []

    for dest, size, dtype, i in inputs:

        src = "fpga_" + dest

        name, desc = sdfg.add_array(
            src,
            shape=[size],
            dtype=dtype,
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
            memlet=Memlet.simple(cpu_out.data, "0:{}".format(size))
        )

        outputs.append(fpga_in)
        names.append(src)

    return (outputs, names)


def fpga_copy_global_to_local_subset(state, src, dest, size, start):

    x_in = state.add_read(src)
    x_buf = state.add_write(dest)

    copyMap_entry, copyMap_exit = state.add_map(
        'copyX_map',
        dict(k = '{0}:{0}+{1}'.format(start, size)),
        schedule=dtypes.ScheduleType.FPGA_Device,
        unroll=True
    )

    copyX_task = state.add_tasklet(
        'copyX_task',
        ['inCon'],
        ['outCon'],
        'outCon = inCon'
    )

    state.add_memlet_path(
        x_in, copyMap_entry, copyX_task,
        dst_conn='inCon',
        memlet=Memlet.simple(x_in.data, "k")
    )

    state.add_memlet_path(
        copyX_task, copyMap_exit, x_buf,
        src_conn='outCon',
        memlet=Memlet.simple(x_buf.data, "k - {}".format(start))
    )


def fpga_copy_global_to_local_tile(state, src, dest, rowSize, rowStart, colSize, colStart, globalCol):

    x_in = state.add_read(src)
    x_buf = state.add_write(dest)

    copyMap_entry, copyMap_exit = state.add_map(
        'copyGlobalToLocal_map_{}'.format(dest),
        dict(k_row = '{0}:{0}+{1}'.format(rowStart, rowSize)),
        schedule=dtypes.ScheduleType.FPGA_Device
    )

    copyMapCol_entry, copyMapCol_exit = state.add_map(
        'copyGlobalToLocalCol_map_{}'.format(dest),
        dict(k_col = '{0}:{0}+{1}'.format(colStart, colSize)),
        schedule=dtypes.ScheduleType.FPGA_Device,
        unroll=True
    )


    copyX_task = state.add_tasklet(
        'copyX_task',
        ['inCon'],
        ['outCon'],
        'outCon = inCon'
    )

    state.add_memlet_path(
        x_in, copyMap_entry,copyMapCol_entry, copyX_task,
        dst_conn='inCon',
        memlet=Memlet.simple(x_in.data, "k_row * {} + k_col".format(globalCol))
    )

    state.add_memlet_path(
        copyX_task, copyMapCol_exit, copyMap_exit, x_buf,
        src_conn='outCon',
        memlet=Memlet.simple(x_buf.data, "(k_row - {0}) * {1} + (k_col - {2})".format(rowStart, colSize, colStart))
    )
