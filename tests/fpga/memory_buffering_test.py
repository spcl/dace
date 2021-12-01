# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
"""
Tests memory buffering in an FPGA SDFG, where memory is read and written using
512-bit wide accesses, and converted (using a "gearbox") to/from the vector
width used by the computational kernel.

Unfortunately this doesn't currently work for Intel, since Intel does not
support vectors of vectors in kernel code.
"""
import dace
from dace.fpga_testing import fpga_test, xilinx_test
from dace.libraries.standard import Gearbox
from dace.transformation.interstate import FPGATransformSDFG, InlineSDFG
import numpy as np

dtype = dace.float32
mem_width = 64 // dtype.bytes
n = dace.symbol("n")


def run_program(sdfg: dace.SDFG):
    size = 16 * mem_width
    input_array = np.ones((size, ), dtype.type)
    output_array = np.empty((size, ), dtype.type)
    sdfg(input_array_host=input_array, output_array_host=output_array, n=size)
    assert all(output_array == input_array + 1)


def memory_buffering(vec_width, use_library_node, elementwise):

    gear_factor = mem_width // vec_width
    kernel_type = dace.vector(dtype, vec_width)
    if elementwise:
        memory_type = dace.vector(dtype, mem_width)
    else:
        memory_type = dace.vector(kernel_type, gear_factor)
    sdfg = dace.SDFG("memory_buffering_library_node")
    state = sdfg.add_state("memory_buffering_library_node")

    sdfg.add_array("input_array", (n / mem_width, ),
                   memory_type,
                   transient=True,
                   storage=dace.StorageType.FPGA_Global)
    sdfg.add_array("output_array", (n / mem_width, ),
                   memory_type,
                   transient=True,
                   storage=dace.StorageType.FPGA_Global)
    sdfg.add_stream("read_to_gearbox",
                    memory_type,
                    transient=True,
                    storage=dace.StorageType.FPGA_Local)
    sdfg.add_stream("gearbox_to_kernel",
                    kernel_type,
                    transient=True,
                    storage=dace.StorageType.FPGA_Local)
    sdfg.add_stream("kernel_to_gearbox",
                    kernel_type,
                    transient=True,
                    storage=dace.StorageType.FPGA_Local)
    sdfg.add_stream("gearbox_to_write",
                    memory_type,
                    transient=True,
                    storage=dace.StorageType.FPGA_Local)

    # Read from memory
    memory_read = state.add_read("input_array")
    read_to_gearbox_write = state.add_write("read_to_gearbox")
    read_entry, read_exit = state.add_map(
        "read", {"i": f"0:n/{mem_width}"},
        schedule=dace.ScheduleType.FPGA_Device)
    read_tasklet = state.add_tasklet("read", {"mem"}, {"to_gearbox"},
                                     "to_gearbox = mem")
    state.add_memlet_path(memory_read,
                          read_entry,
                          read_tasklet,
                          dst_conn="mem",
                          memlet=dace.Memlet(f"input_array[i]"))
    state.add_memlet_path(read_tasklet,
                          read_exit,
                          read_to_gearbox_write,
                          src_conn="to_gearbox",
                          memlet=dace.Memlet(f"read_to_gearbox[0]"))

    # Gearbox input
    read_to_gearbox_read = state.add_read("read_to_gearbox")
    gearbox_to_kernel_write = state.add_write("gearbox_to_kernel")
    if use_library_node:
        read_gearbox = Gearbox(n / mem_width, name="read_gearbox")
        state.add_node(read_gearbox)
        state.add_memlet_path(read_to_gearbox_read,
                              read_gearbox,
                              dst_conn="from_memory",
                              memlet=dace.Memlet("read_to_gearbox[0]",
                                                 volume=n / mem_width))
        state.add_memlet_path(read_gearbox,
                              gearbox_to_kernel_write,
                              src_conn="to_kernel",
                              memlet=dace.Memlet("gearbox_to_kernel[0]",
                                                 volume=n / vec_width))
    else:
        sdfg.add_array("read_buffer", (1, ),
                       memory_type,
                       storage=dace.StorageType.FPGA_Local,
                       transient=True)
        read_buffer_read = state.add_read("read_buffer")
        read_buffer_write = state.add_write("read_buffer")
        read_gearbox_entry, read_gearbox_exit = state.add_map(
            "gearbox_read", {
                "i": f"0:n/{mem_width}",
                "j": f"0:{gear_factor}"
            },
            schedule=dace.ScheduleType.FPGA_Device)
        read_gearbox_tasklet = state.add_tasklet(
            "gearbox_read", {
                "from_memory": memory_type,
                "buffer_in": None
            }, {"to_kernel", "buffer_out"}, """\
wide = from_memory if j == 0 else buffer_in
to_kernel = wide[j]
buffer_out = wide""")
        state.add_memlet_path(read_to_gearbox_read,
                              read_gearbox_entry,
                              read_gearbox_tasklet,
                              dst_conn="from_memory",
                              memlet=dace.Memlet("read_to_gearbox[0]",
                                                 dynamic=True))
        state.add_memlet_path(read_buffer_read,
                              read_gearbox_entry,
                              read_gearbox_tasklet,
                              dst_conn="buffer_in",
                              memlet=dace.Memlet("read_buffer[0]"))
        state.add_memlet_path(read_gearbox_tasklet,
                              read_gearbox_exit,
                              gearbox_to_kernel_write,
                              src_conn="to_kernel",
                              memlet=dace.Memlet("gearbox_to_kernel[0]"))
        state.add_memlet_path(read_gearbox_tasklet,
                              read_gearbox_exit,
                              read_buffer_write,
                              src_conn="buffer_out",
                              memlet=dace.Memlet("read_buffer[0]"))

    # Some fictional compute
    gearbox_to_kernel_read = state.add_read("gearbox_to_kernel")
    kernel_to_gearbox_write = state.add_write("kernel_to_gearbox")
    compute_entry, compute_exit = state.add_map(
        "compute", {"i": f"0:n/{vec_width}"},
        schedule=dace.ScheduleType.FPGA_Device)
    compute_tasklet = state.add_tasklet("compute", {"val_in"}, {"val_out"},
                                        "val_out = val_in + 1")
    state.add_memlet_path(gearbox_to_kernel_read,
                          compute_entry,
                          compute_tasklet,
                          dst_conn="val_in",
                          memlet=dace.Memlet("gearbox_to_kernel[0]"))
    state.add_memlet_path(compute_tasklet,
                          compute_exit,
                          kernel_to_gearbox_write,
                          src_conn="val_out",
                          memlet=dace.Memlet("kernel_to_gearbox[0]"))

    # Gearbox output
    kernel_to_gearbox_read = state.add_write("kernel_to_gearbox")
    gearbox_to_write_write = state.add_read("gearbox_to_write")
    if use_library_node:
        write_gearbox = Gearbox(n / mem_width, name="write_gearbox")
        state.add_node(write_gearbox)
        state.add_memlet_path(kernel_to_gearbox_read,
                              write_gearbox,
                              dst_conn="from_kernel",
                              memlet=dace.Memlet("kernel_to_gearbox[0]",
                                                 volume=n / vec_width))
        state.add_memlet_path(write_gearbox,
                              gearbox_to_write_write,
                              src_conn="to_memory",
                              memlet=dace.Memlet("gearbox_to_write[0]",
                                                 volume=n / mem_width))
    else:
        sdfg.add_array("write_buffer", (1, ),
                       memory_type,
                       storage=dace.StorageType.FPGA_Local,
                       transient=True)
        write_buffer_read = state.add_read("write_buffer")
        write_buffer_write = state.add_write("write_buffer")
        write_gearbox_entry, write_gearbox_exit = state.add_map(
            "gearbox_write", {
                "i": f"0:n/{mem_width}",
                "j": f"0:{gear_factor}"
            },
            schedule=dace.ScheduleType.FPGA_Device)
        write_gearbox_tasklet = state.add_tasklet(
            "gearbox_write", {"from_kernel", "buffer_in"},
            {"to_memory", "buffer_out"}, f"""\
wide = buffer_in
wide[j] = from_kernel
if j == {gear_factor} - 1:
    to_memory = wide
buffer_out = wide""")
        state.add_memlet_path(kernel_to_gearbox_read,
                              write_gearbox_entry,
                              write_gearbox_tasklet,
                              dst_conn="from_kernel",
                              memlet=dace.Memlet("kernel_to_gearbox[0]"))
        state.add_memlet_path(write_buffer_read,
                              write_gearbox_entry,
                              write_gearbox_tasklet,
                              dst_conn="buffer_in",
                              memlet=dace.Memlet("write_buffer[0]"))
        state.add_memlet_path(write_gearbox_tasklet,
                              write_gearbox_exit,
                              gearbox_to_write_write,
                              src_conn="to_memory",
                              memlet=dace.Memlet("gearbox_to_write[0]",
                                                 dynamic=True))
        state.add_memlet_path(write_gearbox_tasklet,
                              write_gearbox_exit,
                              write_buffer_write,
                              src_conn="buffer_out",
                              memlet=dace.Memlet("write_buffer[0]"))

    # Write memory
    gearbox_to_write_read = state.add_read("gearbox_to_write")
    memory_write = state.add_write("output_array")
    write_entry, write_exit = state.add_map(
        "write", {"i": f"0:n/{mem_width}"},
        schedule=dace.ScheduleType.FPGA_Device)
    write_tasklet = state.add_tasklet("write", {"from_gearbox"}, {"mem"},
                                      "mem = from_gearbox")
    state.add_memlet_path(gearbox_to_write_read,
                          write_entry,
                          write_tasklet,
                          dst_conn="from_gearbox",
                          memlet=dace.Memlet("gearbox_to_write[0]"))
    state.add_memlet_path(write_tasklet,
                          write_exit,
                          memory_write,
                          src_conn="mem",
                          memlet=dace.Memlet("output_array[i]"))

    # Copy data to the FPGA
    sdfg.add_array("input_array_host", (n, ), dtype)
    pre_state = sdfg.add_state("host_to_device")
    host_to_device_read = pre_state.add_read("input_array_host")
    host_to_device_write = pre_state.add_write("input_array")
    pre_state.add_memlet_path(
        host_to_device_read,
        host_to_device_write,
        memlet=dace.Memlet(f"input_array[0:n/{mem_width}]"))

    # Copy data back to the host
    sdfg.add_array("output_array_host", (n, ), dtype)
    post_state = sdfg.add_state("device_to_host")
    device_to_host_read = post_state.add_read("output_array")
    device_to_host_write = post_state.add_write("output_array_host")
    post_state.add_memlet_path(
        device_to_host_read,
        device_to_host_write,
        memlet=dace.Memlet(f"output_array[0:n/{mem_width}]"))

    # Link states
    sdfg.add_edge(pre_state, state, dace.InterstateEdge())
    sdfg.add_edge(state, post_state, dace.InterstateEdge())

    run_program(sdfg)

    return sdfg


@xilinx_test()
def test_memory_buffering_manual():
    return memory_buffering(4, False, False)


@xilinx_test()
def test_memory_buffering_manual_scalar():
    return memory_buffering(1, False, False)


@xilinx_test()
def test_memory_buffering_library_node():
    return memory_buffering(4, True, False)


@xilinx_test()
def test_memory_buffering_library_node_scalar():
    return memory_buffering(1, True, False)


@fpga_test()
def test_memory_buffering_library_node_elementwise():
    return memory_buffering(4, True, True)


@fpga_test()
def test_memory_buffering_library_node_elementwise_scalar():
    return memory_buffering(1, True, True)


if __name__ == "__main__":
    test_memory_buffering_manual(None)
    test_memory_buffering_library_node(None)
    test_memory_buffering_library_node_scalar(None)
    test_memory_buffering_library_node_elementwise(None)
