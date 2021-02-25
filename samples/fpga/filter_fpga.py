# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
from __future__ import print_function

import argparse
import dace
import math
import numpy as np

N = dace.symbol("N", positive=True)


def make_copy_to_device(sdfg):

    pre_state = sdfg.add_state("copy_to_device")

    A_host = pre_state.add_read("A")

    A_device = pre_state.add_write("A_device")

    pre_state.add_edge(A_host, None, A_device, None,
                       dace.memlet.Memlet.simple(A_device, "0:N"))

    return pre_state


def make_copy_to_host(sdfg):

    post_state = sdfg.add_state("copy_to_host")

    B_device = post_state.add_read("B_device")
    outsize_device = post_state.add_read("outsize_device")

    B_host = post_state.add_write("B")
    outsize_host = post_state.add_write("outsize")

    post_state.add_edge(B_device, None, B_host, None,
                        dace.memlet.Memlet.simple(B_device, "0:N"))
    post_state.add_edge(outsize_device, None, outsize_host, None,
                        dace.memlet.Memlet.simple(outsize_device, "0"))

    return post_state


def make_compute_state(sdfg):

    state = sdfg.add_state("compute_state")

    A = state.add_read("A_device")
    ratio = state.add_read("ratio")

    outsize = state.add_write("outsize_device")
    B = state.add_write("B_device")

    for_loop_sdfg = make_nested_sdfg(state)
    nested_sdfg = state.add_nested_sdfg(for_loop_sdfg, sdfg,
                                        {"A_nested", "ratio_nested"},
                                        {"B_nested", "outsize_nested"})

    state.add_edge(A, None, nested_sdfg, "A_nested",
                   dace.memlet.Memlet.simple(A, "0:N"))
    state.add_edge(ratio, None, nested_sdfg, "ratio_nested",
                   dace.memlet.Memlet.simple(ratio, "0"))
    state.add_edge(nested_sdfg, "B_nested", B, None,
                   dace.memlet.Memlet.simple(B, "0:N"))
    state.add_edge(nested_sdfg, "outsize_nested", outsize, None,
                   dace.memlet.Memlet.simple(outsize, "0"))

    return state


def make_nested_sdfg(parent):

    sdfg = dace.SDFG("filter_nested")

    sdfg.add_scalar("outsize_buffer",
                    dtype=dace.uint32,
                    transient=True,
                    storage=dace.dtypes.StorageType.FPGA_Registers)
    sdfg.add_array("outsize_nested", [1],
                   dtype=dace.uint32,
                   storage=dace.dtypes.StorageType.FPGA_Global)
    sdfg.add_array("A_nested", [N],
                   dtype=dace.float32,
                   storage=dace.dtypes.StorageType.FPGA_Global)
    sdfg.add_array("B_nested", [N],
                   dtype=dace.float32,
                   storage=dace.dtypes.StorageType.FPGA_Global)
    sdfg.add_scalar("ratio_nested",
                    dtype=dace.float32,
                    storage=dace.dtypes.StorageType.FPGA_Global)

    set_zero = make_set_zero(sdfg)
    loop_entry = sdfg.add_state("loop_entry")
    loop_body = make_loop_body(sdfg)
    write_out_size = make_write_out_size(sdfg)

    sdfg.add_edge(set_zero, loop_entry,
                  dace.sdfg.InterstateEdge(assignments={"i": 0}))

    sdfg.add_edge(
        loop_entry, loop_body,
        dace.sdfg.InterstateEdge(
            condition=dace.properties.CodeProperty.from_string(
                "i < N", language=dace.dtypes.Language.Python)))

    sdfg.add_edge(
        loop_entry, write_out_size,
        dace.sdfg.InterstateEdge(
            condition=dace.properties.CodeProperty.from_string(
                "i >= N", language=dace.dtypes.Language.Python)))

    sdfg.add_edge(loop_body, loop_entry,
                  dace.sdfg.InterstateEdge(assignments={"i": "i + 1"}))

    return sdfg


def make_set_zero(sdfg):

    set_zero = sdfg.add_state("set_zero")
    tasklet = set_zero.add_tasklet("set_zero", {}, {"size_zero"},
                                   "size_zero = 0")
    outsize = set_zero.add_write("outsize_buffer")
    set_zero.add_edge(tasklet, "size_zero", outsize, None,
                      dace.memlet.Memlet.simple(outsize, "0"))

    return set_zero


def make_write_out_size(sdfg):

    write_out = sdfg.add_state("write_out")
    outsize_buffer = write_out.add_read("outsize_buffer")
    outsize = write_out.add_write("outsize_nested")
    write_out.add_edge(outsize_buffer, None, outsize, None,
                       dace.memlet.Memlet.simple(outsize, "0"))

    return write_out


def make_loop_body(sdfg):

    state = sdfg.add_state("loop_body")

    A = state.add_read("A_nested")
    B = state.add_write("B_nested")
    ratio = state.add_read("ratio_nested")

    outsize_buffer_in = state.add_read("outsize_buffer")
    outsize_buffer_out = state.add_write("outsize_buffer")

    tasklet = state.add_tasklet(
        "filter", {"a", "write_index", "r"}, {"b", "size_out"}, "if a > r:"
        "\n\tb[write_index] = a"
        "\n\tsize_out = write_index + 1")

    state.add_edge(A, None, tasklet, "a", dace.memlet.Memlet.simple(A, "i"))
    state.add_edge(ratio, None, tasklet, "r",
                   dace.memlet.Memlet.simple(ratio, "0"))
    state.add_edge(
        tasklet, "b", B, None,
        dace.memlet.Memlet.simple(B,
                                  dace.subsets.Range.from_array(B.desc(sdfg)),
                                  num_accesses=-1))
    state.add_edge(outsize_buffer_in, None, tasklet, "write_index",
                   dace.memlet.Memlet.simple(outsize_buffer_in, "0"))
    state.add_edge(
        tasklet, "size_out", outsize_buffer_out, None,
        dace.memlet.Memlet.simple(outsize_buffer_out, "0", num_accesses=-1))

    return state


def make_sdfg(specialize):

    if not specialize:
        sdfg = dace.SDFG("filter_fpga")
    else:
        sdfg = dace.SDFG("filter_fpga_{}".format(N.get()))

    sdfg.add_array("A_device", [N],
                   dtype=dace.float32,
                   transient=True,
                   storage=dace.dtypes.StorageType.FPGA_Global)
    sdfg.add_array("A", [N], dtype=dace.float32)

    sdfg.add_array("B_device", [N],
                   dtype=dace.float32,
                   transient=True,
                   storage=dace.dtypes.StorageType.FPGA_Global)
    sdfg.add_array("outsize_device", [1],
                   dtype=dace.uint32,
                   transient=True,
                   storage=dace.dtypes.StorageType.FPGA_Global)
    sdfg.add_array("B", [N], dtype=dace.float32)
    sdfg.add_array("outsize", [1], dtype=dace.uint32)
    sdfg.add_scalar("ratio",
                    storage=dace.dtypes.StorageType.FPGA_Global,
                    dtype=dace.float32)

    copy_to_device_state = make_copy_to_device(sdfg)
    compute_state = make_compute_state(sdfg)
    copy_to_host_state = make_copy_to_host(sdfg)

    sdfg.add_edge(copy_to_device_state, compute_state,
                  dace.sdfg.InterstateEdge())
    sdfg.add_edge(compute_state, copy_to_host_state, dace.sdfg.InterstateEdge())

    return sdfg


def regression(A, ratio):
    return A[np.where(A > ratio)]


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("N", type=int)
    parser.add_argument("ratio", type=float)
    parser.add_argument("-specialize",
                        default=False,
                        action="store_true",
                        help="Fix all symbols at compile time/in hardware")
    args = vars(parser.parse_args())

    N.set(args["N"])

    A = dace.ndarray([N], dtype=dace.float32)
    B = dace.ndarray([N], dtype=dace.float32)
    outsize = dace.scalar(dace.uint32)
    outsize[0] = 0

    ratio = np.float32(args["ratio"])

    print("Predicate-Based Filter. size={}, ratio={} ({}specialized)".format(
        N.get(), ratio, "" if args["specialize"] else "not "))

    A[:] = np.random.rand(N.get()).astype(dace.float32.type)
    B[:] = dace.float32(0)

    sdfg = make_sdfg(args["specialize"])
    if args["specialize"]:
        sdfg.specialize(dict(N=N))
        sdfg(A=A, B=B, outsize=outsize, ratio=ratio)
    else:
        sdfg(A=A, B=B, outsize=outsize, ratio=ratio, N=N)

    if dace.Config.get_bool('profiling'):
        dace.timethis('filter', 'numpy', 0, regression, A, ratio)

    filtered = regression(A, ratio)

    if len(filtered) != outsize[0]:
        print(
            "Difference in number of filtered items: %d (DaCe) vs. %d (numpy)" %
            (outsize[0], len(filtered)))
        totalitems = min(outsize[0], N.get())
        print('DaCe:', B[:totalitems].view(type=np.ndarray))
        print('Regression:', filtered.view(type=np.ndarray))
        exit(1)

    # Sort the outputs
    filtered = np.sort(filtered)
    B[:outsize[0]] = np.sort(B[:outsize[0]])

    if len(filtered) == 0:
        print("==== Program end ====")
        exit(0)

    diff = np.linalg.norm(filtered - B[:outsize[0]]) / float(outsize[0])
    print("Difference:", diff)
    if diff > 1e-5:
        totalitems = min(outsize[0], N.get())
        print('DaCe:', B[:totalitems].view(type=np.ndarray))
        print('Regression:', filtered.view(type=np.ndarray))

    print("==== Program end ====")
    exit(0 if diff <= 1e-5 else 1)
