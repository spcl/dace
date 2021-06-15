# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
from __future__ import print_function

import argparse
import dace
import math
import numpy as np

from dace.dtypes import StorageType, Language
from dace.sdfg import SDFG
from dace.memlet import Memlet
from dace.subsets import Indices

N = dace.symbol("N", positive=True)
W = dace.symbol("W", positive=True)
dtype = dace.float32
vtype = dace.vector(dtype, W)
buffer_size = 2048  # Of internal FIFOs


def make_copy_to_device(sdfg):

    pre_state = sdfg.add_state("copy_to_device")

    A_host = pre_state.add_array("A", [N / W], dtype=vtype)

    A_device = pre_state.add_array("A_device", [N / W],
                                   dtype=vtype,
                                   transient=True,
                                   storage=StorageType.FPGA_Global)

    pre_state.add_edge(A_host, None, A_device, None,
                       dace.memlet.Memlet.simple(A_device, "0:N/W"))

    return pre_state


def make_copy_to_host(sdfg):

    post_state = sdfg.add_state("copy_to_host")

    B_device = post_state.add_array("B_device", [N / W],
                                    dtype=vtype,
                                    transient=True,
                                    storage=StorageType.FPGA_Global)
    outsize_device = post_state.add_array("outsize_device", [1],
                                          dtype=dace.uint32,
                                          transient=True,
                                          storage=StorageType.FPGA_Global)

    B_host = post_state.add_array("B", [N / W], dtype=vtype)
    outsize_host = post_state.add_array("outsize", [1], dtype=dace.uint32)

    post_state.add_edge(B_device, None, B_host, None,
                        dace.memlet.Memlet.simple(B_device, "0:N/W"))
    post_state.add_edge(outsize_device, None, outsize_host, None,
                        dace.memlet.Memlet.simple(outsize_device, "0"))

    return post_state


def make_iteration_space(sdfg, add_one=False):

    loop_begin = sdfg.add_state("loop_begin")
    loop_entry = sdfg.add_state("loop_entry")
    loop_body = sdfg.add_state("loop_body")
    loop_end = sdfg.add_state("loop_end")

    sdfg.add_edge(loop_begin, loop_entry,
                  dace.sdfg.InterstateEdge(assignments={"i": 0}))

    sdfg.add_edge(
        loop_entry, loop_body,
        dace.sdfg.InterstateEdge(
            condition=dace.properties.CodeProperty.from_string(
                "i < N" + (" + W" if add_one else ""),
                language=dace.dtypes.Language.Python)))

    sdfg.add_edge(
        loop_entry, loop_end,
        dace.sdfg.InterstateEdge(
            condition=dace.properties.CodeProperty.from_string(
                "i >= N" + (" + W" if add_one else ""),
                language=dace.dtypes.Language.Python)))

    sdfg.add_edge(loop_body, loop_entry,
                  dace.sdfg.InterstateEdge(assignments={"i": "i + W"}))

    return loop_body


def make_compute_state(state):

    A_pipe = state.add_stream("_A_pipe",
                              dtype=vtype,
                              buffer_size=buffer_size,
                              storage=StorageType.FPGA_Global)
    B_pipe = state.add_stream("_B_pipe",
                              dtype=vtype,
                              buffer_size=buffer_size,
                              storage=StorageType.FPGA_Global)
    valid_pipe = state.add_stream("_valid_pipe",
                                  dtype=dace.bool,
                                  storage=StorageType.FPGA_Global)
    ratio = state.add_scalar("ratio_nested",
                             dtype=dtype,
                             storage=StorageType.FPGA_Global)

    count = state.add_scalar("count",
                             dtype=dace.uint32,
                             storage=StorageType.FPGA_Registers)

    # Inline Vivado HLS code
    code = """\
constexpr int num_stages = 2 * W;
using Stage_t = ap_uint<hlslib::ConstLog2(num_stages)>;
using Count_t = ap_uint<hlslib::ConstLog2(W)>;
using Vec_t = typename std::remove_reference<decltype(A_pipe_in)>::type::Data_t;
using Data_t = decltype(ratio_in);

Vec_t stages[num_stages];
#pragma HLS ARRAY_PARTITION variable=stages complete

Stage_t num_shifts[num_stages][W];
#pragma HLS ARRAY_PARTITION variable=num_shifts complete

bool non_zero[num_stages][W];
#pragma HLS ARRAY_PARTITION variable=non_zero complete

Vec_t output, next;

Count_t elements_in_output = 0;

unsigned _count = 0;

for (unsigned i = 0; i < N / W + 1; ++i) {
  #pragma HLS PIPELINE II=1

  if (i < N / W) {
    stages[0] = A_pipe_in.pop();
  } else {
    stages[0] = Vec_t(static_cast<Data_t>(0));
  }

  // Initialize shift counters
  Stage_t empty_slots_left = 0;
  const Count_t elements_in_output_local = elements_in_output;
  Count_t additional_elements = 0;
  for (unsigned w = 0; w < W; ++w) {
    #pragma HLS UNROLL
    if (stages[0][w] < ratio_in) {
      ++empty_slots_left;
      non_zero[0][w] = false;
      num_shifts[0][w] = 0;
    } else {
      non_zero[0][w] = true;
      num_shifts[0][w] =
          W - elements_in_output_local + empty_slots_left;
      ++additional_elements;
    }
  }
  elements_in_output =
      (elements_in_output_local + additional_elements) % W;

  // Merge stages
  for (int s = 1; s < num_stages; ++s) {
    #pragma HLS UNROLL
    for (unsigned w = 0; w < W; ++w) {
      #pragma HLS UNROLL
      // If we're already in the correct place, just propagate forward
      if (num_shifts[s - 1][w] == 0 && non_zero[s - 1][w]) {
        stages[s][w] = stages[s - 1][w];
        num_shifts[s][w] = 0;
        non_zero[s][w] = true;
      } else {
        // Otherwise shift if the value is non-zero
        const Stage_t shifts = num_shifts[s - 1][(w + 1) % W];
        if (shifts > 0) {
          stages[s][w] = stages[s - 1][(w + 1) % W];
          num_shifts[s][w] = shifts - 1;
          non_zero[s][w] = true;
        } else {
          stages[s][w] = 0;
          num_shifts[s][w] = 0;
          non_zero[s][w] = false;
        }
      }
    }
  }

  // Fill up vector
  Count_t num_curr = 0;
  Count_t num_next = 0;
  for (unsigned w = 0; w < W; ++w) {
    #pragma HLS UNROLL
    const bool is_taken = w < elements_in_output_local;
    const bool is_non_zero = non_zero[num_stages - 1][w];
    if (!is_taken) {
      if (is_non_zero) {
        ++num_curr;
        output[w] = stages[num_stages - 1][w];
      }
    } else {
      ++num_curr;
      if (is_non_zero) {
        next[w] = stages[num_stages - 1][w];
        ++num_next;
      }
    }
  }

  const bool is_full = num_curr == W;
  const bool last_iter = i == N / W;
  B_pipe_out.push(output);
  valid_pipe_out.push(is_full || last_iter);
  if (is_full) {
    output = next;
    next = Vec_t(Data_t(0));
    ++_count;
  }

} // End loop

count_out = std::min<unsigned>(W * _count + elements_in_output, N);"""

    tasklet = state.add_tasklet("filter", {"A_pipe_in", "ratio_in"},
                                {"B_pipe_out", "valid_pipe_out", "count_out"},
                                code,
                                language=Language.CPP)

    state.add_memlet_path(A_pipe,
                          tasklet,
                          dst_conn="A_pipe_in",
                          memlet=Memlet.simple(A_pipe, "0", num_accesses="N"))
    state.add_memlet_path(ratio,
                          tasklet,
                          dst_conn="ratio_in",
                          memlet=Memlet.simple(ratio, "0"))
    state.add_memlet_path(tasklet,
                          B_pipe,
                          src_conn="B_pipe_out",
                          memlet=Memlet.simple(B_pipe,
                                               "0",
                                               num_accesses="N",
                                               dynamic=True))
    state.add_memlet_path(tasklet,
                          valid_pipe,
                          src_conn="valid_pipe_out",
                          memlet=Memlet.simple(valid_pipe,
                                               "0",
                                               dynamic=True,
                                               num_accesses="N"))
    state.add_memlet_path(tasklet,
                          count,
                          src_conn="count_out",
                          memlet=Memlet.simple(count, "0"))


def make_compute_sdfg():

    sdfg = SDFG("filter_compute")

    state = sdfg.add_state("compute")

    make_compute_state(state)

    return sdfg


def make_write_sdfg():

    sdfg = SDFG("filter_write")

    loop_begin = sdfg.add_state("loop_begin")
    loop_entry = sdfg.add_state("loop_entry")
    state = sdfg.add_state("loop_body")
    loop_end = sdfg.add_state("loop_end")

    i_write_zero = loop_begin.add_scalar("i_write",
                                         dtype=dace.dtypes.uint32,
                                         transient=True,
                                         storage=StorageType.FPGA_Registers)
    zero_tasklet = loop_begin.add_tasklet("zero", {}, {"i_write_out"},
                                          "i_write_out = 0")
    loop_begin.add_memlet_path(zero_tasklet,
                               i_write_zero,
                               src_conn="i_write_out",
                               memlet=Memlet.simple(i_write_zero, "0"))

    sdfg.add_edge(loop_begin, loop_entry,
                  dace.sdfg.InterstateEdge(assignments={"i": 0}))

    sdfg.add_edge(
        loop_entry, state,
        dace.sdfg.InterstateEdge(
            condition=dace.properties.CodeProperty.from_string(
                "i < N + W", language=dace.dtypes.Language.Python)))

    sdfg.add_edge(
        loop_entry, loop_end,
        dace.sdfg.InterstateEdge(
            condition=dace.properties.CodeProperty.from_string(
                "i >= N + W", language=dace.dtypes.Language.Python)))

    sdfg.add_edge(state, loop_entry,
                  dace.sdfg.InterstateEdge(assignments={"i": "i + W"}))

    B = state.add_array("B_mem", [N / W],
                        dtype=vtype,
                        storage=StorageType.FPGA_Global)
    B_pipe = state.add_stream("_B_pipe",
                              dtype=vtype,
                              buffer_size=buffer_size,
                              storage=StorageType.FPGA_Local)
    valid_pipe = state.add_stream("_valid_pipe",
                                  dtype=dace.dtypes.bool,
                                  buffer_size=buffer_size,
                                  storage=StorageType.FPGA_Local)
    i_write_in = state.add_scalar("i_write",
                                  dtype=dace.dtypes.uint32,
                                  transient=True,
                                  storage=StorageType.FPGA_Registers)
    i_write_out = state.add_scalar("i_write",
                                   dtype=dace.dtypes.uint32,
                                   transient=True,
                                   storage=StorageType.FPGA_Registers)

    tasklet = state.add_tasklet(
        "write", {"b_in", "valid_in", "i_write_in"}, {"b_out", "i_write_out"},
        "if valid_in:"
        "\n\tb_out[i_write_in] = b_in"
        "\n\ti_write_out = i_write_in + 1"
        "\nelse:"
        "\n\ti_write_out = i_write_in")

    state.add_memlet_path(B_pipe,
                          tasklet,
                          dst_conn="b_in",
                          memlet=Memlet.simple(B_pipe, "0"))
    state.add_memlet_path(valid_pipe,
                          tasklet,
                          dst_conn="valid_in",
                          memlet=Memlet.simple(valid_pipe, "0"))
    state.add_memlet_path(i_write_in,
                          tasklet,
                          dst_conn="i_write_in",
                          memlet=Memlet.simple(i_write_in, "0"))
    state.add_memlet_path(tasklet,
                          i_write_out,
                          src_conn="i_write_out",
                          memlet=Memlet.simple(i_write_out, "0"))
    state.add_memlet_path(tasklet,
                          B,
                          src_conn="b_out",
                          memlet=Memlet.simple(B, "0:N"))

    return sdfg


def make_main_state(sdfg):

    state = sdfg.add_state("filter")

    A = state.add_array("A_device", [N / W],
                        dtype=vtype,
                        transient=True,
                        storage=StorageType.FPGA_Global)
    ratio = state.add_scalar("ratio",
                             storage=StorageType.FPGA_Global,
                             dtype=dtype)

    outsize = state.add_array("outsize_device", [1],
                              dtype=dace.uint32,
                              transient=True,
                              storage=StorageType.FPGA_Global)
    B = state.add_array("B_device", [N / W],
                        dtype=vtype,
                        transient=True,
                        storage=StorageType.FPGA_Global)

    A_pipe_in = state.add_stream("A_pipe",
                                 dtype=vtype,
                                 buffer_size=buffer_size,
                                 transient=True,
                                 storage=StorageType.FPGA_Local)
    A_pipe_out = state.add_stream("A_pipe",
                                  dtype=vtype,
                                  buffer_size=buffer_size,
                                  transient=True,
                                  storage=StorageType.FPGA_Local)
    B_pipe_in = state.add_stream("B_pipe",
                                 dtype=vtype,
                                 buffer_size=buffer_size,
                                 transient=True,
                                 storage=StorageType.FPGA_Local)
    B_pipe_out = state.add_stream("B_pipe",
                                  dtype=vtype,
                                  buffer_size=buffer_size,
                                  transient=True,
                                  storage=StorageType.FPGA_Local)
    valid_pipe_in = state.add_stream("valid_pipe",
                                     dtype=dace.dtypes.bool,
                                     buffer_size=buffer_size,
                                     transient=True,
                                     storage=StorageType.FPGA_Local)
    valid_pipe_out = state.add_stream("valid_pipe",
                                      dtype=dace.dtypes.bool,
                                      buffer_size=buffer_size,
                                      transient=True,
                                      storage=StorageType.FPGA_Local)

    compute_sdfg = make_compute_sdfg()
    compute_tasklet = state.add_nested_sdfg(compute_sdfg, sdfg,
                                            {"_A_pipe", "ratio_nested"},
                                            {"_B_pipe", "_valid_pipe", "count"})

    write_sdfg = make_write_sdfg()
    write_tasklet = state.add_nested_sdfg(write_sdfg, sdfg,
                                          {"_B_pipe", "_valid_pipe"}, {"B_mem"})

    state.add_memlet_path(A,
                          A_pipe_out,
                          memlet=Memlet.simple(A, "0:N/W", num_accesses="N/W"))

    state.add_memlet_path(A_pipe_in,
                          compute_tasklet,
                          dst_conn="_A_pipe",
                          memlet=Memlet.simple(A_pipe_in,
                                               "0",
                                               num_accesses="N/W"))
    state.add_memlet_path(ratio,
                          compute_tasklet,
                          dst_conn="ratio_nested",
                          memlet=Memlet.simple(ratio, "0"))
    state.add_memlet_path(compute_tasklet,
                          B_pipe_out,
                          src_conn="_B_pipe",
                          memlet=Memlet.simple(B_pipe_out,
                                               "0",
                                               num_accesses="N/W",
                                               dynamic=True))
    state.add_memlet_path(compute_tasklet,
                          valid_pipe_out,
                          src_conn="_valid_pipe",
                          memlet=Memlet.simple(valid_pipe_out,
                                               "0",
                                               dynamic=True,
                                               num_accesses="N"))
    state.add_memlet_path(compute_tasklet,
                          outsize,
                          src_conn="count",
                          memlet=Memlet.simple(outsize, "0"))

    state.add_memlet_path(B_pipe_in,
                          write_tasklet,
                          dst_conn="_B_pipe",
                          memlet=Memlet.simple(B_pipe_in,
                                               "0",
                                               num_accesses="N/W"))
    state.add_memlet_path(valid_pipe_in,
                          write_tasklet,
                          dst_conn="_valid_pipe",
                          memlet=Memlet.simple(valid_pipe_in,
                                               "0",
                                               num_accesses="N"))
    state.add_memlet_path(write_tasklet,
                          B,
                          src_conn="B_mem",
                          memlet=Memlet.simple(B, "0:N/W", dynamic=True))

    return state


def make_sdfg(specialize):

    if not specialize:
        sdfg = dace.SDFG("filter_fpga_vectorized_{}".format(W.get()))
    else:
        sdfg = dace.SDFG("filter_fpga_vectorized_{}_{}".format(
            W.get(), N.get()))

    copy_to_device_state = make_copy_to_device(sdfg)
    compute_state = make_main_state(sdfg)
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
    parser.add_argument("W", type=int)
    parser.add_argument("ratio", type=float)
    parser.add_argument("-specialize",
                        default=False,
                        action="store_true",
                        help="Fix all symbols at compile time/in hardware")
    args = vars(parser.parse_args())

    # Specialize vector width regardless
    W.set(args["W"])
    num_stages = 2 * W.get() - 1
    vtype.veclen = W.get()

    if args["specialize"]:
        N.set(args["N"])
        sdfg = make_sdfg(True)
        sdfg.specialize(dict(W=W, N=N))
    else:
        sdfg = make_sdfg(False)
        sdfg.specialize(dict(W=W))
        N.set(args["N"])
    sdfg.add_constant("num_stages", dace.int32(num_stages))

    ratio = dtype(args["ratio"])

    print("Predicate-Based Filter. size={}, ratio={} ({}specialized)".format(
        N.get(), ratio, "" if args["specialize"] else "not "))

    A = dace.ndarray([N], dtype=dtype)
    B = dace.ndarray([N], dtype=dtype)
    outsize = dace.scalar(dace.uint32)
    outsize[0] = 0

    A[:] = np.random.rand(N.get()).astype(dtype.type)
    B[:] = dtype(0)

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

    if len(filtered) == 0:
        print("==== Program end ====")
        exit(0)

    diff = np.abs(filtered - B[:outsize[0]])
    mismatches = np.transpose(np.nonzero(diff > 1e-3 * filtered))
    if mismatches.size > 0:
        print("Mismatches found:")
        for i in mismatches:
            print("{} (should be {})".format(B[i], filtered[i]))
        totalitems = min(outsize[0], N.get())
        print('DaCe:', B[:totalitems].view(type=np.ndarray))
        print('Regression:', filtered.view(type=np.ndarray))
    else:
        print("Results successfully verified.")

    print("==== Program end ====")
    exit(1 if mismatches.size > 0 else 0)
