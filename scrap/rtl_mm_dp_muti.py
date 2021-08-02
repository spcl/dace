# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
#
#        ap_aclk               s_axis_a             s_axis_b           s_axis_c_in
#           │                     │                    │                    │
#           │                     │                    │                    │
#           │                     │32 bit              │64 bit              │64 bit
#           │               ┌─────▼──────┐             │                    │
#           │               │ Data       │             │                    │
#           │               │            │             │                    │
#           │               │ Doubler    │             │                    │
#           │               └─────┬──────┘             │                    │
#           │                     │64 bit              │                    │
#           │1x Freq              │                    │                    │
#  ┌────────┼────────────┬────────│──────────┬─────────│──────────┐         │
#  │        │            │        │          │         │          │         │
#  │ ┌──────▼─────┐      │  ┌─────▼──────┐   │   ┌─────▼──────┐   │   ┌─────▼──────┐
#  │ │Clock       │      └─►│            │   └──►│            │   └──►│            │
#  │ │            │2x Freq  │ Clock Sync │       │ Clock Sync │       │ Clock Sync │
#  │ │ Multiplier ├──────┬─►│            │   ┌──►│            │   ┌──►│            │
#  │ └────────────┘      │  └─────┬──────┘   │   └─────┬──────┘   │   └─────┬──────┘
#  │                     │        │          │         │          │         │
#  │                     ├────────│──────────┼─────────│──────────┤         │
#  │                     │        │          │         │          │         │
#  │                     │        │64 bit    │         │64 bit    │         │64 bit
#  │                     │  ┌─────▼──────┐   │   ┌─────▼──────┐   │   ┌─────▼──────┐
#  │                     │  │            │   │   │            │   │   │            │
#  │                     ├─►│ Data Issue │   └──►│ Data Issue │   └──►│ Data Issue │
#  │                     │  │            │       │            │       │            │
#  │                     │  └──────┬─────┘       └─────┬──────┘       └─────┬──────┘
#  │                     │         │32 bit             │                    │
#  │                     │         │     ┌─────────────┘                    │
#  │                     │         │     │                                  │
#  │                     │         │     │                                  │
#  │                     │     ┌───▼─────▼──┐                               │
#  │                     │     │            │                               │
#  │                     ├────►│ Multiplier │                               │
#  │                     │     │            │                               │
#  │                     │     └───────┬────┘                               │
#  │                     │             │                                    │
#  │                     │             │    ┌───────────────────────────────┘
#  │                     │             │    │
#  │                     │         ┌───▼────▼───┐
#  │                     │         │            │
#  │                     ├────────►│   Adder    │
#  │                     │         │            │
#  │                     │         └─────┬──────┘
#  │                     │               │
#  │                     │               │
#  │                     │               │32 bit
#  │                     │         ┌─────▼──────┐
#  │                     │         │ Data       │
#  │                     ├────────►│            │
#  │                     │         │ Packer     │
#  │                     │         └─────┬──────┘
#  │                     │               │
#  │                     │               │64 bit
#  │                     │         ┌─────▼──────┐
#  │                     └────────►│            │
#  │                               │ Clock Sync │
#  └──────────────────────────────►│            │
#                                  └─────┬──────┘
#                                        │
#                                        │
#                                        │
#                                        │
#                                        ▼
#                                   m_axis_c_out

import argparse
import dace
import numpy as np

N = dace.symbol("N")
K = dace.symbol("K")
M = dace.symbol("M")
twoM = dace.symbol("twoM")
P = dace.symbol("P")
twoP = dace.symbol("twoP")

sdfg = dace.SDFG("mm_rtl_dp_systolic_multi_fM_c0_ABC2P")

# vectorization
veclen = 2
sdfg.add_constant('VECLEN', veclen)

def make_copy_to_fpga_state(sdfg):

    ###########################################################################
    # Copy data to FPGA

    state = sdfg.add_state("copy_to_device")

    sdfg.add_array("A", [N, K], dtype=dace.float32)
    sdfg.add_array("B", [K, M // veclen],
                    dtype=dace.vector(dace.float32, veclen))
    sdfg.add_array("C", [N, M // veclen],
                    dtype=dace.vector(dace.float32, veclen))
    A_host = state.add_read("A")
    B_host = state.add_read("B")
    C_host = state.add_read("C")

    sdfg.add_array("A_device", [N, K],
                   dtype=dace.float32,
                   transient=True,
                   storage=dace.dtypes.StorageType.FPGA_Global)
    sdfg.add_array("B_device", [K, M // veclen],
                   dtype=dace.vector(dace.float32, veclen),
                   transient=True,
                   storage=dace.dtypes.StorageType.FPGA_Global)
    sdfg.add_array("C_device", [N, M // veclen],
                   dtype=dace.vector(dace.float32, veclen),
                   transient=True,
                   storage=dace.dtypes.StorageType.FPGA_Global)
    A_device = state.add_write("A_device")
    B_device = state.add_write("B_device")
    C_device = state.add_write("C_device")

    state.add_memlet_path(A_host,
                          A_device,
                          memlet=dace.Memlet("A_device[0:N, 0:K]"))
    state.add_memlet_path(B_host,
                          B_device,
                          memlet=dace.Memlet("B_device[0:K, 0:M//VECLEN]"))
    state.add_memlet_path(C_host,
                          C_device,
                          memlet=dace.Memlet("C_device[0:N, 0:M//VECLEN]"))

    return state


def make_copy_to_host_state(sdfg):

    ###########################################################################
    # Copy data to FPGA

    state = sdfg.add_state("copy_to_host")

    C_device = state.add_read("C_device")
    C_host = state.add_write("C")

    state.add_memlet_path(C_device, C_host, memlet=dace.Memlet("C[0:N, 0:M//VECLEN]"))

    return state


# reads A into A_pipe
def make_read_A(state):

    entry, exit = state.add_map("read_A", {
        "n0": "0:N/P",
        "k": "0:K",
        "n1": "0:P"
    },
                                schedule=dace.ScheduleType.FPGA_Device)

    mem = state.add_read("A_device")
    pipe = state.add_write("A_pipe")
    tasklet = state.add_tasklet("read_A", {"from_memory"}, {"to_kernel"},
                                "to_kernel = from_memory")

    state.add_memlet_path(mem,
                          entry,
                          tasklet,
                          dst_conn="from_memory",
                          memlet=dace.Memlet("A_device[n0 * P + n1, k]"))
    state.add_memlet_path(tasklet,
                          exit,
                          pipe,
                          src_conn="to_kernel",
                          memlet=dace.Memlet("A_pipe[0]"))


# reads B into B_pipe
def make_read_B(state):

    entry, exit = state.add_map("read_B", {
        "n": "0:N/P",
        "k": "0:K",
        "m": "0:M//VECLEN"
    },
                                schedule=dace.ScheduleType.FPGA_Device)

    mem = state.add_read("B_device")
    pipe = state.add_write("B_pipe")
    tasklet = state.add_tasklet("read_B", {"from_memory"}, {"to_kernel"},
                                "to_kernel = from_memory")

    state.add_memlet_path(mem,
                          entry,
                          tasklet,
                          dst_conn="from_memory",
                          memlet=dace.Memlet("B_device[k, m]"))
    state.add_memlet_path(tasklet,
                          exit,
                          pipe,
                          src_conn="to_kernel",
                          memlet=dace.Memlet("B_pipe[0]"))


# writes the output C_pipe[P-1] into C
def make_write_C(state):

    pipe = state.add_read("C_pipe")
    mem = state.add_write("C_device")

    state.add_memlet_path(pipe,
                          mem,
                          memlet=dace.Memlet("C_device[0:N, 0:M//VECLEN]",
                                             other_subset="P - 1"))

# does all preprocessing and forwarding for a
# produces onto a comp_A_stream which is feed to the multiply accumulate compute tasklet
def make_prep_a(sdfg, state):

    A_pipe_in = state.add_read("A_pipe")
    A_pipe_carry_out = state.add_write("A_pipe")

    comp_A_pipe_out = state.add_write("comp_A_pipe")

    entry_n0, exit_n0 = state.add_map("n0", {
        "n0": "0:N/P",
    },
                                      schedule=dace.ScheduleType.FPGA_Device)
    entry_k, exit_k = state.add_map("k", {"k": "0:K"},
                                    schedule=dace.ScheduleType.FPGA_Device)
    entry_a, exit_a = state.add_map("buffer_A", {"n1": "0:P"},
                                    schedule=dace.ScheduleType.FPGA_Device)
    entry_m, exit_m = state.add_map("m", {"m": "0:M//VECLEN"},
                                    schedule=dace.ScheduleType.FPGA_Device)

    # Instantiate buffers
    sdfg.add_scalar("A_reg",
                    dtype=dace.float32,
                    transient=True,
                    storage=dace.dtypes.StorageType.FPGA_Registers)
    A_reg = state.add_write("A_reg")

    buffer_a_tasklet = state.add_tasklet(
        "buffer_a", {"a_in"}, {"a_reg", "a_out"}, """\
if n1 == P - p - 1:
    a_reg = a_in
if p < P - 1:
    a_out = a_in""")

    # Unroll processing elements
    preprocess_entry, preprocess_exit = state.add_map(
        "unroll_preprocess", {"p": "0:P"},
        schedule=dace.ScheduleType.FPGA_Device,
        unroll=True)

    state.add_memlet_path(A_pipe_in,
                          preprocess_entry,
                          entry_n0,
                          entry_k,
                          entry_a,
                          buffer_a_tasklet,
                          memlet=dace.Memlet("A_pipe[p]", dynamic=False),
                          dst_conn="a_in")
    state.add_memlet_path(buffer_a_tasklet,
                          exit_a,
                          A_reg,
                          memlet=dace.Memlet("A_reg[0]", dynamic=True),
                          src_conn="a_reg")
    state.add_memlet_path(buffer_a_tasklet,
                          exit_a,
                          exit_k,
                          exit_n0,
                          preprocess_exit,
                          A_pipe_carry_out,
                          memlet=dace.Memlet("A_pipe[p + 1]", dynamic=True),
                          src_conn="a_out")

    preprocess_tasklet = state.add_tasklet(
        "preprocess_tasklet", {"a_in"}, {"a_out"}, """\
a_out = a_in
""")

    state.add_memlet_path(A_reg,
                          entry_m,
                          preprocess_tasklet,
                          dst_conn="a_in",
                          memlet=dace.Memlet("A_reg[0]", dynamic=False))
    state.add_memlet_path(preprocess_tasklet,
                          exit_m,
                          exit_k,
                          exit_n0,
                          preprocess_exit,
                          comp_A_pipe_out,
                          memlet=dace.Memlet("comp_A_pipe[p]", dynamic=False),
                          src_conn="a_out")

    # Bring data nodes into scope
#    state.add_memlet_path(preprocess_entry, A_pipe_in, memlet=dace.memlet.Memlet())
#    state.add_memlet_path(A_pipe_carry_out, preprocess_exit, memlet=dace.memlet.Memlet())
#    state.add_memlet_path(comp_A_pipe_out, preprocess_exit, memlet=dace.memlet.Memlet())

# does all preprocessing and forwarding for b
# produces onto a comp_B_stream which is feed to the multiply accumulate compute tasklet
def make_prep_b(state):

    B_pipe_in = state.add_read("B_pipe")
    B_pipe_carry_out = state.add_write("B_pipe")


    comp_B_pipe_out = state.add_write("comp_B_pipe")

    entry_n0, exit_n0 = state.add_map("n0", {
        "n0": "0:N/P",
    },
                                      schedule=dace.ScheduleType.FPGA_Device)
    entry_k, exit_k = state.add_map("k", {"k": "0:K"},
                                    schedule=dace.ScheduleType.FPGA_Device)
    entry_m, exit_m = state.add_map("m", {"m": "0:M//VECLEN"},
                                    schedule=dace.ScheduleType.FPGA_Device)


    preprocess_tasklet = state.add_tasklet(
        "preprocess_tasklet", { "b_in"}, {"b_comp_out", "b_carry_out"}, """\
b_comp_out = b_in
if p < P - 1:
    b_carry_out = b_in""")

    # Unroll processing elements
    preprocess_entry, preprocess_exit = state.add_map(
        "unroll_preprocess", {"p": "0:P"},
        schedule=dace.ScheduleType.FPGA_Device,
        unroll=True)

    state.add_memlet_path(B_pipe_in,
                          preprocess_entry,
                          entry_n0,
                          entry_k,
                          entry_m,
                          preprocess_tasklet,
                          memlet=dace.Memlet("B_pipe[p]", dynamic=False),
                          dst_conn="b_in")
    state.add_memlet_path(preprocess_tasklet,
                          exit_m,
                          exit_k,
                          exit_n0,
                          preprocess_exit,
                          comp_B_pipe_out,
                          memlet=dace.Memlet("comp_B_pipe[p]", dynamic=False),
                          src_conn="b_comp_out")
    state.add_memlet_path(preprocess_tasklet,
                          exit_m,
                          exit_k,
                          exit_n0,
                          preprocess_exit,
                          B_pipe_carry_out,
                          memlet=dace.Memlet("B_pipe[p + 1]", dynamic=True),
                          src_conn="b_carry_out")

    # Bring data nodes into scope
#    state.add_memlet_path(preprocess_entry, B_pipe_in, memlet=dace.memlet.Memlet())
#    state.add_memlet_path(B_pipe_carry_out, preprocess_exit, memlet=dace.memlet.Memlet())
#    state.add_memlet_path(comp_B_pipe_out, preprocess_exit, memlet=dace.memlet.Memlet())

# does all preprocessing on c - for this it consumes the C_feedback stream (goes backwards)
# produces onto a comp_C_stream which is feed to the multiply accumulate compute tasklet
def make_prep_c(state):

    C_feedback_in = state.add_read("C_feedback")

    comp_C_out= state.add_write("comp_C_pipe")

    entry_n0, exit_n0 = state.add_map("n0", {
        "n0": "0:N/P",
    },
                                      schedule=dace.ScheduleType.FPGA_Device)
    entry_k, exit_k = state.add_map("k", {"k": "0:K"},
                                    schedule=dace.ScheduleType.FPGA_Device)
    entry_m, exit_m = state.add_map("m", {"m": "0:M//VECLEN"},
                                    schedule=dace.ScheduleType.FPGA_Device)

    preprocess_tasklet = state.add_tasklet(
        "preprocess_tasklet", {"c_in"}, {"c_out"}, """\
dace::vec<float, 2> zero_vec;
zero_vec[0] = 0.0;
zero_vec[1] = 0.0;
dace::vec<float, 2> c = k == 0 ? zero_vec : c_in;
comp_C_pipe[p].push(c);
""",
    language=dace.Language.CPP)

    # Unroll processing elements
    preprocess_entry, preprocess_exit = state.add_map(
        "unroll_preprocess", {"p": "0:P"},
        schedule=dace.ScheduleType.FPGA_Device,
        unroll=True)

    state.add_memlet_path(C_feedback_in,
                          preprocess_entry,
                          entry_n0,
                          entry_k,
                          entry_m,
                          preprocess_tasklet,
                          dst_conn="c_in",
                          memlet=dace.Memlet("C_feedback[p]", dynamic=True))
    state.add_memlet_path(preprocess_tasklet,
                          exit_m,
                          exit_k,
                          exit_n0,
                          preprocess_exit,
                          comp_C_out,
                          memlet=dace.Memlet("comp_C_pipe[p]", dynamic=False),
                          src_conn="c_out")

    # Bring data nodes into scope
#    state.add_memlet_path(preprocess_entry, C_feedback_in, memlet=dace.Memlet())
#    state.add_memlet_path(comp_C_out, preprocess_exit, memlet=dace.memlet.Memlet())

# does all postprocessing on the results from comp_result_pipe
# will also consume C_pipe to forward values from it to use here
# produces onto C_feedback and C_pipe
def make_post_compute(state):

    comp_result_in = state.add_read("comp_result_pipe")
    C_pipe_out = state.add_write("C_pipe")
    C_pipe_in = state.add_read("C_pipe")

    C_feedback_out = state.add_write("C_feedback")

    entry_n0, exit_n0 = state.add_map("n0", {
        "n0": "0:N/P",
    },
                                      schedule=dace.ScheduleType.FPGA_Device)
    entry_k, exit_k = state.add_map("k", {"k": "0:K"},
                                    schedule=dace.ScheduleType.FPGA_Device)
    entry_m, exit_m = state.add_map("m", {"m": "0:M//VECLEN"},
                                    schedule=dace.ScheduleType.FPGA_Device)
    entry_c, exit_c = state.add_map("write_C", {
        "n1": "0:P",
        "m": "0:M//VECLEN"
    },
                                    schedule=dace.ScheduleType.FPGA_Device)

    # Unroll processing elements
    postprocess_entry, postprocess_exit = state.add_map(
        "unroll_postprocess", {"p": "0:P"},
        schedule=dace.ScheduleType.FPGA_Device,
        unroll=True)


    # Instantiate buffers
    C_buffer = state.add_array("C_buffer", [M // veclen],
                    dtype=dace.vector(dace.float32, veclen),
                    transient=True,
                    storage=dace.dtypes.StorageType.FPGA_Registers)

    postprocess_tasklet = state.add_tasklet(
        "postprocess_tasklet", {"c_in"}, {"c_out1", "c_out2"}, """\
c_out1 = c_in
if k < K-1:
    c_out2 = c_in""")

    state.add_memlet_path(comp_result_in,
                          entry_n0,
                          entry_k,
                          entry_m,
                          postprocess_tasklet,
                          memlet=dace.Memlet("comp_result_pipe[p]", dynamic=False),
                          dst_conn="c_in")
    state.add_memlet_path(postprocess_tasklet,
                          exit_m,
                          exit_k,
                          C_buffer,
                          memlet=dace.Memlet("C_buffer[m]", dynamic=False),
                          src_conn="c_out1")
    state.add_memlet_path(postprocess_tasklet,
                          exit_m,
                          exit_k,
                          exit_n0,
                          C_feedback_out,
                          memlet=dace.Memlet("C_feedback[p]", dynamic=True),
                          src_conn="c_out2")

    # Write back
    write_c_tasklet = state.add_tasklet(
        "write_c", {"buffer_in", "forward_in"}, {"c_out"}, """\
if n1 <= p:
    c_out = forward_in if p > 0 and n1 > 0 else buffer_in""")

    state.add_memlet_path(C_buffer,
                          entry_c,
                          write_c_tasklet,
                          memlet=dace.Memlet("C_buffer[m]", dynamic=True),
                          dst_conn="buffer_in")
    state.add_memlet_path(C_pipe_in,
                          entry_n0,
                          entry_c,
                          write_c_tasklet,
                          memlet=dace.Memlet("C_pipe[p-1]", dynamic=True),
                          dst_conn="forward_in")
    state.add_memlet_path(write_c_tasklet,
                          exit_c,
                          exit_n0,
                          C_pipe_out,
                          memlet=dace.Memlet("C_pipe[p]", dynamic=True),
                          src_conn="c_out")

    # Bring data nodes into scope
    state.add_memlet_path(postprocess_entry, comp_result_in, memlet=dace.memlet.Memlet())
    state.add_memlet_path(postprocess_entry, C_pipe_in, memlet=dace.memlet.Memlet())
    state.add_memlet_path(C_pipe_out, postprocess_exit, memlet=dace.memlet.Memlet())
    state.add_memlet_path(C_feedback_out, postprocess_exit, memlet=dace.memlet.Memlet())

# "core" compute section
# consumes all comp input streams and computes "comp_result_pipe = comp_C_pipe + comp_A_pipe*comp_B_pipe"
def make_rtl_compute(sdfg, state):

    comp_A_in = state.add_read("comp_A_pipe")
    comp_B_in = state.add_read("comp_B_pipe")
    comp_C_in = state.add_read("comp_C_pipe")

    comp_result_out = state.add_write("comp_result_pipe")

    base_clk_freq = dace.Config.get('compiler', 'xilinx', 'frequency')
    if base_clk_freq == '':
        base_clk_freq='300'
    double_clk_freq = str(2 * int(base_clk_freq))

    compute_tasklet = state.add_tasklet(
        name="rtl_dp_ma",
        inputs={"a", "b", "c_in"},
        outputs={"c_out"},
        code='''
    assign ap_done = 1; // free-running

    wire clk_sp;
    wire clk_dp;
    wire rstn_sp;
    wire rstn_dp;

    clk_wiz_0 clock_multiplier (
        .clk_in1(ap_aclk),
        .clk_out1(clk_sp),
        .clk_out2(clk_dp)
    );

    rst_clk_wiz rst_clk_wiz_sp (
        .slowest_sync_clk(clk_sp),
        .ext_reset_in(ap_areset),
        .peripheral_aresetn(rstn_sp),
        .aux_reset_in(0),
        .dcm_locked(1),
        .mb_debug_sys_rst(0)
    );

    rst_clk_wiz rst_clk_wiz_dp (
        .slowest_sync_clk(clk_dp),
        .ext_reset_in(ap_areset),
        .peripheral_aresetn(rstn_dp),
        .aux_reset_in(0),
        .dcm_locked(1),
        .mb_debug_sys_rst(0)
    );

    wire        axis_a_dpclk_tvalid;
    wire [63:0] axis_a_dpclk_tdata;
    wire        axis_a_dpclk_tready;

    slow_to_fast_clk clock_sync_a (
        .s_axis_aclk(clk_sp),
        .s_axis_aresetn(rstn_sp),
        .m_axis_aclk(clk_dp),
        .m_axis_aresetn(rstn_dp),

        .s_axis_tvalid(s_axis_a_tvalid),
        .s_axis_tdata({s_axis_a_tdata, s_axis_a_tdata}),
        .s_axis_tready(s_axis_a_tready),

        .m_axis_tvalid(axis_a_dpclk_tvalid),
        .m_axis_tdata(axis_a_dpclk_tdata),
        .m_axis_tready(axis_a_dpclk_tready)
    );

    wire        axis_a_dp_tvalid;
    wire [31:0] axis_a_dp_tdata;
    wire        axis_a_dp_tready;

    slow_to_fast_data data_issue_a (
        .aclk(clk_dp),
        .aresetn(rstn_dp),

        .s_axis_tvalid(axis_a_dpclk_tvalid),
        .s_axis_tdata(axis_a_dpclk_tdata),
        .s_axis_tready(axis_a_dpclk_tready),

        .m_axis_tvalid(axis_a_dp_tvalid),
        .m_axis_tdata(axis_a_dp_tdata),
        .m_axis_tready(axis_a_dp_tready)
    );

    wire        axis_b_dpclk_tvalid;
    wire [63:0] axis_b_dpclk_tdata;
    wire        axis_b_dpclk_tready;

    slow_to_fast_clk clock_sync_b (
        .s_axis_aclk(clk_sp),
        .s_axis_aresetn(rstn_sp),
        .m_axis_aclk(clk_dp),
        .m_axis_aresetn(rstn_dp),

        .s_axis_tvalid(s_axis_b_tvalid),
        .s_axis_tdata(s_axis_b_tdata),
        .s_axis_tready(s_axis_b_tready),

        .m_axis_tvalid(axis_b_dpclk_tvalid),
        .m_axis_tdata(axis_b_dpclk_tdata),
        .m_axis_tready(axis_b_dpclk_tready)
    );

    wire        axis_b_dp_tvalid;
    wire [31:0] axis_b_dp_tdata;
    wire        axis_b_dp_tready;

    slow_to_fast_data data_issue_b (
        .aclk(clk_dp),
        .aresetn(rstn_dp),

        .s_axis_tvalid(axis_b_dpclk_tvalid),
        .s_axis_tdata(axis_b_dpclk_tdata),
        .s_axis_tready(axis_b_dpclk_tready),

        .m_axis_tvalid(axis_b_dp_tvalid),
        .m_axis_tdata(axis_b_dp_tdata),
        .m_axis_tready(axis_b_dp_tready)
    );

    wire        axis_ab_tvalid;
    wire [31:0] axis_ab_tdata;
    wire        axis_ab_tready;

    floating_point_mult fl_mult (
        .aclk(clk_dp),
        .aresetn(rstn_dp),

        .s_axis_a_tvalid(axis_a_dp_tvalid),
        .s_axis_a_tdata(axis_a_dp_tdata),
        .s_axis_a_tready(axis_a_dp_tready),

        .s_axis_b_tvalid(axis_b_dp_tvalid),
        .s_axis_b_tdata(axis_b_dp_tdata),
        .s_axis_b_tready(axis_b_dp_tready),

        .m_axis_result_tvalid(axis_ab_tvalid),
        .m_axis_result_tdata(axis_ab_tdata),
        .m_axis_result_tready(axis_ab_tready)
    );

    wire        axis_c_in_dpclk_tvalid;
    wire [63:0] axis_c_in_dpclk_tdata;
    wire        axis_c_in_dpclk_tready;

    slow_to_fast_clk clock_sync_c_in (
        .s_axis_aclk(clk_sp),
        .s_axis_aresetn(rstn_sp),
        .m_axis_aclk(clk_dp),
        .m_axis_aresetn(rstn_dp),

        .s_axis_tvalid(s_axis_c_in_tvalid),
        .s_axis_tdata(s_axis_c_in_tdata),
        .s_axis_tready(s_axis_c_in_tready),

        .m_axis_tvalid(axis_c_in_dpclk_tvalid),
        .m_axis_tdata(axis_c_in_dpclk_tdata),
        .m_axis_tready(axis_c_in_dpclk_tready)
    );

    wire        axis_c_in_dp_tvalid;
    wire [31:0] axis_c_in_dp_tdata;
    wire        axis_c_in_dp_tready;

    slow_to_fast_data data_issue_c (
        .aclk(clk_dp),
        .aresetn(rstn_dp),

        .s_axis_tvalid(axis_c_in_dpclk_tvalid),
        .s_axis_tdata(axis_c_in_dpclk_tdata),
        .s_axis_tready(axis_c_in_dpclk_tready),

        .m_axis_tvalid(axis_c_in_dp_tvalid),
        .m_axis_tdata(axis_c_in_dp_tdata),
        .m_axis_tready(axis_c_in_dp_tready)
    );

    wire        axis_c_out_dp_tvalid;
    wire [31:0] axis_c_out_dp_tdata;
    wire        axis_c_out_dp_tready;

    floating_point_add fl_add (
        .aclk(clk_dp),
        .aresetn(rstn_dp),

        .s_axis_a_tvalid(axis_c_in_dp_tvalid),
        .s_axis_a_tdata(axis_c_in_dp_tdata),
        .s_axis_a_tready(axis_c_in_dp_tready),

        .s_axis_b_tvalid(axis_ab_tvalid),
        .s_axis_b_tdata(axis_ab_tdata),
        .s_axis_b_tready(axis_ab_tready),

        .m_axis_result_tvalid(axis_c_out_dp_tvalid),
        .m_axis_result_tdata(axis_c_out_dp_tdata),
        .m_axis_result_tready(axis_c_out_dp_tready)
    );

    wire        axis_c_out_dpclk_tvalid;
    wire [63:0] axis_c_out_dpclk_tdata;
    wire        axis_c_out_dpclk_tready;

    fast_to_slow_data data_packer_c_out (
        .aclk(clk_dp),
        .aresetn(rstn_dp),

        .s_axis_tvalid(axis_c_out_dp_tvalid),
        .s_axis_tdata(axis_c_out_dp_tdata),
        .s_axis_tready(axis_c_out_dp_tready),

        .m_axis_tvalid(axis_c_out_dpclk_tvalid),
        .m_axis_tdata(axis_c_out_dpclk_tdata),
        .m_axis_tready(axis_c_out_dpclk_tready)
    );

    fast_to_slow_clk clock_sync_result (
        .s_axis_aclk(clk_dp),
        .s_axis_aresetn(rstn_dp),
        .m_axis_aclk(clk_sp),
        .m_axis_aresetn(rstn_sp),

        .s_axis_tvalid(axis_c_out_dpclk_tvalid),
        .s_axis_tdata(axis_c_out_dpclk_tdata),
        .s_axis_tready(axis_c_out_dpclk_tready),

        .m_axis_tvalid(m_axis_c_out_tvalid),
        .m_axis_tdata(m_axis_c_out_tdata),
        .m_axis_tready(m_axis_c_out_tready)
    );
''', language=dace.Language.SystemVerilog)

    compute_tasklet.add_ip_core(
    'floating_point_mult', 'floating_point', 'xilinx.com', '7.1', {
        "CONFIG.Operation_Type": "Multiply",
        "CONFIG.C_Mult_Usage": "Max_Usage",
        "CONFIG.Axi_Optimize_Goal": "Performance",
        "CONFIG.A_Precision_Type": "Single",
        "CONFIG.C_A_Exponent_Width": "8",
        "CONFIG.C_A_Fraction_Width": "24",
        "CONFIG.Result_Precision_Type": "Single",
        "CONFIG.C_Result_Exponent_Width": "8",
        "CONFIG.C_Result_Fraction_Width": "24",
        "CONFIG.C_Latency": "9",
        "CONFIG.C_Rate": "1",
        'CONFIG.Has_ARESETn': 'true',
        "CONFIG.Flow_Control": "Blocking"
    })

    compute_tasklet.add_ip_core(
    'floating_point_add', 'floating_point', 'xilinx.com', '7.1', {
        "CONFIG.Add_Sub_Value": "Add",
        "CONFIG.Axi_Optimize_Goal": "Performance",
        'CONFIG.Has_ARESETn': 'true',
        "CONFIG.Flow_Control": "Blocking"
    })

    compute_tasklet.add_ip_core(
        "clk_wiz_0", "clk_wiz", "xilinx.com", "6.0", {
            "CONFIG.PRIMITIVE": "Auto",
            "CONFIG.PRIM_IN_FREQ": base_clk_freq,
            "CONFIG.CLKOUT2_USED": "true",
            "CONFIG.CLKOUT1_REQUESTED_OUT_FREQ": base_clk_freq,
            "CONFIG.CLKOUT2_REQUESTED_OUT_FREQ": double_clk_freq,
            "CONFIG.CLKIN1_JITTER_PS": "33.330000000000005",
            "CONFIG.CLKOUT1_DRIVES": "Buffer",
            "CONFIG.CLKOUT2_DRIVES": "Buffer",
            "CONFIG.CLKOUT3_DRIVES": "Buffer",
            "CONFIG.CLKOUT4_DRIVES": "Buffer",
            "CONFIG.CLKOUT5_DRIVES": "Buffer",
            "CONFIG.CLKOUT6_DRIVES": "Buffer",
            "CONFIG.CLKOUT7_DRIVES": "Buffer",
            "CONFIG.FEEDBACK_SOURCE": "FDBK_AUTO",
            "CONFIG.USE_LOCKED": "false",
            "CONFIG.USE_RESET": "false",
            "CONFIG.MMCM_DIVCLK_DIVIDE": "1",
            "CONFIG.MMCM_BANDWIDTH": "OPTIMIZED",
            "CONFIG.MMCM_CLKFBOUT_MULT_F": "4",
            "CONFIG.MMCM_CLKIN1_PERIOD": "3.333",
            "CONFIG.MMCM_CLKIN2_PERIOD": "10.0",
            "CONFIG.MMCM_COMPENSATION": "AUTO",
            "CONFIG.MMCM_CLKOUT0_DIVIDE_F": "4",
            "CONFIG.MMCM_CLKOUT1_DIVIDE": "2",
            "CONFIG.NUM_OUT_CLKS": "2",
            "CONFIG.CLKOUT1_JITTER": "81.814",
            "CONFIG.CLKOUT1_PHASE_ERROR": "77.836",
            "CONFIG.CLKOUT2_JITTER": "71.438",
            "CONFIG.CLKOUT2_PHASE_ERROR": "77.836",
            "CONFIG.AUTO_PRIMITIVE": "Auto"
        })
    
    compute_tasklet.add_ip_core('rst_clk_wiz', 'proc_sys_reset', 'xilinx.com', '5.0',
                            {})

    compute_tasklet.add_ip_core('slow_to_fast_clk', 'axis_clock_converter',
                            'xilinx.com', '1.1', {
                                "CONFIG.TDATA_NUM_BYTES": "8",
                                "CONFIG.SYNCHRONIZATION_STAGES": "4"
                            })

    compute_tasklet.add_ip_core('fast_to_slow_clk', 'axis_clock_converter',
                            'xilinx.com', '1.1', {
                                "CONFIG.TDATA_NUM_BYTES": "8",
                                "CONFIG.SYNCHRONIZATION_STAGES": "4"
                            })

    compute_tasklet.add_ip_core('slow_to_fast_data', 'axis_dwidth_converter',
                            'xilinx.com', '1.1', {
                                "CONFIG.S_TDATA_NUM_BYTES": "8",
                                "CONFIG.M_TDATA_NUM_BYTES": "4"
                            })

    compute_tasklet.add_ip_core('fast_to_slow_data', 'axis_dwidth_converter',
                            'xilinx.com', '1.1', {
                                "CONFIG.S_TDATA_NUM_BYTES": "4",
                                "CONFIG.M_TDATA_NUM_BYTES": "8"
                            })

    # Unroll processing elements
    compute_entry, compute_exit = state.add_map(
        "unroll_compute", {"p": "0:P"},
        schedule=dace.ScheduleType.FPGA_Device,
        unroll=True)

    state.add_memlet_path(comp_A_in,
                          compute_entry,
                          compute_tasklet,
                          dst_conn="a",
                          memlet=dace.Memlet("comp_A_pipe[p]"))
    state.add_memlet_path(comp_B_in,
                          compute_entry,
                          compute_tasklet,
                          dst_conn="b",
                          memlet=dace.Memlet("comp_B_pipe[p]"))
    state.add_memlet_path(comp_C_in,
                          compute_entry,
                          compute_tasklet,
                          dst_conn="c_in",
                          memlet=dace.Memlet("comp_C_pipe[p]"))
    state.add_memlet_path(compute_tasklet,
                          compute_exit,
                          comp_result_out,
                          src_conn="c_out",
                          memlet=dace.Memlet("comp_result_pipe[p]"))

# "core" compute section
# consumes all comp input streams and computes "comp_result_pipe = comp_C_pipe + comp_A_pipe*comp_B_pipe"
def make_hls_compute(sdfg, state):

    comp_A_in = state.add_read("comp_A_pipe")
    comp_B_in = state.add_read("comp_B_pipe")
    comp_C_in = state.add_read("comp_C_pipe")

    comp_result_out = state.add_write("comp_result_pipe")

    compute_tasklet = state.add_tasklet(
        name="rtl_dp_ma",
        inputs={"a", "b", "c_in"},
        outputs={"c_out"},
        code='''
dace::vec<float,2> c_out;
c_out[0] = c_in[0] + a*b[0];
c_out[1] = c_in[1] + a*b[1];

comp_result_pipe[p].push(c_out);''',
        language=dace.Language.CPP)

    # Unroll processing elements
    compute_entry, compute_exit = state.add_map(
        "unroll_compute", {"p": "0:P"},
        schedule=dace.ScheduleType.FPGA_Device,
        unroll=True)

    entry_n0, exit_n0 = state.add_map("n0", {
        "n0": "0:N/P",
    },
                                      schedule=dace.ScheduleType.FPGA_Device)
    entry_k, exit_k = state.add_map("k", {"k": "0:K"},
                                    schedule=dace.ScheduleType.FPGA_Device)
    entry_m, exit_m = state.add_map("m", {"m": "0:M//VECLEN"},
                                    schedule=dace.ScheduleType.FPGA_Device)

    state.add_memlet_path(comp_A_in,
                          compute_entry,
                          entry_n0,
                          entry_k,
                          entry_m,
                          compute_tasklet,
                          dst_conn="a",
                          memlet=dace.Memlet("comp_A_pipe[p]"))
    state.add_memlet_path(comp_B_in,
                          compute_entry,
                          entry_n0,
                          entry_k,
                          entry_m,
                          compute_tasklet,
                          dst_conn="b",
                          memlet=dace.Memlet("comp_B_pipe[p]"))
    state.add_memlet_path(comp_C_in,
                          compute_entry,
                          entry_n0,
                          entry_k,
                          entry_m,
                          compute_tasklet,
                          dst_conn="c_in",
                          memlet=dace.Memlet("comp_C_pipe[p]"))
    state.add_memlet_path(compute_tasklet,
                          exit_m,
                          exit_k,
                          exit_n0,
                          compute_exit,
                          comp_result_out,
                          src_conn="c_out",
                          memlet=dace.Memlet("comp_result_pipe[p]"))

def make_fpga_state(sdfg):

    state = sdfg.add_state("mm")

    sdfg.add_stream("A_pipe",
                    dace.float32,
                    transient=True,
                    shape=(P + 1, ),
                    buffer_size="twoP",
                    storage=dace.dtypes.StorageType.FPGA_Local)
    sdfg.add_stream("B_pipe",
                    dace.vector(dace.float32, veclen),
                    transient=True,
                    shape=(P + 1, ),
                    buffer_size="twoP",
                    storage=dace.dtypes.StorageType.FPGA_Local)
    sdfg.add_stream("C_pipe",
                    dace.vector(dace.float32, veclen),
                    transient=True,
                    shape=(P, ),
                    buffer_size="twoP",
                    storage=dace.dtypes.StorageType.FPGA_Local)

    sdfg.add_stream("comp_A_pipe",
                    dace.float32,
                    transient=True,
                    shape=(P, ),
                    storage=dace.dtypes.StorageType.FPGA_Local)
    sdfg.add_stream("comp_B_pipe",
                    dace.vector(dace.float32, veclen),
                    transient=True,
                    shape=(P, ),
                    storage=dace.dtypes.StorageType.FPGA_Local)
    sdfg.add_stream("comp_C_pipe",
                    dace.vector(dace.float32, veclen),
                    transient=True,
                    shape=(P, ),
                    storage=dace.dtypes.StorageType.FPGA_Local)
    sdfg.add_stream("comp_result_pipe",
                    dace.vector(dace.float32, veclen),
                    transient=True,
                    shape=(P, ),
                    storage=dace.dtypes.StorageType.FPGA_Local)

    sdfg.add_stream("C_feedback",
                    dace.vector(dace.float32, veclen),
                    transient=True,
                    shape=(P, ),
                    storage=dace.dtypes.StorageType.FPGA_Local,
                    buffer_size="M")

    make_read_A(state)
    make_read_B(state)
    make_prep_a(sdfg, state)
    make_prep_b(state)
    make_prep_c(state)
    make_rtl_compute(sdfg, state)
#    make_hls_compute(sdfg, state)
    make_post_compute(state)
    make_write_C(state)

    return state


def make_sdfg():

    pre_state = make_copy_to_fpga_state(sdfg)
    compute_state = make_fpga_state(sdfg)
    post_state = make_copy_to_host_state(sdfg)

    sdfg.add_edge(pre_state, compute_state, dace.sdfg.InterstateEdge())
    sdfg.add_edge(compute_state, post_state, dace.sdfg.InterstateEdge())

    return sdfg


if __name__ == "__main__":
    print("==== Program start ====")

    parser = argparse.ArgumentParser()
    parser.add_argument("M", type=int)
    parser.add_argument("N", type=int)
    parser.add_argument("K", type=int)
    parser.add_argument("P", type=int)
    parser.add_argument("-specialize",
                        default=False,
                        action="store_true",
                        help="Fix all loop bounds at compile time/in hardware")
    args = vars(parser.parse_args())

    P.set(args["P"])
    twoP.set(2*args["P"])
    M.set(args["M"])
    twoM.set(2*args["M"])
    N.set(args["N"])
    K.set(args["K"])
    # M must always be specialized, as it's used for the static buffer size
    sdfg = make_sdfg()
#    sdfg.specialize(dict(P=P, M=M))
    sdfg.specialize(dict(P=P, M=M, twoP=twoP, twoM=twoM))

    print("Matrix multiplication {}x{}x{} with {} PEs ({}specialized)".format(
        M.get(), N.get(), K.get(), P.get(),
        "" if args["specialize"] else "not "))

    # Initialize arrays: Randomize A and B, zero C
    A = np.ndarray([N.get(), K.get()], dtype=dace.float32.type)
    B = np.ndarray([K.get(), M.get()], dtype=dace.float32.type)
    C = np.ndarray([N.get(), M.get()], dtype=dace.float32.type)
    A[:] = np.random.rand(N.get(), K.get()).astype(dace.float32.type)
    B[:] = np.random.rand(K.get(), M.get()).astype(dace.float32.type)
#    A[:] = dace.float32(0)
#    C[:] = dace.float32(0)

#    for i in range(N.get()):
#        for j in range(K.get()):
#            if i == j:
#                A[i,j] = dace.float32(1)

#    for i in range(K.get()):
#        for j in range(M.get()):
#            B[i,j] = dace.float32(i*K.get() + j)

    sdfg(A=A, B=B, C=C, N=N, K=K)
    
#    print("=== B 0-10 ===")
#    print(B[0:10, 0:10])
#    print("=== C 0-10 ===")
#    print(C[0:10, 0:10])
#    print("##########")
#    print("##########")
#    print("##########")
#    print("=== B -10 - max ===")
#    print(B[K.get()-10:K.get(), M.get()-10:M.get()])
#    print("=== C -10 - max ===")
#    print(C[N.get()-10:N.get(), M.get()-10:M.get()])
#    print("=========")

    diff = np.linalg.norm((A @ B) - C) / float(M.get() * K.get())
    if diff > 1e-6:
        raise ValueError(f"Verification failed, difference: {diff}")
    else:
        print("Results successfully verified.")
