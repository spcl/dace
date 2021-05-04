# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
#
# This sample shows the AXPY BLAS routine. It is implemented through Xilinx
# IPs in order to utilize double pumping, which doubles the performance per
# consumed FPGA resource. The block diagram of the design (with reset
# synchronization omitted) is:
#
#          ap_aclk          s_axis_y_in        s_axis_x_in     a
#             │                  │                  │          │
#             │                  │                  │          │
#             │                  │                  │          │
#     ┌───────┼─────────┬────────┼─────────┐        │          │
#     │       │         │        │         │        │          │
#     │       │         │        ▼         │        ▼          │
#     │       │         │  ┌────────────┐  │  ┌────────────┐   │
#     │       │         └─►│            │  └─►│            │   │
#     │       │            │ Clock sync │     │ Clock sync │   │
#     │       │         ┌─►│            │  ┌─►│            │   │
#     │       ▼ 300 MHz │  └─────┬──────┘  │  └─────┬──────┘   │
#     │ ┌────────────┐  │        │         │        │          │
#     │ │ Clock      │  │        │         │        │          │
#     │ │            │  ├────────┼─────────┤        │          │
#     │ │ Multiplier │  │        │         │        │          │
#     │ └─────┬──────┘  │        ▼ 64 bit  │        ▼ 64 bit   │
#     │       │ 600 MHz │  ┌────────────┐  │  ┌────────────┐   │
#     │       │         │  │            │  │  │            │   │
#     │       └─────────┼─►│ Data issue │  └─►│ Data issue │   │
#     │                 │  │            │     │            │   │
#     │                 │  └─────┬──────┘     └─────┬──────┘   │
#     │                 │        │ 32 bit           │ 32 bit   │
#     │                 │        │                  │          │
#     │                 │        │                  │          │
#     │                 │        │                  ▼          ▼
#     │                 │        │                 ┌────────────┐
#     │                 │        │                 │            │
#     │                 ├────────┼────────────────►│ Multiplier │
#     │                 │        │                 │            │
#     │                 │        │                 └─────┬──────┘
#     │                 │        │                       │
#     │                 │        │        ┌──────────────┘
#     │                 │        │        │
#     │                 │        ▼        ▼
#     │                 │      ┌────────────┐
#     │                 │      │            │
#     │                 ├─────►│    Adder   │
#     │                 │      │            │
#     │                 │      └─────┬──────┘
#     │                 │            │
#     │                 │            ▼ 32 bit
#     │                 │      ┌─────────────┐
#     │                 │      │             │
#     │                 ├─────►│ Data packer │
#     │                 │      │             │
#     │                 │      └─────┬───────┘
#     │                 │            │ 64 bit
#     │                 │            ▼
#     │                 │      ┌────────────┐
#     │                 └─────►│            │
#     │                        │ Clock sync │
#     └───────────────────────►│            │
#                              └─────┬──────┘
#                                    │
#                                    ▼
#                            m_axis_result_out
#
# It is intended for running hardware_emulation or hardware xilinx targets.

import dace
import numpy as np

# add symbol
N = dace.symbol('N')

# add sdfg
sdfg = dace.SDFG('axpy_double_pump')

# add state
state = sdfg.add_state('device_state')

# add parametr
veclen = 2
sdfg.add_constant('VECLEN', veclen)
sdfg.add_constant('DATA_WIDTH', 64)
sdfg.add_constant('WORD_WIDTH', 32)
sdfg.add_constant('RATIO', 2)

# add arrays
sdfg.add_scalar('a', dtype=dace.float32, storage=dace.StorageType.FPGA_Global)
sdfg.add_array('x', [N // veclen],
               dtype=dace.vector(dace.float32, veclen),
               storage=dace.StorageType.CPU_Heap)
sdfg.add_array('y', [N // veclen],
               dtype=dace.vector(dace.float32, veclen),
               storage=dace.StorageType.CPU_Heap)
sdfg.add_array('result', [N // veclen],
               dtype=dace.vector(dace.float32, veclen),
               storage=dace.StorageType.CPU_Heap)
sdfg.add_array('fpga_x', [N // veclen],
               dtype=dace.vector(dace.float32, veclen),
               transient=True,
               storage=dace.StorageType.FPGA_Global)
sdfg.add_array('fpga_y', [N // veclen],
               dtype=dace.vector(dace.float32, veclen),
               transient=True,
               storage=dace.StorageType.FPGA_Global)
sdfg.add_array('fpga_result', [N // veclen],
               dtype=dace.vector(dace.float32, veclen),
               transient=True,
               storage=dace.StorageType.FPGA_Global)

# add streams
sdfg.add_stream('x_stream',
                buffer_size=32,
                dtype=dace.vector(dace.float32, veclen),
                transient=True,
                storage=dace.StorageType.FPGA_Local)
sdfg.add_stream('y_stream',
                buffer_size=32,
                dtype=dace.vector(dace.float32, veclen),
                transient=True,
                storage=dace.StorageType.FPGA_Local)
sdfg.add_stream('result_stream',
                buffer_size=32,
                dtype=dace.vector(dace.float32, veclen),
                transient=True,
                storage=dace.StorageType.FPGA_Local)

# add custom rtl tasklet
rtl_tasklet = state.add_tasklet(name='rtl_tasklet',
                                inputs={'a_in', 'x_in', 'y_in'},
                                outputs={'result_out'},
                                code='''
    /*
        Convention:
           |--------------------------------------------------------|
           |                                                        |
        -->| ap_aclk (clock input)                                  |
        -->| ap_areset (reset input, rst on high)                   |
        -->| ap_start (start pulse from host)                       |
        <--| ap_done (tells the host that the kernel is done)       |
           |                                                        |
           | For each input:             For each output:           |
           |                                                        |
        -->|     s_axis_{input}_tvalid   reg m_axis_{output}_tvalid |-->
        -->|     s_axis_{input}_tdata    reg m_axis_{output}_tdata  |-->
        <--| reg s_axis_{input}_tready       m_axis_{output}_tready |<--
        -->|     s_axis_{input}_tkeep    reg m_axis_{output}_tkeep  |-->
        -->|     s_axis_{input}_tlast    reg m_axis_{output}_tlast  |-->
           |                                                        |
           |--------------------------------------------------------|
    */
    assign ap_done = 1; // free-running kernel
    wire clk_300;
    wire clk_600;
    wire rstn_300;
    wire rstn_600;

    clk_wiz_0 clock_multiplier (
        .clk_in1(ap_aclk),
        .clk_out1(clk_300),
        .clk_out2(clk_600)
    );

    rst_clk_wiz rst_clk_wiz_300 (
        .slowest_sync_clk(clk_300),
        .ext_reset_in(ap_areset),
        .peripheral_aresetn(rstn_300),
        .aux_reset_in(0),
        .dcm_locked(1),
        .mb_debug_sys_rst(0)
    );

    rst_clk_wiz rst_clk_wiz_600 (
        .slowest_sync_clk(clk_600),
        .ext_reset_in(ap_areset),
        .peripheral_aresetn(rstn_600),
        .aux_reset_in(0),
        .dcm_locked(1),
        .mb_debug_sys_rst(0)
    );

    wire        axis_x_clk_data_tvalid;
    wire [63:0] axis_x_clk_data_tdata;
    wire        axis_x_clk_data_tready;

    slow_to_fast_clk clock_sync_x (
        .s_axis_aclk(clk_300),
        .s_axis_aresetn(rstn_300),
        .m_axis_aclk(clk_600),
        .m_axis_aresetn(rstn_600),

        .s_axis_tvalid(s_axis_x_in_tvalid),
        .s_axis_tdata( s_axis_x_in_tdata),
        .s_axis_tready(s_axis_x_in_tready),

        .m_axis_tvalid(axis_x_clk_data_tvalid),
        .m_axis_tdata( axis_x_clk_data_tdata),
        .m_axis_tready(axis_x_clk_data_tready)
    );

    wire        axis_y_clk_data_tvalid;
    wire [63:0] axis_y_clk_data_tdata;
    wire        axis_y_clk_data_tready;

    slow_to_fast_clk clock_sync_y (
        .s_axis_aclk(clk_300),
        .s_axis_aresetn(rstn_300),
        .m_axis_aclk(clk_600),
        .m_axis_aresetn(rstn_600),

        .s_axis_tvalid(s_axis_y_in_tvalid),
        .s_axis_tdata( s_axis_y_in_tdata),
        .s_axis_tready(s_axis_y_in_tready),

        .m_axis_tvalid(axis_y_clk_data_tvalid),
        .m_axis_tdata( axis_y_clk_data_tdata),
        .m_axis_tready(axis_y_clk_data_tready)
    );

    wire        axis_x_data_fl_tvalid;
    wire [31:0] axis_x_data_fl_tdata;
    wire        axis_x_data_fl_tready;

    slow_to_fast_data data_issue_x (
        .aclk(clk_600),
        .aresetn(rstn_600),

        .s_axis_tvalid(axis_x_clk_data_tvalid),
        .s_axis_tdata( axis_x_clk_data_tdata),
        .s_axis_tready(axis_x_clk_data_tready),

        .m_axis_tvalid(axis_x_data_fl_tvalid),
        .m_axis_tdata( axis_x_data_fl_tdata),
        .m_axis_tready(axis_x_data_fl_tready)
    );

    wire        axis_y_data_fl_tvalid;
    wire [31:0] axis_y_data_fl_tdata;
    wire        axis_y_data_fl_tready;

    slow_to_fast_data data_issue_y (
        .aclk(clk_600),
        .aresetn(rstn_600),

        .s_axis_tvalid(axis_y_clk_data_tvalid),
        .s_axis_tdata( axis_y_clk_data_tdata),
        .s_axis_tready(axis_y_clk_data_tready),

        .m_axis_tvalid(axis_y_data_fl_tvalid),
        .m_axis_tdata( axis_y_data_fl_tdata),
        .m_axis_tready(axis_y_data_fl_tready)
    );

    wire        axis_ax_tvalid;
    wire [31:0] axis_ax_tdata;
    wire        axis_ax_tready;

    floating_point_mult multiplier (
        .aclk(clk_600),

        .s_axis_a_tvalid(1),
        .s_axis_a_tdata(a_in),
        //.s_axis_a_tready(),

        .s_axis_b_tvalid(axis_x_data_fl_tvalid),
        .s_axis_b_tdata( axis_x_data_fl_tdata),
        .s_axis_b_tready(axis_x_data_fl_tready),

        .m_axis_result_tvalid(axis_ax_tvalid),
        .m_axis_result_tdata( axis_ax_tdata),
        .m_axis_result_tready(axis_ax_tready)
    );

    wire        axis_result_tvalid;
    wire [31:0] axis_result_tdata;
    wire        axis_result_tready;

    floating_point_add adder (
        .aclk(clk_600),

        .s_axis_a_tvalid(axis_ax_tvalid),
        .s_axis_a_tdata( axis_ax_tdata),
        .s_axis_a_tready(axis_ax_tready),

        .s_axis_b_tvalid(axis_y_data_fl_tvalid),
        .s_axis_b_tdata( axis_y_data_fl_tdata),
        .s_axis_b_tready(axis_y_data_fl_tready),

        .m_axis_result_tvalid(axis_result_tvalid),
        .m_axis_result_tdata( axis_result_tdata),
        .m_axis_result_tready(axis_result_tready)
    );

    wire        axis_result_data_clk_tvalid;
    wire [63:0] axis_result_data_clk_tdata;
    wire        axis_result_data_clk_tready;

    fast_to_slow_data data_packer (
        .aclk(clk_600),
        .aresetn(rstn_600),

        .s_axis_tvalid(axis_result_tvalid),
        .s_axis_tdata( axis_result_tdata),
        .s_axis_tready(axis_result_tready),

        .m_axis_tvalid(axis_result_data_clk_tvalid),
        .m_axis_tdata( axis_result_data_clk_tdata),
        .m_axis_tready(axis_result_data_clk_tready)
    );

    fast_to_slow_clk clock_sync_result (
        .s_axis_aclk(clk_600),
        .s_axis_aresetn(rstn_600),
        .m_axis_aclk(clk_300),
        .m_axis_aresetn(rstn_300),

        .s_axis_tvalid(axis_result_data_clk_tvalid),
        .s_axis_tdata( axis_result_data_clk_tdata),
        .s_axis_tready(axis_result_data_clk_tready),

        .m_axis_tvalid(m_axis_result_out_tvalid),
        .m_axis_tdata( m_axis_result_out_tdata),
        .m_axis_tready(m_axis_result_out_tready)
    );
    ''',
                                language=dace.Language.SystemVerilog)

rtl_tasklet.add_ip_core(
    "clk_wiz_0", "clk_wiz", "xilinx.com", "6.0", {
        "CONFIG.PRIMITIVE": "Auto",
        "CONFIG.PRIM_IN_FREQ": "300",
        "CONFIG.CLKOUT2_USED": "true",
        "CONFIG.CLKOUT1_REQUESTED_OUT_FREQ": "300",
        "CONFIG.CLKOUT2_REQUESTED_OUT_FREQ": "600",
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
        "CONFIG.AUTO_PRIMITIVE": "PLL"
    })

rtl_tasklet.add_ip_core('rst_clk_wiz', 'proc_sys_reset', 'xilinx.com', '5.0',
                        {})

rtl_tasklet.add_ip_core('slow_to_fast_clk', 'axis_clock_converter',
                        'xilinx.com', '1.1', {
                            "CONFIG.TDATA_NUM_BYTES": "8",
                            "CONFIG.SYNCHRONIZATION_STAGES": "8"
                        })

rtl_tasklet.add_ip_core('slow_to_fast_data', 'axis_dwidth_converter',
                        'xilinx.com', '1.1', {
                            "CONFIG.S_TDATA_NUM_BYTES": "8",
                            "CONFIG.M_TDATA_NUM_BYTES": "4"
                        })

rtl_tasklet.add_ip_core(
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
        "CONFIG.C_Rate": "1"
    })

rtl_tasklet.add_ip_core(
    'floating_point_add', 'floating_point', 'xilinx.com', '7.1', {
        "CONFIG.Add_Sub_Value": "Add",
        "CONFIG.Axi_Optimize_Goal": "Performance",
        "CONFIG.C_Latency": "14"
    })

rtl_tasklet.add_ip_core('fast_to_slow_data', 'axis_dwidth_converter',
                        'xilinx.com', '1.1', {
                            "CONFIG.S_TDATA_NUM_BYTES": "4",
                            "CONFIG.M_TDATA_NUM_BYTES": "8"
                        })

rtl_tasklet.add_ip_core('fast_to_slow_clk', 'axis_clock_converter',
                        'xilinx.com', '1.1', {
                            "CONFIG.TDATA_NUM_BYTES": "8",
                            "CONFIG.SYNCHRONIZATION_STAGES": "8"
                        })

# add read and write tasklets
read_x = state.add_tasklet('read_x', {'inp'}, {'out'}, 'out = inp')
read_y = state.add_tasklet('read_y', {'inp'}, {'out'}, 'out = inp')
write_result = state.add_tasklet('write_result', {'inp'}, {'out'}, 'out = inp')

# add read and write maps
read_x_entry, read_x_exit = state.add_map(
    'read_x_map', dict(i='0:N//VECLEN'), schedule=dace.ScheduleType.FPGA_Device)
read_y_entry, read_y_exit = state.add_map(
    'read_y_map', dict(i='0:N//VECLEN'), schedule=dace.ScheduleType.FPGA_Device)
write_result_entry, write_result_exit = state.add_map(
    'write_result_map',
    dict(i='0:N//VECLEN'),
    schedule=dace.ScheduleType.FPGA_Device)

# add read_a memlets and access nodes
read_x_inp = state.add_read('fpga_x')
read_x_out = state.add_write('x_stream')
state.add_memlet_path(read_x_inp,
                      read_x_entry,
                      read_x,
                      dst_conn='inp',
                      memlet=dace.Memlet('fpga_x[i]'))
state.add_memlet_path(read_x,
                      read_x_exit,
                      read_x_out,
                      src_conn='out',
                      memlet=dace.Memlet('x_stream[0]'))

read_y_inp = state.add_read('fpga_y')
read_y_out = state.add_write('y_stream')
state.add_memlet_path(read_y_inp,
                      read_y_entry,
                      read_y,
                      dst_conn='inp',
                      memlet=dace.Memlet('fpga_y[i]'))
state.add_memlet_path(read_y,
                      read_y_exit,
                      read_y_out,
                      src_conn='out',
                      memlet=dace.Memlet('y_stream[0]'))

# add tasklet memlets
a = state.add_read('a')
x = state.add_read('x_stream')
y = state.add_read('y_stream')
result = state.add_write('result_stream')
state.add_memlet_path(a,
                      rtl_tasklet,
                      dst_conn='a_in',
                      memlet=dace.Memlet('a[0]'))
state.add_memlet_path(x,
                      rtl_tasklet,
                      dst_conn='x_in',
                      memlet=dace.Memlet('x_stream[0]'))
state.add_memlet_path(y,
                      rtl_tasklet,
                      dst_conn='y_in',
                      memlet=dace.Memlet('y_stream[0]'))
state.add_memlet_path(rtl_tasklet,
                      result,
                      src_conn='result_out',
                      memlet=dace.Memlet('result_stream[0]'))

# add write_c memlets and access nodes
write_result_inp = state.add_read('result_stream')
write_result_out = state.add_write('fpga_result')
state.add_memlet_path(write_result_inp,
                      write_result_entry,
                      write_result,
                      dst_conn='inp',
                      memlet=dace.Memlet('result_stream[0]'))
state.add_memlet_path(write_result,
                      write_result_exit,
                      write_result_out,
                      src_conn='out',
                      memlet=dace.Memlet('fpga_result[i]'))

# add copy to device state
copy_to_device = sdfg.add_state('copy_to_device')
cpu_x = copy_to_device.add_read('x')
cpu_y = copy_to_device.add_read('y')
dev_x = copy_to_device.add_write('fpga_x')
dev_y = copy_to_device.add_write('fpga_y')
copy_to_device.add_memlet_path(cpu_x,
                               dev_x,
                               memlet=dace.Memlet('x[0:N//VECLEN]'))
copy_to_device.add_memlet_path(cpu_y,
                               dev_y,
                               memlet=dace.Memlet('y[0:N//VECLEN]'))
sdfg.add_edge(copy_to_device, state, dace.InterstateEdge())

# add copy to host state
copy_to_host = sdfg.add_state('copy_to_host')
dev_result = copy_to_host.add_read('fpga_result')
cpu_result = copy_to_host.add_write('result')
copy_to_host.add_memlet_path(dev_result,
                             cpu_result,
                             memlet=dace.Memlet('result[0:N//VECLEN]'))
sdfg.add_edge(state, copy_to_host, dace.InterstateEdge())

# validate sdfg
sdfg.validate()

######################################################################

if __name__ == '__main__':

    # init data structures
    N.set(4096)
    a = np.random.rand(1)[0].astype(np.float32)
    x = np.random.rand(N.get()).astype(np.float32)
    y = np.random.rand(N.get()).astype(np.float32)
    result = np.zeros((N.get(), )).astype(np.float32)

    # show initial values
    print("a={}, x={}, y={}".format(a, x, y))

    # call program
    sdfg(a=a, x=x, y=y, result=result, N=N)

    # show result
    print("result={}".format(result))

    # check result
    expected = a * x + y
    diff = np.linalg.norm(expected - result) / N.get()
    print("Difference:", diff)
    exit(0 if diff <= 1e-5 else 1)
