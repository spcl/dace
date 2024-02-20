# Copyright 2019-2023 ETH Zurich and the DaCe authors. All rights reserved.
"""
    This sample shows the AXPY BLAS routine. It is implemented through Xilinx IPs in order to utilize floating point
    operations.

    It is intended for running hardware_emulation or hardware xilinx targets.
"""

import dace
import numpy as np

# add symbol
N = dace.symbol('N')


def make_sdfg(veclen=2):
    # add sdfg
    sdfg = dace.SDFG('axpy')

    # add state
    state = sdfg.add_state('device_state')

    # add parameter
    sdfg.add_constant('VECLEN', veclen)

    # add arrays
    sdfg.add_scalar('a', dtype=dace.float32, storage=dace.StorageType.FPGA_Global)
    sdfg.add_array('x', [N // veclen], dtype=dace.vector(dace.float32, veclen), storage=dace.StorageType.CPU_Heap)
    sdfg.add_array('y', [N // veclen], dtype=dace.vector(dace.float32, veclen), storage=dace.StorageType.CPU_Heap)
    sdfg.add_array('result', [N // veclen], dtype=dace.vector(dace.float32, veclen), storage=dace.StorageType.CPU_Heap)
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
         -->| ap_aclk (slow clock input)                             |
         -->| ap_areset (slow reset input, rst on high)              |
         -->| ap_aclk (fast clock input)                             |
         -->| ap_areset_2 (fast reset input, rst on high)            |
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

        wire        axis_ax_tvalid;
        wire [31:0] axis_ax_tdata;
        wire        axis_ax_tready;

        reg [VECLEN-1:0] s_axis_x_in_tready_tmp;
        reg [VECLEN-1:0] s_axis_y_in_tready_tmp;
        reg [VECLEN-1:0] m_axis_result_out_tvalid_tmp;

        generate for (genvar i = 0; i < VECLEN; i++) begin

            wire        axis_ax_tvalid;
            wire [31:0] axis_ax_tdata;
            wire        axis_ax_tready;

            floating_point_mult multiplier (
                .aclk(ap_aclk),

                .s_axis_a_tvalid(scalars_valid),
                .s_axis_a_tdata(a_in),
                //.s_axis_a_tready(),

                .s_axis_b_tvalid(s_axis_x_in_tvalid),
                .s_axis_b_tdata( s_axis_x_in_tdata[i]),
                .s_axis_b_tready(s_axis_x_in_tready_tmp[i]),

                .m_axis_result_tvalid(axis_ax_tvalid),
                .m_axis_result_tdata( axis_ax_tdata),
                .m_axis_result_tready(axis_ax_tready)
            );

            floating_point_add adder (
                .aclk(ap_aclk),

                .s_axis_a_tvalid(axis_ax_tvalid),
                .s_axis_a_tdata( axis_ax_tdata),
                .s_axis_a_tready(axis_ax_tready),

                .s_axis_b_tvalid(s_axis_y_in_tvalid),
                .s_axis_b_tdata( s_axis_y_in_tdata[i]),
                .s_axis_b_tready(s_axis_y_in_tready_tmp[i]),

                .m_axis_result_tvalid(m_axis_result_out_tvalid_tmp[i]),
                .m_axis_result_tdata( m_axis_result_out_tdata[i]),
                .m_axis_result_tready(m_axis_result_out_tready)
            );

        end endgenerate

        assign s_axis_x_in_tready = &s_axis_x_in_tready_tmp;
        assign s_axis_y_in_tready = &s_axis_y_in_tready_tmp;
        assign m_axis_result_out_tvalid = &m_axis_result_out_tvalid_tmp;
        ''',
                                    language=dace.Language.SystemVerilog)

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

    rtl_tasklet.add_ip_core('floating_point_add', 'floating_point', 'xilinx.com', '7.1', {
        "CONFIG.Add_Sub_Value": "Add",
        "CONFIG.Axi_Optimize_Goal": "Performance",
        "CONFIG.C_Latency": "14"
    })

    # add read and write tasklets
    read_x = state.add_tasklet('read_x', {'inp'}, {'out'}, 'out = inp')
    read_y = state.add_tasklet('read_y', {'inp'}, {'out'}, 'out = inp')
    write_result = state.add_tasklet('write_result', {'inp'}, {'out'}, 'out = inp')

    # add read and write maps
    read_x_entry, read_x_exit = state.add_map('read_x_map',
                                              dict(i='0:N//VECLEN'),
                                              schedule=dace.ScheduleType.FPGA_Device)
    read_y_entry, read_y_exit = state.add_map('read_y_map',
                                              dict(i='0:N//VECLEN'),
                                              schedule=dace.ScheduleType.FPGA_Device)
    write_result_entry, write_result_exit = state.add_map('write_result_map',
                                                          dict(i='0:N//VECLEN'),
                                                          schedule=dace.ScheduleType.FPGA_Device)

    # add read_a memlets and access nodes
    read_x_inp = state.add_read('fpga_x')
    read_x_out = state.add_write('x_stream')
    state.add_memlet_path(read_x_inp, read_x_entry, read_x, dst_conn='inp', memlet=dace.Memlet('fpga_x[i]'))
    state.add_memlet_path(read_x, read_x_exit, read_x_out, src_conn='out', memlet=dace.Memlet('x_stream[0]'))

    read_y_inp = state.add_read('fpga_y')
    read_y_out = state.add_write('y_stream')
    state.add_memlet_path(read_y_inp, read_y_entry, read_y, dst_conn='inp', memlet=dace.Memlet('fpga_y[i]'))
    state.add_memlet_path(read_y, read_y_exit, read_y_out, src_conn='out', memlet=dace.Memlet('y_stream[0]'))

    # add tasklet memlets
    a = state.add_read('a')
    x = state.add_read('x_stream')
    y = state.add_read('y_stream')
    result = state.add_write('result_stream')
    state.add_memlet_path(a, rtl_tasklet, dst_conn='a_in', memlet=dace.Memlet('a[0]'))
    state.add_memlet_path(x, rtl_tasklet, dst_conn='x_in', memlet=dace.Memlet('x_stream[0]'))
    state.add_memlet_path(y, rtl_tasklet, dst_conn='y_in', memlet=dace.Memlet('y_stream[0]'))
    state.add_memlet_path(rtl_tasklet, result, src_conn='result_out', memlet=dace.Memlet('result_stream[0]'))

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
    copy_to_device.add_memlet_path(cpu_x, dev_x, memlet=dace.Memlet('x[0:N//VECLEN]'))
    copy_to_device.add_memlet_path(cpu_y, dev_y, memlet=dace.Memlet('y[0:N//VECLEN]'))
    sdfg.add_edge(copy_to_device, state, dace.InterstateEdge())

    # add copy to host state
    copy_to_host = sdfg.add_state('copy_to_host')
    dev_result = copy_to_host.add_read('fpga_result')
    cpu_result = copy_to_host.add_write('result')
    copy_to_host.add_memlet_path(dev_result, cpu_result, memlet=dace.Memlet('result[0:N//VECLEN]'))
    sdfg.add_edge(state, copy_to_host, dace.InterstateEdge())

    # validate sdfg
    sdfg.validate()

    return sdfg


######################################################################

if __name__ == '__main__':
    with dace.config.set_temporary('compiler', 'xilinx', 'mode', value='hardware_emulation'):
        # init data structures
        N = 4096
        a = np.random.rand(1)[0].astype(np.float32)
        x = np.random.rand(N).astype(np.float32)
        y = np.random.rand(N).astype(np.float32)
        result = np.zeros((N, )).astype(np.float32)

        # show initial values
        print("a={}, x={}, y={}".format(a, x, y))

        # Build the SDFG
        sdfg = make_sdfg()

        # call program
        sdfg(a=a, x=x, y=y, result=result, N=N)

        # show result
        print("result={}".format(result))

        # check result
        expected = a * x + y
        diff = np.linalg.norm(expected - result) / N
        print("Difference:", diff)
        assert diff <= 1e-5
