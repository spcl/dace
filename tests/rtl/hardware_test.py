# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
import dace
from dace.fpga_testing import xilinx_test
import numpy as np


def make_sdfg(veclen=8):
    # add sdfg
    sdfg = dace.SDFG('floating_point_vector_plus_scalar')

    # add state
    state = sdfg.add_state('device_state')

    # add parameter
    sdfg.add_constant('VECLEN', veclen)

    # add arrays
    sdfg.add_array('A', [N // veclen], dtype=dace.vector(dace.float32, veclen), storage=dace.StorageType.CPU_Heap)
    sdfg.add_scalar('B', dace.float32, storage=dace.StorageType.FPGA_Registers)
    sdfg.add_array('C', [N // veclen], dtype=dace.vector(dace.float32, veclen), storage=dace.StorageType.CPU_Heap)
    sdfg.add_array('fpga_A', [N // veclen],
                   dtype=dace.vector(dace.float32, veclen),
                   transient=True,
                   storage=dace.StorageType.FPGA_Global)
    sdfg.add_array('fpga_C', [N // veclen],
                   dtype=dace.vector(dace.float32, veclen),
                   transient=True,
                   storage=dace.StorageType.FPGA_Global)

    # add streams
    sdfg.add_stream('A_stream',
                    buffer_size=32,
                    dtype=dace.vector(dace.float32, veclen),
                    transient=True,
                    storage=dace.StorageType.FPGA_Local)
    sdfg.add_stream('C_stream',
                    buffer_size=32,
                    dtype=dace.vector(dace.float32, veclen),
                    transient=True,
                    storage=dace.StorageType.FPGA_Local)

    # add custom rtl tasklet
    rtl_tasklet = state.add_tasklet(name='rtl_tasklet',
                                    inputs={'a', 'b'},
                                    outputs={'c'},
                                    code='''
        wire ap_aresetn = ~ap_areset;
        reg [31:0] b_local;
        reg b_valid = 0;
        wire [VECLEN-1:0] a_tready;
        wire [VECLEN-1:0] c_tvalid;

        assign ap_done = 1;
        assign s_axis_a_tready = &a_tready;
        assign m_axis_c_tvalid = &c_tvalid;

        always @(posedge ap_aclk) begin
            if (ap_areset) begin
                b_local = 0;
                b_valid = 0;
            end else if (ap_start) begin
                b_local = b;
                b_valid = 1;
            end
        end

        genvar i;
        generate
            for (i = 0; i < VECLEN; i = i + 1) begin
                floating_point_add add(
                    .aclk(ap_aclk),
                    .aresetn(ap_aresetn),

                    .s_axis_a_tvalid(s_axis_a_tvalid),
                    .s_axis_a_tdata(s_axis_a_tdata[i]),
                    .s_axis_a_tready(a_tready[i]),

                    .s_axis_b_tvalid(b_valid),
                    .s_axis_b_tdata(b_local),

                    .m_axis_result_tvalid(c_tvalid[i]),
                    .m_axis_result_tdata(m_axis_c_tdata[i]),
                    .m_axis_result_tready(m_axis_c_tready)
                );
            end
        endgenerate
        ''',
                                    language=dace.Language.SystemVerilog)

    rtl_tasklet.add_ip_core('floating_point_add', 'floating_point', 'xilinx.com', '7.1', {
        'CONFIG.Add_Sub_Value': 'Add',
        'CONFIG.Has_ARESETn': 'true'
    })

    # add read and write tasklets
    read_a = state.add_tasklet('read_a', {'inp'}, {'out'}, 'out = inp')
    write_c = state.add_tasklet('write_c', {'inp'}, {'out'}, 'out = inp')

    # add read and write maps
    read_a_entry, read_a_exit = state.add_map('read_a_map',
                                              dict(i='0:N//VECLEN'),
                                              schedule=dace.ScheduleType.FPGA_Device)
    write_c_entry, write_c_exit = state.add_map('write_c_map',
                                                dict(i='0:N//VECLEN'),
                                                schedule=dace.ScheduleType.FPGA_Device)

    # add read_a memlets and access nodes
    read_a_inp = state.add_read('fpga_A')
    read_a_out = state.add_write('A_stream')
    state.add_memlet_path(read_a_inp, read_a_entry, read_a, dst_conn='inp', memlet=dace.Memlet('fpga_A[i]'))
    state.add_memlet_path(read_a, read_a_exit, read_a_out, src_conn='out', memlet=dace.Memlet('A_stream[0]'))

    # add tasklet memlets
    A = state.add_read('A_stream')
    B = state.add_read('B')
    C = state.add_write('C_stream')
    state.add_memlet_path(A, rtl_tasklet, dst_conn='a', memlet=dace.Memlet('A_stream[0]'))
    state.add_memlet_path(B, rtl_tasklet, dst_conn='b', memlet=dace.Memlet('B[0]'))
    state.add_memlet_path(rtl_tasklet, C, src_conn='c', memlet=dace.Memlet('C_stream[0]'))

    # add write_c memlets and access nodes
    write_c_inp = state.add_read('C_stream')
    write_c_out = state.add_write('fpga_C')
    state.add_memlet_path(write_c_inp, write_c_entry, write_c, dst_conn='inp', memlet=dace.Memlet('C_stream[0]'))
    state.add_memlet_path(write_c, write_c_exit, write_c_out, src_conn='out', memlet=dace.Memlet('fpga_C[i]'))

    # add copy to device state
    copy_to_device = sdfg.add_state('copy_to_device')
    cpu_a = copy_to_device.add_read('A')
    dev_a = copy_to_device.add_write('fpga_A')
    copy_to_device.add_memlet_path(cpu_a, dev_a, memlet=dace.Memlet('A[0:N//VECLEN]'))
    sdfg.add_edge(copy_to_device, state, dace.InterstateEdge())

    # add copy to host state
    copy_to_host = sdfg.add_state('copy_to_host')
    dev_c = copy_to_host.add_read('fpga_C')
    cpu_c = copy_to_host.add_write('C')
    copy_to_host.add_memlet_path(dev_c, cpu_c, memlet=dace.Memlet('C[0:N//VECLEN]'))
    sdfg.add_edge(state, copy_to_host, dace.InterstateEdge())

    # validate sdfg
    sdfg.validate()

    return sdfg


# add symbol
N = dace.symbol('N')


@xilinx_test()
def test_hardware():
    N.set(4096)
    veclen = 16
    sdfg = make_sdfg(veclen)
    a = np.random.randint(0, 100, N.get()).astype(np.float32)
    b = np.random.rand(1)[0].astype(np.float32)
    c = np.zeros((N.get(), )).astype(np.float32)

    # call program
    sdfg(A=a, B=b, C=c, N=N)

    expected = a + b
    diff = np.linalg.norm(expected - c) / N.get()
    assert diff <= 1e-5

    return sdfg


if __name__ == '__main__':
    test_hardware(None)
