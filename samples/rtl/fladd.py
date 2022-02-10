# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
#
# This sample shows how to utilize an IP core in an RTL tasklet. This is done
# through the vector add problem, which adds two floating point vectors
# together.
#
# It is intended for running hardware_emulation or hardware xilinx targets.

import dace
import numpy as np

# add symbol
N = dace.symbol('N')

# add sdfg
sdfg = dace.SDFG('fladd')

# add state
state = sdfg.add_state('device_state')

# add parameter
veclen = 1
sdfg.add_constant('VECLEN', veclen)

# add arrays
sdfg.add_array('A', [N // veclen], dtype=dace.vector(dace.float32, veclen), storage=dace.StorageType.CPU_Heap)
sdfg.add_array('B', [N // veclen], dtype=dace.vector(dace.float32, veclen), storage=dace.StorageType.CPU_Heap)
sdfg.add_array('C', [N // veclen], dtype=dace.vector(dace.float32, veclen), storage=dace.StorageType.CPU_Heap)
sdfg.add_array('fpga_A', [N // veclen],
               dtype=dace.vector(dace.float32, veclen),
               transient=True,
               storage=dace.StorageType.FPGA_Global)
sdfg.add_array('fpga_B', [N // veclen],
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
sdfg.add_stream('B_stream',
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

    wire ap_aresetn = ~ap_areset; // IP core is active-low reset

    floating_point_add add(
        .aclk(ap_aclk),
        .aresetn(ap_aresetn),

        .s_axis_a_tvalid(s_axis_a_tvalid),
        .s_axis_a_tdata(s_axis_a_tdata),
        .s_axis_a_tready(s_axis_a_tready),

        .s_axis_b_tvalid(s_axis_b_tvalid),
        .s_axis_b_tdata(s_axis_b_tdata),
        .s_axis_b_tready(s_axis_b_tready),

        .m_axis_result_tvalid(m_axis_c_tvalid),
        .m_axis_result_tdata(m_axis_c_tdata),
        .m_axis_result_tready(m_axis_c_tready)
    );
    ''',
                                language=dace.Language.SystemVerilog)

rtl_tasklet.add_ip_core('floating_point_add', 'floating_point', 'xilinx.com', '7.1', {
    'CONFIG.Add_Sub_Value': 'Add',
    'CONFIG.Has_ARESETn': 'true'
})

# add read and write tasklets
read_a = state.add_tasklet('read_a', {'inp'}, {'out'}, 'out = inp')
read_b = state.add_tasklet('read_b', {'inp'}, {'out'}, 'out = inp')
write_c = state.add_tasklet('write_c', {'inp'}, {'out'}, 'out = inp')

# add read and write maps
read_a_entry, read_a_exit = state.add_map('read_a_map', dict(i='0:N//VECLEN'), schedule=dace.ScheduleType.FPGA_Device)
read_b_entry, read_b_exit = state.add_map('read_b_map', dict(i='0:N//VECLEN'), schedule=dace.ScheduleType.FPGA_Device)
write_c_entry, write_c_exit = state.add_map('write_c_map',
                                            dict(i='0:N//VECLEN'),
                                            schedule=dace.ScheduleType.FPGA_Device)

# add read_a memlets and access nodes
read_a_inp = state.add_read('fpga_A')
read_a_out = state.add_write('A_stream')
state.add_memlet_path(read_a_inp, read_a_entry, read_a, dst_conn='inp', memlet=dace.Memlet('fpga_A[i]'))
state.add_memlet_path(read_a, read_a_exit, read_a_out, src_conn='out', memlet=dace.Memlet('A_stream[0]'))

read_b_inp = state.add_read('fpga_B')
read_b_out = state.add_write('B_stream')
state.add_memlet_path(read_b_inp, read_b_entry, read_b, dst_conn='inp', memlet=dace.Memlet('fpga_B[i]'))
state.add_memlet_path(read_b, read_b_exit, read_b_out, src_conn='out', memlet=dace.Memlet('B_stream[0]'))

# add tasklet memlets
A = state.add_read('A_stream')
B = state.add_read('B_stream')
C = state.add_write('C_stream')
state.add_memlet_path(A, rtl_tasklet, dst_conn='a', memlet=dace.Memlet('A_stream[0]'))
state.add_memlet_path(B, rtl_tasklet, dst_conn='b', memlet=dace.Memlet('B_stream[0]'))
state.add_memlet_path(rtl_tasklet, C, src_conn='c', memlet=dace.Memlet('C_stream[0]'))

# add write_c memlets and access nodes
write_c_inp = state.add_read('C_stream')
write_c_out = state.add_write('fpga_C')
state.add_memlet_path(write_c_inp, write_c_entry, write_c, dst_conn='inp', memlet=dace.Memlet('C_stream[0]'))
state.add_memlet_path(write_c, write_c_exit, write_c_out, src_conn='out', memlet=dace.Memlet('fpga_C[i]'))

# add copy to device state
copy_to_device = sdfg.add_state('copy_to_device')
cpu_a = copy_to_device.add_read('A')
cpu_b = copy_to_device.add_read('B')
dev_a = copy_to_device.add_write('fpga_A')
dev_b = copy_to_device.add_write('fpga_B')
copy_to_device.add_memlet_path(cpu_a, dev_a, memlet=dace.Memlet('A[0:N//VECLEN]'))
copy_to_device.add_memlet_path(cpu_b, dev_b, memlet=dace.Memlet('B[0:N//VECLEN]'))
sdfg.add_edge(copy_to_device, state, dace.InterstateEdge())

# add copy to host state
copy_to_host = sdfg.add_state('copy_to_host')
dev_c = copy_to_host.add_read('fpga_C')
cpu_c = copy_to_host.add_write('C')
copy_to_host.add_memlet_path(dev_c, cpu_c, memlet=dace.Memlet('C[0:N//VECLEN]'))
sdfg.add_edge(state, copy_to_host, dace.InterstateEdge())

# validate sdfg
sdfg.validate()

######################################################################

if __name__ == '__main__':

    # init data structures
    N.set(8192)
    a = np.random.randint(0, 100, N.get()).astype(np.float32)
    b = np.random.randint(0, 100, N.get()).astype(np.float32)
    c = np.zeros((N.get() // veclen, )).astype(np.float32)
    print(a.shape, b.shape, c.shape)

    # show initial values
    print("a={}, b={}".format(a, b))

    # call program
    sdfg(A=a, B=b, C=c, N=N)

    # show result
    print("a={}, b={}, c={}".format(a, b, c))

    # check result
    expected = a + b
    diff = np.linalg.norm(expected - c) / N.get()
    print("Difference:", diff)
    exit(0 if diff <= 1e-5 else 1)
