# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
#
# This sample shows adding a constant integer value to a stream of integers.
#
# It is intended for running hardware_emulation or hardware xilinx targets.

import dace
import numpy as np

# add symbol
N = dace.symbol('N')

# add sdfg
sdfg = dace.SDFG('add_fortytwo')

# add state
state = sdfg.add_state('device_state')

# add arrays
sdfg.add_array('A', [N], dtype=dace.int32, storage=dace.StorageType.CPU_Heap)
sdfg.add_array('B', [N], dtype=dace.int32, storage=dace.StorageType.CPU_Heap)
sdfg.add_array('fpga_A', [N], dtype=dace.int32, transient=True, storage=dace.StorageType.FPGA_Global)
sdfg.add_array('fpga_B', [N], dtype=dace.int32, transient=True, storage=dace.StorageType.FPGA_Global)

# add streams
sdfg.add_stream('A_stream', dtype=dace.int32, transient=True, storage=dace.StorageType.FPGA_Local)
sdfg.add_stream('B_stream', dtype=dace.int32, transient=True, storage=dace.StorageType.FPGA_Local)

# add custom rtl tasklet
rtl_tasklet = state.add_tasklet(name='rtl_tasklet',
                                inputs={'a'},
                                outputs={'b'},
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

    always@(posedge ap_aclk) begin
        if (ap_areset) begin // case: reset
            s_axis_a_tready <= 1'b1;
            m_axis_b_tvalid <= 1'b0;
            m_axis_b_tdata <= 0;
        end else if (s_axis_a_tvalid && s_axis_a_tready) begin
            s_axis_a_tready <= 1'b0;
            m_axis_b_tvalid <= 1'b1;
            m_axis_b_tdata <= s_axis_a_tdata + 42;
        end else if (!s_axis_a_tready && m_axis_b_tvalid && m_axis_b_tready) begin
            s_axis_a_tready <= 1'b1;
            m_axis_b_tvalid <= 1'b0;
        end
    end
    ''',
                                language=dace.Language.SystemVerilog)

# add read and write tasklets
read_a = state.add_tasklet('read_a', {'inp'}, {'out'}, 'out = inp')
write_b = state.add_tasklet('write_b', {'inp'}, {'out'}, 'out = inp')

# add read and write maps
read_a_entry, read_a_exit = state.add_map('read_a_map', dict(i='0:N'), schedule=dace.ScheduleType.FPGA_Device)
write_b_entry, write_b_exit = state.add_map('write_b_map', dict(i='0:N'), schedule=dace.ScheduleType.FPGA_Device)

# add read_a memlets and access nodes
read_a_inp = state.add_read('fpga_A')
read_a_out = state.add_write('A_stream')
state.add_memlet_path(read_a_inp, read_a_entry, read_a, dst_conn='inp', memlet=dace.Memlet('fpga_A[i]'))
state.add_memlet_path(read_a, read_a_exit, read_a_out, src_conn='out', memlet=dace.Memlet('A_stream[0]'))

# add tasklet memlets
A = state.add_read('A_stream')
B = state.add_write('B_stream')
state.add_memlet_path(A, rtl_tasklet, dst_conn='a', memlet=dace.Memlet('A_stream[0]'))
state.add_memlet_path(rtl_tasklet, B, src_conn='b', memlet=dace.Memlet('B_stream[0]'))

# add write_b memlets and access nodes
write_b_inp = state.add_read('B_stream')
write_b_out = state.add_write('fpga_B')
state.add_memlet_path(write_b_inp, write_b_entry, write_b, dst_conn='inp', memlet=dace.Memlet('B_stream[0]'))
state.add_memlet_path(write_b, write_b_exit, write_b_out, src_conn='out', memlet=dace.Memlet('fpga_B[i]'))

# add copy to device state
copy_to_device = sdfg.add_state('copy_to_device')
cpu_a = copy_to_device.add_read('A')
dev_a = copy_to_device.add_write('fpga_A')
copy_to_device.add_memlet_path(cpu_a, dev_a, memlet=dace.Memlet('A[0:N]'))
sdfg.add_edge(copy_to_device, state, dace.InterstateEdge())

# add copy to host state
copy_to_host = sdfg.add_state('copy_to_host')
dev_b = copy_to_host.add_read('fpga_B')
cpu_b = copy_to_host.add_write('B')
copy_to_host.add_memlet_path(dev_b, cpu_b, memlet=dace.Memlet('B[0:N]'))
sdfg.add_edge(state, copy_to_host, dace.InterstateEdge())

# validate sdfg
sdfg.validate()

######################################################################

if __name__ == '__main__':

    # init data structures
    N.set(8192)
    a = np.random.randint(0, 100, N.get()).astype(np.int32)
    b = np.zeros((N.get(), )).astype(np.int32)

    # show initial values
    print("a={}, b={}".format(a, b))

    # call program
    sdfg(A=a, B=b, N=N)

    # show result
    print("a={}, b={}".format(a, b))

    # check result
    for i in range(N.get()):
        assert b[i] == a[i] + 42
