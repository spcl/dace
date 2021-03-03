import dace
from dace.transformation.dataflow import StreamingMemory
from dace.transformation.interstate import FPGATransformState
from dace.transformation.dataflow import TrivialMapElimination
import numpy as np

# add symbol
N = dace.symbol('N')

# add sdfg
sdfg = dace.SDFG('add_fortytwo_vec')

# add state
state = sdfg.add_state('device_state')

# add parametr
veclen = 8
sdfg.add_constant('VECLEN', veclen)

# add arrays
sdfg.add_array('A', [N // veclen],
               dtype=dace.vector(dace.int32, veclen),
               storage=dace.StorageType.CPU_Heap)
sdfg.add_array('B', [N // veclen],
               dtype=dace.int32,
               storage=dace.StorageType.CPU_Heap)
sdfg.add_array('fpga_A', [N // veclen],
               dtype=dace.vector(dace.int32, veclen),
               transient=True,
               storage=dace.StorageType.FPGA_Global)
sdfg.add_array('fpga_B', [N // veclen],
               dtype=dace.int32,
               transient=True,
               storage=dace.StorageType.FPGA_Global)

# add streams
sdfg.add_stream('A_stream',
                buffer_size=32,
                dtype=dace.vector(dace.int32, veclen),
                transient=True,
                storage=dace.StorageType.FPGA_Local)
sdfg.add_stream('B_stream',
                buffer_size=32,
                dtype=dace.int32,
                transient=True,
                storage=dace.StorageType.FPGA_Local)

# add custom rtl tasklet
rtl_tasklet = state.add_tasklet(name='rtl_tasklet',
                                inputs={'a'},
                                outputs={'b'},
                                code='''
    /*
        This tasklet tests whether a contineous stream of data can be processed
        Convention:
           |--------------------------------------------------------|
           |                                                        |
        -->| ap_aclk (clock input)                                  |
        -->| ap_areset (reset input, rst on high)                   |
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

    assign ap_done = 1;

    reg ready;
    reg [31:0] accum;

    always@(posedge ap_aclk) begin
        if (ap_areset) begin // case: reset
            s_axis_a_tready <= 1'b1;
            m_axis_b_tvalid <= 1'b0;
            m_axis_b_tdata <= 0;
            ready <= 1'b1;
        end else if (ready && s_axis_a_tvalid && s_axis_a_tready) begin
            s_axis_a_tready <= 1'b0;
            m_axis_b_tvalid <= 1'b1;
            accum = 0;
            for (integer i = 0; i < VECLEN; i = i + 1) begin
                accum = accum + s_axis_a_tdata[i];
            end
            m_axis_b_tdata = accum;
            ready <= 1'b0;
        end else if (!ready && m_axis_b_tvalid && m_axis_b_tready) begin
            s_axis_a_tready <= 1'b1;
            m_axis_b_tvalid <= 1'b0;
            ready <= 1'b1;
        end
    end
    ''',
                                language=dace.Language.SystemVerilog)

# add read and write tasklets
read_a = state.add_tasklet('read_a', {'inp'}, {'out'}, 'out = inp')
write_b = state.add_tasklet('write_b', {'inp'}, {'out'}, 'out = inp')

# add read and write maps
read_a_entry, read_a_exit = state.add_map(
    'read_a_map', dict(i='0:N//VECLEN'), schedule=dace.ScheduleType.FPGA_Device)
write_b_entry, write_b_exit = state.add_map(
    'write_b_map',
    dict(i='0:N//VECLEN'),
    schedule=dace.ScheduleType.FPGA_Device)

# add read_a memlets and access nodes
read_a_inp = state.add_read('fpga_A')
read_a_out = state.add_write('A_stream')
state.add_memlet_path(read_a_inp,
                      read_a_entry,
                      read_a,
                      dst_conn='inp',
                      memlet=dace.Memlet.simple('fpga_A', 'i'))
state.add_memlet_path(read_a,
                      read_a_exit,
                      read_a_out,
                      src_conn='out',
                      memlet=dace.Memlet.simple('A_stream', '0'))

# add tasklet memlets
A = state.add_read('A_stream')
B = state.add_write('B_stream')
state.add_memlet_path(A,
                      rtl_tasklet,
                      dst_conn='a',
                      memlet=dace.Memlet.simple('A_stream', '0'))
state.add_memlet_path(rtl_tasklet,
                      B,
                      src_conn='b',
                      memlet=dace.Memlet.simple('B_stream', '0'))

# add write_b memlets and access nodes
write_b_inp = state.add_read('B_stream')
write_b_out = state.add_write('fpga_B')
state.add_memlet_path(write_b_inp,
                      write_b_entry,
                      write_b,
                      dst_conn='inp',
                      memlet=dace.Memlet.simple('B_stream', '0'))
state.add_memlet_path(write_b,
                      write_b_exit,
                      write_b_out,
                      src_conn='out',
                      memlet=dace.Memlet.simple('fpga_B', 'i'))

# add copy to device state
copy_to_device = sdfg.add_state('copy_to_device')
cpu_a = copy_to_device.add_read('A')
dev_a = copy_to_device.add_write('fpga_A')
copy_to_device.add_memlet_path(cpu_a,
                               dev_a,
                               memlet=dace.Memlet.simple('A', '0:N//VECLEN'))
sdfg.add_edge(copy_to_device, state, dace.InterstateEdge())

# add copy to host state
copy_to_host = sdfg.add_state('copy_to_host')
dev_b = copy_to_host.add_read('fpga_B')
cpu_b = copy_to_host.add_write('B')
copy_to_host.add_memlet_path(dev_b,
                             cpu_b,
                             memlet=dace.Memlet.simple('B', '0:N//VECLEN'))
sdfg.add_edge(state, copy_to_host, dace.InterstateEdge())

# validate sdfg
sdfg.validate()

######################################################################

if __name__ == '__main__':

    # init data structures
    N.set(8192)
    a = np.random.randint(0, 100, N.get()).astype(np.int32)
    b = np.zeros((N.get() // veclen, )).astype(np.int32)

    # show initial values
    print("a={}, b={}".format(a, b))

    # call program
    sdfg(A=a, B=b, N=N)

    # show result
    print("a={}, b={}".format(a, b))

    # check result
    for i in range(N.get() // veclen):
        assert b[i] == a[i * veclen:(i + 1) * veclen].sum()
