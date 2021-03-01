import dace
from dace.transformation.dataflow import StreamingMemory
from dace.transformation.interstate import FPGATransformState
from dace.transformation.dataflow import TrivialMapElimination
import numpy as np

# add symbol
N = dace.symbol('N')

# add sdfg
sdfg = dace.SDFG('reciprog_double_pump')

# add state
state = sdfg.add_state('device_state')

# add parameter
veclen = 2
sdfg.add_constant('VECLEN', veclen)
sdfg.add_constant('DATA_WIDTH', 64)
sdfg.add_constant('WORD_WIDTH', 32)
sdfg.add_constant('RATIO', 2)

# add arrays
sdfg.add_array('A', [N // veclen],
               dtype=dace.vector(dace.float32, veclen),
               storage=dace.StorageType.CPU_Heap)
sdfg.add_array('C', [N // veclen],
               dtype=dace.vector(dace.float32, veclen),
               storage=dace.StorageType.CPU_Heap)
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
                                inputs={'input'},
                                outputs={'output'},
                                code='''
    wire areset = ~ap_areset;
    wire clk_300;
    wire clk_600;
    wire rstn_300;
    wire rstn_600;

    clk_wiz_0 clk_gen (
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

    wire        axis_clk_to_data_tvalid;
    wire [63:0] axis_clk_to_data_tdata;
    wire        axis_clk_to_data_tready;

    slow_to_fast_clk slow_to_fast_clk_0 (
        .s_axis_aclk(clk_300),
        .m_axis_aclk(clk_600),
        .s_axis_aresetn(rstn_300),
        .m_axis_aresetn(rstn_600),

        .s_axis_tvalid(s_axis_input_tvalid),
        .s_axis_tdata(s_axis_input_tdata),
        .s_axis_tready(s_axis_input_tready),

        .m_axis_tvalid(axis_clk_to_data_tvalid),
        .m_axis_tdata(axis_clk_to_data_tdata),
        .m_axis_tready(axis_clk_to_data_tready)
    );

    wire        axis_data_to_fl_tvalid;
    wire [31:0] axis_data_to_fl_tdata;
    wire        axis_data_to_fl_tready;

    slow_to_fast_data slow_to_fast_data_0 (
        .aclk(clk_600),
        .aresetn(rstn_600),

        .s_axis_tvalid(axis_clk_to_data_tvalid),
        .s_axis_tdata(axis_clk_to_data_tdata),
        .s_axis_tready(axis_clk_to_data_tready),

        .m_axis_tvalid(axis_data_to_fl_tvalid),
        .m_axis_tdata(axis_data_to_fl_tdata),
        .m_axis_tready(axis_data_to_fl_tready)
    );

    wire        axis_fl_to_data_tvalid;
    wire [31:0] axis_fl_to_data_tdata;
    wire        axis_fl_to_data_tready;

    floating_point_reciprog floating_point_0 (
        .aclk(clk_600),

        .s_axis_a_tvalid(axis_data_to_fl_tvalid),
        .s_axis_a_tdata(axis_data_to_fl_tdata),
        .s_axis_a_tready(axis_data_to_fl_tready),

        .m_axis_result_tvalid(axis_fl_to_data_tvalid),
        .m_axis_result_tdata(axis_fl_to_data_tdata),
        .m_axis_result_tready(axis_fl_to_data_tready)
    );

    wire        axis_data_to_clk_tvalid;
    wire [63:0] axis_data_to_clk_tdata;
    wire        axis_data_to_clk_tready;

    fast_to_slow_data fast_to_slow_data_0 (
        .aclk(clk_600),
        .aresetn(rstn_600),

        .s_axis_tvalid(axis_fl_to_data_tvalid),
        .s_axis_tdata( axis_fl_to_data_tdata),
        .s_axis_tready(axis_fl_to_data_tready),

        .m_axis_tvalid(axis_data_to_clk_tvalid),
        .m_axis_tdata( axis_data_to_clk_tdata),
        .m_axis_tready(axis_data_to_clk_tready)
    );

    fast_to_slow_clk fast_to_slow_clk_0 (
        .s_axis_aclk(clk_600),
        .m_axis_aclk(clk_300),
        .s_axis_aresetn(rstn_600),
        .m_axis_aresetn(rstn_300),

        .s_axis_tvalid(axis_data_to_clk_tvalid),
        .s_axis_tdata( axis_data_to_clk_tdata),
        .s_axis_tready(axis_data_to_clk_tready),

        .m_axis_tvalid(m_axis_output_tvalid),
        .m_axis_tdata( m_axis_output_tdata),
        .m_axis_tready(m_axis_output_tready)
    );

    assign ap_done = 1;
    ''',
                                language=dace.Language.SystemVerilog)

rtl_tasklet.add_ip_core('slow_to_fast_data', 'axis_dwidth_converter', 'xilinx.com', '1.1', {
        'CONFIG.S_TDATA_NUM_BYTES' : '8',
        'CONFIG.M_TDATA_NUM_BYTES' : '4',
        'CONFIG.Component_Name' : 'slow_to_fast_data',
    })
rtl_tasklet.add_ip_core('fast_to_slow_data', 'axis_dwidth_converter', 'xilinx.com', '1.1', {
        'CONFIG.S_TDATA_NUM_BYTES' : '4',
        'CONFIG.M_TDATA_NUM_BYTES' : '8',
        'CONFIG.Component_Name' : 'fast_to_slow_data',
    })
rtl_tasklet.add_ip_core('slow_to_fast_clk', 'axis_clock_converter', 'xilinx.com', '1.1', {
        'CONFIG.TDATA_NUM_BYTES' : '8',
        'CONFIG.SYNCHRONIZATION_STAGES' : '8',
        'CONFIG.Component_Name' : 'slow_to_fast_clk',
    })
rtl_tasklet.add_ip_core('fast_to_slow_clk', 'axis_clock_converter', 'xilinx.com', '1.1', {
        'CONFIG.TDATA_NUM_BYTES' : '8',
        'CONFIG.SYNCHRONIZATION_STAGES' : '8',
        'CONFIG.Component_Name' : 'fast_to_slow_clk',
    })
rtl_tasklet.add_ip_core('floating_point_reciprog', 'floating_point', 'xilinx.com', '7.1', {
        'CONFIG.Operation_Type' : 'Reciprocal',
        'CONFIG.Add_Sub_Value' : 'Both',
        'CONFIG.Result_Precision_Type' : 'Single',
        'CONFIG.C_Result_Exponent_Width' : '8',
        'CONFIG.C_Result_Fraction_Width' : '24',
        'CONFIG.C_Mult_Usage' : 'Full_Usage',
        'CONFIG.C_Latency' : '32',
        'CONFIG.C_Rate' : '1',
    })
rtl_tasklet.add_ip_core('clk_wiz_0', 'clk_wiz', 'xilinx.com', '6.0', {
        'CONFIG.PRIMITIVE' : 'Auto',
        'CONFIG.PRIM_IN_FREQ' : '300',
        'CONFIG.CLKOUT2_USED' : 'true',
        'CONFIG.CLKOUT1_REQUESTED_OUT_FREQ' : '300',
        'CONFIG.CLKOUT2_REQUESTED_OUT_FREQ' : '600',
        'CONFIG.CLKIN1_JITTER_PS' : '33.330000000000005',
        'CONFIG.CLKOUT1_DRIVES' : 'Buffer',
        'CONFIG.CLKOUT2_DRIVES' : 'Buffer',
        'CONFIG.CLKOUT3_DRIVES' : 'Buffer',
        'CONFIG.CLKOUT4_DRIVES' : 'Buffer',
        'CONFIG.CLKOUT5_DRIVES' : 'Buffer',
        'CONFIG.CLKOUT6_DRIVES' : 'Buffer',
        'CONFIG.CLKOUT7_DRIVES' : 'Buffer',
        'CONFIG.FEEDBACK_SOURCE' : 'FDBK_AUTO',
        'CONFIG.USE_LOCKED' : 'false',
        'CONFIG.USE_RESET' : 'false',
        'CONFIG.MMCM_DIVCLK_DIVIDE' : '1',
        'CONFIG.MMCM_BANDWIDTH' : 'OPTIMIZED',
        'CONFIG.MMCM_CLKFBOUT_MULT_F' : '4',
        'CONFIG.MMCM_CLKIN1_PERIOD' : '3.333',
        'CONFIG.MMCM_CLKIN2_PERIOD' : '10.0',
        'CONFIG.MMCM_COMPENSATION' : 'AUTO',
        'CONFIG.MMCM_CLKOUT0_DIVIDE_F' : '4',
        'CONFIG.MMCM_CLKOUT1_DIVIDE' : '2',
        'CONFIG.NUM_OUT_CLKS' : '2',
        'CONFIG.CLKOUT1_JITTER' : '81.814',
        'CONFIG.CLKOUT1_PHASE_ERROR' : '77.836',
        'CONFIG.CLKOUT2_JITTER' : '71.438',
        'CONFIG.CLKOUT2_PHASE_ERROR' : '77.836',
        'CONFIG.AUTO_PRIMITIVE' : 'PL',
    })
rtl_tasklet.add_ip_core('rst_clk_wiz', 'proc_sys_reset', 'xilinx.com', '5.0', {
        'CONFIG.Component_Name' : 'rst_clk_wiz'
    })

# add read and write tasklets
read_a = state.add_tasklet('read_a', {'inp'}, {'out'}, 'out = inp')
write_c = state.add_tasklet('write_c', {'inp'}, {'out'}, 'out = inp')

# add read and write maps
read_a_entry, read_a_exit = state.add_map(
    'read_a_map', dict(i='0:N//VECLEN'), schedule=dace.ScheduleType.FPGA_Device)
write_c_entry, write_c_exit = state.add_map(
    'write_c_map',
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
C = state.add_write('C_stream')
state.add_memlet_path(A,
                      rtl_tasklet,
                      dst_conn='input',
                      memlet=dace.Memlet.simple('A_stream', '0'))
state.add_memlet_path(rtl_tasklet,
                      C,
                      src_conn='output',
                      memlet=dace.Memlet.simple('C_stream', '0'))

# add write_c memlets and access nodes
write_c_inp = state.add_read('C_stream')
write_c_out = state.add_write('fpga_C')
state.add_memlet_path(write_c_inp,
                      write_c_entry,
                      write_c,
                      dst_conn='inp',
                      memlet=dace.Memlet.simple('C_stream', '0'))
state.add_memlet_path(write_c,
                      write_c_exit,
                      write_c_out,
                      src_conn='out',
                      memlet=dace.Memlet.simple('fpga_C', 'i'))

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
dev_c = copy_to_host.add_read('fpga_C')
cpu_c = copy_to_host.add_write('C')
copy_to_host.add_memlet_path(dev_c,
                             cpu_c,
                             memlet=dace.Memlet.simple('C', '0:N//VECLEN'))
sdfg.add_edge(state, copy_to_host, dace.InterstateEdge())

# validate sdfg
sdfg.validate()

######################################################################

if __name__ == '__main__':

    # init data structures
    N.set(262144 * 128)
    a = np.random.randint(1, 100, N.get()).astype(np.float32)
    c = np.zeros((N.get(), )).astype(np.float32)
    print (a.shape, c.shape)

    # show initial values
    print("a={}".format(a))

    # call program
    sdfg(A=a, C=c, N=N)

    # show result
    print("a={}, c={}".format(a, c))

    # check result
    for i in range(N.get() // veclen):
        assert abs(c[i] - (1. / a[i])) < .001

    print("Assert passed!")
