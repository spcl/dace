# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
import dace
from dace.fpga_testing import rtl_test
import numpy as np
import importlib.util
from pathlib import Path
import pytest
from dace.transformation.dataflow import StreamingMemory, Vectorization
from dace.transformation.interstate import FPGATransformState
from dace.transformation.subgraph import TemporalVectorization


def make_vadd_sdfg(N: dace.symbol, veclen: int = 8):
    '''
    Function for generating a simple vector addition SDFG that adds a vector `A` of `N` elements to a scalar `B` into a vector `C` of `N` elements, all using SystemVerilog.
    The tasklet creates `veclen` instances of a floating point adder that operates on `N` elements. 

    :param N: The number of elements the SDFG takes as input and output.
    :param veclen: The number of floating point adders to instantiate.
    :return: An SDFG that has arguments `A`, `B` and `C`.
    '''
    # add sdfg
    sdfg = dace.SDFG('floating_point_vector_plus_scalar')

    # add state
    state = sdfg.add_state('device_state')

    # add parameter
    sdfg.add_constant('VECLEN', veclen)

    # add arrays
    sdfg.add_array('A', [N // veclen], dtype=dace.vector(dace.float32, veclen), storage=dace.StorageType.CPU_Heap)
    sdfg.add_scalar('B', dace.float32, storage=dace.StorageType.FPGA_Global)
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
        assign ap_done = 1;
        wire ap_aresetn = ~ap_areset;

        wire [VECLEN-1:0]       a_tvalid;
        wire [VECLEN-1:0][31:0] a_tdata;
        wire [VECLEN-1:0]       a_tready;

        wire [VECLEN-1:0]       c_tvalid;
        wire [VECLEN-1:0][31:0] c_tdata;
        wire [VECLEN-1:0]       c_tready;

        axis_broadcaster_0 ab0(
            .aclk    (ap_aclk),
            .aresetn (ap_aresetn),

            .s_axis_tvalid (s_axis_a_tvalid),
            .s_axis_tdata  (s_axis_a_tdata),
            .s_axis_tready (s_axis_a_tready),

            .m_axis_tvalid (a_tvalid),
            .m_axis_tdata  (a_tdata),
            .m_axis_tready (a_tready)
        );

        generate
            for (genvar i = 0; i < VECLEN; i = i + 1) begin
                floating_point_add add(
                    .aclk    (ap_aclk),
                    .aresetn (ap_aresetn),

                    .s_axis_a_tvalid (a_tvalid[i]),
                    .s_axis_a_tdata  (a_tdata[i]),
                    .s_axis_a_tready (a_tready[i]),

                    .s_axis_b_tvalid (scalars_valid),
                    .s_axis_b_tdata  (b),

                    .m_axis_result_tvalid (c_tvalid[i]),
                    .m_axis_result_tdata  (c_tdata[i]),
                    .m_axis_result_tready (c_tready[i])
                );
            end
        endgenerate

        axis_combiner_0 ac0(
            .aclk    (ap_aclk),
            .aresetn (ap_aresetn),

            .s_axis_tvalid (c_tvalid),
            .s_axis_tdata  (c_tdata),
            .s_axis_tready (c_tready),

            .m_axis_tvalid (m_axis_c_tvalid),
            .m_axis_tdata  (m_axis_c_tdata),
            .m_axis_tready (m_axis_c_tready)
        );
        ''',
                                    language=dace.Language.SystemVerilog)

    rtl_tasklet.add_ip_core(
        'floating_point_add', 'floating_point', 'xilinx.com', '7.1', {
            'CONFIG.Add_Sub_Value': 'Add',
            'CONFIG.Has_ARESETn': 'true',
            'CONFIG.Axi_Optimize_Goal': 'Performance',
            'CONFIG.C_Latency': '14'
        })
    rtl_tasklet.add_ip_core(
        'axis_broadcaster_0', 'axis_broadcaster', 'xilinx.com', '1.1',
        dict({
            'CONFIG.NUM_MI': f'{veclen}',
            'CONFIG.M_TDATA_NUM_BYTES': '4',
            'CONFIG.S_TDATA_NUM_BYTES': f'{veclen*4}'
        }, **{f'CONFIG.M{i:02}_TDATA_REMAP': f'tdata[{((i+1)*32)-1}:{i*32}]'
              for i in range(veclen)}))
    rtl_tasklet.add_ip_core('axis_combiner_0', 'axis_combiner', 'xilinx.com', '1.1', {
        'CONFIG.TDATA_NUM_BYTES': '4',
        'CONFIG.NUM_SI': f'{veclen}'
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


def make_vadd_multi_sdfg(N, M):
    '''
    Function for constructing an SDFG that adds a constant (42) to a an array `A` of `N` elements into a vector `B`.
    Each instance of an adder is within its own Processing Element (PE), along with a reader and writer to/from global memory.
    This SDFG also utilizes array of streams, giving each compute PE its own set of in- and output streams.

    :param N: The number of elements to compute on.
    :param M: The number of compute PEs to initialize.
    :return: An SDFG that has arguments `A` and `B`. 
    '''
    # add sdfg
    sdfg = dace.SDFG(f'integer_vector_plus_42_multiple_kernels_{N.get() // M.get()}')

    # add state
    state = sdfg.add_state('device_state')

    # add arrays
    sdfg.add_array('A', [N], dtype=dace.int32, storage=dace.StorageType.CPU_Heap)
    sdfg.add_array('B', [N], dtype=dace.int32, storage=dace.StorageType.CPU_Heap)
    sdfg.add_array('fpga_A', [N], dtype=dace.int32, transient=True, storage=dace.StorageType.FPGA_Global)
    sdfg.add_array('fpga_B', [N], dtype=dace.int32, transient=True, storage=dace.StorageType.FPGA_Global)

    # add streams
    sdfg.add_stream('A_stream',
                    shape=(int(N.get() / M.get()), ),
                    dtype=dace.int32,
                    transient=True,
                    storage=dace.StorageType.FPGA_Local)
    sdfg.add_stream('B_stream',
                    shape=(int(N.get() / M.get()), ),
                    dtype=dace.int32,
                    transient=True,
                    storage=dace.StorageType.FPGA_Local)

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
    read_a_entry, read_a_exit = state.add_map('read_a_map',
                                              dict(i='0:N//M', j='0:M'),
                                              schedule=dace.ScheduleType.FPGA_Device)
    write_b_entry, write_b_exit = state.add_map('write_b_map',
                                                dict(i='0:N//M', j='0:M'),
                                                schedule=dace.ScheduleType.FPGA_Device)
    compute_entry, compute_exit = state.add_map('compute_map',
                                                dict(i='0:N//M'),
                                                schedule=dace.ScheduleType.FPGA_Device,
                                                unroll=True)

    # add read_a memlets and access nodes
    read_a_inp = state.add_read('fpga_A')
    read_a_out = state.add_write('A_stream')
    state.add_memlet_path(read_a_inp, read_a_entry, read_a, dst_conn='inp', memlet=dace.Memlet('fpga_A[i*M+j]'))
    state.add_memlet_path(read_a, read_a_exit, read_a_out, src_conn='out', memlet=dace.Memlet('A_stream[i]'))

    # add tasklet memlets
    A = state.add_read('A_stream')
    B = state.add_write('B_stream')
    state.add_memlet_path(A, compute_entry, rtl_tasklet, dst_conn='a', memlet=dace.Memlet('A_stream[i]'))
    state.add_memlet_path(rtl_tasklet, compute_exit, B, src_conn='b', memlet=dace.Memlet('B_stream[i]'))

    # add write_b memlets and access nodes
    write_b_inp = state.add_read('B_stream')
    write_b_out = state.add_write('fpga_B')
    state.add_memlet_path(write_b_inp, write_b_entry, write_b, dst_conn='inp', memlet=dace.Memlet('B_stream[i]'))
    state.add_memlet_path(write_b, write_b_exit, write_b_out, src_conn='out', memlet=dace.Memlet('fpga_B[i*M+j]'))

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

    return sdfg


@rtl_test()
def test_hardware_vadd():
    '''
    Test for the simple vector addition. 
    '''

    # add symbol
    N = dace.symbol('N')
    N.set(32)
    veclen = 4
    sdfg = make_vadd_sdfg(N, veclen)
    a = np.random.randint(0, 100, N.get()).astype(np.float32)
    b = np.random.randint(1, 100, 1)[0].astype(np.float32)
    c = np.zeros((N.get(), )).astype(np.float32)

    # call program
    sdfg(A=a, B=b, C=c, N=N)

    expected = a + b
    diff = np.linalg.norm(expected - c) / N.get()
    assert diff <= 1e-5

    return sdfg


@rtl_test()
def test_hardware_add42_single():
    '''
    Test for adding a constant using a single PE. 
    '''
    N = dace.symbol('N')
    M = dace.symbol('M')

    # init data structures
    N.set(32)  # elements
    M.set(32)  # elements per kernel
    a = np.random.randint(0, 100, N.get()).astype(np.int32)
    b = np.zeros((N.get(), )).astype(np.int32)
    sdfg = make_vadd_multi_sdfg(N, M)
    sdfg.specialize(dict(N=N, M=M))

    # call program
    sdfg(A=a, B=b)

    # check result
    for i in range(N.get()):
        assert b[i] == a[i] + 42

    return sdfg


@pytest.mark.skip(reason="This test is covered by the Xilinx tests.")
def test_hardware_axpy_double_pump(veclen=2):
    '''
    Tests manual application of the multi-pumping optimization applied to the AXPY program from BLAS.

    :param veclen: The vectorization length to instantiate. Must be a multiple of 2.
    '''
    with dace.config.set_temporary('compiler', 'xilinx', 'frequency', value='"0:300\\|1:600"'):
        # Grab the double pumped AXPY implementation the samples directory
        spec = importlib.util.spec_from_file_location(
            "axpy",
            Path(__file__).parent.parent.parent / "samples" / "fpga" / "rtl" / "axpy_double_pump.py")
        axpy = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(axpy)

        # init data structures
        N = dace.symbol('N')
        N.set(32)
        a = np.random.rand(1)[0].astype(np.float32)
        x = np.random.rand(N.get()).astype(np.float32)
        y = np.random.rand(N.get()).astype(np.float32)
        result = np.zeros((N.get(), )).astype(np.float32)

        # Build the SDFG
        sdfg = axpy.make_sdfg(veclen)

        # call program
        sdfg(a=a, x=x, y=y, result=result, N=N)

        # check result
        expected = a * x + y
        diff = np.linalg.norm(expected - result) / N.get()

    assert diff <= 1e-5

    return sdfg


@rtl_test()
def test_hardware_axpy_double_pump_vec2():
    '''
    Tests double pumping with a vector length of 2.
    '''
    return test_hardware_axpy_double_pump(veclen=2)


@rtl_test()
def test_hardware_axpy_double_pump_vec4():
    '''
    Tests double pumping with a vector length of 4.
    '''
    return test_hardware_axpy_double_pump(veclen=4)


@rtl_test()
def test_hardware_vadd_temporal_vectorization():
    '''
    Tests whether the multi-pumping optimization can be applied automatically by applying the temporal vectorization transformation. It starts from a numpy vector addition for generating the SDFG. This SDFG is then optimized by applying the vectorization, streaming memory, fpga and temporal vectorization transformations in that order.
    '''
    # TODO !!!!! THIS TEST STALLS WITH VITIS 2021.2 !!!!!
    # But it works fine for 2020.2
    with dace.config.set_temporary('compiler', 'xilinx', 'frequency', value='"0:300\\|1:600"'):
        # Generate the test data and expected results
        size_n = 1024
        veclen = 4
        N = dace.symbol('N')
        N.set(size_n)
        x = np.random.rand(N.get()).astype(np.float32)
        y = np.random.rand(N.get()).astype(np.float32)
        result = np.zeros(N.get(), dtype=np.float32)
        expected = x + y

        # Generate the initial SDFG
        def np_vadd(x: dace.float32[N], y: dace.float32[N]):
            return x + y

        sdfg = dace.program(np_vadd).to_sdfg()

        # Remove underscores as Xilinx does not like them
        for dn in sdfg.nodes()[0].data_nodes():
            if '__' in dn.data:
                new_name = dn.data.replace('__', '') + 'new'
                sdfg.replace(dn.data, new_name)

        # Apply vectorization transformation
        ambles = size_n % veclen != 0
        map_entry = [n for n, _ in sdfg.all_nodes_recursive() if isinstance(n, dace.nodes.MapEntry)][0]
        applied = sdfg.apply_transformations(
            Vectorization, {
                'vector_len': veclen,
                'preamble': ambles,
                'postamble': ambles,
                'propagate_parent': True,
                'strided_map': False,
                'map_entry': map_entry
            })
        assert (applied == 1)

        # Transform to an FPGA implementation
        applied = sdfg.apply_transformations(FPGATransformState)
        assert (applied == 1)

        # Apply streaming memory transformation
        applied = sdfg.apply_transformations_repeated(StreamingMemory, {
            'storage': dace.StorageType.FPGA_Local,
            'buffer_size': 1
        })
        assert (applied == 3)

        # Apply temporal vectorization transformation
        sgs = dace.sdfg.concurrent_subgraphs(sdfg.states()[0])
        sf = TemporalVectorization()
        cba = [TemporalVectorization.can_be_applied(sf, sdfg, sg) for sg in sgs]
        assert (sum(cba) == 1)
        [TemporalVectorization.apply_to(sdfg, sg) for i, sg in enumerate(sgs) if cba[i]]

        # Run the program and verify the results
        sdfg.specialize({'N': N.get()})
        sdfg(x=x, y=y, returnnew=result)
        assert (np.allclose(expected, result))
        
        return sdfg


# TODO disabled due to problem with array of streams in Vitis 2021.1
#rtl_test()
#def test_hardware_add42_multi():
#    N = dace.symbol('N')
#    M = dace.symbol('M')
#
#    # init data structures
#    N.set(32)  # elements
#    M.set(8)  # elements per kernel
#    a = np.random.randint(0, 100, N.get()).astype(np.int32)
#    b = np.zeros((N.get(), )).astype(np.int32)
#    sdfg = make_vadd_multi_sdfg(N, M)
#    sdfg.specialize(dict(N=N, M=M))
#
#    # call program
#    sdfg(A=a, B=b)
#
#    # check result
#    for i in range(N.get()):
#        assert b[i] == a[i] + 42
#
#    return sdfg

if __name__ == '__main__':
    # These tests should only be run in hardware* mode
    with dace.config.set_temporary('compiler', 'xilinx', 'mode', value='hardware_emulation'):
        test_hardware_vadd(None)
        test_hardware_vadd_temporal_vectorization(None)
        test_hardware_add42_single(None)
        # TODO disabled due to problem with array of streams in Vitis 2021.1
        #test_hardware_add42_multi(None)
        test_hardware_axpy_double_pump_vec2(None)
        test_hardware_axpy_double_pump_vec4(None)
