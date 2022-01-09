# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.

import dace
import numpy as np
import pytest

@pytest.mark.verilator
def test_tasklet_array():
    """
        Test the simple array execution sample.
    """

    n = 128
    N = dace.symbol('N')
    N.set(n)

    # add sdfg
    sdfg = dace.SDFG('rtl_tasklet_array')

    # add state
    state = sdfg.add_state()

    # add arrays
    sdfg.add_array('A', [N], dtype=dace.int32)
    sdfg.add_array('B', [N], dtype=dace.int32)

    # add custom cpp tasklet
    tasklet = state.add_tasklet(name='rtl_tasklet',
                                inputs={'a'},
                                outputs={'b'},
                                code='''
        always@(posedge ap_aclk) begin
            if (ap_areset) begin
                s_axis_a_tready <= 1;
                m_axis_b_tvalid <= 0;
                m_axis_b_tdata <= 0;
            end else if (s_axis_a_tvalid && s_axis_a_tready) begin
                s_axis_a_tready <= 0;
                m_axis_b_tvalid <= 1;
                m_axis_b_tdata <= s_axis_a_tdata + 42;
            end else if (m_axis_b_tvalid && m_axis_b_tready) begin
                s_axis_a_tready <= 1;
                m_axis_b_tvalid <= 0;
                m_axis_b_tdata <= 0;
            end
        end
        ''',
                                language=dace.Language.SystemVerilog)

    # add input/output array
    A = state.add_read('A')
    B = state.add_write('B')

    # connect input/output array with the tasklet
    state.add_edge(A, None, tasklet, 'a', dace.Memlet('A[0:N]'))
    state.add_edge(tasklet, 'b', B, None, dace.Memlet('B[0:N]'))

    # validate sdfg
    sdfg.specialize({'N': N.get()})
    sdfg.validate()

    # init data structures
    a = np.random.randint(0, 100, N.get()).astype(np.int32)
    b = np.zeros((N.get(),)).astype(np.int32)

    # call program
    sdfg(A=a, B=b)

    # check result
    assert (b == a + 42).all()

@pytest.mark.verilator
def test_tasklet_scalar():
    """
        Test the simple scalar execution sample.
    """

    # add sdfg
    sdfg = dace.SDFG('rtl_tasklet_scalar')

    # add state
    state = sdfg.add_state()

    # add arrays
    sdfg.add_scalar('A', dtype=dace.int32)
    sdfg.add_array('B', [1], dtype=dace.int32)

    # add custom cpp tasklet
    tasklet = state.add_tasklet(name='rtl_tasklet',
                                inputs={'a'},
                                outputs={'b'},
                                code='''
        always@(posedge ap_aclk) begin
            if (ap_areset) begin // case: reset
                m_axis_b_tvalid <= 0;
                m_axis_b_tdata <= 0;
            end else if (m_axis_b_tdata < a) begin // case: increment counter b
                m_axis_b_tvalid <= 0;
                m_axis_b_tdata <= m_axis_b_tdata + 1;
            end else begin
                m_axis_b_tvalid <= 1;
            end
        end
        ''',
                                language=dace.Language.SystemVerilog)

    # add input/output array
    A = state.add_read('A')
    B = state.add_write('B')

    # connect input/output array with the tasklet
    state.add_edge(A, None, tasklet, 'a', dace.Memlet('A[0]'))
    state.add_edge(tasklet, 'b', B, None, dace.Memlet('B[0]'))

    # validate sdfg
    sdfg.validate()

    # Execute

    # init data structures
    a = np.random.randint(0, 100, 1).astype(np.int32)
    b = np.zeros((1,)).astype(np.int32)

    # call program
    sdfg(A=a[0], B=b)

    # check result
    assert b[0] == a[0]


@pytest.mark.verilator
def test_tasklet_parameter():
    """
        Test the sv parameter support.
    """

    # add sdfg
    sdfg = dace.SDFG('rtl_tasklet_parameter')

    # add state
    state = sdfg.add_state()

    # add arrays
    sdfg.add_array('A', [1], dtype=dace.int32)
    sdfg.add_array('B', [1], dtype=dace.int32)

    # add parameter(s)
    sdfg.add_constant("MAX_VAL", 42)

    # add custom cpp tasklet
    tasklet = state.add_tasklet(name='rtl_tasklet',
                                inputs={'a'},
                                outputs={'b'},
                                code='''
        /*
            Convention:
               |---------------------------------------------------------------------|
            -->| ap_aclk (clock input)                                               |
            -->| ap_areset (reset input, rst on high)                                |
               |                                                                     |
            -->| {inputs}                                              reg {outputs} |-->
               |                                                                     |
            <--| s_axis_a_tready (ready for data)       (data avail) m_axis_b_tvalid |-->
            -->| s_axis_a_tvalid (new data avail)    (data consumed) m_axis_b_tready |<--
               |---------------------------------------------------------------------|
        */

        typedef enum [1:0] {READY, BUSY, DONE} state_e;
        state_e state;

        always@(posedge ap_aclk) begin
            if (ap_areset) begin // case: reset
                m_axis_b_tdata <= 0;
                s_axis_a_tready <= 1'b1;
                state <= READY;
            end else if (s_axis_a_tvalid && state == READY) begin // case: load a
                m_axis_b_tdata <= s_axis_a_tdata;
                s_axis_a_tready <= 1'b0;
                state <= BUSY;
            end else if (m_axis_b_tdata < MAX_VAL) // case: increment counter b
                m_axis_b_tdata <= m_axis_b_tdata + 1;
            else
                m_axis_b_tdata <= m_axis_b_tdata;
                state <= DONE;
        end

        assign m_axis_b_tvalid  = (m_axis_b_tdata >= MAX_VAL) ? 1'b1:1'b0;
        ''',
                                language=dace.Language.SystemVerilog)

    # add input/output array
    A = state.add_read('A')
    B = state.add_write('B')

    # connect input/output array with the tasklet
    state.add_edge(A, None, tasklet, 'a', dace.Memlet('A[0]'))
    state.add_edge(tasklet, 'b', B, None, dace.Memlet('B[0]'))

    # validate sdfg
    sdfg.validate()

    # execute

    # init data structures
    a = np.random.randint(0, 100, 1).astype(np.int32)
    b = np.random.randint(0, 100, 1).astype(np.int32)

    # call program
    sdfg(A=a, B=b)

    # check result
    assert b == sdfg.constants["MAX_VAL"]


@pytest.mark.verilator
def test_tasklet_vector_add():
    """
        Test rtl tasklet vector support.
    """

    # add symbol
    W = dace.symbol('W')

    # add sdfg
    sdfg = dace.SDFG('rtl_tasklet_vector_add')

    # define compile-time constant
    sdfg.specialize(dict(W=4))

    # add state
    state = sdfg.add_state()

    # add arrays
    sdfg.add_array('A', [1], dtype=dace.vector(dace.int32, dace.symbolic.evaluate(W, sdfg.constants)))
    sdfg.add_array('B', [1], dtype=dace.vector(dace.int32, dace.symbolic.evaluate(W, sdfg.constants)))

    # add custom cpp tasklet
    tasklet = state.add_tasklet(name='rtl_tasklet',
                                inputs={'a'},
                                outputs={'b'},
                                code='''
    always@(posedge ap_aclk) begin
        if (ap_areset) begin
            s_axis_a_tready <= 1;
            m_axis_b_tvalid <= 0;
            m_axis_b_tdata <= 0;
        end else if (s_axis_a_tvalid && s_axis_a_tready) begin
            s_axis_a_tready <= 0;
            m_axis_b_tvalid <= 1;
            for (int i = 0; i < W; i++) begin
                m_axis_b_tdata[i] <= s_axis_a_tdata[i] + 42;
            end
        end else if (m_axis_b_tvalid && m_axis_b_tready) begin
            s_axis_a_tready <= 1;
            m_axis_b_tvalid <= 0;
            m_axis_b_tdata <= 0;
        end
    end
        ''',
                                language=dace.Language.SystemVerilog)

    # add input/output array
    A = state.add_read('A')
    B = state.add_write('B')

    # connect input/output array with the tasklet
    state.add_edge(A, None, tasklet, 'a', dace.Memlet('A[0]'))
    state.add_edge(tasklet, 'b', B, None, dace.Memlet('B[0]'))

    # validate sdfg
    sdfg.validate()

    # Execute

    # init data structures
    a = np.random.randint(0, 100, (dace.symbolic.evaluate(W, sdfg.constants),)).astype(np.int32)
    b = np.zeros((dace.symbolic.evaluate(W, sdfg.constants),)).astype(np.int32)

    # call program
    sdfg(A=a, B=b)

    # check result
    print (a)
    print (b)
    assert (b == a + 42).all()

@pytest.mark.verilator
def test_tasklet_vector_conversion():
    """
        Test rtl tasklet vector conversion support.
    """

    # add symbol
    N = dace.symbol('N')

    # add sdfg
    sdfg = dace.SDFG('rtl_tasklet_vector_conversion')

    # define compile-time constant
    sdfg.specialize(dict(N=4))

    # add state
    state = sdfg.add_state()

    # add arrays
    sdfg.add_array('A', [N], dtype=dace.int32)
    sdfg.add_array('B', [1], dtype=dace.int32)

    # add custom cpp tasklet
    tasklet = state.add_tasklet(name='rtl_tasklet',
                                inputs={'a': dace.vector(dace.int32, N)},
                                outputs={'b'},
                                code='''
        /*
            Convention:
               |---------------------------------------------------------------------|
            -->| ap_aclk (clock input)                                               |
            -->| ap_areset (reset input, rst on high)                                |
               |                                                                     |
            -->| {inputs}                                              reg {outputs} |-->
               |                                                                     |
            <--| s_axis_a_tready (ready for data)       (data avail) m_axis_b_tvalid |-->
            -->| s_axis_a_tvalid (new data avail)    (data consumed) m_axis_b_tready |<--
               |---------------------------------------------------------------------|
        */

        typedef enum [1:0] {READY, BUSY, DONE} state_e;
        state_e state;

        always@(posedge ap_aclk) begin
            if (ap_areset) begin // case: reset
                m_axis_b_tdata <= 0;
                s_axis_a_tready <= 1'b1;
                state <= READY;
            end else if (s_axis_a_tvalid && state == READY) begin // case: load a
                m_axis_b_tdata <= s_axis_a_tdata[0];
                s_axis_a_tready <= 1'b0;
                state <= BUSY;
            end else if (m_axis_b_tdata < s_axis_a_tdata[0] + s_axis_a_tdata[1] && state == BUSY) begin // case: increment counter b
                m_axis_b_tdata <= m_axis_b_tdata + 1;
            end else if (state == BUSY) begin
                m_axis_b_tdata <= m_axis_b_tdata;
                state <= DONE;
            end
        end

        assign m_axis_b_tvalid  = (m_axis_b_tdata >= s_axis_a_tdata[0] + s_axis_a_tdata[1] && (state == BUSY || state == DONE)) ? 1'b1:1'b0;
        ''',
                                language=dace.Language.SystemVerilog)

    # add input/output array
    A = state.add_read('A')
    B = state.add_write('B')

    # connect input/output array with the tasklet
    state.add_edge(A, None, tasklet, 'a', dace.Memlet('A[0:N]'))
    state.add_edge(tasklet, 'b', B, None, dace.Memlet('B[0]'))

    # validate sdfg
    sdfg.validate()

    # Execute

    # init data structures
    a = np.random.randint(0, 100, dace.symbolic.evaluate(N, sdfg.constants)).astype(np.int32)
    b = np.array([0]).astype(np.int32)

    # call program
    sdfg(A=a, B=b)

    # check result
    assert b == a[0] + a[1]


@pytest.mark.verilator
def test_multi_tasklet():
    """
        Test multiple rtl tasklet support.
    """

    # add sdfg
    sdfg = dace.SDFG('rtl_multi_tasklet')

    # add state
    state = sdfg.add_state()

    # add arrays
    sdfg.add_array('A', [1], dtype=dace.int32)
    sdfg.add_array('B', [1], dtype=dace.int32)
    sdfg.add_array('C', [1], dtype=dace.int32)

    # add custom cpp tasklet
    tasklet0 = state.add_tasklet(name='rtl_tasklet0',
                                 inputs={'a'},
                                 outputs={'b'},
                                 code='''
        typedef enum [1:0] {READY, BUSY, DONE} state_e;
        state_e state;

        always@(posedge ap_aclk) begin
            if (ap_areset) begin // case: reset
                m_axis_b_tdata <= 0;
                s_axis_a_tready <= 1'b1;
                state <= READY;
            end else if (s_axis_a_tvalid && state == READY) begin // case: load a
                m_axis_b_tdata <= s_axis_a_tdata;
                s_axis_a_tready <= 1'b0;
                state <= BUSY;
            end else if (m_axis_b_tdata < 80) // case: increment counter b
                m_axis_b_tdata <= m_axis_b_tdata + 1;
            else
                m_axis_b_tdata <= m_axis_b_tdata;
                state <= DONE;
        end

        assign m_axis_b_tvalid = (m_axis_b_tdata >= 80) ? 1'b1:1'b0;
        ''',
                                 language=dace.Language.SystemVerilog)

    tasklet1 = state.add_tasklet(name='rtl_tasklet1',
                                 inputs={'b'},
                                 outputs={'c'},
                                 code='''
        typedef enum [1:0] {READY, BUSY, DONE} state_e;
        state_e state;

        always@(posedge ap_aclk) begin
            if (ap_areset) begin // case: reset
                m_axis_c_tdata <= 0;
                s_axis_b_tready <= 1'b1;
                state <= READY;
            end else if (s_axis_b_tvalid && state == READY) begin // case: load a
                m_axis_c_tdata <= s_axis_b_tdata;
                s_axis_b_tready <= 1'b0;
                state <= BUSY;
            end else if (m_axis_c_tdata < 100) // case: increment counter b
                m_axis_c_tdata <= m_axis_c_tdata + 1;
            else
                m_axis_c_tdata <= m_axis_c_tdata;
                state <= DONE;
        end

        assign m_axis_c_tvalid = (m_axis_c_tdata >= 100) ? 1'b1:1'b0;
        ''',
                                 language=dace.Language.SystemVerilog)

    # add input/output array
    A = state.add_read('A')
    B_w = state.add_write('B')
    B_r = state.add_read('B')
    C = state.add_write('C')

    # connect input/output array with the tasklet
    state.add_edge(A, None, tasklet0, 'a', dace.Memlet('A[0]'))
    state.add_edge(tasklet0, 'b', B_w, None, dace.Memlet('B[0]'))
    state.add_edge(B_r, None, tasklet1, 'b', dace.Memlet('B[0]'))
    state.add_edge(tasklet1, 'c', C, None, dace.Memlet('C[0]'))

    # validate sdfg
    sdfg.validate()

    # Execute

    # init data structures
    a = np.random.randint(0, 80, 1).astype(np.int32)
    b = np.array([0]).astype(np.int32)
    c = np.array([0]).astype(np.int32)

    # call program
    sdfg(A=a, B=b, C=c)

    # check result
    assert b == 80
    assert c == 100

@pytest.mark.verilator
def test_tasklet_map():
    '''
        Test the unrolled map support for M tasklets on N vectors of size W.
    '''
    # add symbols
    n = 512
    m = 8
    w = 4
    N = dace.symbol('N')
    M = dace.symbol('M')
    W = dace.symbol('W')
    N.set(n)
    M.set(m)
    W.set(w)

    # add sdfg
    sdfg = dace.SDFG('rtl_tasklet_map')

    # add state
    state = sdfg.add_state()

    # add arrays
    sdfg.add_array('A', [M,N], dtype=dace.vector(dace.int32, W.get()))
    sdfg.add_array('B', [M,N], dtype=dace.vector(dace.int32, W.get()))
    sdfg.add_array('C', [M,N], dtype=dace.vector(dace.int32, W.get()))

    mentry, mexit = state.add_map('compute_map', {'k': '0:M'})

    tasklet = state.add_tasklet(name='rtl_tasklet1',
                                 inputs={'a','b'},
                                 outputs={'c'},
                                 code='''
reg [W-1:0][31:0] a_data;
reg a_valid;
reg [W-1:0][31:0] b_data;
reg b_valid;

// Read A
always@(posedge ap_aclk) begin
    if (ap_areset) begin
        s_axis_a_tready <= 0;
        a_valid <= 0;
        a_data <= 0;
    end else begin
        if (s_axis_a_tready && s_axis_a_tvalid) begin
            a_valid <= 1;
            a_data <= s_axis_a_tdata;
            s_axis_a_tready <= 0;
        end else if (m_axis_c_tvalid && m_axis_c_tready) begin
            a_valid <= 0;
            s_axis_a_tready <= 1;
        end else begin
            s_axis_a_tready <= ~a_valid;
        end
    end
end

// Read B
always@(posedge ap_aclk) begin
    if (ap_areset) begin
        s_axis_b_tready <= 0;
        b_valid <= 0;
        b_data <= 0;
    end else begin
        if (s_axis_b_tready && s_axis_b_tvalid) begin
            b_valid <= 1;
            b_data <= s_axis_b_tdata;
            s_axis_b_tready <= 0;
        end else if (m_axis_c_tvalid && m_axis_c_tready) begin
            b_valid <= 0;
            b_data <= 0;
            s_axis_b_tready <= 1;
        end else begin
            s_axis_b_tready <= ~b_valid;
        end
    end
end

// Compute and write C
always@(posedge ap_aclk) begin
    if (ap_areset) begin
        m_axis_c_tvalid <= 0;
        m_axis_c_tdata <= 0;
    end else begin
        if (m_axis_c_tvalid && m_axis_c_tready) begin
            m_axis_c_tvalid <= 0;
        end else if (a_valid && b_valid) begin
            m_axis_c_tvalid <= 1;
            m_axis_c_tdata <= a_data + b_data;
        end
    end
end''',
                                 language=dace.Language.SystemVerilog)

    A = state.add_read('A')
    B = state.add_read('B')
    C = state.add_write('C')

    state.add_memlet_path(A, mentry, tasklet, memlet=dace.Memlet('A[k,0:N]'), dst_conn='a')
    state.add_memlet_path(B, mentry, tasklet, memlet=dace.Memlet('B[k,0:N]'), dst_conn='b')
    state.add_memlet_path(tasklet, mexit, C, memlet=dace.Memlet('C[k,0:N]'), src_conn='c')

    sdfg.specialize({'M': M, 'N': N, 'W': W})
    sdfg.validate()

    # init data structures
    a = np.random.randint(0, 100, m*n*w).reshape((m,n,w)).astype(np.int32)
    b = np.random.randint(0, 100, m*n*w).reshape((m,n,w)).astype(np.int32)
    c = np.zeros((m,n,w)).astype(np.int32)

    # call program
    sdfg(A=a, B=b, C=c)

    # check result
    assert (c == a + b).all()

if __name__ == '__main__':
    #test_multi_tasklet()
    #test_tasklet_array()
    #test_tasklet_map()
    #test_tasklet_parameter()
    #test_tasklet_scalar()
    test_tasklet_vector_add()
    #test_tasklet_vector_conversion()