# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.

import dace
import numpy as np
import pytest


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
    sdfg.add_array('A', [1], dtype=dace.int32)
    sdfg.add_array('B', [1], dtype=dace.int32)

    # add custom cpp tasklet
    tasklet = state.add_tasklet(name='rtl_tasklet',
                                inputs={'a'},
                                outputs={'b'},
                                code='''
        /*
            Convention:
               |----------------------------------------------------|
            -->| clk_i (clock input)                                |
            -->| rst_i (reset input, rst on high)                   |
               |                                                    |
            -->| {inputs}                             reg {outputs} |-->
               |                                                    |
            <--| ready_o (ready for data)       (data avail) valid_o|-->
            -->| valid_i (new data avail)    (data consumed) ready_i|<--
               |----------------------------------------------------|
        */

        typedef enum [1:0] {READY, BUSY, DONE} state_e;
        state_e state;
        
        always@(posedge clk_i) begin
            if (rst_i) begin // case: reset
                b <= 0;
                ready_o <= 1'b1;
                state <= READY;
            end else if (valid_i && state == READY) begin // case: load a 
                b <= a;
                ready_o <= 1'b0;
                state <= BUSY;
            end else if (b < 100) // case: increment counter b
                b <= b + 1;
            else
                b <= b;
                state <= DONE;
        end    
        
        assign valid_o = (b >= 100) ? 1'b1:1'b0; 
        ''',
                                language=dace.Language.SystemVerilog)

    # add input/output array
    A = state.add_read('A')
    B = state.add_write('B')

    # connect input/output array with the tasklet
    state.add_edge(A, None, tasklet, 'a', dace.Memlet.simple('A', '0'))
    state.add_edge(tasklet, 'b', B, None, dace.Memlet.simple('B', '0'))

    # validate sdfg
    sdfg.validate()

    # Execute

    # init data structures
    a = np.random.randint(0, 100, 1).astype(np.int32)
    b = np.random.randint(0, 100, 1).astype(np.int32)

    # call program
    sdfg(A=a, B=b)

    # check result
    assert b == 100


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
               |----------------------------------------------------|
            -->| clk_i (clock input)                                |
            -->| rst_i (reset input, rst on high)                   |
               |                                                    |
            -->| {inputs}                             reg {outputs} |-->
               |                                                    |
            <--| ready_o (ready for data)       (data avail) valid_o|-->
            -->| valid_i (new data avail)    (data consumed) ready_i|<--
               |----------------------------------------------------|
        */

        typedef enum [1:0] {READY, BUSY, DONE} state_e;
        state_e state;
    
        always@(posedge clk_i) begin
            if (rst_i) begin // case: reset
                b <= 0;
                ready_o <= 1'b1;
                state <= READY;
            end else if (valid_i && state == READY) begin // case: load a 
                b <= a;
                ready_o <= 1'b0;
                state <= BUSY;
            end else if (b < MAX_VAL) // case: increment counter b
                b <= b + 1;
            else
                b <= b;
                state <= DONE;
        end    
    
        assign valid_o = (b >= MAX_VAL) ? 1'b1:1'b0;
        ''',
                                language=dace.Language.SystemVerilog)

    # add input/output array
    A = state.add_read('A')
    B = state.add_write('B')

    # connect input/output array with the tasklet
    state.add_edge(A, None, tasklet, 'a', dace.Memlet.simple('A', '0'))
    state.add_edge(tasklet, 'b', B, None, dace.Memlet.simple('B', '0'))

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
def test_tasklet_vector():
    """
        Test rtl tasklet vector support.
    """

    # add symbol
    N = dace.symbol('N')

    # add sdfg
    sdfg = dace.SDFG('rtl_tasklet_vector')

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
               |----------------------------------------------------|
            -->| clk_i (clock input)                                |
            -->| rst_i (reset input, rst on high)                   |
               |                                                    |
            -->| {inputs}                             reg {outputs} |-->
               |                                                    |
            <--| ready_o (ready for data)       (data avail) valid_o|-->
            -->| valid_i (new data avail)    (data consumed) ready_i|<--
               |----------------------------------------------------|
        */

        typedef enum [1:0] {READY, BUSY, DONE} state_e;
        state_e state;
    
        always@(posedge clk_i) begin
            if (rst_i) begin // case: reset
                b <= 0;
                ready_o <= 1'b1;
                state <= READY;
            end else if (valid_i && state == READY) begin // case: load a 
                b <= a[0];
                ready_o <= 1'b0;
                state <= BUSY;
            end else if (b < a[0] + a[1] && state == BUSY) begin // case: increment counter b
                b <= b + 1;
            end else if (state == BUSY) begin
                b <= b;
                state <= DONE;
            end
        end    
    
        assign valid_o = (b >= a[0] + a[1] && (state == BUSY || state == DONE)) ? 1'b1:1'b0; 
        ''',
                                language=dace.Language.SystemVerilog)

    # add input/output array
    A = state.add_read('A')
    B = state.add_write('B')

    # connect input/output array with the tasklet
    state.add_edge(A, None, tasklet, 'a', dace.Memlet.simple('A', '0:N-1'))
    state.add_edge(tasklet, 'b', B, None, dace.Memlet.simple('B', '0'))

    # validate sdfg
    sdfg.validate()

    # Execute

    # init data structures
    a = np.random.randint(0, 100, dace.symbolic.evaluate(
        N, sdfg.constants)).astype(np.int32)
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
    
        always@(posedge clk_i) begin
            if (rst_i) begin // case: reset
                b <= 0;
                ready_o <= 1'b1;
                state <= READY;
            end else if (valid_i && state == READY) begin // case: load a 
                b <= a;
                ready_o <= 1'b0;
                state <= BUSY;
            end else if (b < 80) // case: increment counter b
                b <= b + 1;
            else
                b <= b;
                state <= DONE;
        end    
    
        assign valid_o = (b >= 80) ? 1'b1:1'b0; 
        ''',
                                 language=dace.Language.SystemVerilog)

    tasklet1 = state.add_tasklet(name='rtl_tasklet1',
                                 inputs={'b'},
                                 outputs={'c'},
                                 code='''
        typedef enum [1:0] {READY, BUSY, DONE} state_e;
        state_e state;
    
        always@(posedge clk_i) begin
            if (rst_i) begin // case: reset
                c <= 0;
                ready_o <= 1'b1;
                state <= READY;
            end else if (valid_i && state == READY) begin // case: load a 
                c <= b;
                ready_o <= 1'b0;
                state <= BUSY;
            end else if (c < 100) // case: increment counter b
                c <= c + 1;
            else
                c <= c;
                state <= DONE;
        end    
    
        assign valid_o = (c >= 100) ? 1'b1:1'b0;  
        ''',
                                 language=dace.Language.SystemVerilog)

    # add input/output array
    A = state.add_read('A')
    B_w = state.add_write('B')
    B_r = state.add_read('B')
    C = state.add_write('C')

    # connect input/output array with the tasklet
    state.add_edge(A, None, tasklet0, 'a', dace.Memlet.simple('A', '0'))
    state.add_edge(tasklet0, 'b', B_w, None, dace.Memlet.simple('B', '0'))
    state.add_edge(B_r, None, tasklet1, 'b', dace.Memlet.simple('B', '0'))
    state.add_edge(tasklet1, 'c', C, None, dace.Memlet.simple('C', '0'))

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


if __name__ == '__main__':
    test_tasklet_scalar()
    test_tasklet_parameter()
    test_tasklet_vector()
    test_multi_tasklet()
