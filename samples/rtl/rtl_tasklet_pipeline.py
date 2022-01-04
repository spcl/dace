# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
"""
    Pipelined, AXI-handshake compliant example that increments b from a up to 100.
"""

import dace
import argparse

import numpy as np

# add symbol
N = dace.symbol('N')

# add sdfg
sdfg = dace.SDFG('rtl_tasklet_pipeline')

# add state
state = sdfg.add_state()

# define compile-time constant
sdfg.specialize(dict(N=4))

# disable sv debugging output
sdfg.add_constant("SYSTEMVERILOG_DEBUG", False)

# add arrays
sdfg.add_array('A', [N], dtype=dace.int32)
sdfg.add_array('B', [N], dtype=dace.int32)

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

    /****
    * Finite State Machine
    *****/
    typedef enum [1:0] {READY, BUSY, DONE} state_e;
    state_e state, state_next;

    always@(posedge ap_aclk)
    begin
        if(ap_areset)
            state <= READY;
        else
            state <= state_next;
    end

    always_comb 
    begin
        state_next = state;
        case(state)
            READY: if(s_axis_a_tvalid) state_next = BUSY;
            BUSY: if(m_axis_b_tdata >= 99) state_next = DONE;
            DONE: if(m_axis_b_tready) state_next = READY;
            default: state_next = state;
        endcase
    end


    /***********
    * Control Logic
    ************/
    always_comb
    begin
        // init default value
        s_axis_a_tready = 0;
        m_axis_b_tvalid = 0;
        // set actual value
        case(state)
            READY:  s_axis_a_tready = 1;
            DONE:   m_axis_b_tvalid = 1;
            default:;
        endcase
    end

    /****
    * Data Path
    ****/
    always@(posedge ap_aclk)
    begin
        case(state)
            READY: if(s_axis_a_tvalid) m_axis_b_tdata <= s_axis_a_tdata;
            BUSY: m_axis_b_tdata <= m_axis_b_tdata + 1;
            DONE: m_axis_b_tdata <= m_axis_b_tdata;
            default: m_axis_b_tdata <= m_axis_b_tdata;
        endcase
    end

    /*****
    * DEBUG
    *****/
    always@(posedge ap_aclk)
    begin
        if(SYSTEMVERILOG_DEBUG)
        begin
            case(state)
                READY: $display("READY");
                BUSY: $display("BUSY");
                DONE: $display("DONE");
                default: $display("Undefined State");
            endcase
        end
    end
    ''',
                            language=dace.Language.SystemVerilog)

# add input/output array
A = state.add_read('A')
B = state.add_write('B')

# connect input/output array with the tasklet
state.add_edge(A, None, tasklet, 'a', dace.Memlet('A[0:N-1]'))
state.add_edge(tasklet, 'b', B, None, dace.Memlet('B[0:N-1]'))

# validate sdfg
sdfg.validate()

######################################################################

if __name__ == '__main__':

    # init data structures
    num_elements = dace.symbolic.evaluate(N, sdfg.constants)
    a = np.random.randint(0, 100, num_elements).astype(np.int32)
    b = np.array([0] * num_elements).astype(np.int32)

    # show initial values
    print("a={}, b={}".format(a, b))

    # call program
    sdfg(A=a, B=b)

    # show result
    print("a={}, b={}".format(a, b))

    assert b[
        0] == 100  # TODO: implement detection of #elements to process, s.t. we can extend the assertion to the whole array
    assert np.all(map((lambda x: x == 0), b[1:-1]))  # should still be at the init value (for the moment)
