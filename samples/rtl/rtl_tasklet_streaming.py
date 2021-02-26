# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
"""
    Free running (autorun) tasklet example.
"""

import dace
import argparse

import numpy as np

# add symbol
N = dace.symbol('N')

# add sdfg
sdfg = dace.SDFG('rtl_tasklet_streaming')

# add state
state = sdfg.add_state()

# define compile-time constant
sdfg.specialize(dict(N=4))

# disable sv debugging output
sdfg.add_constant("SYSTEMVERILOG_DEBUG", False)

# add arrays
sdfg.add_array('A', dtype=dace.int32, shape=[N])
sdfg.add_array('B', dtype=dace.int32, shape=[N])

# add streams
sdfg.add_stream("PIPE_IN", dtype=dace.int32, transient=True, buffer_size=1)
sdfg.add_stream("PIPE_OUT", dtype=dace.int32, transient=True, buffer_size=1)

# add custom cpp tasklet
tasklet_read = state.add_tasklet(name='tasklet_read',
                            inputs={'a'},
                            outputs={'pipe_in'},
                            code='''
for(int i = 0; i < N; i++){
    PIPE_IN.push(a[i]);
}
                            ''',
                            language=dace.Language.CPP)

tasklet_rtl = state.add_tasklet(name='tasklet_rtl',
                            inputs={'pipe_in'},
                            outputs={'pipe_out'},
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
        -->| valid_i (new data avail)    (data consumed) ready_i |<--
           |----------------------------------------------------|
    */

    /****
    * Finite State Machine
    *****/
    typedef enum [1:0] {READY, BUSY, DONE} state_e;
    state_e state, state_next;

    always@(posedge clk_i)
    begin
        if(rst_i)
            state <= READY;
        else
            state <= state_next;
    end

    always_comb 
    begin
        state_next = state;
        case(state)
            READY: if(valid_i) state_next = BUSY;
            BUSY: if(pipe_out >= 99) state_next = DONE;
            DONE: if(ready_i) state_next = READY;
            default: state_next = state;
        endcase
    end


    /***********
    * Control Logic
    ************/
    always_comb
    begin
        // init default value
        ready_o = 0;
        valid_o = 0;
        // set actual value
        case(state)
            READY:  ready_o = 1;
            DONE:   valid_o = 1;
            default:;
        endcase
    end

    /****
    * Data Path
    ****/
    always@(posedge clk_i)
    begin
        case(state)
            READY: if(valid_i) pipe_out <= pipe_in;
            BUSY: pipe_out <= pipe_out + 1;
            DONE: pipe_out <= pipe_out;
            default: pipe_out <= pipe_out;
        endcase
    end

    /*****
    * DEBUG
    *****/
    always@(posedge clk_i)
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

tasklet_write = state.add_tasklet(name='tasklet_write',
                            inputs={'pipe_out'},
                            outputs={'b'},
                            code='''
for(int i = 0; i < N; i++){
    b[i] = PIPE_OUT.pop();
}
                            ''',
                            language=dace.Language.CPP)


# add input/output array
A = state.add_read('A')
PIPE_IN_W = state.add_write('PIPE_IN')
PIPE_IN_R = state.add_read('PIPE_IN')
PIPE_OUT_W = state.add_write('PIPE_OUT')
PIPE_OUT_R = state.add_read('PIPE_OUT')
B = state.add_write('B')

# connect input/output array with the tasklet
state.add_edge(A, None, tasklet_read, 'a', dace.Memlet.simple('A', '0:N-1'))  # array -> read tasklet
state.add_edge(tasklet_read, 'pipe_in', PIPE_IN_W, None, dace.Memlet.simple('PIPE_IN', '0:N-1'))  # read tasklet -> pipe
state.add_edge(PIPE_IN_R, None, tasklet_rtl, 'pipe_in', dace.Memlet.simple('PIPE_IN', '0:N-1'))  # pipe -> rtl tasklet
state.add_edge(tasklet_rtl, 'pipe_out', PIPE_OUT_W, None, dace.Memlet.simple('PIPE_OUT', '0:N-1'))  # rtl tasklet -> pipe
state.add_edge(PIPE_OUT_R, None, tasklet_write, 'pipe_out', dace.Memlet.simple('PIPE_OUT', '0:N-1'))  # pipe -> write tasklet
state.add_edge(tasklet_write, 'b', B, None, dace.Memlet.simple('B', '0:N-1'))  # write tasklet -> array

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

    assert np.all(map((lambda x: x == 0), b))
