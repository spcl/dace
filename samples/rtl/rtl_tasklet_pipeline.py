#!/usr/bin/env python3

"""
    Pipelined, axi-handshake compliant example that increments b from a up to 100.
"""

import dace

import numpy as np

# add symbol
N = dace.symbol('N')

# add sdfg
sdfg = dace.SDFG('rtl_tasklet_pipeline')

# add state
state = sdfg.add_state()

# define compile-time constant
sdfg.specialize(dict(N=4))

# add arrays
sdfg.add_array('A', [N], dtype=dace.int32)
sdfg.add_array('B', [N], dtype=dace.int32)

# enable debugging output
sdfg.add_constant("DEBUG", 1)

# add custom cpp tasklet
tasklet = state.add_tasklet(
    name='rtl_tasklet',
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
        -->| valid_i (new data avail)    (data consumed) yumi_i |<--
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
            BUSY: if(b >= 99) state_next = DONE;
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
            READY: if(valid_i) b <= a;
            BUSY: b <= b + 1;
            DONE: b <= b;
            default: b <= b;
        endcase
    end

    /*****
    * DEBUG
    *****/
    always@(posedge clk_i)
    begin
        if(DEBUG)
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
    language=dace.Language.RTL)

# add input/output array
A = state.add_read('A')
B = state.add_write('B')

# connect input/output array with the tasklet
state.add_edge(A, None, tasklet, 'a', dace.Memlet.simple('A', '0:N-1'))
state.add_edge(tasklet, 'b', B, None, dace.Memlet.simple('B', '0:N-1'))

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
