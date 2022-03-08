# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
"""
    Simple RTL tasklet with a single scalar input and a single scalar output. It increments b from a up to 100.
"""

import dace
import argparse

import numpy as np

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

    typedef enum logic [1:0] {READY, BUSY, DONE} state_e;
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
        end else if (m_axis_b_tdata < 100) // case: increment counter b
            m_axis_b_tdata <= m_axis_b_tdata + 1;
        else begin
            m_axis_b_tdata <= m_axis_b_tdata;
            state <= DONE;
        end
    end

    assign m_axis_b_tvalid = (m_axis_b_tdata >= 100) ? 1'b1:1'b0;
    ''',
                            language=dace.Language.SystemVerilog)

# add input/output array
A = state.add_read('A')
B = state.add_write('B')

# connect input/output array with the tasklet
state.add_memlet_path(A, tasklet, dst_conn='a', memlet=dace.Memlet('A[0]'))
state.add_memlet_path(tasklet, B, src_conn='b', memlet=dace.Memlet('B[0]'))

# validate sdfg
sdfg.validate()

######################################################################

if __name__ == '__main__':

    # init data structures
    a = np.random.randint(0, 100, 1).astype(np.int32)
    b = np.array([0]).astype(np.int32)

    # show initial values
    print("a={}, b={}".format(a, b))

    # call program
    sdfg(A=a, B=b)

    # show result
    print("a={}, b={}".format(a, b))

    # check result
    assert b == 100
