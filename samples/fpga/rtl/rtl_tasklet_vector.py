# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
"""
    RTL tasklet with a vector input of 4 int32 (width=128bits) and a single scalar output. It increments b from a[31:0] up to 100.
"""

import dace
import argparse

import numpy as np

# add symbol
WIDTH = dace.symbol('WIDTH')

# add sdfg
sdfg = dace.SDFG('rtl_tasklet_vector')

# define compile-time constant
sdfg.specialize(dict(WIDTH=4))

# add state
state = sdfg.add_state()

# add arrays
sdfg.add_array('A', [WIDTH], dtype=dace.int32)
sdfg.add_array('B', [1], dtype=dace.int32)

# add custom cpp tasklet
tasklet = state.add_tasklet(name='rtl_tasklet',
                            inputs={'a': dace.vector(dace.int32, WIDTH)},
                            outputs={'b'},
                            code='''
        /*
            Convention:
               |--------------------------------------------------------------------|
            -->| ap_aclk (clock input)                                              |
            -->| ap_areset (reset input, rst on high)                               |
               |                                                                    |
            -->| {inputs}                                             reg {outputs} |-->
               |                                                                    |
            <--| s_axis_a_tready (ready for data)       (data avail) m_axis_b_tvalid|-->
            -->| s_axis_a_tvalid (new data avail)    (data consumed) m_axis_b_tready|<--
               |--------------------------------------------------------------------|
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
    
        assign m_axis_b_tvalid = (m_axis_b_tdata >= s_axis_a_tdata[0] + s_axis_a_tdata[1] && (state == BUSY || state == DONE)) ? 1'b1:1'b0; 
    ''',
                            language=dace.Language.SystemVerilog)

# add input/output array
A = state.add_read('A')
B = state.add_write('B')

# connect input/output array with the tasklet
state.add_edge(A, None, tasklet, 'a', dace.Memlet('A[0:WIDTH-1]'))
state.add_edge(tasklet, 'b', B, None, dace.Memlet('B[0]'))

# validate sdfg
sdfg.validate()

######################################################################

if __name__ == '__main__':

    # init data structures
    a = np.random.randint(0, 100, dace.symbolic.evaluate(WIDTH, sdfg.constants)).astype(np.int32)
    b = np.array([0]).astype(np.int32)

    # show initial values
    print("a={}, b={}".format(a, b))

    # call program
    sdfg(A=a, B=b)

    # show result
    print("a={}, b={}".format(a, b))

    # check result
    assert b == a[0] + a[1]
