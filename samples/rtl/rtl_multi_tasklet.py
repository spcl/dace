# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
"""
    Two sequential RTL tasklets connected through a memlet.
"""

import dace
import argparse

import numpy as np

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
                             code="""\
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
""",
                             language=dace.Language.SystemVerilog)

tasklet1 = state.add_tasklet(name='rtl_tasklet1',
                             inputs={'b'},
                             outputs={'c'},
                             code="""\
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
""",
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

######################################################################

if __name__ == '__main__':

    # init data structures
    a = np.random.randint(0, 80, 1).astype(np.int32)
    b = np.array([0]).astype(np.int32)
    c = np.array([0]).astype(np.int32)

    # show initial values
    print("a={}, b={}, c={}".format(a, b, c))

    # call program
    sdfg(A=a, B=b, C=c)

    # show result
    print("a={}, b={}, c={}".format(a, b, c))

    # check result
    assert b == 80
    assert c == 100
