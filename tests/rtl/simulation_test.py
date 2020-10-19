
import dace

import numpy as np


def test_tasklet_scalar():
    """
        Test the simple scalar execution sample.
    """

    # add sdfg
    sdfg = dace.SDFG('rtl_tasklet_demo')

    # add state
    state = sdfg.add_state()

    # add arrays
    sdfg.add_array('A', [1], dtype=dace.int32)
    sdfg.add_array('B', [1], dtype=dace.int32)

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
            -->| valid_i (new data avail)    (data consumed) ready_i|<--
               |----------------------------------------------------|
        */

        always@(posedge clk_i) begin
            if (rst_i) begin // case: reset
                b <= 0;
                ready_o <= 1'b1;
            end else if (valid_i) begin // case: load a 
                b <= a;
                ready_o <= 1'b0;
            end else if (b < 100) // case: increment counter b
                b <= b + 1;
            else
                b <= b; 
        end    

        assign valid_o = (b >= 100) ? 1'b1:1'b0; 
        ''',
        language=dace.Language.RTL)

    # add input/output array
    A = state.add_read('A')
    B = state.add_write('B')

    # connect input/output array with the tasklet
    state.add_edge(A, None, tasklet, 'a', dace.Memlet.simple('A', '0'))
    state.add_edge(tasklet, 'b', B, None, dace.Memlet.simple('B', '0'))

    # validate sdfg
    sdfg.validate()

    """
        Execute    
    """
    # init data structures
    a = np.random.randint(0, 100, 1).astype(np.int32)
    b = np.random.randint(0, 100, 1).astype(np.int32)

    # show initial values
    print("a={}, b={}".format(a, b))

    # call program
    sdfg(A=a, B=b)

    # show result
    print("a={}, b={}".format(a, b))

    # check result
    assert b == 100
