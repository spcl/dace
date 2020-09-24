''' example rtl tasklet '''
import dace
import numpy as np

# add symbol
N = dace.symbol('N')
BIT_WIDTH = dace.symbol('BIT_WIDTH')

# add sdfg
sdfg = dace.SDFG('rtl_tasklet_demo')

# add state
state = sdfg.add_state()

# add arrays
sdfg.add_array('A', [N], dtype=dace.int32)
sdfg.add_array('B', [N], dtype=dace.int32)

# add custom cpp tasklet
tasklet = state.add_tasklet(
    name='rtl_tasklet',
    inputs={'a'},
    outputs={'b'},
    code='''            
            module top
              #(
                parameter  WIDTH = 32
              ) 
              ( input                  clk_i  // convention: clk_i clocks the design
              , input                  rst_i  // convention: rst_i resets the design
              , input      [WIDTH-1:0] a
              , output reg [WIDTH-1:0] b
              );
            
              always@(posedge clk_i) begin
                if (rst_i)
                  b <= a;
                else
                  b <= b + 1;
              end    
            
              always@(*) begin
                  if (b >= 100) begin
                    $finish; // convention: $finish; must eventually be called
                  end
               end
                    
            endmodule
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
    n = 3
    a = np.random.randint(0, 100, n).astype(np.int32)
    b = np.random.randint(0, 100, n).astype(np.int32)

    # show initial values
    print("a={}, b={}".format(a, b))

    # call program
    sdfg(A=a, B=b, N=n, BIT_WIDTH=32)

    # show result
    print("a={}, b={}".format(a, b))
