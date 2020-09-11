''' example rtl tasklet '''
import dace
import os

import numpy as np

# add symbol
#N = dace.symbol('N')
#BIT_WIDTH = dace.symbol('BIT_WIDTH')

# add sdfg
sdfg = dace.SDFG('rtl_tasklet_demo')

# add state
state = sdfg.add_state()

# add arrays
sdfg.add_array('A', [1], dtype=dace.int32)
sdfg.add_array('B', [1], dtype=dace.int32)

# add header includes (verilator & verilated rtl design)
sdfg.append_global_code(cpp_code='''
        // include common routines
        #include <verilated.h>
        
        #include <iostream>
        
        // include model header, generated from verilating "top.v"
        #include "Vtop.h"
        #include "Vtop.cpp"
        #include "Vtop__Slow.cpp"
        #include "Vtop__Syms.cpp"
        #include "verilated.cpp"

        // current simulation time (64-bit unsigned, cycles, global)
        vluint64_t main_time = 0;
        ''')

# add custom cpp tasklet
tasklet = state.add_tasklet(
    name='rtl_tasklet',
    inputs={'a'},
    outputs={'b'},
    code='''
        // Construct the Verilated model, from Vtop.h generated from Verilating "top.v"
        Vtop* top = new Vtop;  // Or use a const unique_ptr, or the VL_UNIQUE_PTR wrapper
    
        // apply initial input values
        top->rst_i = 0;
        top->clk_i = 0;
        top->a = a;
        top->eval();
    
        // apply reset signal and simulate
        top->rst_i = 1;
        top->clk_i = !top->clk_i;
        top->eval();
        top->clk_i = !top->clk_i;
        top->eval();
        top->rst_i = 0;
        top->eval();
        top->clk_i = !top->clk_i;
        top->eval();
    
        // simulate until $finish
        while (!Verilated::gotFinish()) {
    
            // increment time
            main_time++;
    
            // positive clock edge
            top->clk_i = !top->clk_i;
    
            // assign some other inputs
            //top->a = main_time;
    
            // evaluate model
            top->eval();
    
            // read outputs
            VL_PRINTF("[%lx] clk_i=%x rst_i=%x a=%x b=%x\\n", main_time, top->clk_i, top->rst_i, top->a, top->b);

            // negative clock edge
            top->clk_i = !top->clk_i;
            top->eval();
        }
    
        // write result
        int b_local = (int)top->b;
        std::cout << b_local << std::endl;
        b = b_local;
    
        // final model cleanup
        top->final();
    
        // destroy model
        delete top;
        top = NULL;
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

# run verilator
unique_name = "top_{}_{}_{}".format(sdfg.sdfg_id, sdfg.node_id(state), state.node_id(tasklet))
base_path = os.path.join(".dacecache", sdfg.name, "src", "rtl")
absolut_path = os.path.abspath(base_path)
code = '''
module top
  #(
    parameter  WIDTH = 32
  ) 
  ( input                  clk_i
  , input                  rst_i
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
        $finish;
      end
   end
        
endmodule
'''

# write verilog to file
import shutil

if os.path.isdir(absolut_path):
    shutil.rmtree(absolut_path)
os.mkdir(absolut_path)
with open(os.path.join(absolut_path, "top.v"), "w") as file:
    file.writelines(code)

######################################################################

if __name__ == '__main__':
    # init data structures
    a = np.random.randint(0, 100, 1).astype(np.int32)
    b = np.random.randint(0, 100, 1).astype(np.int32)

    # show initial values
    print("a={}, b={}".format(a, b))

    # call program
    sdfg(A=a, B=b)

    # show result
    print("a={}, b={}".format(a, b))
