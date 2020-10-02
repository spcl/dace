import dace
from dace import registry
from dace.config import Config
from dace.codegen.targets.target import TargetCodeGenerator
from dace.codegen.targets.cpp import unparse_tasklet
import dace.codegen.prettycode

import os
import shutil


# class RTLCodeGen(CPUCodeGen):
@registry.autoregister_params(name='rtl')
class RTLCodeGen(TargetCodeGenerator):
    """ RTL (SystemVerilog) Code Generator """

    title = 'RTL'
    target_name = 'rtl'
    language = 'rtl'

    def __init__(self, frame_codegen, sdfg, *args, **kwargs):
        self._frame = frame_codegen
        self._dispatcher = frame_codegen.dispatcher
        # register dispatchers
        self.codegen = self._dispatcher.get_generic_node_dispatcher()
        self.sdfg = sdfg


    # define cpp code templates
    header_template = """
                            // generic includes
                            #include <iostream>
                    
                            // verilator includes
                            #include <verilated.h>
                            
                            // include model header, generated from verilating the sv design
                            #include "V{name}.h"
                            
                            // global simulation time cycle counter
                            vluint64_t main_time = 0;
                            //long main_time = 0;
                            
                            // set debug level
                            bool DEBUG = {debug};
                            """
    main_template = """
                        // instantiate model
                        V{name}* model = new V{name};
                    
                        // apply initial input values
                        model->rst_i = 0;
                        model->clk_i = 0;
                        model->a = a; // TODO: make generic for all types of inputs (and multiple)
                        model->eval();
                    
                        // reset design
                        model->rst_i = 1;
                        model->clk_i = !model->clk_i;
                        model->eval();
                        model->clk_i = !model->clk_i;
                        model->eval();
                        model->rst_i = 0;
                        model->eval();
                        model->clk_i = !model->clk_i;
                        model->eval();
                    
                        // simulate until $finish
                        while (!Verilated::gotFinish()) {{
                    
                            // increment time
                            main_time++;
                    
                            // positive clock edge
                            model->clk_i = !model->clk_i;
                            model->eval();
                    
                            // report internal state
                            if(false){{
                                VL_PRINTF("[%lx] clk_i=%x rst_i=%x a=%x b=%x\\n", main_time, model->clk_i, model->rst_i, model->a, model->b);
                            }}
                            
                            // negative clock edge
                            model->clk_i = !model->clk_i;
                            model->eval();
                        }}
                    
                        // write result TODO: make generic for all type of outputs
                        int b_local = (int)model->b;
                        b = b_local;

                        // report result
                        if(DEBUG){{
                            std::cout << b_local << std::endl;
                        }}
                    
                        // final model cleanup
                        model->final();
                    
                        // clean up resources
                        delete model;
                        model = NULL;
                        """
    rtl_header = """
module top
  #(
  parameter  WIDTH = 32
  ) 
  ( input                  clk_i  // convention: clk_i clocks the design
  , input                  rst_i  // convention: rst_i resets the design
  {inputs}
  {outputs}
  );

  logic valid_o, ready_o, valid_i, yumi_i;
    """

    rtl_footer = """
  always@(*) begin
    if (valid_o)
      $finish; // convention: $finish; must eventually be called
  end
endmodule
    """

    @staticmethod
    def unparse_rtl_tasklet(sdfg, state_id, dfg, node, function_stream, callsite_stream: dace.codegen.prettycode.CodeIOStream,
                    locals, ldepth, toplevel_schedule, codegen):

        # extract data
        state = sdfg.nodes()[state_id]
        tasklet = node

        # construct paths
        unique_name = "top_{}_{}_{}".format(sdfg.sdfg_id, sdfg.node_id(state), state.node_id(tasklet))
        base_path = os.path.join(".dacecache", sdfg.name, "src", "rtl")
        absolut_path = os.path.abspath(base_path)

        # construct input / output module header
        inputs = [", input [{}:0] {}".format(tasklet.in_connectors[inp].bytes*8-1, inp) for inp in tasklet.in_connectors]
        outputs = [", output reg [{}:0] {}".format(tasklet.out_connectors[inp].bytes*8-1, inp) for inp in tasklet.out_connectors]


        # write verilog to file
        if os.path.isdir(absolut_path):
            shutil.rmtree(absolut_path)
        os.makedirs(absolut_path)
        with open(os.path.join(absolut_path, "{}.sv".format(unique_name)), "w") as file:
            file.writelines(RTLCodeGen.rtl_header.format(inputs="\n".join(inputs), outputs="\n".join(outputs)))
            file.writelines(tasklet.code.code)
            file.writelines(RTLCodeGen.rtl_footer)

        sdfg.append_global_code(cpp_code=RTLCodeGen.header_template.format(name=unique_name,
                                                                           debug="true"))

        callsite_stream.write(contents=RTLCodeGen.main_template.format(name=unique_name),
                              sdfg=sdfg,
                              state_id=state_id,
                              node_id=node)


    @staticmethod
    def cmake_options():
        """ Prepare CMake options. """
        # get flags from config
        verilator_flags = Config.get("compiler", "rtl", "verilator_flags")  # COVERAGE, TRACE
        # create options list
        options = [
            "-DDACE_RTL_VERILATOR_FLAGS=\"{}\"".format(verilator_flags)
        ]
        return options
