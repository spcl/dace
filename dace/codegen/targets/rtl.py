import dace
from dace import registry
from dace.config import Config
from dace.codegen.targets.target import TargetCodeGenerator
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
    header_template = """\
                            // generic includes
                            #include <iostream>
                    
                            // verilator includes
                            #include <verilated.h>
                            
                            // include model header, generated from verilating the sv design
                            #include "V{name}.h"
                            
                            // global simulation time cycle counter
                            vluint64_t main_time = 0;
                            //long main_time = 0;\
                            """
    main_template = """
                        // instantiate model
                        V{name}* model = new V{name};
                    
                        // apply initial input values
                        model->rst_i = 0;
                        model->clk_i = 0;
                        
                        // read inputs
                        {inputs}
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
                            if(DEBUG){{
                                VL_PRINTF("[%lx] clk_i=%x rst_i=%x a=%x b=%x\\n", main_time, model->clk_i, model->rst_i, model->a, model->b);
                            }}
                            
                            // negative clock edge
                            model->clk_i = !model->clk_i;
                            model->eval();
                        }}
                    
                        // write result
                        {outputs}
                        
                        // report result
                        if(DEBUG){{
                            std::cout << b << std::endl;
                        }}
                    
                        // final model cleanup
                        model->final();
                    
                        // clean up resources
                        delete model;
                        model = NULL;
                        """
    rtl_header = """\
module {name}
{parameters}
( input                  clk_i  // convention: clk_i clocks the design
, input                  rst_i  // convention: rst_i resets the design
, input                  valid_i
, input                  ready_i
{inputs}
, output reg             ready_o
, output reg             valid_o
{outputs}
);"""
    rtl_footer = """\
  always@(*) begin
    if (valid_o)
      $finish; // convention: $finish; must eventually be called
  end
endmodule"""

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

        # construct parameters module header
        if len(sdfg.constants) == 0:
            parameter_string = ""
        else:
            parameter_string = """\
#(
{}
)""".format("\n".join(["{} parameter {} = {}".format("," if i > 0 else "", key, sdfg.constants[key]) for i, key in enumerate(sdfg.constants)]))

        # set default value for DEBUG to 'false'
        if "DEBUG" not in sdfg.constants:
            sdfg.add_constant("DEBUG", 0)

        # construct input / output module header
        MAX_PADDING = 17
        inputs = [", input{padding}[{}:0] {}".format(tasklet.in_connectors[inp].bytes*8-1,
                                                     inp,
                                                     padding=" "*(MAX_PADDING-len("[{}:0]".format(tasklet.in_connectors[inp].bytes*8-1))))
                  for inp in tasklet.in_connectors]
        MAX_PADDING = 12
        outputs = [", output reg{padding}[{}:0] {}".format(tasklet.out_connectors[inp].bytes*8-1,
                                                  inp,
                                                  padding=" "*(MAX_PADDING-len("[{}:0]".format(tasklet.out_connectors[inp].bytes*8-1))))
                   for inp in tasklet.out_connectors]

        # generate cpp input reading/output writing code
        """
        input:
        for vectors:
            for (int i = 0; i < 4; i++){{ 
                model->a[i] = a[i];
            }}
        for scalars:
            model->a = a;
            
        output:
        for vectors:
            for(int i = 0; i < 4; i++){{
                b[i] = (int)model->b[i];
            }}
        for scalars:
            b = (int)model->b;
        """
        input_read_string = "\n".join(["model->{name} = {name};".format(name=var_name)
                                       if tasklet.in_connectors[var_name].veclen == 1 else
                                       """\
                                       for(int i = 0; i < {veclen}; i++){{
                                         model->{name}[i] = {name}[i];
                                       }}\
                                       """.format(veclen=tasklet.in_connectors[var_name].veclen, name=var_name)
                                       for var_name in tasklet.in_connectors])

        output_read_string = "\n".join(["{name} = (int)model->{name};".format(name=var_name)
                                       if tasklet.out_connectors[var_name].veclen == 1 else
                                       """\
                                       for(int i = 0; i < {veclen}; i++){{
                                         {name}[i] = (int)model->{name}[i];
                                       }}\
                                       """.format(veclen=tasklet.out_connectors[var_name].veclen, name=var_name)
                                       for var_name in tasklet.out_connectors])
        # write verilog to file
        if os.path.isdir(absolut_path):
            shutil.rmtree(absolut_path)
        os.makedirs(absolut_path)
        with open(os.path.join(absolut_path, "{}.sv".format(unique_name)), "w") as file:
            file.writelines(RTLCodeGen.rtl_header.format(name=unique_name, parameters=parameter_string, inputs="\n".join(inputs), outputs="\n".join(outputs)))
            file.writelines(tasklet.code.code)
            file.writelines(RTLCodeGen.rtl_footer)

        sdfg.append_global_code(cpp_code=RTLCodeGen.header_template.format(name=unique_name))

        callsite_stream.write(contents=RTLCodeGen.main_template.format(name=unique_name, inputs=input_read_string, outputs=output_read_string),
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
