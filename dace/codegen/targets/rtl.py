import dace
from dace import registry, symbolic
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

    @staticmethod
    def cmake_options():
        """ Prepare CMake options. """
        # get flags from config
        verbose = Config.get_bool("compiler", "rtl", "verbose")
        verilator_flags = Config.get("compiler", "rtl", "verilator_flags")
        verilator_lint_warnings = Config.get_bool("compiler", "rtl", "verilator_lint_warnings")
        # create options list
        options = [
            "-DDACE_RTL_VERBOSE=\"{}\"".format(verbose),
            "-DDACE_RTL_VERILATOR_FLAGS=\"{}\"".format(verilator_flags),
            "-DDACE_RTL_VERILATOR_LINT_WARNINGS=\"{}\"".format(verilator_lint_warnings)
        ]
        return options


    # define cpp code templates
    header_template = """
                            // generic includes
                            #include <iostream>
                    
                            // verilator includes
                            #include <verilated.h>
                            
                            // include model header, generated from verilating the sv design
                            #include "V{name}.h"
                                                       
                            {debug}
                            """
    main_template = """
                        std::cout << "SIM START" << std::endl;

                        vluint64_t main_time = 0;
                        
                        // instantiate model
                        V{name}* model = new V{name};
                    
                        // apply initial input values
                        model->rst_i = 0;  // no reset
                        model->clk_i = 0; // neg clock
                        model->valid_i = 0; // not valid
                        model->ready_i = 0; // not ready 
                        model->eval();
                        
                        // read inputs
                        //{{inputs}}
                        //model->eval();
                        
                        // init vector
                        {vector_init}
                    
                        // reset design
                        model->rst_i = 1;
                        model->clk_i = 1; // rising
                        model->eval();
                        model->clk_i = 0; // falling
                        model->eval();
                        model->rst_i = 0;
                        model->clk_i = 1; // rising
                        model->eval();
                        model->clk_i = 0; // falling
                        model->eval();
                        
                        // simulate until in_handshakes = out_handshakes = num_elements
                        bool read_input_hs = false, write_output_hs = false;
                        int in_ptr = 0, out_ptr = 0;
                        {num_elements}
                        
                        while (out_ptr < num_elements) {{
                    
                            // increment time
                            main_time++;
                    
                            // check if valid_i and ready_o have been asserted at the rising clock edge -> input read handshake
                            if (model->ready_o == 1 && model->valid_i == 1){{
                                read_input_hs = true;
                            }} 
                            // feed new element
                            if(model->valid_i == 0 && in_ptr < num_elements){{
                                std::cout << "feed new element" << std::endl;
                                //model->a = a[in_ptr++];
                                {inputs}
                                model->valid_i = 1;
                            }}
                    
                            // export element
                            if(model->valid_o == 1){{
                                std::cout << "export element" << std::endl;
                                //b[out_ptr++] = model->b;
                                {outputs}
                                model->ready_i = 1;
                            }}
                            // check if valid_o and ready_i have been asserted at the rising clock edge -> output write handshake
                            if (model->ready_i == 1 && model->valid_o == 1){{
                                write_output_hs = true;
                            }}
                    
                            // positive clock edge
                            model->clk_i = !model->clk_i;
                            model->eval();
                    
                            // report internal state
                            if(DEBUG){{
                                VL_PRINTF("[t=%lu] clk_i=%u rst_i=%u valid_i=%u ready_i=%u valid_o=%u ready_o=%u \\n", main_time, model->clk_i, model->rst_i, model->valid_i, model->ready_i, model->valid_o, model->ready_o);
                                VL_PRINTF("{internal_state_str}", {internal_state_var});
                                std::cout << std::endl;
                            }}
                            
                            // check if valid_i and ready_o have been asserted at the rising clock edge
                            if (read_input_hs){{
                                // remove valid_i flag
                                std::cout << "remove read_input_hs flag" << std::endl;
                                model->valid_i = 0;
                                read_input_hs = false;
                            }}
                    
                            // check if valid_o and ready_i have been asserted at the rising clock edge
                            if (write_output_hs){{
                                // remove valid_i flag
                                std::cout << "remove write_output_hs flag" << std::endl;
                                model->ready_i = 0;
                                write_output_hs = false;
                            }}
                            
                            
                            // negative clock edge
                            model->clk_i = !model->clk_i;
                            model->eval();
                        }}
                   
                        // report internal state
                        if(DEBUG){{
                            VL_PRINTF("[t=%lu] clk_i=%u rst_i=%u valid_i=%u ready_i=%u valid_o=%u ready_o=%u \\n", main_time, model->clk_i, model->rst_i, model->valid_i, model->ready_i, model->valid_o, model->ready_o);
                            VL_PRINTF("{internal_state_str}", {internal_state_var});
                            std::cout << std::endl;
                        }}
                   
                        // write result
                        //{{outputs}}
                                            
                        // final model cleanup
                        model->final();
                    
                        // clean up resources
                        delete model;
                        model = NULL;
                        
                        std::cout << "SIM END" << std::endl;
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
    rtl_footer = """
endmodule
"""

    @staticmethod
    def unparse_rtl_tasklet(sdfg, state_id, dfg, node, function_stream,
                            callsite_stream: dace.codegen.prettycode.CodeIOStream,
                            locals, ldepth, toplevel_schedule, codegen):

        # extract data
        state = sdfg.nodes()[state_id]
        tasklet = node

        # construct paths
        unique_name = "top_{}_{}_{}".format(sdfg.sdfg_id, sdfg.node_id(state), state.node_id(tasklet))
        base_path = os.path.join(sdfg.build_folder, "src", "rtl")
        absolut_path = os.path.abspath(base_path)

        # construct parameters module header
        if len(sdfg.constants) == 0:
            parameter_string = ""
        else:
            parameter_string = """\
#(
{}
)""".format(" " + "\n".join(["{} parameter {} = {}".format("," if i > 0 else "", key, sdfg.constants[key]) for i, key in
                       enumerate(sdfg.constants)]))

        # construct input / output module header
        MAX_PADDING = 17
        inputs = list()
        for inp in tasklet.in_connectors:
            # add vector index
            idx_str = ""
            # catch symbolic (compile time variables)
            if symbolic.issymbolic(tasklet.in_connectors[inp].veclen, sdfg.constants):
                raise RuntimeError("Please use sdfg.specialize to specialize the symbol in expression: {}".format(tasklet.in_connectors[inp].veclen))
            if symbolic.issymbolic(tasklet.in_connectors[inp].bytes, sdfg.constants):
                raise RuntimeError("Please use sdfg.specialize to specialize the symbol in expression: {}".format(tasklet.in_connectors[inp].bytes))
            # extract parameters
            vec_len = int(symbolic.evaluate(tasklet.in_connectors[inp].veclen, sdfg.constants))
            total_size = int(symbolic.evaluate(tasklet.in_connectors[inp].bytes, sdfg.constants))
            # generate vector representation
            if vec_len > 1:
                idx_str = "[{}:0]".format(vec_len - 1)
            # add element index
            idx_str += "[{}:0]".format(int(total_size / vec_len) * 8 - 1)
            # generate padded string and add to list
            inputs.append(", input{padding}{idx_str} {name}".format(padding=" " * (MAX_PADDING-len(idx_str)),
                                                                    idx_str=idx_str,
                                                                    name=inp))
        MAX_PADDING = 12
        outputs = list()
        for inp in tasklet.out_connectors:
            # add vector index
            idx_str = ""
            # catch symbolic (compile time variables)
            if symbolic.issymbolic(tasklet.out_connectors[inp].veclen, sdfg.constants):
                raise RuntimeError("Please use sdfg.specialize to specialize the symbol in expression: {}".format(tasklet.in_connectors[inp].veclen))
            if symbolic.issymbolic(tasklet.out_connectors[inp].bytes, sdfg.constants):
                raise RuntimeError("Please use sdfg.specialize to specialize the symbol in expression: {}".format(tasklet.in_connectors[inp].bytes))
            # extract parameters
            vec_len = int(symbolic.evaluate(tasklet.out_connectors[inp].veclen, sdfg.constants))
            total_size = int(symbolic.evaluate(tasklet.out_connectors[inp].bytes, sdfg.constants))
            # generate vector representation
            if vec_len > 1:
                idx_str = "[{}:0]".format(vec_len - 1)
            # add element index
            idx_str += "[{}:0]".format(int(total_size / vec_len) * 8 - 1)
            # generate padded string and add to list
            outputs.append(", output reg{padding}{idx_str} {name}".format(padding=" " * (MAX_PADDING-len(idx_str)),
                                                                          idx_str=idx_str,
                                                                          name=inp))
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

        input_read_string = "\n".join(["model->{name} = {name}[in_ptr++];".format(veclen=tasklet.in_connectors[var_name].veclen, name=var_name)
                                       if isinstance(tasklet.in_connectors[var_name], dace.dtypes.pointer) else
                                       """\
                                       for(int i = 0; i < {veclen}; i++){{
                                         model->{name}[i] = {name}[i];
                                       }}\
                                       """.format(veclen=tasklet.in_connectors[var_name].veclen, name=var_name)
                                       if isinstance(tasklet.in_connectors[var_name], dace.dtypes.vector)  else
                                      "model->{name} = {name}; in_ptr++;".format(name=var_name)
                                      for var_name in tasklet.in_connectors])

        output_read_string = "\n".join(["{name}[out_ptr++] = (int)model->{name};".format(veclen=tasklet.out_connectors[var_name].veclen, name=var_name)
                                        if isinstance(tasklet.out_connectors[var_name], dace.dtypes.pointer) else
                                        """\
                                        for(int i = 0; i < {veclen}; i++){{
                                          {name}[i] = (int)model->{name}[i];
                                        }}\
                                        """.format(veclen=tasklet.out_connectors[var_name].veclen, name=var_name)
                                        if isinstance(tasklet.out_connectors[var_name], dace.dtypes.vector) else
                                        "{name} = (int)model->{name}; out_ptr++;".format(name=var_name)
                                        for var_name in tasklet.out_connectors])

        init_vector_string = "\n".join(["""\
                                       for(int i = 0; i < {veclen}; i++){{
                                         model->{name}[i] = 0;
                                       }}\
                                       """.format(veclen=tasklet.in_connectors[var_name].veclen, name=var_name)
                                        if isinstance(tasklet.in_connectors[var_name], dace.dtypes.vector) else ""
                                        for var_name in tasklet.in_connectors])

        # write verilog to file
        os.makedirs(absolut_path, exist_ok=True)
        with open(os.path.join(absolut_path, "{}.sv".format(unique_name)), "w") as file:
            file.writelines(
                RTLCodeGen.rtl_header.format(name=unique_name, parameters=parameter_string, inputs="\n".join(inputs),
                                             outputs="\n".join(outputs)))
            file.writelines(tasklet.code.code)
            file.writelines(RTLCodeGen.rtl_footer)

        # compute num_elements=#elements that enter/leave the pipeline, for now we assume in_elem=out_elem (i.e. no reduction)
        num_elements_string = "int num_elements = {};".format(1)

        sdfg.append_global_code(cpp_code=RTLCodeGen.header_template.format(name=unique_name,
                                                                           debug="// enable/disable debug log\n" +
                                                                                 "bool DEBUG = false;" if "DEBUG" not in sdfg.constants else ""))
        #dace.config.Config.get()
        callsite_stream.write(contents=RTLCodeGen.main_template.format(name=unique_name,
                                                                       inputs=input_read_string,
                                                                       outputs=output_read_string,
                                                                       num_elements=num_elements_string,
                                                                       vector_init=init_vector_string,
                                                                       internal_state_str=" ".join(
                                                                           ["{}=0x%x".format(var_name) for var_name in
                                                                            {**tasklet.in_connectors,
                                                                             **tasklet.out_connectors}]),
                                                                       internal_state_var=", ".join(
                                                                           ["model->{}".format(var_name) for var_name in
                                                                            {**tasklet.in_connectors,
                                                                             **tasklet.out_connectors}])),
                              sdfg=sdfg,
                              state_id=state_id,
                              node_id=node)

