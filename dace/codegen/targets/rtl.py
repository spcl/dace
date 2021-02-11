# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.

import itertools

from typing import List, Tuple, Dict

from dace import dtypes, config, registry, symbolic, nodes, sdfg
from dace.sdfg import graph, state, find_input_arraynode, find_output_arraynode
from dace.codegen import codeobject, dispatcher, prettycode
from dace.codegen.targets import target, framecode
from dace.codegen.targets.common import sym2cpp


@registry.autoregister_params(name='rtl')
class RTLCodeGen(target.TargetCodeGenerator):
    """ RTL Code Generator (SystemVerilog) """

    title = 'RTL'
    target_name = 'rtl'
    languages = [dtypes.Language.SystemVerilog]

    def __init__(self, frame_codegen: framecode.DaCeCodeGenerator,
                 sdfg: sdfg.SDFG):
        # store reference to sdfg
        self.sdfg: sdfg.SDFG = sdfg
        # store reference to frame code generator
        self.frame: framecode.DaCeCodeGenerator = frame_codegen
        # get dispatcher to register callbacks for allocation/nodes/.. code generators
        self.dispatcher: dispatcher.TargetDispatcher = frame_codegen.dispatcher
        # register node dispatcher -> generate_node(), predicate: process tasklets only
        self.dispatcher.register_node_dispatcher(
            self, lambda sdfg, node: isinstance(node, nodes.Tasklet) and node.
            language == dtypes.Language.SystemVerilog)
        # register all storage types that connect from/to an RTL tasklet
        for src_storage, dst_storage in itertools.product(
                dtypes.StorageType, dtypes.StorageType):
            self.dispatcher.register_copy_dispatcher(
                src_storage, dst_storage, None, self,
                lambda sdfg, dfg, src_node, dest_node:
                (isinstance(src_node, nodes.Tasklet) and src_node.language ==
                 dtypes.Language.SystemVerilog) or
                (isinstance(dest_node, nodes.Tasklet) and dest_node.language ==
                 dtypes.Language.SystemVerilog))
        # local variables
        self.verilator_debug: bool = config.Config.get_bool(
            "compiler", "rtl", "verilator_enable_debug")
        self.code_objects: List[codeobject.CodeObject] = list()
        self.cpp_general_header_added: bool = False

    def generate_node(self, sdfg: sdfg.SDFG, dfg: state.StateSubgraphView,
                      state_id: int, node: nodes.Node,
                      function_stream: prettycode.CodeIOStream,
                      callsite_stream: prettycode.CodeIOStream):
        # check instance type
        if isinstance(node, nodes.Tasklet):
            """ 
            handle Tasklet: 
                (1) generate in->tasklet
                (2) generate tasklet->out 
                (3) generate tasklet 
            """
            # generate code to handle data input to the tasklet
            for edge in dfg.in_edges(node):
                # find input array
                src_node = find_input_arraynode(dfg, edge)
                # dispatch code gen (copy_memory)
                self.dispatcher.dispatch_copy(src_node, node, edge, sdfg, dfg,
                                              state_id, function_stream,
                                              callsite_stream)
            # generate code to handle data output from the tasklet
            for edge in dfg.out_edges(node):
                # find output array
                dst_node = find_output_arraynode(dfg, edge)
                # dispatch code gen (define_out_memlet)
                self.dispatcher.dispatch_output_definition(
                    node, dst_node, edge, sdfg, dfg, state_id, function_stream,
                    callsite_stream)
            # generate tasklet code
            self.unparse_tasklet(sdfg, dfg, state_id, node, function_stream,
                                 callsite_stream)
        else:
            raise RuntimeError(
                "Only tasklets are handled here, not {}. This should have been filtered by the predicate"
                .format(type(node)))

    def copy_memory(self, sdfg: sdfg.SDFG, dfg: state.StateSubgraphView,
                    state_id: int, src_node: nodes.Node, dst_node: nodes.Node,
                    edge: graph.MultiConnectorEdge,
                    function_stream: prettycode.CodeIOStream,
                    callsite_stream: prettycode.CodeIOStream):
        """
            Generate input/output memory copies from the array references to local variables (i.e. for the tasklet code).
        """
        if isinstance(edge.src, nodes.AccessNode) and isinstance(
                edge.dst, nodes.Tasklet):  # handle AccessNode->Tasklet
            if isinstance(dst_node.in_connectors[edge.dst_conn],
                          dtypes.pointer):  # pointer accessor
                line: str = "{} {} = &{}[0];".format(
                    dst_node.in_connectors[edge.dst_conn].ctype, edge.dst_conn,
                    edge.src.data)
            elif isinstance(dst_node.in_connectors[edge.dst_conn],
                            dtypes.vector):  # vector accessor
                line: str = "{} {} = *({} *)(&{}[0]);".format(
                    dst_node.in_connectors[edge.dst_conn].ctype, edge.dst_conn,
                    dst_node.in_connectors[edge.dst_conn].ctype, edge.src.data)
            else:  # scalar accessor
                line: str = "{}* {} = &{}[0];".format(
                    dst_node.in_connectors[edge.dst_conn].ctype, edge.dst_conn,
                    edge.src.data)
        else:
            raise RuntimeError(
                "Not handling copy_memory case of type {} -> {}.".format(
                    type(edge.src), type(edge.dst)))
        # write accessor to file
        callsite_stream.write(line)

    def define_out_memlet(self, sdfg: sdfg.SDFG, dfg: state.StateSubgraphView,
                          state_id: int, src_node: nodes.Node,
                          dst_node: nodes.Node, edge: graph.MultiConnectorEdge,
                          function_stream: prettycode.CodeIOStream,
                          callsite_stream: prettycode.CodeIOStream):
        """
            Generate output copy code (handled within the rtl tasklet code).
        """
        if isinstance(edge.src, nodes.Tasklet) and isinstance(
                edge.dst, nodes.AccessNode):
            if isinstance(src_node.out_connectors[edge.src_conn],
                          dtypes.pointer):  # pointer accessor
                line: str = "{} {} = &{}[0];".format(
                    src_node.out_connectors[edge.src_conn].ctype, edge.src_conn,
                    edge.dst.data)
            elif isinstance(src_node.out_connectors[edge.src_conn],
                            dtypes.vector):  # vector accessor
                line: str = "{} {} = *({} *)(&{}[0]);".format(
                    src_node.out_connectors[edge.src_conn].ctype, edge.src_conn,
                    src_node.out_connectors[edge.src_conn].ctype, edge.dst.data)
            else:  # scalar accessor
                line: str = "{}* {} = &{}[0];".format(
                    src_node.out_connectors[edge.src_conn].ctype, edge.src_conn,
                    edge.dst.data)
        else:
            raise RuntimeError(
                "Not handling define_out_memlet case of type {} -> {}.".format(
                    type(edge.src), type(edge.dst)))
        # write accessor to file
        callsite_stream.write(line)

    def get_generated_codeobjects(self):
        """
            Return list of code objects (that are later generating code files).
        """
        return self.code_objects

    @property
    def has_initializer(self):
        """
            Disable initializer method generation.
        """
        return False

    @property
    def has_finalizer(self):
        """
            Disable exit/finalizer method generation.
        """
        return False

    @staticmethod
    def cmake_options():
        """
            Process variables to be exposed to the CMakeList.txt script.
        """
        # get flags from config
        verbose = config.Config.get_bool("compiler", "rtl", "verbose")
        verilator_flags = config.Config.get("compiler", "rtl",
                                            "verilator_flags")
        verilator_lint_warnings = config.Config.get_bool(
            "compiler", "rtl", "verilator_lint_warnings")
        # create options list
        options = [
            "-DDACE_RTL_VERBOSE=\"{}\"".format(verbose),
            "-DDACE_RTL_VERILATOR_FLAGS=\"{}\"".format(verilator_flags),
            "-DDACE_RTL_VERILATOR_LINT_WARNINGS=\"{}\"".format(
                verilator_lint_warnings)
        ]
        return options

    def generate_rtl_parameters(self, constants):
        # construct parameters module header
        if len(constants) == 0:
            return str()
        else:
            return "#(\n{}\n)".format(" " + "\n".join([
                "{} parameter {} = {}".format("," if i > 0 else "", key,
                                              sym2cpp(constants[key]))
                for i, key in enumerate(constants)
            ]))

    def generate_rtl_inputs_outputs(self, sdfg, tasklet):
        # construct input / output module header
        inputs = list()
        for inp in tasklet.in_connectors:
            # add vector index
            idx_str = ""
            # catch symbolic (compile time variables)
            check_issymbolic([
                tasklet.in_connectors[inp].veclen,
                tasklet.in_connectors[inp].bytes
            ], sdfg)
            # extract parameters
            vec_len = int(
                symbolic.evaluate(tasklet.in_connectors[inp].veclen,
                                  sdfg.constants))
            total_size = int(
                symbolic.evaluate(tasklet.in_connectors[inp].bytes,
                                  sdfg.constants))
            # generate vector representation
            if vec_len > 1:
                idx_str = "[{}:0]".format(vec_len - 1)
            # add element index
            idx_str += "[{}:0]".format(int(total_size / vec_len) * 8 - 1)
            # generate padded string and add to list
            inputs.append(", input{padding}{idx_str} {name}".format(
                padding=" " * (17 - len(idx_str)), idx_str=idx_str, name=inp))
        outputs = list()
        for inp in tasklet.out_connectors:
            # add vector index
            idx_str = ""
            # catch symbolic (compile time variables)
            check_issymbolic([
                tasklet.out_connectors[inp].veclen,
                tasklet.out_connectors[inp].bytes
            ], sdfg)
            # extract parameters
            vec_len = int(
                symbolic.evaluate(tasklet.out_connectors[inp].veclen,
                                  sdfg.constants))
            total_size = int(
                symbolic.evaluate(tasklet.out_connectors[inp].bytes,
                                  sdfg.constants))
            # generate vector representation
            if vec_len > 1:
                idx_str = "[{}:0]".format(vec_len - 1)
            # add element index
            idx_str += "[{}:0]".format(int(total_size / vec_len) * 8 - 1)
            # generate padded string and add to list
            outputs.append(", output reg{padding}{idx_str} {name}".format(
                padding=" " * (12 - len(idx_str)), idx_str=idx_str, name=inp))
        return inputs, outputs

    def generate_cpp_inputs_outputs(self, tasklet):

        # generate cpp input reading/output writing code
        """
        input:
        for vectors:
            for (int i = 0; i < WIDTH; i++){{
                model->a[i] = a[i];
            }}
        for scalars:
            model->a = a[0];

        output:
        for vectors:
            for(int i = 0; i < WIDTH; i++){{
                b[i] = (int)model->b[i];
            }}
        for scalars:
            b[0] = (int)model->b;
        """
        input_read_string = "\n".join([
            "model->{name} = {name}[in_ptr++];".format(
                name=var_name) if isinstance(tasklet.in_connectors[var_name],
                                             dtypes.pointer) else """\
 for(int i = 0; i < {veclen}; i++){{
   model->{name}[i] = {name}[i];
 }}\
 """.format(veclen=tasklet.in_connectors[var_name].veclen, name=var_name)
            if isinstance(tasklet.in_connectors[var_name], dtypes.vector) else
            "model->{name} = {name}[in_ptr++];".format(name=var_name)
            for var_name in tasklet.in_connectors
        ])

        output_read_string = "\n".join([
            "{name}[out_ptr++] = (int)model->{name};".format(
                name=var_name) if isinstance(tasklet.out_connectors[var_name],
                                             dtypes.pointer) else """\
for(int i = 0; i < {veclen}; i++){{
  {name}[i] = (int)model->{name}[i];
}}\
""".format(veclen=tasklet.out_connectors[var_name].veclen, name=var_name)
            if isinstance(tasklet.out_connectors[var_name], dtypes.vector) else
            "{name}[out_ptr++] = (int)model->{name};".format(name=var_name)
            for var_name in tasklet.out_connectors
        ])
        # return generated strings
        return input_read_string, output_read_string

    def generate_cpp_vector_init(self, tasklet):
        init_vector_string = "\n".join([
            """\
        for(int i = 0; i < {veclen}; i++){{
         model->{name}[i] = 0;
        }}\
        """.format(veclen=tasklet.in_connectors[var_name].veclen, name=var_name)
            if isinstance(tasklet.in_connectors[var_name], dtypes.vector) else
            "" for var_name in tasklet.in_connectors
        ])
        return "// initialize vector\n" if len(
            init_vector_string) > 0 else "" + init_vector_string

    def generate_cpp_num_elements(self):
        # TODO: compute num_elements=#elements that enter/leave the pipeline, for now we assume in_elem=out_elem (i.e. no reduction)
        return "int num_elements = {};".format(1)

    def generate_cpp_internal_state(self, tasklet):
        internal_state_str = " ".join([
            "{}=0x%x".format(var_name) for var_name in {
                **tasklet.in_connectors,
                **tasklet.out_connectors
            }
        ])
        internal_state_var = ", ".join([
            "model->{}".format(var_name) for var_name in {
                **tasklet.in_connectors,
                **tasklet.out_connectors
            }
        ])
        return internal_state_str, internal_state_var

    def unparse_tasklet(self, sdfg: sdfg.SDFG, dfg: state.StateSubgraphView,
                        state_id: int, node: nodes.Node,
                        function_stream: prettycode.CodeIOStream,
                        callsite_stream: prettycode.CodeIOStream):

        # extract data
        state = sdfg.nodes()[state_id]
        tasklet = node

        # construct variables paths
        unique_name: str = "top_{}_{}_{}".format(sdfg.sdfg_id,
                                                 sdfg.node_id(state),
                                                 state.node_id(tasklet))

        # generate system verilog module components
        parameter_string: str = self.generate_rtl_parameters(sdfg.constants)
        inputs, outputs = self.generate_rtl_inputs_outputs(sdfg, tasklet)

        # create rtl code object (that is later written to file)
        self.code_objects.append(
            codeobject.CodeObject(
                name="{}".format(unique_name),
                code=RTLCodeGen.RTL_HEADER.format(name=unique_name,
                                                  parameters=parameter_string,
                                                  inputs="\n".join(inputs),
                                                  outputs="\n".join(outputs)) +
                tasklet.code.code + RTLCodeGen.RTL_FOOTER,
                language="sv",
                target=RTLCodeGen,
                title="rtl",
                target_type="",
                additional_compiler_kwargs="",
                linkable=True,
                environments=None))

        # generate verilator simulation cpp code components
        inputs, outputs = self.generate_cpp_inputs_outputs(tasklet)
        vector_init = self.generate_cpp_vector_init(tasklet)
        num_elements = self.generate_cpp_num_elements()
        internal_state_str, internal_state_var = self.generate_cpp_internal_state(
            tasklet)

        # add header code to stream
        if not self.cpp_general_header_added:
            sdfg.append_global_code(
                cpp_code=RTLCodeGen.CPP_GENERAL_HEADER_TEMPLATE.format(
                    debug_include="// generic includes\n#include <iostream>"
                    if self.verilator_debug else ""))
            self.cpp_general_header_added = True
        sdfg.append_global_code(
            cpp_code=RTLCodeGen.CPP_MODEL_HEADER_TEMPLATE.format(
                name=unique_name))

        # add main cpp code to stream
        callsite_stream.write(contents=RTLCodeGen.CPP_MAIN_TEMPLATE.format(
            name=unique_name,
            inputs=inputs,
            outputs=outputs,
            num_elements=num_elements,
            vector_init=vector_init,
            internal_state_str=internal_state_str,
            internal_state_var=internal_state_var,
            debug_sim_start="std::cout << \"SIM {name} START\" << std::endl;"
            if self.verilator_debug else "",
            debug_feed_element="std::cout << \"feed new element\" << std::endl;"
            if self.verilator_debug else "",
            debug_export_element="std::cout << \"export element\" << std::endl;"
            if self.verilator_debug else "",
            debug_internal_state="""
// report internal state 
VL_PRINTF("[t=%lu] clk_i=%u rst_i=%u valid_i=%u ready_i=%u valid_o=%u ready_o=%u \\n", main_time, model->clk_i, model->rst_i, model->valid_i, model->ready_i, model->valid_o, model->ready_o);
VL_PRINTF("{internal_state_str}\\n", {internal_state_var});
std::cout << std::flush;
""".format(internal_state_str=internal_state_str,
           internal_state_var=internal_state_var)
            if self.verilator_debug else "",
            debug_read_input_hs=
            "std::cout << \"remove read_input_hs flag\" << std::endl;"
            if self.verilator_debug else "",
            debug_output_hs=
            "std::cout << \"remove write_output_hs flag\" << std::endl;"
            if self.verilator_debug else "",
            debug_sim_end="std::cout << \"SIM {name} END\" << std::endl;"
            if self.verilator_debug else ""),
                              sdfg=sdfg,
                              state_id=state_id,
                              node_id=node)

    CPP_GENERAL_HEADER_TEMPLATE = """\
{debug_include}
// verilator includes
#include <verilated.h>
"""

    CPP_MODEL_HEADER_TEMPLATE = """\
// include model header, generated from verilating the sv design
#include "V{name}.h"
"""

    CPP_MAIN_TEMPLATE = """\
{debug_sim_start}

vluint64_t main_time = 0;

// instantiate model
V{name}* model = new V{name};

// apply initial input values
model->rst_i = 0;  // no reset
model->clk_i = 0; // neg clock
model->valid_i = 0; // not valid
model->ready_i = 0; // not ready 
model->eval();

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
        {debug_feed_element}
        {inputs}
        model->valid_i = 1;
    }}

    // export element
    if(model->valid_o == 1){{
        {debug_export_element}
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

    {debug_internal_state}

    // check if valid_i and ready_o have been asserted at the rising clock edge
    if (read_input_hs){{
        // remove valid_i flag
        model->valid_i = 0;
        {debug_read_input_hs}
        read_input_hs = false;
    }}

    // check if valid_o and ready_i have been asserted at the rising clock edge
    if (write_output_hs){{
        // remove ready_i flag
        model->ready_i = 0;
        {debug_output_hs}
        write_output_hs = false;
    }}

    // negative clock edge
    model->clk_i = !model->clk_i;
    model->eval();
}}

{debug_internal_state}

// final model cleanup
model->final();

// clean up resources
delete model;
model = NULL;

{debug_sim_end}
"""

    RTL_HEADER = """\
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
);
"""

    RTL_FOOTER = """\
endmodule
"""


def check_issymbolic(iterator: iter, sdfg):
    for item in iterator:
        # catch symbolic (compile time variables)
        if symbolic.issymbolic(item, sdfg.constants):
            raise ValueError(
                "Please use sdfg.specialize to make the following symbol(s) constant: {}"
                .format(", ".join([
                    str(x) for x in item.free_symbols
                    if str(x) not in sdfg.constants
                ])))
