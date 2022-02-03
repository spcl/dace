# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.

import itertools

from typing import List, Tuple, Dict

from dace import dtypes, config, registry, symbolic, nodes, sdfg, data
from dace.sdfg import graph, state, find_input_arraynode, find_output_arraynode
from dace.codegen import codeobject, dispatcher, prettycode
from dace.codegen.targets import target, framecode
from dace.codegen.targets.common import sym2cpp

from dace.external.rtllib.templates.control import generate_from_config as rtllib_control
from dace.external.rtllib.templates.package import generate_from_config as rtllib_package
from dace.external.rtllib.templates.synth import generate_from_config as rtllib_synth
from dace.external.rtllib.templates.top import generate_from_config as rtllib_top


@registry.autoregister_params(name='rtl')
class RTLCodeGen(target.TargetCodeGenerator):
    """ RTL Code Generator (SystemVerilog) """

    title = 'RTL'
    target_name = 'rtl'
    languages = [dtypes.Language.SystemVerilog]
    n_unrolled: Dict[str, int] = {}

    def __init__(self, frame_codegen: framecode.DaCeCodeGenerator, sdfg: sdfg.SDFG):
        # store reference to sdfg
        self.sdfg: sdfg.SDFG = sdfg
        # store reference to frame code generator
        self.frame: framecode.DaCeCodeGenerator = frame_codegen
        self._frame = self.frame
        # get dispatcher to register callbacks for allocation/nodes/.. code generators
        self.dispatcher: dispatcher.TargetDispatcher = frame_codegen.dispatcher
        # register node dispatcher -> generate_node(), predicate: process tasklets only
        self.dispatcher.register_node_dispatcher(
            self, lambda sdfg, state, node: isinstance(node, nodes.Tasklet) and node.language == dtypes.Language.
            SystemVerilog)
        # register all storage types that connect from/to an RTL tasklet
        for src_storage, dst_storage in itertools.product(dtypes.StorageType, dtypes.StorageType):
            self.dispatcher.register_copy_dispatcher(
                src_storage, dst_storage, None, self, lambda sdfg, dfg, src_node, dest_node:
                (isinstance(src_node, nodes.Tasklet) and src_node.language == dtypes.Language.SystemVerilog) or
                (isinstance(dest_node, nodes.Tasklet) and dest_node.language == dtypes.Language.SystemVerilog))
        # local variables
        self.verilator_debug: bool = config.Config.get_bool("compiler", "rtl", "verilator_enable_debug")
        self.code_objects: List[codeobject.CodeObject] = list()
        self.cpp_general_header_added: bool = False
        self.vendor: str = config.Config.get("compiler", "fpga", "vendor")
        self.hardware_target: bool = config.Config.get("compiler", "xilinx", "mode").startswith("hardware")
        self.frequencies: str = config.Config.get("compiler", "xilinx", "frequency")

    def generate_node(self, sdfg: sdfg.SDFG, dfg: state.StateSubgraphView, state_id: int, node: nodes.Node,
                      function_stream: prettycode.CodeIOStream, callsite_stream: prettycode.CodeIOStream):
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
                self.dispatcher.dispatch_copy(src_node, node, edge, sdfg, dfg, state_id, function_stream,
                                              callsite_stream)
            # generate code to handle data output from the tasklet
            for edge in dfg.out_edges(node):
                # find output array
                dst_node = find_output_arraynode(dfg, edge)
                # dispatch code gen (define_out_memlet)
                self.dispatcher.dispatch_output_definition(node, dst_node, edge, sdfg, dfg, state_id, function_stream,
                                                           callsite_stream)
            # generate tasklet code
            self.unparse_tasklet(sdfg, dfg, state_id, node, function_stream, callsite_stream)
        else:
            raise RuntimeError(
                "Only tasklets are handled here, not {}. This should have been filtered by the predicate".format(
                    type(node)))

    def copy_memory(self, sdfg: sdfg.SDFG, dfg: state.StateSubgraphView, state_id: int, src_node: nodes.Node,
                    dst_node: nodes.Node, edge: graph.MultiConnectorEdge, function_stream: prettycode.CodeIOStream,
                    callsite_stream: prettycode.CodeIOStream):
        """
            Generate input/output memory copies from the array references to local variables (i.e. for the tasklet code).
        """
        if isinstance(edge.src, nodes.AccessNode) and isinstance(edge.dst, nodes.Tasklet):  # handle AccessNode->Tasklet
            if isinstance(dst_node.in_connectors[edge.dst_conn], dtypes.pointer):  # pointer accessor
                line: str = "{} {} = &{}[0];".format(dst_node.in_connectors[edge.dst_conn].ctype, edge.dst_conn,
                                                     edge.src.data)
            elif isinstance(dst_node.in_connectors[edge.dst_conn], dtypes.vector):  # vector accessor
                line: str = "{} {} = *({} *)(&{}[0]);".format(dst_node.in_connectors[edge.dst_conn].ctype,
                                                              edge.dst_conn,
                                                              dst_node.in_connectors[edge.dst_conn].ctype,
                                                              edge.src.data)
            else:  # scalar accessor
                arr = sdfg.arrays[edge.data.data]
                if isinstance(arr, data.Array):
                    line: str = "{}* {} = &{}[0];".format(dst_node.in_connectors[edge.dst_conn].ctype, edge.dst_conn,
                                                          edge.src.data)
                elif isinstance(arr, data.Scalar):
                    line: str = "{} {} = {};".format(dst_node.in_connectors[edge.dst_conn].ctype, edge.dst_conn,
                                                     edge.src.data)
        elif isinstance(edge.src, nodes.MapEntry) and isinstance(edge.dst, nodes.Tasklet):
            rtl_name = self.unique_name(edge.dst, sdfg.nodes()[state_id], sdfg)
            self.n_unrolled[rtl_name] = symbolic.evaluate(edge.src.map.range[0][1] + 1, sdfg.constants)
            line: str = f'{dst_node.in_connectors[edge.dst_conn]} {edge.dst_conn} = &{edge.data.data}[{edge.src.map.params[0]}*{edge.data.volume}];'
        else:
            raise RuntimeError("Not handling copy_memory case of type {} -> {}.".format(type(edge.src), type(edge.dst)))
        # write accessor to file
        callsite_stream.write(line)

    def define_out_memlet(self, sdfg: sdfg.SDFG, dfg: state.StateSubgraphView, state_id: int, src_node: nodes.Node,
                          dst_node: nodes.Node, edge: graph.MultiConnectorEdge,
                          function_stream: prettycode.CodeIOStream, callsite_stream: prettycode.CodeIOStream):
        """
            Generate output copy code (handled within the rtl tasklet code).
        """
        if isinstance(edge.src, nodes.Tasklet) and isinstance(edge.dst, nodes.AccessNode):
            if isinstance(src_node.out_connectors[edge.src_conn], dtypes.pointer):  # pointer accessor
                line: str = "{} {} = &{}[0];".format(src_node.out_connectors[edge.src_conn].ctype, edge.src_conn,
                                                     edge.dst.data)
            elif isinstance(src_node.out_connectors[edge.src_conn], dtypes.vector):  # vector accessor
                line: str = "{}* {} = ({} *)(&{}[0]);".format(src_node.out_connectors[edge.src_conn].ctype,
                                                              edge.src_conn,
                                                              src_node.out_connectors[edge.src_conn].ctype,
                                                              edge.dst.data)
            else:  # scalar accessor
                line: str = "{}* {} = &{}[0];".format(src_node.out_connectors[edge.src_conn].ctype, edge.src_conn,
                                                      edge.dst.data)
        elif isinstance(edge.src, nodes.Tasklet) and isinstance(edge.dst, nodes.MapExit):
            line: str = f'{src_node.out_connectors[edge.src_conn].ctype} {edge.src_conn} = &{edge.data.data}[{edge.dst.map.params[0]}*{edge.data.volume}];'
        else:
            raise RuntimeError("Not handling define_out_memlet case of type {} -> {}.".format(
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
        verilator_flags = config.Config.get("compiler", "rtl", "verilator_flags")
        verilator_lint_warnings = config.Config.get_bool("compiler", "rtl", "verilator_lint_warnings")
        mode: str = config.Config.get("compiler", "rtl", "mode")

        # create options list
        options = [
            "-DDACE_RTL_VERBOSE=\"{}\"".format(verbose), "-DDACE_RTL_VERILATOR_FLAGS=\"{}\"".format(verilator_flags),
            "-DDACE_RTL_VERILATOR_LINT_WARNINGS=\"{}\"".format(verilator_lint_warnings),
            "-DDACE_RTL_MODE=\"{}\"".format(mode)
        ]
        return options

    def generate_rtl_parameters(self, constants):
        """
        Construct parameters module header
        """
        if len(constants) == 0:
            return str()
        else:
            return "#(\n{}\n)".format(" " + "\n".join([
                "{} parameter {} = {}".format("," if i > 0 else "", key, sym2cpp(constants[key]))
                for i, key in enumerate(constants)
            ]))

    def generate_padded_axis(self, is_output, name, total_size, veclen):
        """
        Generates a padded list of strings for pretty printing streaming
        AXI port definitions. E.g. for a streaming input port named "a", the
        output would be:

        , input             s_axis_a_tvalid
        , input      [31:0] s_axis_a_tdata
        , output reg        s_axis_a_tready
        , input       [3:0] s_axis_a_tkeep
        , input             s_axis_a_tlast
        """
        vec_str = '' if veclen <= 1 else f'[{veclen-1}:0]'
        bits_str = f'[{(total_size // veclen) * 8 - 1}:0]'
        bytes_str = f'[{total_size - 1}:0]'
        dir_str = 'output reg' if is_output else 'input     '
        ndir_str = 'input     ' if is_output else 'output reg'
        prefix = f'm_axis_{name}' if is_output else f's_axis_{name}'
        padding = ' ' * (len(bits_str) + len(vec_str))
        bytes_padding = ' ' * ((len(bits_str) - len(bytes_str)) + len(vec_str))
        return [
            f', {dir_str} {padding} {prefix}_tvalid',
            f', {dir_str} {vec_str}{bits_str} {prefix}_tdata',
            f', {ndir_str} {padding} {prefix}_tready',
            f', {dir_str} {bytes_padding}{bytes_str} {prefix}_tkeep',
            f', {dir_str} {padding} {prefix}_tlast',
        ]

    def generate_rtl_inputs_outputs(self, buses, scalars):
        """
        Generates all of the input and output ports for the tasklet
        """
        inputs = []
        outputs = []
        for scalar, (is_output, total_size) in scalars.items():
            inputs += [f', {"output" if is_output else "input"} [{(total_size*8)-1}:0] {scalar}']

        for bus, (_, is_output, total_size, vec_len, volume) in buses.items():
            if is_output:
                inputs += self.generate_padded_axis(True, bus, total_size, vec_len)
            else:
                outputs += self.generate_padded_axis(False, bus, total_size, vec_len)

        return inputs, outputs

    def generate_clk_from_cfg(self):
        """
        Generate the clock handling initialization expressions.
        """
        # Default case: no frequency specified
        if self.frequencies == '':
            return '1', '{ 300 }', '{ &(model->ap_aclk) }'

        freqs = self.frequencies.strip('"').split('\\|')
        nclks = len(freqs)
        clks = ['&(model->ap_aclk)'] + [f'&(model->ap_aclk_{i+2})' for i in range(nclks - 1)]
        freqs = f'{{ {", ".join([freq.split(":")[1] for freq in freqs])} }}'
        nclks = str(nclks)
        clks = f'{{ {", ".join(clks)} }}'
        return nclks, freqs, clks

    def generate_cpp_zero_inits(self, buses, scalars):
        """
        Generate zero initialization statements
        """
        valids = []
        readys = []
        for name, (arr, is_output, bytes, veclen, volume) in buses.items():
            if is_output:
                readys.append(f'model->m_axis_{name}_tready = 0;')
            else:
                valids.append(f'model->s_axis_{name}_tvalid = 0;')

        scals = [f'model->{name} = {name};' for name, _ in scalars.items()]
        return valids, readys, scals

    def generate_cpp_inputs_outputs(self, tasklet, buses):

        # generate cpp input reading/output writing code
        """
        input:
        for arrays:
            model->a = a[in_ptr_a++];
        for vectors:
            tmp = a[in_ptr_a++];
            for (int i = 0; i < WIDTH; i++) {{
                model->a[i] = tmp[i];
            }}
        for scalars:
            model->a = a[0];

        output:
        for arrays:
            b[out_ptr_b++] = (int)model->b
        for vectors:
            for(int i = 0; i < WIDTH; i++) {{
                tmp[i] = (int)model->b[i];
            }}
            b[out_ptr_b++] = tmp;
        for scalars:
            b[0] = (int)model->b;
        """
        inputs = {}
        outputs = {}

        for name, (arr, is_output, bytes, veclen, volume) in buses.items():
            if is_output:
                conn = tasklet.out_connectors[name]
                if isinstance(conn, dtypes.vector):
                    outputs[name] = f'''out_ptr_{name}++;
for (int j = 0; j < {veclen}; j++) {{
    {name}[0][j] = (int)(model->m_axis_{name}_tdata[j]);
}}'''
                elif isinstance(conn, dtypes.pointer):
                    if isinstance(conn.base_type, dtypes.vector):
                        outputs[name] = f'''int idx = out_ptr_{name}++;
for (int j = 0; j < {veclen}; j++) {{
    {name}[idx][j] = (int)(model->m_axis_{name}_tdata[j]);
}}'''
                    else:
                        outputs[name] = f'{name}[out_ptr_{name}++] = (int)(model->m_axis_{name}_tdata);'
                else:
                    outputs[name] = f'{name}[out_ptr_{name}++] = (int)(model->m_axis_{name}_tdata);'

            else:  # input
                conn = tasklet.in_connectors[name]
                if isinstance(conn, dtypes.vector):
                    inputs[name] = f'''in_ptr_{name}++;
for (int j = 0; j < {veclen}; j++) {{
    model->s_axis_{name}_tdata[j] = {name}[j];
}}'''
                elif isinstance(conn, dtypes.pointer):
                    if isinstance(conn.base_type, dtypes.vector):
                        inputs[name] = f'''int idx = in_ptr_{name}++;
for (int j = 0; j < {veclen}; j++) {{
    model->s_axis_{name}_tdata[j] = (int){name}[idx][j];
}}'''
                    else:
                        inputs[name] = f'model->s_axis_{name}_tdata = {name}[in_ptr_{name}++];'
                else:
                    inputs[name] = f'''in_ptr_{name}++;
model->s_axis_{name}_tdata = {name}[0];'''

        return inputs, outputs

    def generate_cpp_vector_init(self, tasklet):
        inits = []
        for name in tasklet.in_connectors:
            conn = tasklet.in_connectors[name]
            if isinstance(conn, dtypes.pointer):
                conn = conn.base_type
            if isinstance(conn, dtypes.vector):
                inits.append(f'''for (int j = 0; j < {conn.veclen}; j++) {{
    model->s_axis_{name}_tdata[j] = 0;
}}''')

        return "\n".join(inits)

    def generate_cpp_num_elements(self, buses):
        # TODO: compute num_elements=#elements that enter/leave the pipeline, for now we assume in_elem=out_elem (i.e. no reduction)
        return [
            f'''int num_elements_{name} = {volume};'''
            for name, (arr, is_output, bytes, veclen, volume) in buses.items()
        ]

    def generate_cpp_internal_state(self, buses):
        internal_state_strs = []
        internal_state_vars = []
        for name, (_, is_output, _, veclen, _) in buses.items():
            prefix = 'm' if is_output else 's'
            data_format = '0x%x' if veclen == 1 else f'[{" ".join(["0x%x"]*veclen)}]'
            internal_state_strs.append(f'{name}={data_format} ready=%u valid=%u')
            data_vars = f'model->{prefix}_axis_{name}_tdata' if veclen == 1 else ', '.join(
                [f'model->{prefix}_axis_{name}_tdata[{i}]' for i in range(veclen)])
            internal_state_vars.append(
                f'{data_vars}, model->{prefix}_axis_{name}_tvalid, model->{prefix}_axis_{name}_tready')
        internal_state_str = " | ".join(internal_state_strs)
        internal_state_var = ", ".join(internal_state_vars)
        return internal_state_str, internal_state_var

    def generate_input_hs(self, buses):
        """
        Generate checking whether input to the tasklet has been consumed
        """
        return [
            f'''if (model->s_axis_{name}_tready == 1 && model->s_axis_{name}_tvalid == 1) {{
                read_input_hs_{name} = true;
            }}''' for name, (arr, is_output, bytes, veclen, volume) in buses.items() if not is_output
        ]

    def generate_feeding(self, tasklet, inputs):
        """
        Generate statements for feeding into a streaming AXI bus
        """
        debug_feed_element = "std::cout << \"feed new element\" << std::endl;\n" if self.verilator_debug else ""
        return [
            f'''if (model->s_axis_{name}_tvalid == 0 && in_ptr_{name} < num_elements_{name}) {{
            {debug_feed_element}{inputs[name]}
            model->s_axis_{name}_tvalid = 1;
        }}''' for name in inputs
        ]

    def generate_ptrs(self, tasklet):
        """
        Generate pointers for the transaction counters
        """
        ins = [f'''int in_ptr_{name} = 0;''' for name in tasklet.in_connectors]
        outs = [f'''int out_ptr_{name} = 0;''' for name in tasklet.out_connectors]
        return ins, outs

    def generate_exporting(self, tasklet, outputs):
        """
        Generate statements for whether an element output by the tasklet is ready.
        """
        debug_export_element = "std::cout << \"export element\" << std::endl;\n" if self.verilator_debug else ""
        return [
            f'''if (model->m_axis_{name}_tvalid == 1) {{
            {debug_export_element}{outputs[name]}
            model->m_axis_{name}_tready = 1;
        }}''' for name in tasklet.out_connectors
        ]

    def generate_write_output_hs(self, tasklet):
        """
        Generate check for whether an element has been consumed from the output of a tasklet.
        """
        return [
            f'''if (model->m_axis_{name}_tready && model->m_axis_{name}_tvalid == 1) {{
            write_output_hs_{name} = true;
        }}''' for name in tasklet.out_connectors
        ]

    def generate_hs_flags(self, buses):
        """
        Generate flags
        """
        return [
            f'bool {"write_out" if is_output else "read_in"}put_hs_{name} = false;'
            for name, (arr, is_output, bytes, veclen, volume) in buses.items()
        ]

    def generate_input_hs_toggle(self, buses):
        """
        Generate statements for toggling input flags.
        """
        debug_read_input_hs = "\nstd::cout << \"remove read_input_hs flag\" << std::endl;" if self.verilator_debug else ""
        return [
            f'''if (read_input_hs_{name}) {{
            // remove valid flag {debug_read_input_hs}
            model->s_axis_{name}_tvalid = 0;
            read_input_hs_{name} = false;
        }}''' for name, (arr, is_output, bytes, veclen, volume) in buses.items() if not is_output
        ]

    def generate_output_hs_toggle(self, buses):
        """
        Generate statements for toggling output flags.
        """
        debug_write_output_hs = "\nstd::cout << \"remove write_output_hs flag\" << std::endl;" if self.verilator_debug else ""
        return [
            f'''if (write_output_hs_{name}) {{
            // remove ready flag {debug_write_output_hs}
            model->m_axis_{name}_tready = 0;
            write_output_hs_{name} = false;
        }}''' for name, (arr, is_output, bytes, veclen, volume) in buses.items() if is_output
        ]

    def generate_running_condition(self, tasklet):
        """
        Generate the condition for whether the simulation should be running.
        """
        # TODO should be changed with free-running kernels. Currently only
        # one element is supported. Additionally, this should not be used as
        # condition, as the amount of input and output elements might not be
        # equal to each other.
        evals = ' && '.join([f'out_ptr_{name} < num_elements_{name}' for name in tasklet.out_connectors])
        return evals

    def unique_name(self, node: nodes.RTLTasklet, state, sdfg):
        return "{}_{}_{}_{}".format(node.name, sdfg.sdfg_id, sdfg.node_id(state), state.node_id(node))

    def unparse_tasklet(self, sdfg: sdfg.SDFG, dfg: state.StateSubgraphView, state_id: int, node: nodes.Node,
                        function_stream: prettycode.CodeIOStream, callsite_stream: prettycode.CodeIOStream):

        # extract data
        state = sdfg.nodes()[state_id]
        tasklet = node

        # construct variables paths
        unique_name: str = self.unique_name(tasklet, state, sdfg)

        # Collect all of the input and output connectors into buses and scalars
        buses = {}  # {tasklet_name: (array_name, output_from_rtl, bytes, veclen)}
        scalars = {}
        for edge in state.in_edges(tasklet):
            arr = sdfg.arrays[edge.data.data]
            conn = tasklet.in_connectors[edge.dst_conn]
            if isinstance(conn, dtypes.pointer):
                conn = conn.base_type

            # catch symbolic (compile time variables)
            check_issymbolic([conn.veclen, conn.bytes], sdfg)

            # extract parameters
            vec_len = int(symbolic.evaluate(conn.veclen, sdfg.constants))
            total_size = int(symbolic.evaluate(conn.bytes, sdfg.constants))
            if isinstance(arr, data.Array):
                if self.hardware_target:
                    raise NotImplementedError('Array input for RTL hardware* targets not implemented')
                else:
                    buses[edge.dst_conn] = (edge.data.data, False, total_size, vec_len, edge.data.volume)
            elif isinstance(arr, data.Stream):
                buses[edge.dst_conn] = (edge.data.data, False, total_size, vec_len, edge.data.volume)
            elif isinstance(arr, data.Scalar):
                scalars[edge.dst_conn] = (False, total_size)

        for edge in state.out_edges(tasklet):
            arr = sdfg.arrays[edge.data.data]
            conn = tasklet.out_connectors[edge.src_conn]
            if isinstance(conn, dtypes.pointer):
                conn = conn.base_type

            # catch symbolic (compile time variables)
            check_issymbolic([conn.veclen, conn.bytes], sdfg)

            # extract parameters
            vec_len = int(symbolic.evaluate(conn.veclen, sdfg.constants))
            total_size = int(symbolic.evaluate(conn.bytes, sdfg.constants))
            if isinstance(arr, data.Array):
                if self.hardware_target:
                    raise NotImplementedError('Array input for RTL hardware* targets not implemented')
                else:
                    buses[edge.src_conn] = (edge.data.data, True, total_size, vec_len, edge.data.volume)
            elif isinstance(arr, data.Stream):
                buses[edge.src_conn] = (edge.data.data, True, total_size, vec_len, edge.data.volume)
            elif isinstance(arr, data.Scalar):
                raise NotImplementedError('Scalar output from RTL kernels not implemented')

        # generate system verilog module components
        parameter_string: str = self.generate_rtl_parameters(sdfg.constants)
        inputs, outputs = self.generate_rtl_inputs_outputs(buses, scalars)

        # create rtl code object (that is later written to file)
        self.code_objects.append(
            codeobject.CodeObject(
                name="{}".format(unique_name),
                code=RTLCodeGen.RTL_HEADER.format(
                    name=unique_name, parameters=parameter_string, inputs="\n".join(inputs), outputs="\n".join(outputs))
                + tasklet.code.code + RTLCodeGen.RTL_FOOTER,
                language="sv",
                target=RTLCodeGen,
                title="rtl",
                target_type="{}".format(unique_name),
                additional_compiler_kwargs="",
                linkable=True,
                environments=None))

        if self.hardware_target:
            if self.vendor == 'xilinx':
                nclks, _, _ = self.generate_clk_from_cfg()
                rtllib_config = {
                    "name": unique_name,
                    "buses": {
                        name: ('m_axis' if is_output else 's_axis', vec_len)
                        for name, (_, is_output, _, vec_len, _) in buses.items()
                    },
                    "params": {
                        "scalars": {
                            name: total_size * 8  # width in bits
                            for name, (_, total_size) in scalars.items()
                        },
                        "memory": {}
                    },
                    "unroll": self.n_unrolled[unique_name] if unique_name in self.n_unrolled else 1,
                    "ip_cores": tasklet.ip_cores if isinstance(tasklet, nodes.RTLTasklet) else {},
                    "clocks": int(nclks)
                }

                self.code_objects.append(
                    codeobject.CodeObject(name=f"{unique_name}_control",
                                          code=rtllib_control(rtllib_config),
                                          language="v",
                                          target=RTLCodeGen,
                                          title="rtl",
                                          target_type="{}".format(unique_name),
                                          additional_compiler_kwargs="",
                                          linkable=True,
                                          environments=None))

                self.code_objects.append(
                    codeobject.CodeObject(name=f"{unique_name}_top",
                                          code=rtllib_top(rtllib_config),
                                          language="v",
                                          target=RTLCodeGen,
                                          title="rtl",
                                          target_type="{}".format(unique_name),
                                          additional_compiler_kwargs="",
                                          linkable=True,
                                          environments=None))

                self.code_objects.append(
                    codeobject.CodeObject(name=f"{unique_name}_package",
                                          code=rtllib_package(rtllib_config),
                                          language="tcl",
                                          target=RTLCodeGen,
                                          title="rtl",
                                          target_type="scripts",
                                          additional_compiler_kwargs="",
                                          linkable=True,
                                          environments=None))

                self.code_objects.append(
                    codeobject.CodeObject(name=f"{unique_name}_synth",
                                          code=rtllib_synth(rtllib_config),
                                          language="tcl",
                                          target=RTLCodeGen,
                                          title="rtl",
                                          target_type="scripts",
                                          additional_compiler_kwargs="",
                                          linkable=True,
                                          environments=None))
            else:  # self.vendor != "xilinx"
                raise NotImplementedError('Only RTL codegen for Xilinx is implemented')
        else:  # not hardware_target
            # generate verilator simulation cpp code components
            inputs, outputs = self.generate_cpp_inputs_outputs(tasklet, buses)
            valid_zeros, ready_zeros, scalar_zeros = self.generate_cpp_zero_inits(buses, scalars)
            vector_init = self.generate_cpp_vector_init(tasklet)
            num_elements = self.generate_cpp_num_elements(buses)
            internal_state_str, internal_state_var = self.generate_cpp_internal_state(buses)
            read_input_hs = self.generate_input_hs(buses)
            feed_elements = self.generate_feeding(tasklet, inputs)
            in_ptrs, out_ptrs = self.generate_ptrs(tasklet)
            export_elements = self.generate_exporting(tasklet, outputs)
            write_output_hs = self.generate_write_output_hs(tasklet)
            hs_flags = self.generate_hs_flags(buses)
            input_hs_toggle = self.generate_input_hs_toggle(buses)
            output_hs_toggle = self.generate_output_hs_toggle(buses)
            running_condition = self.generate_running_condition(tasklet)
            nclks, freqs, clks = self.generate_clk_from_cfg()

            # add header code to stream
            if not self.cpp_general_header_added:
                sdfg.append_global_code(cpp_code=RTLCodeGen.CPP_GENERAL_HEADER_TEMPLATE.format(
                    debug_include="// generic includes\n#include <iostream>" if self.verilator_debug else ""))
                self.cpp_general_header_added = True
            sdfg.append_global_code(cpp_code=RTLCodeGen.CPP_MODEL_HEADER_TEMPLATE.format(name=unique_name))

            # add main cpp code to stream
            callsite_stream.write(contents=RTLCodeGen.CPP_MAIN_TEMPLATE.format(
                name=unique_name,
                inputs=inputs,
                outputs=outputs,
                num_elements=str.join('\n', num_elements),
                vector_init=vector_init,
                valid_zeros=str.join('\n', valid_zeros),
                ready_zeros=str.join('\n', ready_zeros),
                scalar_zeros=str.join('\n', scalar_zeros),
                read_input_hs=str.join('\n', read_input_hs),
                feed_elements=str.join('\n', feed_elements),
                in_ptrs=str.join('\n', in_ptrs),
                out_ptrs=str.join('\n', out_ptrs),
                export_elements=str.join('\n', export_elements),
                write_output_hs=str.join('\n', write_output_hs),
                hs_flags=str.join('\n', hs_flags),
                input_hs_toggle=str.join('\n', input_hs_toggle),
                output_hs_toggle=str.join('\n', output_hs_toggle),
                nclks=nclks,
                freqs=freqs,
                clks=clks,
                running_condition=running_condition,
                internal_state_str=internal_state_str,
                internal_state_var=internal_state_var,
                debug_sim_start="std::cout << \"SIM {name} START\" << std::endl;" if self.verilator_debug else "",
                debug_internal_state=f'''
// report internal state
VL_PRINTF("[t=%lu] ap_aclk=%u ap_areset=%u\\n", main_time, model->ap_aclk, model->ap_areset);
VL_PRINTF("{internal_state_str}\\n", {internal_state_var});
std::cout << std::flush;
''' if self.verilator_debug else '',
                debug_sim_end="\nstd::cout << \"SIM {name} END\" << std::endl;" if self.verilator_debug else "",
            ),
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

// instantiate model(s)
V{name}* model = new V{name};

// instantiate clock handling
int nclks = {nclks};
int freqs[nclks] = {freqs};
CData* clks[nclks] = {clks};
double periods[nclks];
for (int i = 0; i < nclks; i++) {{
    periods[i] = (1000.0 / freqs[i]);
}}
double ttf[nclks]; // time to flip
auto tick = [&]() {{
    // Lambda function for driving all of the clock signals of the model, until the first clock signal have had a full in-between-rising-edge cycle
    bool first_rised = *(clks[0]);
    while (true) {{
        int next = 0;
        for (int i = 1; i < nclks; i++) {{
            if (ttf[i] < ttf[next]) {{
                next = i;
            }}
        }}
        double time = ttf[next];
        for (int i = 0; i < nclks; i++) {{
            if (i == next) {{
                ttf[i] = periods[i] / 2;
                *(clks[i]) = !*(clks[i]);
            }} else {{
                ttf[i] -= time;
            }}
        }}
        main_time += time;
        model->eval();
        if (next == 0 && *(clks[0]) == 1) {{
            if (first_rised)
                break;
            else
                first_rised = true;
        }}
    }}
}};

// apply initial input values
model->ap_areset = 0;  // no reset
{valid_zeros}
{ready_zeros}
{scalar_zeros}

model->eval();

// Initialize vectors
{vector_init}

// reset design
model->ap_areset = 1;
tick();
model->ap_areset = 0;
tick();

// simulate until in_handshakes = out_handshakes = num_elements
{hs_flags}
{in_ptrs}
{out_ptrs}
{num_elements}

while ({running_condition}) {{
    // increment time
    main_time++;

    // feed elements
{feed_elements}
    // export elements
{export_elements}

    // check if valid and ready have been asserted at the rising clock edge -> input read handshake
{read_input_hs}
    // check if valid and ready have been asserted at the rising clock edge -> output write handshake
{write_output_hs}

    tick(); {debug_internal_state}

    // check if valid and ready has been asserted for each input at the rising clock edge
{input_hs_toggle}

    // check if valid and ready haæ been asserted for each output at the rising clock edge
{output_hs_toggle}
}} {debug_internal_state}

// final model cleanup
model->final();

// clean up resources
delete model;
model = NULL;
{debug_sim_end}"""

    RTL_HEADER = """\
module {name}
{parameters}
( input  ap_aclk     // convention: ap_aclk clocks the design, specifically the external ports
, input  ap_areset   // convention: ap_areset resets the design
, input  ap_aclk_2   // convention: ap_aclk_2 is the secondary clock, which can be used inside
, input  ap_areset_2 // convention: ap_areset_2 resets the components clocked by ap_aclk_2
, input  ap_start    // convention: ap_start indicates a start from host
, output ap_done     // convention: ap_done tells the host that the kernel has finished

{inputs}

{outputs}
);
"""

    RTL_FOOTER = '''
endmodule
'''


def check_issymbolic(iterator: iter, sdfg):
    for item in iterator:
        # catch symbolic (compile time variables)
        if symbolic.issymbolic(item, sdfg.constants):
            raise ValueError("Please use sdfg.specialize to make the following symbol(s) constant: {}".format(", ".join(
                [str(x) for x in item.free_symbols if str(x) not in sdfg.constants])))
