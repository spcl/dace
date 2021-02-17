# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
from typing import Optional, Set, Tuple

import collections
import copy
import dace
import functools
import re
from dace.codegen import control_flow as cflow
from dace.codegen import dispatcher as disp
from dace.codegen.prettycode import CodeIOStream
from dace.codegen.targets.common import codeblock_to_cpp, sym2cpp
from dace.codegen.targets.cpp import unparse_interstate_edge
from dace.codegen.targets.target import TargetCodeGenerator
from dace.sdfg import SDFG, SDFGState, ScopeSubgraphView
from dace.sdfg import nodes
from dace.sdfg.infer_types import set_default_schedule_and_storage_types
from dace import dtypes, data, config
from typing import Any, List

from dace.frontend.python import wrappers

import networkx as nx
import numpy as np


class DaCeCodeGenerator(object):
    """ DaCe code generator class that writes the generated code for SDFG
        state machines, and uses a dispatcher to generate code for
        individual states based on the target. """
    def __init__(self, *args, **kwargs):
        self._dispatcher = disp.TargetDispatcher(self)
        self._dispatcher.register_state_dispatcher(self)
        self._initcode = CodeIOStream()
        self._exitcode = CodeIOStream()
        self.statestruct: List[str] = []
        self.environments: List[Any] = []

    ##################################################################
    # Target registry

    @property
    def dispatcher(self):
        return self._dispatcher

    ##################################################################
    # Code generation

    def generate_constants(self, sdfg: SDFG, callsite_stream: CodeIOStream):
        # Write constants
        for cstname, (csttype, cstval) in sdfg.constants_prop.items():
            if isinstance(csttype, data.Array):
                const_str = "constexpr " + csttype.dtype.ctype + \
                    " " + cstname + "[" + str(cstval.size) + "] = {"
                it = np.nditer(cstval, order='C')
                for i in range(cstval.size - 1):
                    const_str += str(it[0]) + ", "
                    it.iternext()
                const_str += str(it[0]) + "};\n"
                callsite_stream.write(const_str, sdfg)
            else:
                callsite_stream.write(
                    "constexpr %s %s = %s;\n" %
                    (csttype.dtype.ctype, cstname, sym2cpp(cstval)), sdfg)

    def generate_fileheader(self,
                            sdfg: SDFG,
                            global_stream: CodeIOStream,
                            backend: str = 'frame'):
        """ Generate a header in every output file that includes custom types
            and constants.
            :param sdfg: The input SDFG.
            :param global_stream: Stream to write to (global).
            :param backend: Whose backend this header belongs to.
        """
        #########################################################
        # Environment-based includes
        for env in self.environments:
            if len(env.headers) > 0:
                global_stream.write(
                    "\n".join("#include \"" + h + "\"" for h in env.headers),
                    sdfg)

        #########################################################
        # Custom types
        datatypes = set()
        # Types of this SDFG
        for _, arrname, arr in sdfg.arrays_recursive():
            if arr is not None:
                datatypes.add(arr.dtype)

        # Emit unique definitions
        wrote_something = False
        for typ in datatypes:
            if hasattr(typ, 'emit_definition'):
                if not wrote_something:
                    global_stream.write("", sdfg)
                wrote_something = True
                global_stream.write(typ.emit_definition(), sdfg)
        if wrote_something:
            global_stream.write("", sdfg)

        #########################################################
        # Write constants
        self.generate_constants(sdfg, global_stream)

        #########################################################
        # Write state struct
        structstr = '\n'.join(self.statestruct)
        global_stream.write(
            f'''
struct {sdfg.name}_t {{
    {structstr}
}};

''', sdfg)

        for sd in sdfg.all_sdfgs_recursive():
            if None in sd.global_code:
                global_stream.write(codeblock_to_cpp(sd.global_code[None]), sd)
            if backend in sd.global_code:
                global_stream.write(codeblock_to_cpp(sd.global_code[backend]),
                                    sd)

    def generate_header(self, sdfg: SDFG, global_stream: CodeIOStream,
                        callsite_stream: CodeIOStream):
        """ Generate the header of the frame-code. Code exists in a separate
            function for overriding purposes.
            :param sdfg: The input SDFG.
            :param global_stream: Stream to write to (global).
            :param callsite_stream: Stream to write to (at call site).
        """
        # Write frame code - header
        global_stream.write(
            '/* DaCe AUTO-GENERATED FILE. DO NOT MODIFY */\n' +
            '#include <dace/dace.h>\n', sdfg)

        # Write header required by environments
        for env in self.environments:
            self.statestruct.extend(env.state_fields)

        # Instrumentation preamble
        if len(self._dispatcher.instrumentation) > 1:
            self.statestruct.append('dace::perf::Report report;')
            # Reset report if written every invocation
            if config.Config.get_bool('instrumentation',
                                      'report_each_invocation'):
                callsite_stream.write('__state->report.reset();', sdfg)

        self.generate_fileheader(sdfg, global_stream, 'frame')

    def generate_footer(self, sdfg: SDFG, global_stream: CodeIOStream,
                        callsite_stream: CodeIOStream):
        """ Generate the footer of the frame-code. Code exists in a separate
            function for overriding purposes.
            :param sdfg: The input SDFG.
            :param global_stream: Stream to write to (global).
            :param callsite_stream: Stream to write to (at call site).
        """
        import dace.library
        fname = sdfg.name
        params = sdfg.signature()
        paramnames = sdfg.signature(False, for_call=True)
        initparams = sdfg.signature(with_arrays=False)
        initparamnames = sdfg.signature(False, for_call=True, with_arrays=False)

        # Invoke all instrumentation providers
        for instr in self._dispatcher.instrumentation.values():
            if instr is not None:
                instr.on_sdfg_end(sdfg, callsite_stream, global_stream)

        sdfg_hash = sdfg.hash_sdfg()

        # Instrumentation saving
        if (config.Config.get_bool('instrumentation', 'report_each_invocation')
                and len(self._dispatcher.instrumentation) > 1):
            callsite_stream.write(
                '''__state->report.save("{path}/perf", "{hash}");'''
                .format(path=sdfg.build_folder.replace('\\', '/'),
                        hash=sdfg_hash), sdfg)

        # Write closing brace of program
        callsite_stream.write('}', sdfg)

        # Write awkward footer to avoid 'extern "C"' issues
        params_comma = (', ' + params) if params else ''
        initparams_comma = (', ' + initparams) if initparams else ''
        paramnames_comma = (', ' + paramnames) if paramnames else ''
        initparamnames_comma = (', ' + initparamnames) if initparamnames else ''
        callsite_stream.write(
            f'''
DACE_EXPORTED void __program_{fname}({fname}_t *__state{params_comma})
{{
    __program_{fname}_internal(__state{paramnames_comma});
}}''', sdfg)

        for target in self._dispatcher.used_targets:
            if target.has_initializer:
                callsite_stream.write(
                    'DACE_EXPORTED int __dace_init_%s(%s_t *__state%s);\n' %
                    (target.target_name, sdfg.name, initparams_comma), sdfg)
            if target.has_finalizer:
                callsite_stream.write(
                    'DACE_EXPORTED int __dace_exit_%s(%s_t *__state);\n' %
                    (target.target_name, sdfg.name), sdfg)

        callsite_stream.write(
            f"""
DACE_EXPORTED {sdfg.name}_t *__dace_init_{sdfg.name}({initparams})
{{
    int __result = 0;
    {sdfg.name}_t *__state = new {sdfg.name}_t;

            """, sdfg)

        for target in self._dispatcher.used_targets:
            if target.has_initializer:
                callsite_stream.write(
                    '__result |= __dace_init_%s(__state%s);' %
                    (target.target_name, initparamnames_comma), sdfg)
        for env in self.environments:
            if env.init_code:
                callsite_stream.write("{  // Environment: " + env.__name__,
                                      sdfg)
                callsite_stream.write(env.init_code)
                callsite_stream.write("}")

        for sd in sdfg.all_sdfgs_recursive():
            if None in sd.init_code:
                callsite_stream.write(codeblock_to_cpp(sd.init_code[None]), sd)
            callsite_stream.write(codeblock_to_cpp(sd.init_code['frame']), sd)

        callsite_stream.write(self._initcode.getvalue(), sdfg)

        callsite_stream.write(
            f"""
    if (__result) {{
        delete __state;
        return nullptr;
    }}
    return __state;
}}

DACE_EXPORTED void __dace_exit_{sdfg.name}({sdfg.name}_t *__state)
{{
""", sdfg)

        # Instrumentation saving
        if (not config.Config.get_bool('instrumentation',
                                       'report_each_invocation')
                and len(self._dispatcher.instrumentation) > 1):
            callsite_stream.write(
                '__state->report.save("%s/perf");' %
                sdfg.build_folder.replace('\\', '/'), sdfg)

        callsite_stream.write(self._exitcode.getvalue(), sdfg)

        for sd in sdfg.all_sdfgs_recursive():
            if None in sd.exit_code:
                callsite_stream.write(codeblock_to_cpp(sd.exit_code[None]), sd)
            callsite_stream.write(codeblock_to_cpp(sd.exit_code['frame']), sd)

        for target in self._dispatcher.used_targets:
            if target.has_finalizer:
                callsite_stream.write(
                    '__dace_exit_%s(__state);' % target.target_name, sdfg)
        for env in reversed(self.environments):
            if env.finalize_code:
                callsite_stream.write("{  // Environment: " + env.__name__,
                                      sdfg)
                callsite_stream.write(env.finalize_code)
                callsite_stream.write("}")

        callsite_stream.write('delete __state;\n}\n', sdfg)

    def generate_state(self,
                       sdfg,
                       state,
                       global_stream,
                       callsite_stream,
                       generate_state_footer=True):

        sid = sdfg.node_id(state)

        # Emit internal transient array allocation
        # Don't allocate transients shared with another state
        data_to_allocate = (set(state.top_level_transients()) -
                            set(sdfg.shared_transients()))
        allocated = set()
        for node in state.data_nodes():
            if node.data not in data_to_allocate or node.data in allocated:
                continue
            allocated.add(node.data)
            self._dispatcher.dispatch_allocate(sdfg, state, sid, node,
                                               global_stream, callsite_stream)

        callsite_stream.write('\n')

        # Emit internal transient array allocation for nested SDFGs
        # TODO: Replace with global allocation management
        gpu_persistent_subgraphs = [
            state.scope_subgraph(node) for node in state.nodes()
            if isinstance(node, dace.nodes.MapEntry)
            and node.map.schedule == dace.ScheduleType.GPU_Persistent
        ]
        nested_allocated = set()
        for sub_graph in gpu_persistent_subgraphs:
            for nested_sdfg in [
                    n.sdfg for n in sub_graph.nodes()
                    if isinstance(n, nodes.NestedSDFG)
            ]:
                nested_shared_transients = set(nested_sdfg.shared_transients())
                for nested_state in nested_sdfg.nodes():
                    nested_sid = nested_sdfg.node_id(nested_state)
                    nested_to_allocate = (
                        set(nested_state.top_level_transients()) -
                        nested_shared_transients)
                    nodes_to_allocate = [
                        n for n in nested_state.data_nodes()
                        if n.data in nested_to_allocate
                        and n.data not in nested_allocated
                    ]
                    for nested_node in nodes_to_allocate:
                        nested_allocated.add(nested_node.data)
                        self._dispatcher.dispatch_allocate(
                            nested_sdfg, nested_state, nested_sid, nested_node,
                            global_stream, callsite_stream)

        callsite_stream.write('\n')

        # Invoke all instrumentation providers
        for instr in self._dispatcher.instrumentation.values():
            if instr is not None:
                instr.on_state_begin(sdfg, state, callsite_stream,
                                     global_stream)

        #####################
        # Create dataflow graph for state's children.

        # DFG to code scheme: Only generate code for nodes whose all
        # dependencies have been executed (topological sort).
        # For different connected components, run them concurrently.

        components = dace.sdfg.concurrent_subgraphs(state)

        if len(components) == 1:
            self._dispatcher.dispatch_subgraph(sdfg,
                                               state,
                                               sid,
                                               global_stream,
                                               callsite_stream,
                                               skip_entry_node=False)
        else:
            if config.Config.get_bool('compiler', 'cpu', 'openmp_sections'):
                callsite_stream.write("#pragma omp parallel sections\n{")
            for c in components:
                if config.Config.get_bool('compiler', 'cpu', 'openmp_sections'):
                    callsite_stream.write("#pragma omp section\n{")
                self._dispatcher.dispatch_subgraph(sdfg,
                                                   c,
                                                   sid,
                                                   global_stream,
                                                   callsite_stream,
                                                   skip_entry_node=False)
                if config.Config.get_bool('compiler', 'cpu', 'openmp_sections'):
                    callsite_stream.write("} // End omp section")
            if config.Config.get_bool('compiler', 'cpu', 'openmp_sections'):
                callsite_stream.write("} // End omp sections")

        #####################
        # Write state footer

        if generate_state_footer:

            # Emit internal transient array deallocation for nested SDFGs
            # TODO: Replace with global allocation management
            gpu_persistent_subgraphs = [
                state.scope_subgraph(node) for node in state.nodes()
                if isinstance(node, dace.nodes.MapEntry)
                and node.map.schedule == dace.ScheduleType.GPU_Persistent
            ]
            nested_deallocated = set()
            for sub_graph in gpu_persistent_subgraphs:
                for nested_sdfg in [
                        n.sdfg for n in sub_graph.nodes()
                        if isinstance(n, nodes.NestedSDFG)
                ]:
                    nested_shared_transients = \
                        set(nested_sdfg.shared_transients())
                    for nested_state in nested_sdfg:
                        nested_sid = nested_sdfg.node_id(nested_state)
                        nested_to_allocate = (
                            set(nested_state.top_level_transients()) -
                            nested_shared_transients)
                        nodes_to_deallocate = [
                            n for n in nested_state.data_nodes()
                            if n.data in nested_to_allocate
                            and n.data not in nested_deallocated
                        ]
                        for nested_node in nodes_to_deallocate:
                            nested_deallocated.add(nested_node.data)
                            self._dispatcher.dispatch_deallocate(
                                nested_sdfg, nested_state, nested_sid,
                                nested_node, global_stream, callsite_stream)

            # Emit internal transient array deallocation
            deallocated = set()
            for node in state.data_nodes():
                if (node.data not in data_to_allocate
                        or node.data in deallocated
                        or (node.data in sdfg.arrays
                            and sdfg.arrays[node.data].transient == False)):
                    continue
                deallocated.add(node.data)
                self._dispatcher.dispatch_deallocate(sdfg, state, sid, node,
                                                     global_stream,
                                                     callsite_stream)

            # Invoke all instrumentation providers
            for instr in self._dispatcher.instrumentation.values():
                if instr is not None:
                    instr.on_state_end(sdfg, state, callsite_stream,
                                       global_stream)

    def generate_states(self, sdfg, global_stream, callsite_stream):
        states_generated = set()

        # Create closure + function for state dispatcher
        def dispatch_state(state: SDFGState) -> str:
            stream = CodeIOStream()
            self._dispatcher.dispatch_state(sdfg, state, global_stream, stream)
            states_generated.add(state)  # For sanity check
            return stream.getvalue()

        # Handle specialized control flow
        if config.Config.get_bool('optimizer', 'detect_control_flow'):
            # Avoid import loop
            from dace.transformation import helpers as xfh
            # Clean up the state machine by separating combined condition and assignment
            # edges.
            xfh.split_interstate_edges(sdfg)

            cft = cflow.structured_control_flow_tree(sdfg, dispatch_state)
        else:
            # If disabled, generate entire graph as general control flow block
            states_topological = list(sdfg.topological_sort(sdfg.start_state))
            last = states_topological[-1]
            cft = cflow.GeneralBlock(dispatch_state, [
                cflow.SingleState(dispatch_state, s, s is last)
                for s in states_topological
            ], [])

        callsite_stream.write(
            cft.as_cpp(self.dispatcher.defined_vars, sdfg.symbols), sdfg)

        # Write exit label
        callsite_stream.write(f'__state_exit_{sdfg.sdfg_id}:;', sdfg)

        return states_generated

    def generate_code(
        self,
        sdfg: SDFG,
        schedule: Optional[dtypes.ScheduleType],
        sdfg_id: str = ""
    ) -> Tuple[str, str, Set[TargetCodeGenerator], Set[str]]:
        """ Generate frame code for a given SDFG, calling registered targets'
            code generation callbacks for them to generate their own code.
            :param sdfg: The SDFG to generate code for.
            :param schedule: The schedule the SDFG is currently located, or
                             None if the SDFG is top-level.
            :param sdfg_id: An optional string id given to the SDFG label
            :return: A tuple of the generated global frame code, local frame
                     code, and a set of targets that have been used in the
                     generation of this SDFG.
        """

        if len(sdfg_id) == 0 and sdfg.sdfg_id != 0:
            sdfg_id = '_%d' % sdfg.sdfg_id

        global_stream = CodeIOStream()
        callsite_stream = CodeIOStream()

        is_top_level = sdfg.parent is None

        # Generate code
        ###########################

        # Keep track of allocated variables
        allocated = set()

        # Add symbol mappings to allocated variables
        if sdfg.parent_nsdfg_node is not None:
            allocated |= sdfg.parent_nsdfg_node.symbol_mapping.keys()

        # Invoke all instrumentation providers
        for instr in self._dispatcher.instrumentation.values():
            if instr is not None:
                instr.on_sdfg_begin(sdfg, callsite_stream, global_stream)

        # Allocate outer-level transients
        shared_transients = sdfg.shared_transients()
        for state in sdfg.nodes():
            for node in state.data_nodes():
                if (node.data in shared_transients
                        and node.data not in allocated):
                    self._dispatcher.dispatch_allocate(sdfg, state, None, node,
                                                       global_stream,
                                                       callsite_stream)
                    allocated.add(node.data)

        # Allocate inter-state variables
        global_symbols = copy.deepcopy(sdfg.symbols)
        global_symbols.update(
            {aname: arr.dtype
             for aname, arr in sdfg.arrays.items()})
        interstate_symbols = {}
        for e in sdfg.edges():
            symbols = e.data.new_symbols(global_symbols)
            # Inferred symbols only take precedence if global symbol not defined
            symbols = {
                k: v if k not in global_symbols else global_symbols[k]
                for k, v in symbols.items()
            }
            interstate_symbols.update(symbols)
            global_symbols.update(symbols)

        for isvarName, isvarType in interstate_symbols.items():
            # Skip symbols that have been declared as outer-level transients
            if isvarName in allocated:
                continue
            isvar = data.Scalar(isvarType)
            callsite_stream.write(
                '%s;\n' % (isvar.as_arg(with_types=True, name=isvarName)), sdfg)
            self.dispatcher.defined_vars.add(isvarName, isvarType,
                                             isvarType.ctype)

        callsite_stream.write('\n', sdfg)

        #######################################################################
        # Generate actual program body

        states_generated = self.generate_states(sdfg, global_stream,
                                                callsite_stream)

        #######################################################################

        # Sanity check
        if len(states_generated) != len(sdfg.nodes()):
            raise RuntimeError(
                "Not all states were generated in SDFG {}!"
                "\n  Generated: {}\n  Missing: {}".format(
                    sdfg.label, [s.label for s in states_generated],
                    [s.label for s in (set(sdfg.nodes()) - states_generated)]))

        # Deallocate transients
        shared_transients = sdfg.shared_transients()
        deallocated = set()
        for state in sdfg.nodes():
            for node in state.data_nodes():
                if (node.data in shared_transients
                        and node.data not in deallocated):
                    self._dispatcher.dispatch_deallocate(
                        sdfg, state, None, node, global_stream, callsite_stream)
                    deallocated.add(node.data)

        # Now that we have all the information about dependencies, generate
        # header and footer
        if is_top_level:
            # Let each target append code to frame code state before generating
            # header and footer
            for target in self._dispatcher.used_targets:
                target.on_target_used()

            header_stream = CodeIOStream()
            header_global_stream = CodeIOStream()
            footer_stream = CodeIOStream()
            footer_global_stream = CodeIOStream()

            # Get all environments used in the generated code, including
            # dependent environments
            import dace.library  # Avoid import loops
            self.environments = dace.library.get_environments_and_dependencies(
                self._dispatcher.used_environments)

            self.generate_header(sdfg, header_global_stream, header_stream)

            # Open program function
            params = sdfg.signature()
            if params:
                params = ', ' + params
            function_signature = (
                'void __program_%s_internal(%s_t *__state%s)\n{\n' %
                (sdfg.name, sdfg.name, params))

            self.generate_footer(sdfg, footer_global_stream, footer_stream)

            header_global_stream.write(global_stream.getvalue())
            header_global_stream.write(footer_global_stream.getvalue())
            generated_header = header_global_stream.getvalue()

            all_code = CodeIOStream()
            all_code.write(function_signature)
            all_code.write(header_stream.getvalue())
            all_code.write(callsite_stream.getvalue())
            all_code.write(footer_stream.getvalue())
            generated_code = all_code.getvalue()
        else:
            generated_header = global_stream.getvalue()
            generated_code = callsite_stream.getvalue()

        # Clean up generated code
        gotos = re.findall(r'goto (.*);', generated_code)
        clean_code = ''
        for line in generated_code.split('\n'):
            # Empty line with semicolon
            if re.match(r'^\s*;\s*', line):
                continue
            # Label that might be unused
            label = re.findall(
                r'^\s*([a-zA-Z_][a-zA-Z_0-9]*):\s*[;]?\s*////.*$', line)
            if len(label) > 0:
                if label[0] not in gotos:
                    continue
            clean_code += line + '\n'

        # Return the generated global and local code strings
        return (generated_header, clean_code, self._dispatcher.used_targets,
                self._dispatcher.used_environments)
