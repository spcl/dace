# Copyright 2019-2020 ETH Zurich and the DaCe authors. All rights reserved.
from typing import Optional, Set, Tuple

import collections
import copy
import dace
import functools
import re
from dace.codegen import control_flow as cflow
from dace.codegen.prettycode import CodeIOStream
from dace.codegen.targets.common import codeblock_to_cpp
from dace.codegen.targets.target import TargetCodeGenerator, TargetDispatcher
from dace.sdfg import SDFG, SDFGState, ScopeSubgraphView
from dace.sdfg import nodes
from dace.sdfg.infer_types import set_default_schedule_and_storage_types
from dace import dtypes, data, config

from dace.frontend.python import wrappers
from dace.codegen import cppunparse

import networkx as nx
import numpy as np


class DaCeCodeGenerator(object):
    """ DaCe code generator class that writes the generated code for SDFG
        state machines, and uses a dispatcher to generate code for
        individual states based on the target. """
    def __init__(self, *args, **kwargs):
        self._dispatcher = TargetDispatcher()
        self._dispatcher.register_state_dispatcher(self)
        self._initcode = CodeIOStream()
        self._exitcode = CodeIOStream()

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
                    (csttype.dtype.ctype, cstname, str(cstval)), sdfg)

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

        for sd in sdfg.all_sdfgs_recursive():
            if None in sd.global_code:
                global_stream.write(codeblock_to_cpp(sd.global_code[None]), sd)
            if backend in sd.global_code:
                global_stream.write(codeblock_to_cpp(sd.global_code[backend]),
                                    sd)

    def generate_header(self, sdfg: SDFG, used_environments: Set[str],
                        global_stream: CodeIOStream,
                        callsite_stream: CodeIOStream):
        """ Generate the header of the frame-code. Code exists in a separate
            function for overriding purposes.
            :param sdfg: The input SDFG.
            :param global_stream: Stream to write to (global).
            :param callsite_stream: Stream to write to (at call site).
        """

        environments = [
            dace.library.get_environment(env_name)
            for env_name in used_environments
        ]

        # Write frame code - header
        global_stream.write(
            '/* DaCe AUTO-GENERATED FILE. DO NOT MODIFY */\n' +
            '#include <dace/dace.h>\n', sdfg)

        # Write header required by environments
        for env in environments:
            if len(env.headers) > 0:
                global_stream.write(
                    "\n".join("#include \"" + h + "\"" for h in env.headers),
                    sdfg)

        global_stream.write("\n", sdfg)

        self.generate_fileheader(sdfg, global_stream, 'frame')

        # Instrumentation preamble
        if len(self._dispatcher.instrumentation) > 1:
            global_stream.write(
                'namespace dace { namespace perf { Report report; } }', sdfg)
            callsite_stream.write('dace::perf::report.reset();', sdfg)

    def generate_footer(self, sdfg: SDFG, used_environments: Set[str],
                        global_stream: CodeIOStream,
                        callsite_stream: CodeIOStream):
        """ Generate the footer of the frame-code. Code exists in a separate
            function for overriding purposes.
            :param sdfg: The input SDFG.
            :param global_stream: Stream to write to (global).
            :param callsite_stream: Stream to write to (at call site).
        """
        fname = sdfg.name
        params = sdfg.signature()
        paramnames = sdfg.signature(False, for_call=True)
        environments = [
            dace.library.get_environment(env_name)
            for env_name in used_environments
        ]

        # Invoke all instrumentation providers
        for instr in self._dispatcher.instrumentation.values():
            if instr is not None:
                instr.on_sdfg_end(sdfg, callsite_stream, global_stream)

        # Instrumentation saving
        if len(self._dispatcher.instrumentation) > 1:
            callsite_stream.write(
                'dace::perf::report.save("%s/perf");' %
                sdfg.build_folder.replace('\\', '/'), sdfg)

        # Write closing brace of program
        callsite_stream.write('}', sdfg)

        # Write awkward footer to avoid 'extern "C"' issues
        callsite_stream.write(
            """
DACE_EXPORTED void __program_%s(%s)
{
    __program_%s_internal(%s);
}
""" % (fname, params, fname, paramnames), sdfg)

        for target in self._dispatcher.used_targets:
            if target.has_initializer:
                callsite_stream.write(
                    'DACE_EXPORTED int __dace_init_%s(%s);\n' %
                    (target.target_name, params), sdfg)
            if target.has_finalizer:
                callsite_stream.write(
                    'DACE_EXPORTED int __dace_exit_%s(%s);\n' %
                    (target.target_name, params), sdfg)

        callsite_stream.write(
            """
DACE_EXPORTED int __dace_init_%s(%s)
{
    int __result = 0;
""" % (sdfg.name, params), sdfg)

        for target in self._dispatcher.used_targets:
            if target.has_initializer:
                callsite_stream.write(
                    '__result |= __dace_init_%s(%s);' %
                    (target.target_name, paramnames), sdfg)
        for env in environments:
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
            """
    return __result;
}

DACE_EXPORTED void __dace_exit_%s(%s)
{
""" % (sdfg.name, params), sdfg)

        callsite_stream.write(self._exitcode.getvalue(), sdfg)

        for sd in sdfg.all_sdfgs_recursive():
            if None in sd.exit_code:
                callsite_stream.write(codeblock_to_cpp(sd.exit_code[None]), sd)
            callsite_stream.write(codeblock_to_cpp(sd.exit_code['frame']), sd)

        for target in self._dispatcher.used_targets:
            if target.has_finalizer:
                callsite_stream.write(
                    '__dace_exit_%s(%s);' % (target.target_name, paramnames),
                    sdfg)
        for env in environments:
            if env.finalize_code:
                callsite_stream.write("{  // Environment: " + env.__name__,
                                      sdfg)
                callsite_stream.write(env.init_code)
                callsite_stream.write("}")

        callsite_stream.write('}\n', sdfg)

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
                            nested_sdfg,
                            nested_state,
                            nested_sid,
                            nested_node,
                            global_stream,
                            callsite_stream,
                        )

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
            callsite_stream.write("#pragma omp parallel sections\n{")
            for c in components:
                callsite_stream.write("#pragma omp section\n{")
                self._dispatcher.dispatch_subgraph(sdfg,
                                                   c,
                                                   sid,
                                                   global_stream,
                                                   callsite_stream,
                                                   skip_entry_node=False)
                callsite_stream.write("} // End omp section")
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

    @staticmethod
    def _generate_assignments(assignments):
        return [
            "{} = {}".format(variable, value)
            for variable, value in assignments.items()
        ]

    @staticmethod
    def _is_always_true(condition_string):
        return condition_string in ["true", "1"]

    def _generate_transition(self, sdfg, sid, callsite_stream, edge,
                             assignments):

        condition_string = cppunparse.cppunparse(edge.data.condition.code,
                                                 False)
        always_true = self._is_always_true(condition_string)

        if not always_true:
            callsite_stream.write("if ({}) {{".format(condition_string), sdfg,
                                  sid)

        if len(assignments) > 0:
            callsite_stream.write(
                ";\n".join(
                    DaCeCodeGenerator._generate_assignments(assignments) +
                    [""]), sdfg, sid)

        callsite_stream.write(
            "goto __state_{}_{};".format(sdfg.sdfg_id, edge.dst.label), sdfg,
            sid)

        if not always_true:
            callsite_stream.write("}")

    def generate_states(self, sdfg, scope_label, control_flow, global_stream,
                        callsite_stream, scope, states_generated,
                        generated_edges):

        states_topological = list(sdfg.topological_sort(sdfg.start_state))
        states_to_generate = collections.deque([
            s for s in states_topological
            if s in scope and s not in states_generated
        ])
        if len(states_to_generate) == 0:
            return

        while len(states_to_generate) > 0:

            state = states_to_generate.popleft()
            # When generating control flow constructs, we will not necessarily
            # move in topological order, so make sure this state has not
            # already been generated.
            if state in states_generated or state not in scope:
                continue
            states_generated.add(state)

            sid = sdfg.node_id(state)

            callsite_stream.write(
                "__state_{}_{}:;\n".format(sdfg.sdfg_id, state.label), sdfg,
                sid)

            # Don't generate brackets and comments for empty states
            if len([n for n in state.nodes()]) > 0:

                callsite_stream.write('{', sdfg, sid)

                self._dispatcher.dispatch_state(sdfg, state, global_stream,
                                                callsite_stream)

                callsite_stream.write('}', sdfg, sid)

            out_edges = sdfg.out_edges(state)

            # Write conditional branches to next states
            for edge in out_edges:

                generate_assignments = True
                generate_transition = True

                # Handle specialized control flow
                if (dace.config.Config.get_bool('optimizer',
                                                'detect_control_flow')):

                    for control in control_flow[edge]:

                        if isinstance(control, cflow.LoopAssignment):
                            # Generate the transition, but leave the
                            # assignments to the loop
                            generate_transition &= True
                            generate_assignments &= False

                        elif isinstance(control, cflow.LoopBack):
                            generate_transition &= False
                            generate_assignments &= False

                        elif isinstance(control, cflow.LoopExit):
                            # Need to strip the condition, so generate it from
                            # the loop entry
                            generate_transition &= False
                            generate_assignments &= True

                        elif isinstance(control, cflow.LoopEntry):
                            generate_transition &= False
                            generate_assignments &= False

                            if control.scope.assignment is not None:
                                assignment_edge = control.scope.assignment.edge
                                init_assignments = ", ".join(
                                    DaCeCodeGenerator._generate_assignments(
                                        assignment_edge.data.assignments))
                                generated_edges.add(assignment_edge)
                            else:
                                init_assignments = ""

                            back_edge = control.scope.back.edge
                            continue_assignments = ", ".join(
                                DaCeCodeGenerator._generate_assignments(
                                    back_edge.data.assignments))
                            generated_edges.add(back_edge)

                            entry_edge = control.scope.entry.edge
                            condition = cppunparse.cppunparse(
                                entry_edge.data.condition.code, False)
                            generated_edges.add(entry_edge)

                            if (len(init_assignments) > 0
                                    or len(continue_assignments) > 0):
                                callsite_stream.write(
                                    "for ({}; {}; {}) {{".format(
                                        init_assignments, condition,
                                        continue_assignments), sdfg, sid)
                            else:
                                callsite_stream.write(
                                    "while ({}) {{".format(condition), sdfg,
                                    sid)

                            # Generate loop body
                            self.generate_states(sdfg,
                                                 entry_edge.src.label + "_loop",
                                                 control_flow, global_stream,
                                                 callsite_stream, control.scope,
                                                 states_generated,
                                                 generated_edges)

                            callsite_stream.write("}", sdfg, sid)

                            exit_edge = control.scope.exit.edge

                            # Update states to generate after nested call
                            states_to_generate = collections.deque([
                                s for s in states_to_generate
                                if s not in states_generated
                            ])
                            # If the next state to be generated is the exit
                            # state, we can omit the goto
                            if (len(states_to_generate) > 0
                                    and states_to_generate[0] == exit_edge.dst
                                    and exit_edge.dst not in states_generated):
                                pass
                            elif edge in generated_edges:
                                # This edge has more roles, goto doesn't apply
                                pass
                            else:
                                callsite_stream.write(
                                    "goto __state_{}_{};".format(
                                        sdfg.sdfg_id,
                                        control.scope.exit.edge.dst))
                                generated_edges.add(control.scope.exit.edge)

                        elif isinstance(control, cflow.IfExit):
                            generate_transition &= True
                            generate_assignments &= True

                        elif isinstance(control, cflow.IfEntry):
                            generate_transition &= False
                            generate_assignments &= True

                            if len(set(control.scope) - states_generated) == 0:
                                continue

                            then_scope = control.scope.if_then_else.then_scope
                            else_scope = control.scope.if_then_else.else_scope

                            then_entry = then_scope.entry.edge

                            condition = cppunparse.cppunparse(
                                then_entry.data.condition.code, False)

                            callsite_stream.write(
                                "if ({}) {{".format(condition), sdfg, sid)
                            generated_edges.add(then_entry)

                            # Generate the then-scope
                            self.generate_states(sdfg, state.label + "_then",
                                                 control_flow, global_stream,
                                                 callsite_stream, then_scope,
                                                 states_generated,
                                                 generated_edges)

                            callsite_stream.write("} else {", sdfg, sid)
                            generated_edges.add(else_scope.entry.edge)

                            # Generate the else-scope
                            self.generate_states(sdfg, state.label + "_else",
                                                 control_flow, global_stream,
                                                 callsite_stream, else_scope,
                                                 states_generated,
                                                 generated_edges)

                            callsite_stream.write("}", sdfg, sid)
                            generated_edges.add(else_scope.exit.edge)

                            # Update states to generate after nested call
                            states_to_generate = collections.deque([
                                s for s in states_to_generate
                                if s not in states_generated
                            ])

                            if_exit_state = control.scope.exit.edge.dst

                            if ((if_exit_state not in states_generated) and
                                ((len(states_to_generate) > 0) and
                                 (states_to_generate[0] == if_exit_state))):
                                pass
                            else:
                                callsite_stream.write(
                                    "goto __state_{}_{};".format(
                                        sdfg.sdfg_id,
                                        control.scope.exit.edge.dst))

                        else:

                            raise TypeError(
                                "Unknown control flow \"{}\"".format(
                                    type(control).__name__))

                if generate_assignments and len(edge.data.assignments) > 0:
                    assignments_to_generate = edge.data.assignments
                else:
                    assignments_to_generate = {}

                if generate_transition:

                    if ((len(out_edges) == 1)
                            and (edge.dst not in states_generated)
                            and ((len(states_to_generate) > 0) and
                                 (states_to_generate[0] == edge.dst))):
                        # If there is only one outgoing edge, the target will
                        # be generated next, we can omit the goto
                        pass
                    elif (len(out_edges) == 1 and len(states_to_generate) == 0
                          and (edge.dst not in scope)):
                        # This scope has ended, and we don't need to generate
                        # any output edge
                        pass
                    else:
                        self._generate_transition(sdfg, sid, callsite_stream,
                                                  edge, assignments_to_generate)
                        # Assignments will be generated in the transition
                        generate_assignments = False

                if generate_assignments:

                    callsite_stream.write(
                        ";\n".join(
                            DaCeCodeGenerator._generate_assignments(
                                assignments_to_generate) + [""]), sdfg, sid)
                generated_edges.add(edge)
                # End of out_edges loop

            if (((len(out_edges) == 0) or
                 (not isinstance(scope, cflow.ControlFlowScope) and
                  (len(states_to_generate) == 0)))
                    and (len(states_generated) != sdfg.number_of_nodes())):
                callsite_stream.write(
                    "goto __state_exit_{}_{};".format(sdfg.sdfg_id,
                                                      scope_label), sdfg, sid)

        # Write exit state
        callsite_stream.write(
            "__state_exit_{}_{}:;".format(sdfg.sdfg_id, scope_label), sdfg)

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

        sdfg_label = sdfg.name + sdfg_id

        global_stream = CodeIOStream()
        callsite_stream = CodeIOStream()

        # Set default storage/schedule types in SDFG
        set_default_schedule_and_storage_types(sdfg, schedule)

        is_top_level = sdfg.parent is None

        # Generate code
        ###########################

        # Invoke all instrumentation providers
        for instr in self._dispatcher.instrumentation.values():
            if instr is not None:
                instr.on_sdfg_begin(sdfg, callsite_stream, global_stream)

        # Allocate outer-level transients
        shared_transients = sdfg.shared_transients()
        allocated = set()
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
        interstate_symbols = {}
        for e in sdfg.edges():
            symbols = e.data.new_symbols(global_symbols)
            interstate_symbols.update(symbols)
            global_symbols.update(symbols)

        for isvarName, isvarType in interstate_symbols.items():
            # Skip symbols that have been declared as outer-level transients
            if isvarName in allocated:
                continue
            isvar = data.Scalar(isvarType)
            callsite_stream.write(
                '%s;\n' % (isvar.as_arg(with_types=True, name=isvarName)), sdfg)

        callsite_stream.write('\n', sdfg)

        states_topological = list(sdfg.topological_sort(sdfg.start_state))

        # {edge: [dace.edges.ControlFlow]}
        control_flow = {e: [] for e in sdfg.edges()}

        if dace.config.Config.get_bool('optimizer', 'detect_control_flow'):

            ####################################################################
            # Loop detection procedure

            all_cycles = list(sdfg.find_cycles())  # Returns a list of lists
            # Order according to topological sort
            all_cycles = [
                sorted(c, key=lambda x: states_topological.index(x))
                for c in all_cycles
            ]
            # Group in terms of starting node
            starting_nodes = [c[0] for c in all_cycles]
            # Order cycles according to starting node in topological sort
            starting_nodes = sorted(starting_nodes,
                                    key=lambda x: states_topological.index(x))
            cycles_by_node = [[c for c in all_cycles if c[0] == n]
                              for n in starting_nodes]
            for cycles in cycles_by_node:

                # Use arbitrary cycle to find the first and last nodes
                first_node = cycles[0][0]
                last_node = cycles[0][-1]

                if not first_node.is_empty():
                    # The entry node should not contain any computations
                    continue

                if not all([c[-1] == last_node for c in cycles]):
                    # There are multiple back edges: not a for or while loop
                    continue

                previous_edge = [
                    e for e in sdfg.in_edges(first_node) if e.src != last_node
                ]
                if len(previous_edge) != 1:
                    # No single starting point: not a for or while
                    continue
                previous_edge = previous_edge[0]

                back_edge = sdfg.edges_between(last_node, first_node)
                if len(back_edge) != 1:
                    raise RuntimeError("Expected exactly one edge in cycle")
                back_edge = back_edge[0]

                # Build a set of all nodes in all cycles associated with this
                # set of start and end node
                internal_nodes = functools.reduce(
                    lambda a, b: a | b, [set(c)
                                         for c in cycles]) - {first_node}

                exit_edge = [
                    e for e in sdfg.out_edges(first_node)
                    if e.dst not in internal_nodes | {first_node}
                ]
                if len(exit_edge) != 1:
                    # No single stopping condition: not a for or while
                    # (we don't support continue or break)
                    continue
                exit_edge = exit_edge[0]

                entry_edge = [
                    e for e in sdfg.out_edges(first_node) if e != exit_edge
                ]
                if len(entry_edge) != 1:
                    # No single starting condition: not a for or while
                    continue
                entry_edge = entry_edge[0]

                # Make sure this is not already annotated to be another construct
                if (len(control_flow[entry_edge]) != 0
                        or len(control_flow[back_edge]) != 0):
                    continue

                # Nested loops case I - previous edge of internal loop is a
                # loop-entry of an external loop (first state in a loop is
                # another loop)
                if (len(control_flow[previous_edge]) == 1 and isinstance(
                        control_flow[previous_edge][0], cflow.LoopEntry)):
                    # Nested loop, mark parent scope
                    loop_parent = control_flow[previous_edge][0].scope
                # Nested loops case II - exit edge of internal loop is a
                # back-edge of an external loop (last state in a loop is another
                # loop)
                elif (len(control_flow[exit_edge]) == 1 and isinstance(
                        control_flow[exit_edge][0], cflow.LoopBack)):
                    # Nested loop, mark parent scope
                    loop_parent = control_flow[exit_edge][0].scope
                elif (len(control_flow[exit_edge]) == 0
                      or len(control_flow[previous_edge]) == 0):
                    loop_parent = None
                else:
                    continue

                if entry_edge == back_edge:
                    # No entry check (we don't support do-loops)
                    # TODO: do we want to add some support for self-loops?
                    continue

                # Now we make sure that there is no other way to exit this
                # cycle, by checking that there's no reachable node *not*
                # included in any cycle between the first and last node.
                if any([len(set(c) - internal_nodes) > 1 for c in cycles]):
                    continue

                # This is a loop! Generate the necessary annotation objects.
                loop_scope = cflow.LoopScope(internal_nodes)

                if ((len(previous_edge.data.assignments) > 0
                     or len(back_edge.data.assignments) > 0) and
                    (len(control_flow[previous_edge]) == 0 or
                     (len(control_flow[previous_edge]) == 1 and
                      control_flow[previous_edge][0].scope == loop_parent))):
                    # Generate assignment edge, if available
                    control_flow[previous_edge].append(
                        cflow.LoopAssignment(loop_scope, previous_edge))
                # Assign remaining control flow constructs
                control_flow[entry_edge].append(
                    cflow.LoopEntry(loop_scope, entry_edge))
                control_flow[exit_edge].append(
                    cflow.LoopExit(loop_scope, exit_edge))
                control_flow[back_edge].append(
                    cflow.LoopBack(loop_scope, back_edge))

            ###################################################################
            # If/then/else detection procedure

            candidates = [
                n for n in states_topological if sdfg.out_degree(n) == 2
            ]
            for candidate in candidates:

                # A valid if occurs when then are no reachable nodes for either
                # path that does not pass through a common dominator.
                dominators = nx.dominance.dominance_frontiers(
                    sdfg.nx, candidate)

                left_entry, right_entry = sdfg.out_edges(candidate)
                if (len(control_flow[left_entry]) > 0
                        or len(control_flow[right_entry]) > 0):
                    # Already assigned to a control flow construct
                    # TODO: carefully allow this in some cases
                    continue

                left, right = left_entry.dst, right_entry.dst
                dominator = dominators[left] & dominators[right]
                if len(dominator) != 1:
                    # There must be a single dominator across both branches,
                    # unless one of the nodes _is_ the next dominator
                    # if (len(dominator) == 0 and dominators[left] == {right}
                    #         or dominators[right] == {left}):
                    #     dominator = dominators[left] | dominators[right]
                    # else:
                    #     continue
                    continue
                dominator = next(iter(dominator))  # Exactly one dominator

                exit_edges = sdfg.in_edges(dominator)
                if len(exit_edges) != 2:
                    # There must be a single entry and a single exit. This
                    # could be relaxed in the future.
                    continue

                left_exit, right_exit = exit_edges
                if (len(control_flow[left_exit]) > 0
                        or len(control_flow[right_exit]) > 0):
                    # Already assigned to a control flow construct
                    # TODO: carefully allow this in some cases
                    continue

                # Now traverse from the source and verify that all possible paths
                # pass through the dominator
                left_nodes = sdfg.all_nodes_between(left, dominator)
                if left_nodes is None:
                    # Not all paths lead to the next dominator
                    continue
                right_nodes = sdfg.all_nodes_between(right, dominator)
                if right_nodes is None:
                    # Not all paths lead to the next dominator
                    continue
                all_nodes = left_nodes | right_nodes

                # Make sure there is no overlap between left and right nodes
                if len(left_nodes & right_nodes) > 0:
                    continue

                # This is a valid if/then/else construct. Generate annotations
                if_then_else = cflow.IfThenElse(candidate, dominator)

                # Arbitrarily assign then/else to the two branches. If one edge
                # has no dominator but leads to the dominator, it means there's
                # only a then clause (and no else).
                has_else = False
                if len(dominators[left]) == 1:
                    then_scope = cflow.IfThenScope(if_then_else, left_nodes)
                    else_scope = cflow.IfElseScope(if_then_else, right_nodes)
                    control_flow[left_entry].append(
                        cflow.IfEntry(then_scope, left_entry))
                    control_flow[left_exit].append(
                        cflow.IfExit(then_scope, left_exit))
                    control_flow[right_exit].append(
                        cflow.IfExit(else_scope, right_exit))
                    if len(dominators[right]) == 1:
                        control_flow[right_entry].append(
                            cflow.IfEntry(else_scope, right_entry))
                        has_else = True
                else:
                    then_scope = cflow.IfThenScope(if_then_else, right_nodes)
                    else_scope = cflow.IfElseScope(if_then_else, left_nodes)
                    control_flow[right_entry].append(
                        cflow.IfEntry(then_scope, right_entry))
                    control_flow[right_exit].append(
                        cflow.IfExit(then_scope, right_exit))
                    control_flow[left_exit].append(
                        cflow.IfExit(else_scope, left_exit))

        #######################################################################
        # Generate actual program body

        states_generated = set()  # For sanity check
        generated_edges = set()
        self.generate_states(sdfg, "sdfg", control_flow,
                             global_stream, callsite_stream,
                             set(states_topological), states_generated,
                             generated_edges)

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
            header_stream = CodeIOStream()
            header_global_stream = CodeIOStream()
            footer_stream = CodeIOStream()
            footer_global_stream = CodeIOStream()
            self.generate_header(sdfg, self._dispatcher.used_environments,
                                 header_global_stream, header_stream)

            # Open program function
            function_signature = 'void __program_%s_internal(%s)\n{\n' % (
                sdfg.name, sdfg.signature())

            self.generate_footer(sdfg, self._dispatcher.used_environments,
                                 footer_global_stream, footer_stream)

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
