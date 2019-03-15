from typing import Set

import collections
import dace
import functools
from dace.codegen.prettycode import CodeIOStream
from dace.codegen.targets.target import TargetCodeGenerator, TargetDispatcher
from dace.sdfg import SDFG, SDFGState, ScopeSubgraphView
from dace.graph import nodes
from dace import types, config

from dace.frontend.python import ndarray
from dace.codegen.instrumentation.perfsettings import PerfSettings, PerfUtils
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
        for cstname, cstval in sdfg.constants.items():
            if isinstance(cstval, np.ndarray):
                if isinstance(cstval, ndarray.ndarray):
                    dtype = cstval.descriptor.dtype
                else:
                    dtype = types.typeclass(cstval.dtype.type)
                const_str = "constexpr " + dtype.ctype + \
                    " " + cstname + "[" + str(cstval.size) + "] = {"
                it = np.nditer(cstval, order='C')
                for i in range(cstval.size - 1):
                    const_str += str(it[0]) + ", "
                    it.iternext()
                const_str += str(it[0]) + "};\n"
                callsite_stream.write(const_str, sdfg)
            else:
                callsite_stream.write(
                    "constexpr auto %s = %s;\n" % (cstname, str(cstval)), sdfg)

    def generate_fileheader(self, sdfg: SDFG, global_stream: CodeIOStream):
        """ Generate a header in every output file that includes custom types
            and constants.
            @param sdfg: The input SDFG.
            @param global_stream: Stream to write to (global).
        """
        #########################################################
        # Custom types
        types = set()
        # Types of this SDFG
        for sdfg, arrname, arr in sdfg.arrays_recursive():
            if arr is not None:
                types.add(arr.dtype)

        # Emit unique definitions
        global_stream.write('\n')
        for typ in types:
            if hasattr(typ, 'emit_definition'):
                global_stream.write(typ.emit_definition(), sdfg)
        global_stream.write('\n')

        #########################################################
        # Write constants
        self.generate_constants(sdfg, global_stream)

    def generate_header(self, sdfg: SDFG, global_stream: CodeIOStream,
                        callsite_stream: CodeIOStream):
        """ Generate the header of the frame-code. Code exists in a separate
            function for overriding purposes.
            @param sdfg: The input SDFG.
            @param global_stream: Stream to write to (global).
            @param callsite_stream: Stream to write to (at call site).
        """
        fname = sdfg.name
        params = sdfg.signature()

        # Write frame code - header
        global_stream.write(
            '/* DaCe AUTO-GENERATED FILE. DO NOT MODIFY */\n' +
            '#include <dace/dace.h>\n', sdfg)

        # Added for instrumentation includes
        if PerfSettings.perf_enable_instrumentation():
            global_stream.write(
                '/* DaCe instrumentation include */\n' +
                '#include <dace/perf/instrumentation.h>\n', sdfg)

        self.generate_fileheader(sdfg, callsite_stream)

        callsite_stream.write(
            'void __program_%s_internal(%s)\n{\n' % (fname, params), sdfg)

        # Define the performance store (autocleanup on destruction)
        if PerfSettings.perf_enable_instrumentation():
            callsite_stream.write(
                'dace_perf::PAPI::init();\n' + 'dace_perf::%s __perf_store;\n'
                % PerfUtils.perf_counter_store_string(
                    PerfSettings.perf_default_papi_counters()), sdfg)

    def generate_footer(self, sdfg: SDFG, global_stream: CodeIOStream,
                        callsite_stream: CodeIOStream):
        """ Generate the footer of the frame-code. Code exists in a separate
            function for overriding purposes.
            @param sdfg: The input SDFG.
            @param global_stream: Stream to write to (global).
            @param callsite_stream: Stream to write to (at call site).
        """
        fname = sdfg.name
        params = sdfg.signature()
        paramnames = sdfg.signature(False)

        # Write frame code - footer
        callsite_stream.write('}\n', sdfg)

        # Write awkward footer to avoid 'extern "C"' issues
        callsite_stream.write(
            """
void __program_%s_internal(%s);
DACE_EXPORTED void __program_%s(%s)
{
    __program_%s_internal(%s);
}
""" % (fname, params, fname, params, fname, paramnames), sdfg)

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
DACE_EXPORTED int __dace_init(%s)
{
    int result = 0;
""" % params, sdfg)

        for target in self._dispatcher.used_targets:
            if target.has_initializer:
                callsite_stream.write(
                    'result |= __dace_init_%s(%s);' % (target.target_name,
                                                       paramnames), sdfg)

        callsite_stream.write(self._initcode.getvalue(), sdfg)

        callsite_stream.write(
            """
    return result;
}

DACE_EXPORTED void __dace_exit(%s)
{
""" % params, sdfg)

        callsite_stream.write(self._exitcode.getvalue(), sdfg)

        for target in self._dispatcher.used_targets:
            if target.has_finalizer:
                callsite_stream.write(
                    '__dace_exit_%s(%s);' % (target.target_name, paramnames),
                    sdfg)

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
        data_to_allocate = (
            set(state.top_level_transients()) - set(sdfg.shared_transients()))
        allocated = set()
        for node in state.data_nodes():
            if node.data not in data_to_allocate or node.data in allocated:
                continue
            allocated.add(node.data)
            self._dispatcher.dispatch_allocate(sdfg, state, sid, node,
                                               global_stream, callsite_stream)
            self._dispatcher.dispatch_initialize(
                sdfg, state, sid, node, global_stream, callsite_stream)

        #####################
        # Create dataflow graph for state's children.

        # DFG to code scheme: Only generate code for nodes whose all
        # dependencies have been executed (topological sort).
        # For different connected components, run them concurrently.

        components = dace.sdfg.concurrent_subgraphs(state)

        if len(components) == 1:
            self._dispatcher.dispatch_subgraph(
                sdfg,
                state,
                sid,
                global_stream,
                callsite_stream,
                skip_entry_node=False)
        else:
            #############################################################
            # Instrumentation: Pre-state
            # We cannot have supersections starting in parallel
            parent_id = PerfUtils.unified_id(-1, sid)
            if PerfSettings.perf_enable_instrumentation():
                callsite_stream.write(
                    "__perf_store.markSuperSectionStart(%d);\n" %
                    PerfUtils.unified_id(-1, sid))
            #############################################################

            callsite_stream.write("#pragma omp parallel sections\n{")
            for c in components:
                c.set_parallel_parent(
                    parent_id
                )  # Keep in mind not to add supersection start markers!
                callsite_stream.write("#pragma omp section\n{")
                self._dispatcher.dispatch_subgraph(
                    sdfg,
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
            # Emit internal transient array deallocation
            deallocated = set()
            for node in state.data_nodes():
                if node.data not in data_to_allocate or node.data in deallocated:
                    continue
                deallocated.add(node.data)
                self._dispatcher.dispatch_deallocate(
                    sdfg, state, sid, node, global_stream, callsite_stream)

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

        condition_string = cppunparse.cppunparse(edge.data.condition, False)
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
            "goto __state_{}_{};".format(sdfg.name, edge.dst.label), sdfg, sid)

        if not always_true:
            callsite_stream.write("}")

    def generate_states(self, sdfg, scope_label, control_flow, global_stream,
                        callsite_stream, scope, states_generated):

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
                "__state_{}_{}:\n".format(sdfg.name, state.label), sdfg, sid)

            # Don't generate brackets and comments for empty states
            if len([
                    n for n in state.nodes()
                    if not isinstance(n, dace.graph.nodes.EmptyTasklet)
            ]) > 0:

                callsite_stream.write('{', sdfg, sid)

                self._dispatcher.dispatch_state(sdfg, state, global_stream,
                                                callsite_stream)

                callsite_stream.write('}', sdfg, sid)

            else:

                callsite_stream.write(";")

            out_edges = sdfg.out_edges(state)

            # Write conditional branches to next states
            for edge in out_edges:

                generate_assignments = True
                generate_transition = True

                # Handle specialized control flow
                if (dace.config.Config.get_bool('optimizer',
                                                'detect_control_flow')):

                    for control in control_flow[edge]:

                        if isinstance(control,
                                      dace.graph.edges.LoopAssignment):
                            # Generate the transition, but leave the
                            # assignments to the loop
                            generate_transition = True
                            generate_assignments = False

                        elif isinstance(control, dace.graph.edges.LoopBack):
                            generate_transition = False
                            generate_assignments = False

                        elif isinstance(control, dace.graph.edges.LoopExit):
                            # Need to strip the condition, so generate it from
                            # the loop entry
                            generate_transition = False
                            generate_assignments = True
                            pass

                        elif isinstance(control, dace.graph.edges.LoopEntry):
                            generate_transition = False
                            generate_assignments = False

                            if control.scope.assignment is not None:
                                assignment_edge = control.scope.assignment.edge
                                init_assignments = ", ".join(
                                    DaCeCodeGenerator._generate_assignments(
                                        assignment_edge.data.assignments))
                            else:
                                init_assignments = ""

                            back_edge = control.scope.back.edge
                            continue_assignments = ", ".join(
                                DaCeCodeGenerator._generate_assignments(
                                    back_edge.data.assignments))

                            entry_edge = control.scope.entry.edge
                            condition = cppunparse.cppunparse(
                                entry_edge.data.condition, False)

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
                            self.generate_states(
                                sdfg, entry_edge.src.label + "_loop",
                                control_flow, global_stream, callsite_stream,
                                control.scope, states_generated)

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
                            else:
                                callsite_stream.write(
                                    "goto __state_{}_{};".format(
                                        sdfg.name,
                                        control.scope.exit.edge.dst))

                        elif isinstance(control, dace.graph.edges.IfExit):
                            generate_transition = True
                            generate_assignments = True

                        elif isinstance(control, dace.graph.edges.IfEntry):
                            generate_transition = False
                            generate_assignments = True

                            if len(set(control.scope) - states_generated) == 0:
                                continue

                            then_scope = control.scope.if_then_else.then_scope
                            else_scope = control.scope.if_then_else.else_scope

                            then_entry = then_scope.entry.edge

                            condition = cppunparse.cppunparse(
                                then_entry.data.condition, False)

                            callsite_stream.write(
                                "if ({}) {{".format(condition), sdfg, sid)

                            # Generate the then-scope
                            self.generate_states(sdfg, state.label + "_then",
                                                 control_flow, global_stream,
                                                 callsite_stream, then_scope,
                                                 states_generated)

                            callsite_stream.write("} else {", sdfg, sid)

                            # Generate the else-scope
                            self.generate_states(sdfg, state.label + "_else",
                                                 control_flow, global_stream,
                                                 callsite_stream, else_scope,
                                                 states_generated)

                            callsite_stream.write("}", sdfg, sid)

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
                                        sdfg.name,
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
                                                  edge,
                                                  assignments_to_generate)
                        # Assignments will be generated in the transition
                        generate_assignments = False

                if generate_assignments:

                    callsite_stream.write(
                        ";\n".join(
                            DaCeCodeGenerator._generate_assignments(
                                assignments_to_generate) + [""]), sdfg, sid)

            if (((len(out_edges) == 0) or
                 (not isinstance(scope, dace.graph.edges.ControlFlowScope) and
                  (len(states_to_generate) == 0)))
                    and (len(states_generated) != sdfg.number_of_nodes())):
                callsite_stream.write(
                    "goto __state_exit_{}_{};".format(sdfg.name, scope_label),
                    sdfg, sid)

        # Write exit state
        callsite_stream.write(
            "__state_exit_{}_{}:;".format(sdfg.name, scope_label), sdfg)

    @staticmethod
    def all_nodes_between(graph, begin, end):
        """Finds all nodes between begin and end. Returns None if there is any
           path starting at begin that does not reach end."""
        to_visit = [begin]
        seen = set()
        while len(to_visit) > 0:
            n = to_visit.pop()
            if n == end:
                continue  # We've reached the end node
            if n in seen:
                continue  # We've already visited this node
            seen.add(n)
            # Keep chasing all paths to reach the end node
            node_out_edges = graph.out_edges(n)
            if len(node_out_edges) == 0:
                # We traversed to the end without finding the end
                return None
            for e in node_out_edges:
                next_node = e.dst
                if next_node != end and next_node not in seen:
                    to_visit.append(next_node)
        return seen

    def generate_code(self,
                      sdfg: SDFG,
                      schedule: types.ScheduleType,
                      sdfg_id: str = ""
                      ) -> (str, str, Set[TargetCodeGenerator]):
        """ Generate frame code for a given SDFG, calling registered targets'
            code generation callbacks for them to generate their own code.
            @param sdfg: The SDFG to generate code for.
            @param schedule: The schedule the SDFG is currently located, or
                             None if the SDFG is top-level.
            @param sdfg_id: An optional string id given to the SDFG label
            @return: A tuple of the generated global frame code, local frame
                     code, and a set of targets that have been used in the
                     generation of this SDFG.
        """

        sdfg_label = sdfg.name + sdfg_id

        global_stream = CodeIOStream()
        callsite_stream = CodeIOStream()

        # Set default storage/schedule types in SDFG
        _set_default_schedule_and_storage_types(sdfg, schedule)

        # Generate preamble (if top-level)
        if schedule is None:
            self.generate_header(sdfg, global_stream, callsite_stream)

        # Generate code
        ###########################

        if sdfg.parent is not None:
            # Nested SDFG
            symbols_available = sdfg.parent.symbols_defined_at(sdfg)
        else:
            symbols_available = sdfg.constants

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
                    self._dispatcher.dispatch_initialize(
                        sdfg, state, None, node, global_stream,
                        callsite_stream)
                    allocated.add(node.data)

        # Allocate inter-state variables
        assigned, _ = sdfg.interstate_symbols()
        for isvarName, isvarType in assigned.items():
            # Skip symbols that have been declared as outer-level transients
            if isvarName in allocated:
                continue
            callsite_stream.write(
                '%s;\n' % (isvarType.signature(
                    with_types=True, name=isvarName)), sdfg)

        # Initialize parameter arrays
        for argnode in types.deduplicate(sdfg.input_arrays() +
                                         sdfg.output_arrays()):
            # Ignore transient arrays
            if argnode.desc(sdfg).transient: continue
            self._dispatcher.dispatch_initialize(
                sdfg, sdfg, None, argnode, global_stream, callsite_stream)

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
                        or len(control_flow[back_edge]) != 0
                        or len(control_flow[exit_edge]) != 0):
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
                loop_scope = dace.graph.edges.LoopScope(internal_nodes)

                if ((len(previous_edge.data.assignments) > 0
                     or len(back_edge.data.assignments) > 0)
                        and len(control_flow[previous_edge]) == 0):
                    # Generate assignment edge, if available
                    control_flow[previous_edge].append(
                        dace.graph.edges.LoopAssignment(
                            loop_scope, previous_edge))
                # Assign remaining control flow constructs
                control_flow[entry_edge].append(
                    dace.graph.edges.LoopEntry(loop_scope, entry_edge))
                control_flow[exit_edge].append(
                    dace.graph.edges.LoopExit(loop_scope, exit_edge))
                control_flow[back_edge].append(
                    dace.graph.edges.LoopBack(loop_scope, back_edge))

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
                left_nodes = DaCeCodeGenerator.all_nodes_between(
                    sdfg, left, dominator)
                if left_nodes is None:
                    # Not all paths lead to the next dominator
                    continue
                right_nodes = DaCeCodeGenerator.all_nodes_between(
                    sdfg, right, dominator)
                if right_nodes is None:
                    # Not all paths lead to the next dominator
                    continue
                all_nodes = left_nodes | right_nodes

                # Make sure there is no overlap between left and right nodes
                if len(left_nodes & right_nodes) > 0:
                    continue

                # This is a valid if/then/else construct. Generate annotations
                if_then_else = dace.graph.edges.IfThenElse(
                    candidate, dominator)

                # Arbitrarily assign then/else to the two branches. If one edge
                # has no dominator but leads to the dominator, it means there's
                # only a then clause (and no else).
                has_else = False
                if len(dominators[left]) == 1:
                    then_scope = dace.graph.edges.IfThenScope(
                        if_then_else, left_nodes)
                    else_scope = dace.graph.edges.IfElseScope(
                        if_then_else, right_nodes)
                    control_flow[left_entry].append(
                        dace.graph.edges.IfEntry(then_scope, left_entry))
                    control_flow[left_exit].append(
                        dace.graph.edges.IfExit(then_scope, left_exit))
                    control_flow[right_exit].append(
                        dace.graph.edges.IfExit(else_scope, right_exit))
                    if len(dominators[right]) == 1:
                        control_flow[right_entry].append(
                            dace.graph.edges.IfEntry(else_scope, right_entry))
                        has_else = True
                else:
                    then_scope = dace.graph.edges.IfThenScope(
                        if_then_else, right_nodes)
                    else_scope = dace.graph.edges.IfElseScope(
                        if_then_else, left_nodes)
                    control_flow[right_entry].append(
                        dace.graph.edges.IfEntry(then_scope, right_entry))
                    control_flow[right_exit].append(
                        dace.graph.edges.IfExit(then_scope, right_exit))
                    control_flow[left_exit].append(
                        dace.graph.edges.IfExit(else_scope, left_exit))

        #######################################################################
        # State transition generation

        states_generated = set()  # For sanity check
        self.generate_states(sdfg, "sdfg", control_flow,
                             global_stream, callsite_stream,
                             set(states_topological), states_generated)

        #############################
        # End of code generation

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
                        sdfg, sdfg, None, node, global_stream, callsite_stream)
                    deallocated.add(node.data)

        ###########################

        # Generate footer (if top-level)
        if schedule is None:
            self.generate_footer(sdfg, global_stream, callsite_stream)

        # Clear out all the annotated control flow

        # Return the generated global and local code strings
        return (global_stream.getvalue(), callsite_stream.getvalue(),
                self._dispatcher.used_targets)


def _set_default_schedule_and_storage_types(sdfg, toplevel_schedule):
    """ Sets default storage and schedule types throughout SDFG. 
        Replaces `ScheduleType.Default` and `StorageType.Default`
        with the corresponding types according to the parent scope's 
        schedule. """
    for state in sdfg.nodes():
        scope_dict = state.scope_dict()
        reverse_scope_dict = state.scope_dict(node_to_children=True)

        def set_default_in_scope(parent_node):
            if parent_node is None:
                parent_schedule = toplevel_schedule
            else:
                parent_schedule = parent_node.map.schedule

            for node in reverse_scope_dict[parent_node]:
                # Set default schedule type
                if isinstance(node, nodes.MapEntry):
                    if node.map.schedule == types.ScheduleType.Default:
                        node.map._schedule = \
                            types.SCOPEDEFAULT_SCHEDULE[parent_schedule]
                    # Also traverse children (recursively)
                    set_default_in_scope(node)
                elif isinstance(node, nodes.ConsumeEntry):
                    if node.consume.schedule == types.ScheduleType.Default:
                        node.consume._schedule = \
                            types.SCOPEDEFAULT_SCHEDULE[parent_schedule]
                    # Also traverse children (recursively)
                    set_default_in_scope(node)
                elif getattr(node, 'schedule', False):
                    if node.schedule == types.ScheduleType.Default:
                        node._schedule = \
                            types.SCOPEDEFAULT_SCHEDULE[parent_schedule]

        ## End of recursive function

        # Start with top-level nodes
        set_default_in_scope(None)

        # Set default storage type
        for node in state.nodes():
            if isinstance(node, nodes.AccessNode):
                if node.desc(sdfg).storage == types.StorageType.Default:
                    if scope_dict[node] is None:
                        parent_schedule = toplevel_schedule
                    else:
                        parent_schedule = scope_dict[node].map.schedule

                    node.desc(sdfg).storage = (
                        types.SCOPEDEFAULT_STORAGE[parent_schedule])
        ### End of storage type loop
