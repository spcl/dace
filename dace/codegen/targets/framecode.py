# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
import collections
import copy
import functools
import re
from typing import Any, DefaultDict, Dict, List, Optional, Set, Tuple, Union

import networkx as nx
import numpy as np

import dace
from dace import config, data, dtypes
from dace.cli import progress
from dace.codegen import control_flow as cflow
from dace.codegen import dispatcher as disp
from dace.codegen.prettycode import CodeIOStream
from dace.codegen.targets.common import codeblock_to_cpp, sym2cpp
from dace.codegen.targets.cpp import unparse_interstate_edge
from dace.codegen.targets.target import TargetCodeGenerator
from dace.frontend.python import wrappers
from dace.sdfg import SDFG, ScopeSubgraphView, SDFGState, nodes
from dace.sdfg import scope as sdscope
from dace.sdfg import utils
from dace.sdfg.infer_types import set_default_schedule_and_storage_types
from dace.transformation.passes.analysis import StateReachability


def _get_or_eval_sdfg_first_arg(func, sdfg):
    if callable(func):
        return func(sdfg)
    return func


class DaCeCodeGenerator(object):
    """ DaCe code generator class that writes the generated code for SDFG
        state machines, and uses a dispatcher to generate code for
        individual states based on the target. """

    def __init__(self, sdfg: SDFG):
        self._dispatcher = disp.TargetDispatcher(self)
        self._dispatcher.register_state_dispatcher(self)
        self._initcode = CodeIOStream()
        self._exitcode = CodeIOStream()
        self.statestruct: List[str] = []
        self.environments: List[Any] = []
        self.targets: Set[TargetCodeGenerator] = set()
        self.to_allocate: DefaultDict[Union[SDFG, SDFGState, nodes.EntryNode],
                                      List[Tuple[int, int, nodes.AccessNode]]] = collections.defaultdict(list)
        self.where_allocated: Dict[Tuple[SDFG, str], SDFG] = {}
        self.fsyms: Dict[int, Set[str]] = {}
        self._symbols_and_constants: Dict[int, Set[str]] = {}
        fsyms = self.free_symbols(sdfg)
        self.arglist = sdfg.arglist(scalars_only=False, free_symbols=fsyms)

        # resolve all symbols and constants
        # first handle root
        self._symbols_and_constants[sdfg.sdfg_id] = sdfg.free_symbols.union(sdfg.constants_prop.keys())
        # then recurse
        for nested, state in sdfg.all_nodes_recursive():
            if isinstance(nested, nodes.NestedSDFG):
                state: SDFGState

                nsdfg = nested.sdfg

                # found a new nested sdfg: resolve symbols and constants
                result = nsdfg.free_symbols.union(nsdfg.constants_prop.keys())

                # check for constant inputs
                for edge in state.in_edges(nested):
                    if edge.data.data in state.parent.constants_prop:
                        # this edge is constant => propagate to nested sdfg
                        result.add(edge.dst_conn)

                self._symbols_and_constants[nsdfg.sdfg_id] = result

    # Cached fields
    def symbols_and_constants(self, sdfg: SDFG):
        return self._symbols_and_constants[sdfg.sdfg_id]

    def free_symbols(self, obj: Any):
        k = id(obj)
        if k in self.fsyms:
            return self.fsyms[k]
        result = obj.free_symbols
        self.fsyms[k] = result
        return result

    ##################################################################
    # Target registry

    @property
    def dispatcher(self):
        return self._dispatcher

    ##################################################################
    # Code generation

    def preprocess(self, sdfg: SDFG) -> None:
        """
        Called before code generation. Used for making modifications on the SDFG prior to code generation.
        :note: Post-conditions assume that the SDFG will NOT be changed after this point.
        :param sdfg: The SDFG to modify in-place.
        """
        pass

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
                callsite_stream.write("constexpr %s %s = %s;\n" % (csttype.dtype.ctype, cstname, sym2cpp(cstval)), sdfg)

    def generate_fileheader(self, sdfg: SDFG, global_stream: CodeIOStream, backend: str = 'frame'):
        """ Generate a header in every output file that includes custom types
            and constants.
            :param sdfg: The input SDFG.
            :param global_stream: Stream to write to (global).
            :param backend: Whose backend this header belongs to.
        """
        # Hash file include
        if backend == 'frame':
            global_stream.write('#include "../../include/hash.h"\n', sdfg)

        #########################################################
        # Environment-based includes
        for env in self.environments:
            if len(env.headers) > 0:
                if not isinstance(env.headers, dict):
                    headers = {'frame': env.headers}
                else:
                    headers = env.headers
                if backend in headers:
                    global_stream.write("\n".join("#include \"" + h + "\"" for h in headers[backend]), sdfg)

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
        global_stream.write(f'''
struct {sdfg.name}_t {{
    {structstr}
}};

''', sdfg)

        for sd in sdfg.all_sdfgs_recursive():
            if None in sd.global_code:
                global_stream.write(codeblock_to_cpp(sd.global_code[None]), sd)
            if backend in sd.global_code:
                global_stream.write(codeblock_to_cpp(sd.global_code[backend]), sd)

    def generate_header(self, sdfg: SDFG, global_stream: CodeIOStream, callsite_stream: CodeIOStream):
        """ Generate the header of the frame-code. Code exists in a separate
            function for overriding purposes.
            :param sdfg: The input SDFG.
            :param global_stream: Stream to write to (global).
            :param callsite_stream: Stream to write to (at call site).
        """
        # Write frame code - header
        global_stream.write('/* DaCe AUTO-GENERATED FILE. DO NOT MODIFY */\n' + '#include <dace/dace.h>\n', sdfg)

        # Write header required by environments
        for env in self.environments:
            self.statestruct.extend(env.state_fields)

        # Instrumentation preamble
        if len(self._dispatcher.instrumentation) > 2:
            self.statestruct.append('dace::perf::Report report;')
            # Reset report if written every invocation
            if config.Config.get_bool('instrumentation', 'report_each_invocation'):
                callsite_stream.write('__state->report.reset();', sdfg)

        self.generate_fileheader(sdfg, global_stream, 'frame')

    def generate_footer(self, sdfg: SDFG, global_stream: CodeIOStream, callsite_stream: CodeIOStream):
        """ Generate the footer of the frame-code. Code exists in a separate
            function for overriding purposes.
            :param sdfg: The input SDFG.
            :param global_stream: Stream to write to (global).
            :param callsite_stream: Stream to write to (at call site).
        """
        import dace.library
        fname = sdfg.name
        params = sdfg.signature(arglist=self.arglist)
        paramnames = sdfg.signature(False, for_call=True, arglist=self.arglist)
        initparams = sdfg.init_signature(free_symbols=self.free_symbols(sdfg))
        initparamnames = sdfg.init_signature(for_call=True, free_symbols=self.free_symbols(sdfg))

        # Invoke all instrumentation providers
        for instr in self._dispatcher.instrumentation.values():
            if instr is not None:
                instr.on_sdfg_end(sdfg, callsite_stream, global_stream)

        # Instrumentation saving
        if (config.Config.get_bool('instrumentation', 'report_each_invocation')
                and len(self._dispatcher.instrumentation) > 2):
            callsite_stream.write(
                '''__state->report.save("{path}/perf", __HASH_{name});'''.format(path=sdfg.build_folder.replace(
                    '\\', '/'),
                                                                                 name=sdfg.name), sdfg)

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
                    'DACE_EXPORTED int __dace_exit_%s(%s_t *__state);\n' % (target.target_name, sdfg.name), sdfg)

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
                    '__result |= __dace_init_%s(__state%s);' % (target.target_name, initparamnames_comma), sdfg)
        for env in self.environments:
            init_code = _get_or_eval_sdfg_first_arg(env.init_code, sdfg)
            if init_code:
                callsite_stream.write("{  // Environment: " + env.__name__, sdfg)
                callsite_stream.write(init_code)
                callsite_stream.write("}")

        for sd in sdfg.all_sdfgs_recursive():
            if None in sd.init_code:
                callsite_stream.write(codeblock_to_cpp(sd.init_code[None]), sd)
            if 'frame' in sd.init_code:
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
        if (not config.Config.get_bool('instrumentation', 'report_each_invocation')
                and len(self._dispatcher.instrumentation) > 2):
            callsite_stream.write(
                '__state->report.save("%s/perf", __HASH_%s);' % (sdfg.build_folder.replace('\\', '/'), sdfg.name), sdfg)

        callsite_stream.write(self._exitcode.getvalue(), sdfg)

        for sd in sdfg.all_sdfgs_recursive():
            if None in sd.exit_code:
                callsite_stream.write(codeblock_to_cpp(sd.exit_code[None]), sd)
            if 'frame' in sd.exit_code:
                callsite_stream.write(codeblock_to_cpp(sd.exit_code['frame']), sd)

        for target in self._dispatcher.used_targets:
            if target.has_finalizer:
                callsite_stream.write('__dace_exit_%s(__state);' % target.target_name, sdfg)
        for env in reversed(self.environments):
            finalize_code = _get_or_eval_sdfg_first_arg(env.finalize_code, sdfg)
            if finalize_code:
                callsite_stream.write("{  // Environment: " + env.__name__, sdfg)
                callsite_stream.write(finalize_code)
                callsite_stream.write("}")

        callsite_stream.write('delete __state;\n}\n', sdfg)

    def generate_state(self, sdfg, state, global_stream, callsite_stream, generate_state_footer=True):

        sid = sdfg.node_id(state)

        # Emit internal transient array allocation
        self.allocate_arrays_in_scope(sdfg, state, global_stream, callsite_stream)

        callsite_stream.write('\n')

        # Invoke all instrumentation providers
        for instr in self._dispatcher.instrumentation.values():
            if instr is not None:
                instr.on_state_begin(sdfg, state, callsite_stream, global_stream)

        #####################
        # Create dataflow graph for state's children.

        # DFG to code scheme: Only generate code for nodes whose all
        # dependencies have been executed (topological sort).
        # For different connected components, run them concurrently.

        components = dace.sdfg.concurrent_subgraphs(state)

        if len(components) <= 1:
            self._dispatcher.dispatch_subgraph(sdfg, state, sid, global_stream, callsite_stream, skip_entry_node=False)
        else:
            if sdfg.openmp_sections:
                callsite_stream.write("#pragma omp parallel sections\n{")
            for c in components:
                if sdfg.openmp_sections:
                    callsite_stream.write("#pragma omp section\n{")
                self._dispatcher.dispatch_subgraph(sdfg, c, sid, global_stream, callsite_stream, skip_entry_node=False)
                if sdfg.openmp_sections:
                    callsite_stream.write("} // End omp section")
            if sdfg.openmp_sections:
                callsite_stream.write("} // End omp sections")

        #####################
        # Write state footer

        if generate_state_footer:
            # Emit internal transient array deallocation
            self.deallocate_arrays_in_scope(sdfg, state, global_stream, callsite_stream)

            # Invoke all instrumentation providers
            for instr in self._dispatcher.instrumentation.values():
                if instr is not None:
                    instr.on_state_end(sdfg, state, callsite_stream, global_stream)

    def generate_states(self, sdfg, global_stream, callsite_stream):
        states_generated = set()

        opbar = progress.OptionalProgressBar(sdfg.number_of_nodes(), title=f'Generating code (SDFG {sdfg.sdfg_id})')

        # Create closure + function for state dispatcher
        def dispatch_state(state: SDFGState) -> str:
            stream = CodeIOStream()
            self._dispatcher.dispatch_state(sdfg, state, global_stream, stream)
            opbar.next()
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
            cft = cflow.GeneralBlock(dispatch_state,
                                     [cflow.SingleState(dispatch_state, s, s is last) for s in states_topological], [],
                                     [], [], [])

        callsite_stream.write(cft.as_cpp(self, sdfg.symbols), sdfg)

        opbar.done()

        # Write exit label
        callsite_stream.write(f'__state_exit_{sdfg.sdfg_id}:;', sdfg)

        return states_generated

    def _get_schedule(self, scope: Union[nodes.EntryNode, SDFGState, SDFG]) -> dtypes.ScheduleType:
        TOP_SCHEDULE = dtypes.ScheduleType.Sequential
        if scope is None:
            return TOP_SCHEDULE
        elif isinstance(scope, nodes.EntryNode):
            return scope.schedule
        elif isinstance(scope, (SDFGState, SDFG)):
            sdfg: SDFG = (scope if isinstance(scope, SDFG) else scope.parent)
            if sdfg.parent_nsdfg_node is None:
                return TOP_SCHEDULE
            return (sdfg.parent_nsdfg_node.schedule or TOP_SCHEDULE)
        else:
            raise TypeError

    def _can_allocate(self, sdfg: SDFG, state: SDFGState, desc: data.Data, scope: Union[nodes.EntryNode, SDFGState,
                                                                                        SDFG]) -> bool:
        schedule = self._get_schedule(scope)
        # if not dtypes.can_allocate(desc.storage, schedule):
        #     return False
        if dtypes.can_allocate(desc.storage, schedule):
            return True

        # Check for device-level memory recursively
        node = scope if isinstance(scope, nodes.EntryNode) else None
        cstate = scope if isinstance(scope, SDFGState) else state
        csdfg = scope if isinstance(scope, SDFG) else sdfg

        if desc.storage in dtypes.FPGA_STORAGES:
            return sdscope.is_devicelevel_fpga(csdfg, cstate, node)
        elif desc.storage in dtypes.GPU_STORAGES:
            return sdscope.is_devicelevel_gpu(csdfg, cstate, node)

        return False

    def determine_allocation_lifetime(self, top_sdfg: SDFG):
        """
        Determines where (at which scope/state/SDFG) each data descriptor
        will be allocated/deallocated.
        :param top_sdfg: The top-level SDFG to determine for.
        """
        # Gather shared transients, free symbols, and first/last appearance
        shared_transients = {}
        fsyms = {}
        reachability = StateReachability().apply_pass(top_sdfg, {})
        access_instances: Dict[int, Dict[str, List[Tuple[SDFGState, nodes.AccessNode]]]] = {}
        for sdfg in top_sdfg.all_sdfgs_recursive():
            shared_transients[sdfg.sdfg_id] = sdfg.shared_transients(check_toplevel=False)
            fsyms[sdfg.sdfg_id] = self.symbols_and_constants(sdfg)

            #############################################
            # Look for all states in which a scope-allocated array is used in
            instances: Dict[str, List[Tuple[SDFGState, nodes.AccessNode]]] = collections.defaultdict(list)
            array_names = sdfg.arrays.keys(
            )  #set(k for k, v in sdfg.arrays.items() if v.lifetime == dtypes.AllocationLifetime.Scope)
            # Iterate topologically to get state-order
            for state in sdfg.topological_sort():
                for node in state.data_nodes():
                    if node.data not in array_names:
                        continue
                    instances[node.data].append((state, node))

                # Look in the surrounding edges for usage
                edge_fsyms: Set[str] = set()
                for e in sdfg.all_edges(state):
                    edge_fsyms |= e.data.free_symbols
                for edge_array in edge_fsyms & array_names:
                    instances[edge_array].append((state, nodes.AccessNode(edge_array)))
            #############################################

            access_instances[sdfg.sdfg_id] = instances

        for sdfg, name, desc in top_sdfg.arrays_recursive():
            if not desc.transient:
                continue
            if name in sdfg.constants_prop:
                # Constants do not need to be allocated
                continue

            # NOTE: In the code below we infer where a transient should be
            # declared, allocated, and deallocated. The information is stored
            # in the `to_allocate` dictionary. The key of each entry is the
            # scope where one of the above actions must occur, while the value
            # is a tuple containing the following information:
            # 1. The SDFG object that containts the transient.
            # 2. The State id where the action should (approx.) take place.
            # 3. The Access Node id of the transient in the above State.
            # 4. True if declaration should take place, otherwise False.
            # 5. True if allocation should take place, otherwise False.
            # 6. True if deallocation should take place, otherwise False.

            first_state_instance, first_node_instance = \
                access_instances[sdfg.sdfg_id].get(name, [(None, None)])[0]
            last_state_instance, last_node_instance = \
                access_instances[sdfg.sdfg_id].get(name, [(None, None)])[-1]

            # Cases
            if desc.lifetime is dtypes.AllocationLifetime.Persistent:
                # Persistent memory is allocated in initialization code and
                # exists in the library state structure

                # If unused, skip
                if first_node_instance is None:
                    continue

                definition = desc.as_arg(name=f'__{sdfg.sdfg_id}_{name}') + ';'
                self.statestruct.append(definition)

                self.to_allocate[top_sdfg].append((sdfg, first_state_instance, first_node_instance, True, True, True))
                self.where_allocated[(sdfg, name)] = top_sdfg
                continue
            elif desc.lifetime is dtypes.AllocationLifetime.Global:
                # Global memory is allocated in the beginning of the program
                # exists in the library state structure (to be passed along
                # to the right SDFG)

                # If unused, skip
                if first_node_instance is None:
                    continue

                definition = desc.as_arg(name=f'__{sdfg.sdfg_id}_{name}') + ';'
                self.statestruct.append(definition)

                self.to_allocate[top_sdfg].append((sdfg, first_state_instance, first_node_instance, True, True, True))
                self.where_allocated[(sdfg, name)] = top_sdfg
                continue

            # The rest of the cases change the starting scope we attempt to
            # allocate from, since the descriptors may only be allocated higher
            # in the hierarchy (e.g., in the case of GPU global memory inside
            # a kernel).
            alloc_scope: Union[nodes.EntryNode, SDFGState, SDFG] = None
            alloc_state: SDFGState = None
            if (name in shared_transients[sdfg.sdfg_id] or desc.lifetime is dtypes.AllocationLifetime.SDFG):
                # SDFG descriptors are allocated in the beginning of their SDFG
                alloc_scope = sdfg
                if first_state_instance is not None:
                    alloc_state = first_state_instance
                # If unused, skip
                if first_node_instance is None:
                    continue
            elif desc.lifetime == dtypes.AllocationLifetime.State:
                # State memory is either allocated in the beginning of the
                # containing state or the SDFG (if used in more than one state)
                curstate: SDFGState = None
                multistate = False
                for state in sdfg.nodes():
                    if any(n.data == name for n in state.data_nodes()):
                        if curstate is not None:
                            multistate = True
                            break
                        curstate = state
                if multistate:
                    alloc_scope = sdfg
                else:
                    alloc_scope = curstate
                    alloc_state = curstate
            elif desc.lifetime == dtypes.AllocationLifetime.Scope:
                # Scope memory (default) is either allocated in the innermost
                # scope (e.g., Map, Consume) it is used in (i.e., greatest
                # common denominator), or in the SDFG if used in multiple states
                curscope: Union[nodes.EntryNode, SDFGState] = None
                curstate: SDFGState = None
                multistate = False

                # Does the array appear in inter-state edges?
                for isedge in sdfg.edges():
                    if name in self.free_symbols(isedge.data):
                        multistate = True

                for state in sdfg.nodes():
                    if multistate:
                        break
                    sdict = state.scope_dict()
                    for node in state.nodes():
                        if not isinstance(node, nodes.AccessNode):
                            continue
                        if node.data != name:
                            continue

                        # If already found in another state, set scope to SDFG
                        if curstate is not None and curstate != state:
                            multistate = True
                            break
                        curstate = state

                        # Current scope (or state object if top-level)
                        scope = sdict[node] or state
                        if curscope is None:
                            curscope = scope
                            continue
                        # States always win
                        if isinstance(scope, SDFGState):
                            curscope = scope
                            continue
                        # Lower/Higher/Disjoint scopes: find common denominator
                        if isinstance(curscope, SDFGState):
                            if scope in curscope.nodes():
                                continue
                        curscope = sdscope.common_parent_scope(sdict, scope, curscope)

                    if multistate:
                        break

                if multistate:
                    alloc_scope = sdfg
                else:
                    alloc_scope = curscope
                    alloc_state = curstate
            else:
                raise TypeError('Unrecognized allocation lifetime "%s"' % desc.lifetime)

            if alloc_scope is None:  # No allocation necessary
                continue

            # If descriptor cannot be allocated in this scope, traverse up the
            # scope tree until it is possible
            cursdfg = sdfg
            curstate = alloc_state
            curscope = alloc_scope
            while not self._can_allocate(cursdfg, curstate, desc, curscope):
                if curscope is None:
                    break
                if isinstance(curscope, nodes.EntryNode):
                    # Go one scope up
                    curscope = curstate.entry_node(curscope)
                    if curscope is None:
                        curscope = curstate
                elif isinstance(curscope, (SDFGState, SDFG)):
                    cursdfg: SDFG = (curscope if isinstance(curscope, SDFG) else curscope.parent)
                    # Go one SDFG up
                    if cursdfg.parent_nsdfg_node is None:
                        curscope = None
                        curstate = None
                        cursdfg = None
                    else:
                        curstate = cursdfg.parent
                        curscope = curstate.entry_node(cursdfg.parent_nsdfg_node)
                        cursdfg = cursdfg.parent_sdfg
                else:
                    raise TypeError

            if curscope is None:
                curscope = top_sdfg

            # Check if Array/View is dependent on non-free SDFG symbols
            # NOTE: Tuple is (SDFG, State, Node, declare, allocate, deallocate)
            fsymbols = fsyms[sdfg.sdfg_id]
            if (not isinstance(curscope, nodes.EntryNode)
                    and utils.is_nonfree_sym_dependent(first_node_instance, desc, first_state_instance, fsymbols)):
                # Allocate in first State, deallocate in last State
                if first_state_instance != last_state_instance:
                    # If any state is not reachable from first state, find common denominators in the form of
                    # dominator and postdominator.
                    instances = access_instances[sdfg.sdfg_id][name]
                    if any(inst not in reachability[sdfg.sdfg_id][first_state_instance] for inst in instances):
                        first_state_instance, last_state_instance = _get_dominator_and_postdominator(sdfg, instances)
                        # Declare in SDFG scope
                        # NOTE: Even if we declare the data at a common dominator, we keep the first and last node
                        # instances. This is especially needed for Views which require both the SDFGState and the
                        # AccessNode.
                        self.to_allocate[curscope].append((sdfg, None, nodes.AccessNode(name), True, False, False))
                    else:
                        self.to_allocate[curscope].append(
                            (sdfg, first_state_instance, first_node_instance, True, False, False))

                    curscope = first_state_instance
                    self.to_allocate[curscope].append(
                        (sdfg, first_state_instance, first_node_instance, False, True, False))
                    curscope = last_state_instance
                    self.to_allocate[curscope].append(
                        (sdfg, last_state_instance, last_node_instance, False, False, True))
                else:
                    curscope = first_state_instance
                    self.to_allocate[curscope].append(
                        (sdfg, first_state_instance, first_node_instance, True, True, True))
            else:
                self.to_allocate[curscope].append((sdfg, first_state_instance, first_node_instance, True, True, True))
            if isinstance(curscope, SDFG):
                self.where_allocated[(sdfg, name)] = curscope
            else:
                self.where_allocated[(sdfg, name)] = cursdfg

    def allocate_arrays_in_scope(self, sdfg: SDFG, scope: Union[nodes.EntryNode, SDFGState, SDFG],
                                 function_stream: CodeIOStream, callsite_stream: CodeIOStream):
        """ Dispatches allocation of all arrays in the given scope. """
        for tsdfg, state, node, declare, allocate, _ in self.to_allocate[scope]:
            if state is not None:
                state_id = tsdfg.node_id(state)
            else:
                state_id = -1

            desc = node.desc(tsdfg)

            self._dispatcher.dispatch_allocate(tsdfg, state, state_id, node, desc, function_stream, callsite_stream,
                                               declare, allocate)

    def deallocate_arrays_in_scope(self, sdfg: SDFG, scope: Union[nodes.EntryNode, SDFGState, SDFG],
                                   function_stream: CodeIOStream, callsite_stream: CodeIOStream):
        """ Dispatches deallocation of all arrays in the given scope. """
        for tsdfg, state, node, _, _, deallocate in self.to_allocate[scope]:
            if not deallocate:
                continue
            if state is not None:
                state_id = tsdfg.node_id(state)
            else:
                state_id = -1

            desc = node.desc(tsdfg)

            self._dispatcher.dispatch_deallocate(tsdfg, state, state_id, node, desc, function_stream, callsite_stream)

    def generate_code(self,
                      sdfg: SDFG,
                      schedule: Optional[dtypes.ScheduleType],
                      sdfg_id: str = "") -> Tuple[str, str, Set[TargetCodeGenerator], Set[str]]:
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

        # Analyze allocation lifetime of SDFG and all nested SDFGs
        if is_top_level:
            self.determine_allocation_lifetime(sdfg)

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
                instr.on_sdfg_begin(sdfg, callsite_stream, global_stream, self)

        # Allocate outer-level transients
        self.allocate_arrays_in_scope(sdfg, sdfg, global_stream, callsite_stream)

        # Define constants as top-level-allocated
        for cname, (ctype, _) in sdfg.constants_prop.items():
            if isinstance(ctype, data.Array):
                self.dispatcher.defined_vars.add(cname, disp.DefinedType.Pointer, ctype.dtype.ctype)
            else:
                self.dispatcher.defined_vars.add(cname, disp.DefinedType.Scalar, ctype.dtype.ctype)

        # Allocate inter-state variables
        global_symbols = copy.deepcopy(sdfg.symbols)
        global_symbols.update({aname: arr.dtype for aname, arr in sdfg.arrays.items()})
        interstate_symbols = {}
        for e in sdfg.dfs_edges(sdfg.start_state):
            symbols = e.data.new_symbols(sdfg, global_symbols)
            # Inferred symbols only take precedence if global symbol not defined or None
            symbols = {
                k: v if (k not in global_symbols or global_symbols[k] is None) else global_symbols[k]
                for k, v in symbols.items()
            }
            interstate_symbols.update(symbols)
            global_symbols.update(symbols)

        for isvarName, isvarType in interstate_symbols.items():
            if isvarType is None:
                raise TypeError(f'Type inference failed for symbol {isvarName}')

            isvar = data.Scalar(isvarType)
            callsite_stream.write('%s;\n' % (isvar.as_arg(with_types=True, name=isvarName)), sdfg)
            self.dispatcher.defined_vars.add(isvarName, disp.DefinedType.Scalar, isvarType.ctype)

        callsite_stream.write('\n', sdfg)

        #######################################################################
        # Generate actual program body

        states_generated = self.generate_states(sdfg, global_stream, callsite_stream)

        #######################################################################

        # Sanity check
        if len(states_generated) != len(sdfg.nodes()):
            raise RuntimeError(
                "Not all states were generated in SDFG {}!"
                "\n  Generated: {}\n  Missing: {}".format(sdfg.label, [s.label for s in states_generated],
                                                          [s.label for s in (set(sdfg.nodes()) - states_generated)]))

        # Deallocate transients
        self.deallocate_arrays_in_scope(sdfg, sdfg, global_stream, callsite_stream)

        # Now that we have all the information about dependencies, generate
        # header and footer
        if is_top_level:
            header_stream = CodeIOStream()
            header_global_stream = CodeIOStream()
            footer_stream = CodeIOStream()
            footer_global_stream = CodeIOStream()

            # Get all environments used in the generated code, including
            # dependent environments
            import dace.library  # Avoid import loops
            self.environments = dace.library.get_environments_and_dependencies(self._dispatcher.used_environments)

            self.generate_header(sdfg, header_global_stream, header_stream)

            # Open program function
            params = sdfg.signature(arglist=self.arglist)
            if params:
                params = ', ' + params
            function_signature = ('void __program_%s_internal(%s_t *__state%s)\n{\n' % (sdfg.name, sdfg.name, params))

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
        gotos = re.findall(r'goto (.*?);', generated_code)
        clean_code = ''
        for line in generated_code.split('\n'):
            # Empty line with semicolon
            if re.match(r'^\s*;\s*', line):
                continue
            # Label that might be unused
            label = re.findall(r'^\s*([a-zA-Z_][a-zA-Z_0-9]*):\s*[;]?\s*////.*$', line)
            if len(label) > 0:
                if label[0] not in gotos:
                    continue
            clean_code += line + '\n'

        # Return the generated global and local code strings
        return (generated_header, clean_code, self._dispatcher.used_targets, self._dispatcher.used_environments)


def _get_dominator_and_postdominator(sdfg: SDFG, accesses: List[Tuple[SDFGState, nodes.AccessNode]]):
    """
    Gets the closest common dominator and post-dominator for a list of states.
    Used for determining allocation of data used in branched states.
    """
    from dace.sdfg.analysis import cfg

    # Get immediate dominators
    idom = nx.immediate_dominators(sdfg.nx, sdfg.start_state)
    alldoms = cfg.all_dominators(sdfg, idom)

    states = [a for a, _ in accesses]
    data_name = accesses[0][1].data

    # Get immediate post-dominators
    ipostdom, allpostdoms = utils.postdominators(sdfg, return_alldoms=True)

    # All dominators and postdominators include the states themselves
    for state in states:
        alldoms[state].add(state)
        allpostdoms[state].add(state)

    start_state = states[0]
    while any(start_state not in alldoms[n] for n in states):
        if idom[start_state] is start_state:
            raise NotImplementedError(f'Could not find an appropriate dominator for allocation of "{data_name}"')
        start_state = idom[start_state]

    end_state = states[-1]
    while any(end_state not in allpostdoms[n] for n in states):
        if ipostdom[end_state] is end_state:
            raise NotImplementedError(f'Could not find an appropriate post-dominator for deallocation of "{data_name}"')
        end_state = ipostdom[end_state]

    # TODO(later): If any of the symbols were not yet defined, or have changed afterwards, fail
    # raise NotImplementedError

    return start_state, end_state
