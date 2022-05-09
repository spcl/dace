# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
""" Implements the PAPI counter performance instrumentation provider.
    Used for collecting CPU performance counters. """

import dace
from dace import dtypes, registry, symbolic
from dace.codegen.instrumentation.provider import InstrumentationProvider
from dace.codegen.targets.common import sym2cpp
from dace.config import Config
from dace.sdfg import nodes
from dace.sdfg.nodes import EntryNode, MapEntry, MapExit, Tasklet
from dace.sdfg.graph import SubgraphView
from dace.memlet import Memlet
from dace.sdfg import scope_contains_scope
from dace.sdfg.state import StateGraphView

import sympy as sp
import os
import ast
import subprocess
from typing import Dict, List, Optional, Set
import warnings

# Default sets of PAPI counters
VECTOR_COUNTER_SET = ('0x40000025', '0x40000026', '0x40000027', '0x40000028', '0x40000021', '0x40000022', '0x40000023',
                      '0x40000024')
MEM_COUNTER_SET = ('PAPI_MEM_WCY', 'PAPI_LD_INS', 'PAPI_SR_INS')
CACHE_COUNTER_SET = ('PAPI_CA_SNP', 'PAPI_CA_SHR', 'PAPI_CA_CLN', 'PAPI_CA_ITV')


def _unified_id(node_id: int, state_id: int) -> int:
    if node_id > 0x0FFFF:
        raise ValueError("Node ID is too large to fit in 16 bits!")
    if state_id > 0x0FFFF:
        raise ValueError("State ID is too large to fit in 16 bits!")
    return (int(state_id) << 16) | int(node_id)


@registry.autoregister_params(type=dtypes.InstrumentationType.PAPI_Counters)
class PAPIInstrumentation(InstrumentationProvider):
    """ Instrumentation provider that reports CPU performance counters using
        the PAPI library. """

    _counters: Optional[Set[str]] = None

    perf_whitelist_schedules = [dtypes.ScheduleType.CPU_Multicore, dtypes.ScheduleType.Sequential]

    def __init__(self):
        self._papi_used = False
        self._unique_counter = 0
        self.perf_should_instrument = False
        PAPIInstrumentation._counters = PAPIInstrumentation._counters or set(
            ast.literal_eval(Config.get('instrumentation', 'papi', 'default_counters')))

    def get_unique_number(self):
        ret = self._unique_counter
        self._unique_counter += 1
        return ret

    def configure_papi(self):
        # Get available PAPI metrics and intersect them with what we can measure
        counters = PAPIUtils.available_counters()
        missing_counters = self._counters - set(counters.keys())
        if len(missing_counters) > 0:
            warnings.warn('Skipping missing PAPI performance counters: %s' % missing_counters)
        PAPIInstrumentation._counters &= set(counters.keys())

        # Compiler arguments for vectorization output
        if Config.get_bool('instrumentation', 'papi', 'vectorization_analysis'):
            Config.append('compiler', 'cpu', 'args', value=' -fopt-info-vec-optimized-missed=../perf/vecreport.txt ')

        # If no PAPI counters are available, disable PAPI
        if len(self._counters) == 0:
            self._papi_used = False
            warnings.warn('No PAPI counters found. Disabling PAPI')
            return

        # Link with libpapi
        Config.append('compiler', 'cpu', 'libs', value=' papi ')

        self._papi_used = True

    def on_sdfg_begin(self, sdfg, local_stream, global_stream, codegen):
        if sdfg.parent is None and PAPIUtils.is_papi_used(sdfg):
            # Configure CMake project and counters
            self.configure_papi()

            if not self._papi_used:
                return

            # Add instrumentation includes and initialize PAPI
            global_stream.write('#include <dace/perf/papi.h>', sdfg)
            local_stream.write(
                '''dace::perf::PAPI::init();
dace::perf::PAPIValueStore<%s> __perf_store (__state->report);''' % (', '.join(self._counters)), sdfg)
            # Get the measured overhead and take the minimum to compensate
            if Config.get_bool('instrumentation', 'papi', 'overhead_compensation'):
                local_stream.write("__perf_store.getMeasuredOverhead();", sdfg)

    def on_sdfg_end(self, sdfg, local_stream, global_stream):
        if not self._papi_used:
            return

        local_stream.write('__perf_store.flush();', sdfg)

    def on_state_begin(self, sdfg, state, local_stream, global_stream):
        if not self._papi_used:
            return
        if state.instrument == dace.InstrumentationType.PAPI_Counters:
            uid = _unified_id(-1, sdfg.node_id(state))
            local_stream.write("__perf_store.markSuperSectionStart(%d);" % uid)

    def on_copy_begin(self, sdfg, state, src_node, dst_node, edge, local_stream, global_stream, copy_shape, src_strides,
                      dst_strides):
        if not self._papi_used:
            return

        state_id = sdfg.node_id(state)
        memlet = edge.data

        # For perfcounters, we have to make sure that:
        # 1) No other measurements are done for the containing scope (no map
        # operation containing this copy is instrumented)
        src_instrumented = PAPIInstrumentation.has_surrounding_perfcounters(src_node, state)
        dst_instrumented = PAPIInstrumentation.has_surrounding_perfcounters(dst_node, state)
        src_storage = src_node.desc(sdfg).storage
        dst_storage = dst_node.desc(sdfg).storage

        cpu_storage_types = [
            dtypes.StorageType.CPU_Heap,
            dtypes.StorageType.CPU_ThreadLocal,
            dtypes.StorageType.CPU_Pinned,
            dtypes.StorageType.Register,
        ]

        perf_cpu_only = (src_storage in cpu_storage_types) and (dst_storage in cpu_storage_types)

        self.perf_should_instrument = (not src_instrumented and not dst_instrumented and perf_cpu_only
                                       and state.instrument == dace.InstrumentationType.PAPI_Counters)

        if self.perf_should_instrument is False:
            return

        unique_cpy_id = self.get_unique_number()

        dst_nodedesc = dst_node.desc(sdfg)
        ctype = dst_nodedesc.dtype.ctype

        fac3 = (" * ".join(sym2cpp(copy_shape)) + " / " + "/".join(sym2cpp(dst_strides)))
        copy_size = "sizeof(%s) * (%s)" % (ctype, fac3)
        node_id = _unified_id(state.node_id(dst_node), state_id)
        # Mark a section start (this is not really a section in itself (it
        # would be a section with 1 entry))
        local_stream.write(
            self.perf_section_start_string(node_id, copy_size, copy_size),
            sdfg,
            state_id,
            [src_node, dst_node],
        )
        local_stream.write(
            '''
dace::perf::{pcs} __perf_cpy_{nodeid}_{unique_id};
auto& __vs_cpy_{nodeid}_{unique_id} = __perf_store.getNewValueSet(
    __perf_cpy_{nodeid}_{unique_id}, {nodeid}, PAPI_thread_id(), {size}, 
    dace::perf::ValueSetType::Copy);
__perf_cpy_{nodeid}_{unique_id}.enterCritical();'''.format(
                pcs=self.perf_counter_string(),
                nodeid=node_id,
                unique_id=unique_cpy_id,
                size=copy_size,
            ),
            sdfg,
            state_id,
            [src_node, dst_node],
        )

    def on_copy_end(self, sdfg, state, src_node, dst_node, edge, local_stream, global_stream):
        if not self._papi_used:
            return

        state_id = sdfg.node_id(state)
        node_id = state.node_id(dst_node)
        if self.perf_should_instrument:
            unique_cpy_id = self._unique_counter

            local_stream.write(
                "__perf_cpy_%d_%d.leaveCritical(__vs_cpy_%d_%d);" % (node_id, unique_cpy_id, node_id, unique_cpy_id),
                sdfg,
                state_id,
                [src_node, dst_node],
            )

            self.perf_should_instrument = False

    def on_node_begin(self, sdfg, state, node, outer_stream, inner_stream, global_stream):
        if not self._papi_used:
            return

        state_id = sdfg.node_id(state)
        unified_id = _unified_id(state.node_id(node), state_id)

        perf_should_instrument = (node.instrument == dace.InstrumentationType.PAPI_Counters
                                  and not PAPIInstrumentation.has_surrounding_perfcounters(node, state))
        if not perf_should_instrument:
            return

        if isinstance(node, nodes.Tasklet):
            inner_stream.write(
                "dace::perf::%s __perf_%s;\n" % (self.perf_counter_string(), node.label),
                sdfg,
                state_id,
                node,
            )
            inner_stream.write(
                'auto& __perf_vs_%s = __perf_store.getNewValueSet(__perf_%s, '
                '    %d, PAPI_thread_id(), 0);\n' % (node.label, node.label, unified_id),
                sdfg,
                state_id,
                node,
            )

            inner_stream.write("__perf_%s.enterCritical();\n" % node.label, sdfg, state_id, node)

    def on_node_end(self, sdfg, state, node, outer_stream, inner_stream, global_stream):
        if not self._papi_used:
            return

        state_id = sdfg.node_id(state)
        node_id = state.node_id(node)
        unified_id = _unified_id(node_id, state_id)

        if isinstance(node, nodes.CodeNode):
            if node.instrument == dace.InstrumentationType.PAPI_Counters:
                if not PAPIInstrumentation.has_surrounding_perfcounters(node, state):
                    inner_stream.write(
                        "__perf_%s.leaveCritical(__perf_vs_%s);" % (node.label, node.label),
                        sdfg,
                        state_id,
                        node,
                    )

                # Add bytes moved
                inner_stream.write(
                    "__perf_store.addBytesMoved(%s);" %
                    PAPIUtils.get_tasklet_byte_accesses(node, state, sdfg, state_id), sdfg, state_id, node)

    def on_scope_entry(self, sdfg, state, node, outer_stream, inner_stream, global_stream):
        if not self._papi_used:
            return
        if isinstance(node, nodes.MapEntry):
            return self.on_map_entry(sdfg, state, node, outer_stream, inner_stream)
        elif isinstance(node, nodes.ConsumeEntry):
            return self.on_consume_entry(sdfg, state, node, outer_stream, inner_stream)
        raise TypeError

    def on_scope_exit(self, sdfg, state, node, outer_stream, inner_stream, global_stream):
        if not self._papi_used:
            return
        state_id = sdfg.node_id(state)
        entry_node = state.entry_node(node)
        if not self.should_instrument_entry(entry_node):
            return

        unified_id = _unified_id(state.node_id(entry_node), state_id)
        perf_end_string = self.perf_counter_end_measurement_string(unified_id)

        # Inner part
        inner_stream.write(perf_end_string, sdfg, state_id, node)

    def on_map_entry(self, sdfg, state, node, outer_stream, inner_stream):
        dfg = state.scope_subgraph(node)
        state_id = sdfg.node_id(state)
        if node.map.instrument != dace.InstrumentationType.PAPI_Counters:
            return

        unified_id = _unified_id(dfg.node_id(node), state_id)

        #########################################################
        # Outer part

        result = outer_stream

        input_size: str = PAPIUtils.get_memory_input_size(node, sdfg, state_id)

        # Emit supersection if possible
        result.write(self.perf_get_supersection_start_string(node, dfg, unified_id))

        if not self.should_instrument_entry(node):
            return

        size = PAPIUtils.accumulate_byte_movement(node, node, dfg, sdfg, state_id)
        size = sym2cpp(sp.simplify(size))

        result.write(self.perf_section_start_string(unified_id, size, input_size))

        #########################################################
        # Inner part
        result = inner_stream

        map_name = node.map.params[-1]

        result.write(self.perf_counter_start_measurement_string(unified_id, map_name), sdfg, state_id, node)

    def on_consume_entry(self, sdfg, state, node, outer_stream, inner_stream):
        state_id = sdfg.node_id(state)
        unified_id = _unified_id(state.node_id(node), state_id)

        # Outer part
        result = outer_stream

        if self.should_instrument_entry(node):
            # Mark the SuperSection start (if possible)
            result.write(
                self.perf_get_supersection_start_string(node, state, unified_id),
                sdfg,
                state_id,
                node,
            )

            # Mark the section start with zeros (due to dynamic accesses)
            result.write(self.perf_section_start_string(unified_id, "0", "0"), sdfg, state_id, node)

            # Generate a thread affinity locker
            result.write(
                "dace::perf::ThreadLockProvider __perf_tlp_%d;\n" % unified_id,
                sdfg,
                state_id,
                node,
            )

        # Inner part
        result = inner_stream

        # Since the consume internally creates threads, it is instrumented like
        # a map. However, since std::threads are used rather than OpenMP,
        # this implementation only allows to measure on a per-task basis
        # (instead of per-thread). This incurs additional overhead.
        if self.should_instrument_entry(node):
            result.write(
                ("auto __perf_tlp_{id}_releaser = __perf_tlp_{id}.enqueue();\n".format(id=unified_id)) +
                self.perf_counter_start_measurement_string(
                    unified_id,
                    "__perf_tlp_{id}.getAndIncreaseCounter()".format(id=unified_id),
                    core_str="dace::perf::getThreadID()",
                ),
                sdfg,
                state_id,
                node,
            )

    @staticmethod
    def perf_get_supersection_start_string(node, dfg, unified_id):
        if node.map.schedule == dtypes.ScheduleType.CPU_Multicore:
            # Nested SuperSections are not supported. Therefore, we mark the
            # outermost section and disallow internal scopes from creating it.
            if not hasattr(node.map, '_can_be_supersection_start'):
                node.map._can_be_supersection_start = True

            children = PAPIUtils.all_maps(node, dfg)
            for x in children:
                if not hasattr(x.map, '_can_be_supersection_start'):
                    x.map._can_be_supersection_start = True
                if x.map.schedule == dtypes.ScheduleType.CPU_Multicore:

                    x.map._can_be_supersection_start = False
                elif x.map.schedule == dtypes.ScheduleType.Sequential:
                    x.map._can_be_supersection_start = False
                else:
                    # Any other type (FPGA, GPU) - not supported by PAPI.
                    x.map._can_be_supersection_start = False

            if (node.map._can_be_supersection_start and not dace.sdfg.is_parallel(dfg)):
                return "__perf_store.markSuperSectionStart(%d);\n" % unified_id

        elif (getattr(node.map, '_can_be_supersection_start', False) and not dace.sdfg.is_parallel(dfg)):
            return "__perf_store.markSuperSectionStart(%d);\n" % unified_id

        # Otherwise, do nothing (empty string)
        return ""

    @staticmethod
    def should_instrument_entry(map_entry: EntryNode) -> bool:
        """ Returns True if this entry node should be instrumented. """
        if map_entry.map.instrument != dace.InstrumentationType.PAPI_Counters:
            return False
        if (map_entry.map.schedule not in PAPIInstrumentation.perf_whitelist_schedules):
            return False
        try:
            cond = not map_entry.fence_instrumentation
        except (AttributeError, NameError):
            cond = True
        return cond

    @staticmethod
    def has_surrounding_perfcounters(node, dfg: StateGraphView):
        """ Returns true if there is a possibility that this node is part of a
            section that is profiled. """
        parent = dfg.entry_node(node)

        if isinstance(parent, MapEntry):
            if (parent.map.schedule not in PAPIInstrumentation.perf_whitelist_schedules):
                return False
            return True

        return False

    @staticmethod
    def perf_counter_string_from_string_list(counterlist: [str]):
        """ Creates a performance counter typename string. """
        if isinstance(counterlist, str):
            print("Wrong format")
            counterlist = eval(counterlist)
        return "PAPIPerf<" + ", ".join(counterlist) + ">"

    def perf_counter_string(self):
        """
        Creates a performance counter template string.
        """
        return PAPIInstrumentation.perf_counter_string_from_string_list(self._counters)

    def perf_counter_start_measurement_string(self,
                                              unified_id: int,
                                              iteration: str,
                                              core_str: str = "PAPI_thread_id()"):
        pcs = self.perf_counter_string()
        return '''dace::perf::{counter_str} __perf_{id};
auto& __vs_{id} = __perf_store.getNewValueSet(__perf_{id}, {id}, {core}, {it});
__perf_{id}.enterCritical();        
        '''.format(counter_str=pcs, id=unified_id, it=iteration, core=core_str)

    @staticmethod
    def perf_counter_end_measurement_string(unified_id):
        return '__perf_{id}.leaveCritical(__vs_{id});\n'.format(id=unified_id)

    @staticmethod
    def perf_section_start_string(unified_id: int, size: str, in_size: str, core_str: str = "PAPI_thread_id()"):
        return '''
__perf_store.markSectionStart(%d, (long long)%s, (long long)%s, %s);''' % (unified_id, size, in_size, core_str)

    @staticmethod
    def perf_supersection_start_string(unified_id):
        return '__perf_store.markSuperSectionStart(%d);\n' % unified_id


class PAPIUtils(object):
    """ General-purpose utilities for working with PAPI. """
    @staticmethod
    def available_counters() -> Dict[str, int]:
        """
        Returns the available PAPI counters on this machine. Only works on
        *nix based systems with ``grep`` and ``papi-tools`` installed.
        :return: A set of available PAPI counters in the form of a dictionary
                 mapping from counter name to the number of native hardware
                 events.
        """
        if os.name == 'nt':
            return {}

        try:
            p = subprocess.Popen("papi_avail -d -a | grep -E '^PAPI_'",
                                 shell=True,
                                 stdout=subprocess.PIPE,
                                 stderr=subprocess.STDOUT,
                                 universal_newlines=True)
            stdout, _ = p.communicate(timeout=60)
        except (subprocess.TimeoutExpired, subprocess.CalledProcessError):
            return {}

        # Analyze counters
        counters = [line.split('\t') for line in stdout.split('\n')]
        result = {}
        for counter in counters:
            if len(counter) >= 3:
                result[counter[0]] = int(counter[2])

        return result

    @staticmethod
    def is_papi_used(sdfg: dace.SDFG) -> bool:
        """ Returns True if any of the SDFG elements includes PAPI counter
            instrumentation. """
        for node, _ in sdfg.all_nodes_recursive():
            if isinstance(node, nodes.EntryNode) and node.map.instrument == dace.InstrumentationType.PAPI_Counters:
                return True
            if hasattr(node, 'instrument') and node.instrument == dace.InstrumentationType.PAPI_Counters:
                return True
        return False

    @staticmethod
    def reduce_iteration_count(begin, end, step, rparams: dict):
        # There are different rules when expanding depending on where the expand
        # should happen

        if isinstance(begin, int):
            start_syms = []
        else:
            start_syms = symbolic.symlist(begin).keys()

        if isinstance(end, int):
            end_syms = []
        else:
            end_syms = symbolic.symlist(end).keys()

        if isinstance(step, int):
            step_syms = []
        else:
            step_syms = symbolic.symlist(step).keys()

        def intersection(lista, listb):
            return [x for x in lista if x in listb]

        start_dyn_syms = intersection(start_syms, rparams.keys())
        end_dyn_syms = intersection(end_syms, rparams.keys())
        step_dyn_syms = intersection(step_syms, rparams.keys())

        def replace_func(element, dyn_syms, retparams):
            # Resolve all symbols using the retparams-dict

            for x in dyn_syms:
                target = sp.functions.Min(retparams[x] * (retparams[x] - 1) / 2, 0)
                bstr = str(element)
                element = symbolic.pystr_to_symbolic(bstr)
                element = element.subs(x, target)  # Add the classic sum formula; going upwards

                # To not have hidden elements that get added again later, we
                # also replace the values in the other itvars...
                for k, v in retparams.items():
                    newv = symbolic.pystr_to_symbolic(str(v))

                    tarsyms = symbolic.symlist(target).keys()
                    if x in tarsyms:
                        continue

                    tmp = newv.subs(x, target)
                    if tmp != v:
                        retparams[k] = tmp

            return element

        if len(start_dyn_syms) > 0:
            pass
            begin = replace_func(begin, start_dyn_syms, rparams)

        if len(end_dyn_syms) > 0:
            pass
            end = replace_func(end, end_dyn_syms, rparams)

        if len(step_dyn_syms) > 0:
            pass
            print("Dynamic step symbols %s!" % str(step))
            raise NotImplementedError

        return begin, end, step

    @staticmethod
    def get_iteration_count(map_entry: MapEntry, mapvars: dict):
        """
        Get the number of iterations for this map, allowing other variables
        as bounds.
        """

        _map = map_entry.map
        _it = _map.params

        retparams = dict(**mapvars)

        for i, r in enumerate(_map.range):
            begin, end, step = r

            end = end + 1  # end is inclusive, but we want it exclusive

            if isinstance(begin, symbolic.SymExpr):
                begin = begin.expr
            if isinstance(end, symbolic.SymExpr):
                end = end.expr
            if isinstance(step, symbolic.SymExpr):
                step = step.expr

            begin, end, step = PAPIUtils.reduce_iteration_count(begin, end, step, retparams)
            num = (end - begin) / step  # The count of iterations
            retparams[_it[i]] = num

        return retparams

    @staticmethod
    def all_maps(map_entry: EntryNode, dfg: SubgraphView) -> List[EntryNode]:
        """ Returns all scope entry nodes within a scope entry. """
        state: dace.SDFGState = dfg.graph
        subgraph = state.scope_subgraph(map_entry, include_entry=False)
        return [n for n in subgraph.nodes() if isinstance(n, EntryNode)]

    @staticmethod
    def get_memlet_byte_size(sdfg: dace.SDFG, memlet: Memlet):
        """
        Returns the memlet size in bytes, depending on its data type.
        :param sdfg: The SDFG in which the memlet resides.
        :param memlet: Memlet to return size in bytes.
        :return: The size as a symbolic expression.
        """
        if memlet.dynamic:  # Ignoring dynamic accesses
            return 0
        memdata = sdfg.arrays[memlet.data]
        return memlet.volume * memdata.dtype.bytes

    @staticmethod
    def get_out_memlet_costs(sdfg: dace.SDFG, state_id: int, node: nodes.Node, dfg: StateGraphView):
        scope_dict = sdfg.node(state_id).scope_dict()

        out_costs = 0
        for edge in dfg.out_edges(node):
            _, uconn, v, _, memlet = edge
            dst_node = dfg.memlet_path(edge)[-1].dst

            if (isinstance(node, nodes.CodeNode) and isinstance(dst_node, nodes.AccessNode)):

                # If the memlet is pointing into an array in an inner scope,
                # it will be handled by the inner scope.
                if (scope_dict[node] != scope_dict[dst_node] and scope_contains_scope(scope_dict, node, dst_node)):
                    continue

                if not uconn:
                    # This would normally raise a syntax error
                    return 0

                if memlet.subset.data_dims() == 0:
                    if memlet.wcr is not None:
                        # write_and_resolve
                        # We have to assume that every reduction costs 3
                        # accesses of the same size (read old, read new, write)
                        out_costs += 3 * PAPIUtils.get_memlet_byte_size(sdfg, memlet)
                    else:
                        # This standard operation is already counted
                        out_costs += PAPIUtils.get_memlet_byte_size(sdfg, memlet)
        return out_costs

    @staticmethod
    def get_tasklet_byte_accesses(tasklet: nodes.CodeNode, dfg: StateGraphView, sdfg: dace.SDFG, state_id: int) -> str:
        """ Get the amount of bytes processed by `tasklet`. The formula is
            sum(inedges * size) + sum(outedges * size) """
        in_accum = []
        out_accum = []
        in_edges = dfg.in_edges(tasklet)

        for ie in in_edges:
            in_accum.append(PAPIUtils.get_memlet_byte_size(sdfg, ie.data))

        out_accum.append(PAPIUtils.get_out_memlet_costs(sdfg, state_id, tasklet, dfg))

        # Merge
        full = in_accum
        full.extend(out_accum)

        return "(" + sym2cpp(sum(full)) + ")"

    @staticmethod
    def get_parents(outermost_node: nodes.Node, node: nodes.Node, sdfg: dace.SDFG, state_id: int) -> List[nodes.Node]:

        parent = None
        # Because dfg is only a subgraph view, it does not contain the entry
        # node for a given entry. This O(n) solution is suboptimal
        for state in sdfg.nodes():
            s_d = state.scope_dict()
            try:
                scope = s_d[node]
            except KeyError:
                continue

            if scope is not None:
                parent = scope
                break
        if parent is None:
            return []
        if parent == outermost_node:
            return [parent]

        return PAPIUtils.get_parents(outermost_node, parent, sdfg, state_id) + [parent]

    @staticmethod
    def get_memory_input_size(node, sdfg, state_id) -> str:
        curr_state = sdfg.nodes()[state_id]

        input_size = 0
        for edge in curr_state.in_edges(node):
            # Accumulate over range size and get the amount of data accessed
            num_accesses = edge.data.num_accesses

            # It might be better to just take the source object size
            bytes_per_element = sdfg.arrays[edge.data.data].dtype.bytes
            input_size = input_size + (bytes_per_element * num_accesses)

        return sym2cpp(input_size)

    @staticmethod
    def accumulate_byte_movement(outermost_node, node, dfg: StateGraphView, sdfg, state_id):

        itvars = dict()  # initialize an empty dict

        # First, get a list of children
        if isinstance(node, MapEntry):
            children = dfg.scope_children()[node]
        else:
            children = []
        assert not (node in children)

        # If there still are children, descend recursively (dfs is fine here)
        if len(children) > 0:
            size = 0
            for x in children:
                size = size + PAPIUtils.accumulate_byte_movement(outermost_node, x, dfg, sdfg, state_id)

            return size
        else:
            if isinstance(node, MapExit):
                return 0  # We can ignore this.

            # If we reached the deepest node, get all parents
            parent_list = PAPIUtils.get_parents(outermost_node, node, sdfg, state_id)
            if isinstance(node, MapEntry):
                map_list = parent_list + [node]
            else:
                map_list = parent_list

            # From all iterations, get the iteration count, replacing inner
            # iteration variables with the next outer variables.
            for x in map_list:
                itvars = PAPIUtils.get_iteration_count(x, itvars)

            itcount = 1
            for x in itvars.values():
                itcount = itcount * x

            if isinstance(node, MapEntry):
                raise ValueError("Unexpected node")  # A map entry should never be the innermost node
            elif isinstance(node, MapExit):
                return 0  # We can ignore this.
            elif isinstance(node, Tasklet):
                return itcount * symbolic.pystr_to_symbolic(
                    PAPIUtils.get_tasklet_byte_accesses(node, dfg, sdfg, state_id))
            elif isinstance(node, nodes.AccessNode):
                return 0
            else:
                raise NotImplementedError
