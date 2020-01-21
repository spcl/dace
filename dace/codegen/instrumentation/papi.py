""" Implements the PAPI counter performance instrumentation provider.
    Used for collecting CPU performance counters. """

import dace
from dace import dtypes, symbolic
from dace.codegen.instrumentation.provider import InstrumentationProvider
from dace.codegen.targets.common import sym2cpp
from dace.config import Config
from dace.data import Array
from dace.graph import nodes
from dace.graph.nodes import (EntryNode, MapEntry, MapExit, Tasklet,
                              ConsumeEntry)
from dace.graph.graph import SubgraphView
from dace.memlet import Memlet
from dace.sdfg import scope_contains_scope

from functools import reduce
import sympy as sp
import operator
import os
import ast
import subprocess
from typing import Dict, Optional, Set
import warnings

# Default sets of PAPI counters
VECTOR_COUNTER_SET = ('0x40000025', '0x40000026', '0x40000027', '0x40000028',
                      '0x40000021', '0x40000022', '0x40000023', '0x40000024')
MEM_COUNTER_SET = ('PAPI_MEM_WCY', 'PAPI_LD_INS', 'PAPI_SR_INS')
CACHE_COUNTER_SET = ('PAPI_CA_SNP', 'PAPI_CA_SHR', 'PAPI_CA_CLN',
                     'PAPI_CA_ITV')


class PAPIInstrumentation(InstrumentationProvider):
    """ Instrumentation provider that reports CPU performance counters using
        the PAPI library. """

    _counters: Optional[Set[str]] = None

    perf_whitelist_schedules = [
        dtypes.ScheduleType.CPU_Multicore, dtypes.ScheduleType.Sequential
    ]

    def __init__(self):
        self._papi_used = False
        self._unique_counter = 0
        self.perf_should_instrument = False
        PAPIInstrumentation._counters = PAPIInstrumentation._counters or set(
            ast.literal_eval(
                Config.get('instrumentation', 'papi', 'default_counters')))

    def get_unique_number(self):
        ret = self._unique_counter
        self._unique_counter += 1
        return ret

    def configure_papi(self):
        # Get available PAPI metrics and intersect them with what we can measure
        counters = PAPIInstrumentation.available_counters()
        missing_counters = self._counters - set(counters.keys())
        if len(missing_counters) > 0:
            warnings.warn('Skipping missing PAPI performance counters: %s' %
                          missing_counters)
        PAPIInstrumentation._counters &= set(counters.keys())

        # Compiler arguments for vectorization output
        if Config.get_bool('instrumentation', 'papi',
                           'vectorization_analysis'):
            Config.append(
                'compiler',
                'cpu',
                'additional_args',
                value=' -fopt-info-vec-optimized-missed=../perf/vecreport.txt '
            )

        # If no PAPI counters are available, disable PAPI
        if len(self._counters) == 0:
            self._papi_used = False
            warnings.warn('No PAPI counters found. Disabling PAPI')
            return

        # Link with libpapi
        Config.append('compiler', 'cpu', 'libs', value=' papi ')

        self._papi_used = True

    def on_sdfg_begin(self, sdfg, local_stream, global_stream):
        if sdfg.parent is None and PAPIUtils.is_papi_used(sdfg):
            # Configure CMake project and counters
            self.configure_papi()

            if not self._papi_used:
                return

            # Add instrumentation includes and initialize PAPI
            global_stream.write('#include <dace/perf/papi.h>', sdfg)
            local_stream.write(
                '''dace::perf::PAPI::init();
dace::perf::PAPIValueStore<%s> __perf_store (dace::perf::report);''' %
                (', '.join(self._counters)), sdfg)
            # Get the measured overhead and take the minimum to compensate
            if Config.get_bool('instrumentation', 'papi',
                               'overhead_compensation'):
                local_stream.write("__perf_store.getMeasuredOverhead();", sdfg)

    def on_sdfg_end(self, sdfg, local_stream, global_stream):
        pass

    @staticmethod
    def unified_id(node_id, state_id):
        if node_id > 0x0FFFF:
            raise ValueError("Node ID is too large to fit in 16 bits!")
        if state_id > 0x0FFFF:
            raise ValueError("State ID is too large to fit in 16 bits!")
        return (int(state_id) << 16) | int(node_id)

    def on_state_begin(self, sdfg, state, local_stream, global_stream):
        if (self._papi_used and
                state.instrument == dace.InstrumentationType.PAPI_Counters):
            uid = PAPIInstrumentation.unified_id(-1, sdfg.node_id(state))
            local_stream.write("__perf_store.markSuperSectionStart(%d);" % uid)

    def on_copy_begin(self, sdfg, state, src_node, dst_node, edge,
                      local_stream, global_stream, copy_shape, src_strides,
                      dst_strides):
        if not self._papi_used:
            return

        state_id = sdfg.node_id(state)
        memlet = edge.data

        # For perfcounters, we have to make sure that:
        # 1) No other measurements are done for the containing scope (no map
        # operation containing this copy is instrumented)
        src_instrumented = PAPIInstrumentation.has_surrounding_perfcounters(
            src_node, state)
        dst_instrumented = PAPIInstrumentation.has_surrounding_perfcounters(
            dst_node, state)
        src_storage = src_node.desc(sdfg).storage
        dst_storage = dst_node.desc(sdfg).storage

        # From cuda.py
        cpu_storage_types = [
            dtypes.StorageType.CPU_Heap,
            dtypes.StorageType.CPU_Stack,
            dtypes.StorageType.CPU_Pinned,
            dtypes.StorageType.Register,
        ]

        perf_cpu_only = (src_storage in cpu_storage_types) and (
            dst_storage in cpu_storage_types)

        self.perf_should_instrument = (
            PAPIInstrumentation.perf_enable_instrumentation_for(sdfg)
            and (not src_instrumented) and (not dst_instrumented)
            and perf_cpu_only)

        if self.perf_should_instrument is False:
            return

        unique_cpy_id = self.get_unique_number()

        dst_nodedesc = dst_node.desc(sdfg)
        ctype = "dace::vec<%s, %d>" % (dst_nodedesc.dtype.ctype, memlet.veclen)

        fac3 = (" * ".join(sym2cpp(copy_shape)) + " / " + "/".join(
            sym2cpp(dst_strides)))
        copy_size = "sizeof(%s) * %s * (%s)" % (ctype, memlet.veclen, fac3)
        node_id = PAPIInstrumentation.unified_id(
            state.node_id(dst_node), state_id)
        # Mark a section start (this is not really a section in itself (it
        # would be a section with 1 entry))
        local_stream.write(
            PAPIInstrumentation.perf_section_start_string(
                node_id, copy_size, copy_size),
            sdfg,
            state_id,
            [src_node, dst_node],
        )
        local_stream.write(
            ("dace::perf::{pcs} __perf_cpy_{nodeid}_{unique_id};\n" +
             "auto& __vs_cpy_{nodeid}_{unique_id} = __perf_store.getNewValueSet(__perf_cpy_{nodeid}_{unique_id}, {nodeid}, PAPI_thread_id(), {size}, dace::perf::ValueSetType::Copy);\n"
             + "__perf_cpy_{nodeid}_{unique_id}.enterCritical();\n").format(
                 pcs=self.perf_counter_string(),
                 nodeid=node_id,
                 unique_id=unique_cpy_id,
                 size=copy_size,
             ),
            sdfg,
            state_id,
            [src_node, dst_node],
        )

    def on_copy_end(self, sdfg, state, src_node, dst_node, edge, local_stream,
                    global_stream):
        if not self._papi_used:
            return

        state_id = sdfg.node_id(state)
        node_id = state.node_id(dst_node)
        if self.perf_should_instrument:
            unique_cpy_id = self._unique_counter

            local_stream.write(
                "__perf_cpy_%d_%d.leaveCritical(__vs_cpy_%d_%d);" %
                (node_id, unique_cpy_id, node_id, unique_cpy_id),
                sdfg,
                state_id,
                [src_node, dst_node],
            )

    def on_node_begin(self, sdfg, state, node, outer_stream, inner_stream,
                      global_stream):
        if not self._papi_used:
            return

        state_id = sdfg.node_id(state)
        unified_id = PAPIInstrumentation.unified_id(
            state.node_id(node), state_id)

        perf_should_instrument = (
            node.instrument == dace.InstrumentationType.PAPI_Counters and
            not PAPIInstrumentation.has_surrounding_perfcounters(node, state)
            and PAPIInstrumentation.perf_enable_instrumentation_for(
                sdfg, node))
        if not perf_should_instrument:
            return

        if isinstance(node, nodes.Tasklet):
            inner_stream.write(
                "dace::perf::%s __perf_%s;\n" % (self.perf_counter_string(),
                                                 node.label),
                sdfg,
                state_id,
                node,
            )
            inner_stream.write(
                "auto& __perf_vs_%s = __perf_store.getNewValueSet(__perf_%s, %d, PAPI_thread_id(), 0);\n"
                % (node.label, node.label, unified_id),
                sdfg,
                state_id,
                node,
            )

            inner_stream.write("__perf_%s.enterCritical();\n" % node.label,
                               sdfg, state_id, node)
        elif isinstance(node, nodes.Reduce):
            unified_id = PAPIInstrumentation.unified_id(
                state.node_id(node), state_id)

            input_size = PAPIUtils.get_memory_input_size(
                node, sdfg, state, state_id)

            # For measuring the memory bandwidth, we analyze the amount of data
            # moved.
            result = outer_stream
            perf_expected_data_movement_sympy = 1

            input_memlet = state.in_edges(node)[0].data
            output_memlet = state.out_edges(node)[0].data
            # If axes were not defined, use all input dimensions
            input_dims = input_memlet.subset.dims()
            output_dims = output_memlet.subset.data_dims()
            axes = node.axes
            if axes is None:
                axes = tuple(range(input_dims))

            for axis in range(output_dims):
                ao = output_memlet.subset[axis]
                perf_expected_data_movement_sympy *= (
                    ao[1] + 1 - ao[0]) / ao[2]

            for axis in axes:
                ai = input_memlet.subset[axis]
                perf_expected_data_movement_sympy *= (
                    ai[1] + 1 - ai[0]) / ai[2]

            if not state.is_parallel():
                # We put a start marker, but only if we are in a serial state
                result.write(
                    PAPIInstrumentation.perf_supersection_start_string(
                        unified_id),
                    sdfg,
                    state_id,
                    node,
                )

            result.write(
                PAPIInstrumentation.perf_section_start_string(
                    unified_id,
                    str(sp.simplify(perf_expected_data_movement_sympy)) +
                    (" * (sizeof(%s) + sizeof(%s))" % (
                        sdfg.arrays[output_memlet.data].dtype.ctype,
                        sdfg.arrays[input_memlet.data].dtype.ctype,
                    )),
                    input_size,
                ),
                sdfg,
                state_id,
                node,
            )

            #############################################################
            # Internal part
            result = inner_stream
            result.write(
                self.perf_counter_start_measurement_string(
                    node, unified_id, '__o%d' % (output_dims - 1)),
                sdfg,
                state_id,
                node,
            )

    def on_node_end(self, sdfg, state, node, outer_stream, inner_stream,
                    global_stream):
        if not self._papi_used:
            return

        state_id = sdfg.node_id(state)
        node_id = state.node_id(node)
        unified_id = PAPIInstrumentation.unified_id(node_id, state_id)

        if isinstance(node, nodes.CodeNode):
            if node.instrument == dace.InstrumentationType.PAPI_Counters:
                if not PAPIInstrumentation.has_surrounding_perfcounters(
                        node, state):
                    inner_stream.write(
                        "__perf_%s.leaveCritical(__perf_vs_%s);" %
                        (node.label, node.label),
                        sdfg,
                        state_id,
                        node,
                    )

                # Add bytes moved
                inner_stream.write(
                    "__perf_store.addBytesMoved(%s);" %
                    PAPIUtils.get_tasklet_byte_accesses(
                        node, state, sdfg, state_id), sdfg, state_id, node)
        elif isinstance(node, nodes.Reduce):
            result = inner_stream
            #############################################################
            # Instrumentation: Post-Reduce (pre-braces)
            byte_moved_measurement = "__perf_store.addBytesMoved(%s);\n"

            # For reductions, we assume Read-Modify-Write for all operations
            # Every reduction statement costs sizeof(input) + sizeof(output).
            # This is wrong with some custom reductions or extending operations
            # (e.g., i32 * i32 => i64)
            # It also is wrong for write-avoiding min/max (min/max that only
            # overwrite the reduced variable when it needs to be changed)

            if node.instrument == dace.InstrumentationType.PAPI_Counters:
                input_memlet = state.in_edges(node)[0].data
                output_memlet = state.out_edges(node)[0].data
                # If axes were not defined, use all input dimensions
                input_dims = input_memlet.subset.dims()
                axes = node.axes
                if axes is None:
                    axes = tuple(range(input_dims))

                num_reduced_inputs = None
                input_size = input_memlet.subset.size()
                for d in range(input_dims):
                    if d in axes:
                        if num_reduced_inputs is None:
                            num_reduced_inputs = input_size[d]
                        else:
                            num_reduced_inputs *= input_size[d]

                result.write(
                    byte_moved_measurement %
                    ("%s * (sizeof(%s) + sizeof(%s))" %
                     (sym2cpp(num_reduced_inputs),
                      sdfg.arrays[output_memlet.data].dtype.ctype,
                      sdfg.arrays[input_memlet.data].dtype.ctype)),
                    sdfg,
                    state_id,
                    node,
                )

                if not PAPIInstrumentation.has_surrounding_perfcounters(
                        node, state):
                    result.write(
                        PAPIInstrumentation.
                        perf_counter_end_measurement_string(unified_id),
                        sdfg,
                        state_id,
                        node,
                    )

    def on_scope_entry(self, sdfg, state, node, outer_stream, inner_stream,
                       global_stream):
        if not self._papi_used:
            return
        if isinstance(node, nodes.MapEntry):
            return self.on_map_entry(sdfg, state, node, outer_stream,
                                     inner_stream, global_stream)
        elif isinstance(node, nodes.ConsumeEntry):
            return self.on_consume_entry(sdfg, state, node, outer_stream,
                                         inner_stream, global_stream)
        raise TypeError

    def on_scope_exit(self, sdfg, state, node, outer_stream, inner_stream,
                      global_stream):
        if not self._papi_used:
            return
        if isinstance(node, nodes.MapExit):
            return self.on_map_exit(sdfg, state, node, outer_stream,
                                    inner_stream, global_stream)
        elif isinstance(node, nodes.ConsumeExit):
            return self.on_consume_exit(sdfg, state, node, outer_stream,
                                        inner_stream, global_stream)
        raise TypeError

    def on_map_entry(self, sdfg, state, node, outer_stream, inner_stream,
                     global_stream):
        dfg = state.scope_subgraph(node)
        state_id = sdfg.node_id(state)
        if node.map.instrument != dace.InstrumentationType.PAPI_Counters:
            return

        unified_id = PAPIInstrumentation.unified_id(
            dfg.node_id(node), state_id)

        #########################################################
        # Outer part

        result = outer_stream

        input_size = PAPIUtils.get_memory_input_size(node, sdfg, dfg, state_id)

        idstr = "// (Node %d)\n" % unified_id
        result.write(idstr)  # Used to identify line numbers later

        # Emit supersection if possible
        result.write(
            PAPIInstrumentation.perf_get_supersection_start_string(
                node, sdfg, dfg, unified_id))

        if PAPIInstrumentation.instrument_entry(
                node) and PAPIInstrumentation.perf_enable_instrumentation_for(
                    sdfg, node):

            size = PAPIUtils.accumulate_byte_movements_v2(
                node, node, dfg, sdfg, state_id)
            size = sp.simplify(size)

            used_symbols = symbolic.symbols_in_sympy_expr(size)
            defined_symbols = sdfg.symbols_defined_at(node)
            undefined_symbols = [
                x for x in used_symbols if x not in defined_symbols
            ]
            if len(undefined_symbols) > 0:
                # We cannot statically determine the size at this point
                print(
                    'Failed to determine size because of undefined symbols ("'
                    + str(undefined_symbols) + '") in "' + str(size) +
                    '", falling back to 0')
                size = 0

            size = sym2cpp(size)

            result.write(
                PAPIInstrumentation.perf_section_start_string(
                    unified_id, size, input_size))

        #########################################################
        # Inner part
        result = inner_stream

        if node.map.flatten:
            # Perfcounters for flattened maps include the calculations
            # made to obtain the different axis indices
            if PAPIInstrumentation.instrument_entry(
                    node
            ) and PAPIInstrumentation.perf_enable_instrumentation_for(
                    sdfg, node):
                map_name = "__DACEMAP_" + str(state_id) + "_" + str(
                    state.node_id(node))
                start_string = self.perf_counter_start_measurement_string(
                    node, unified_id, map_name + "_iter")
                result.write(start_string, sdfg, state_id, node)

                # remember which map has the counters enabled
                node.map._has_papi_counters = True
        else:
            var = node.map.params[-1]
            if (PAPIInstrumentation.instrument_entry(node)
                    and PAPIInstrumentation.perf_enable_instrumentation_for(
                        sdfg, node)):
                start_string = self.perf_counter_start_measurement_string(
                    node, unified_id, var)
                result.write(start_string, sdfg, state_id, node)
                # remember which map has the counters enabled
                node.map._has_papi_counters = True

    def on_consume_entry(self, sdfg, state, node, outer_stream, inner_stream,
                         global_stream):
        state_id = sdfg.node_id(state)
        unified_id = PAPIInstrumentation.unified_id(
            state.node_id(node), state_id)

        # Outer part
        result = outer_stream

        if PAPIInstrumentation.instrument_entry(node):
            # Mark the SuperSection start (if possible)
            # TODO: Safety checks
            result.write(
                PAPIInstrumentation.perf_get_supersection_start_string(
                    node, sdfg, state, unified_id),
                sdfg,
                state_id,
                node,
            )

            # Mark the section start
            # TODO
            size = 0
            result.write(
                PAPIInstrumentation.perf_section_start_string(
                    unified_id, size, 0),
                sdfg,
                state_id,
                node,
            )

            # Generate a thread locker (could be used for dependency injection)
            result.write(
                "dace::perf::ThreadLockProvider __perf_tlp_%d;\n" % unified_id,
                sdfg,
                state_id,
                node,
            )

        # Inner part
        result = inner_stream

        # Instrumenting this is a bit flaky: Since the consume interally
        # creates threads, it must be instrumented like a normal map. However,
        # it seems to spawn normal std::threads (instead of going for openMP)
        # This implementation only allows to measure on a per-task basis
        # (instead of per-thread). This is much more overhead.
        if PAPIInstrumentation.instrument_entry(node):
            result.write(
                ("auto __perf_tlp_{id}_releaser = __perf_tlp_{id}.enqueue();\n"
                 .format(id=unified_id)) +
                self.perf_counter_start_measurement_string(
                    node,
                    unified_id,
                    "__perf_tlp_{id}.getAndIncreaseCounter()".format(
                        id=unified_id),
                    core_str="dace::perf::getThreadID()",
                ),
                sdfg,
                state_id,
                node,
            )

    def on_map_exit(self, sdfg, state, node, outer_stream, inner_stream,
                    global_stream):
        state_id = sdfg.node_id(state)
        entry_node = state.entry_node(node)
        if node.map.instrument != dace.InstrumentationType.PAPI_Counters:
            return

        unified_id = PAPIInstrumentation.unified_id(
            state.node_id(entry_node), state_id)

        perf_end_string = \
            PAPIInstrumentation.perf_counter_end_measurement_string(
                unified_id)

        # Inner part
        result = inner_stream
        if node.map.flatten:
            if node.map._has_papi_counters:
                result.write(perf_end_string, sdfg, state_id, node)

        else:
            if node.map._has_papi_counters:
                result.write(perf_end_string, sdfg, state_id, node)

    def on_consume_exit(self, sdfg, state, node, outer_stream, inner_stream,
                        global_stream):
        state_id = sdfg.node_id(state)
        entry_node = state.entry_node(node)
        unified_id = PAPIInstrumentation.unified_id(
            state.node_id(entry_node), state_id)

        result = inner_stream
        if PAPIInstrumentation.instrument_entry(entry_node):
            result.write(
                PAPIInstrumentation.perf_counter_end_measurement_string(
                    unified_id),
                sdfg,
                state_id,
                node,
            )

    @staticmethod
    def perf_get_supersection_start_string(node, sdfg, dfg, unified_id):
        if node.map.schedule == dtypes.ScheduleType.CPU_Multicore:

            if not hasattr(node.map, '_can_be_supersection_start'):
                node.map._can_be_supersection_start = True

            # Find out if we should mark a section start here or later.
            children = PAPIUtils.all_maps(node, dfg)
            for x in children:
                if not hasattr(x.map, '_can_be_supersection_start'):
                    x.map._can_be_supersection_start = True
                if x.map.schedule == dtypes.ScheduleType.CPU_Multicore:
                    # Nested SuperSections are not supported
                    # We have to mark the outermost section,
                    # which also means that we have to somehow tell the lower
                    # nodes to not mark the section start.
                    x.map._can_be_supersection_start = False
                elif x.map.schedule == dtypes.ScheduleType.Sequential:
                    x.map._can_be_supersection_start = False
                else:
                    # Any other type (FPGA, GPU) - not supported by PAPI.
                    x.map._can_be_supersection_start = False

            if PAPIInstrumentation.perf_enable_instrumentation_for(
                    sdfg, node
            ) and node.map._can_be_supersection_start and not dfg.is_parallel(
            ):
                return "__perf_store.markSuperSectionStart(%d);\n" % unified_id

        elif (PAPIInstrumentation.perf_enable_instrumentation_for(sdfg, node)
              and node.map._can_be_supersection_start
              and not dfg.is_parallel()):
            # Also put this here if it is _REALLY_ safe to do so. (Basically,
            # even if the schedule is sequential, we can serialize to keep
            # buffer usage low)
            return "__perf_store.markSuperSectionStart(%d);\n" % unified_id

        # Otherwise, do nothing (empty string)
        return ""

    @staticmethod
    def instrument_entry(mapEntry: MapEntry):
        # Skip consume entries
        if isinstance(mapEntry, ConsumeEntry):
            return False
        if mapEntry.map.instrument != dace.InstrumentationType.PAPI_Counters:
            return False
        if (mapEntry.map.schedule not in
                PAPIInstrumentation.perf_whitelist_schedules):
            return False
        try:
            cond = not mapEntry.fence_instrumentation
        except (AttributeError, NameError):
            cond = True
        return cond

    @staticmethod
    def has_surrounding_perfcounters(node, dfg: SubgraphView):
        """ Returns true if there is a possibility that this node is part of a
            section that is profiled. """
        parent = dfg.scope_dict()[node]

        if isinstance(parent, MapEntry):
            if (parent.map.schedule not in
                    PAPIInstrumentation.perf_whitelist_schedules):
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
        return PAPIInstrumentation.perf_counter_string_from_string_list(
            self._counters)

    def perf_counter_start_measurement_string(self,
                                              node,
                                              unified_id,
                                              iteration,
                                              core_str="PAPI_thread_id()"):
        pcs = self.perf_counter_string()
        return (
            'dace::perf::{counter_str} __perf_{id};\n' +
            'auto& __vs_{id} = __perf_store.getNewValueSet(__perf_{id}, {id}, {core}, {it});\n'
            + '__perf_{id}.enterCritical();\n').format(
                counter_str=pcs, id=unified_id, it=iteration, core=core_str)

    @staticmethod
    def perf_counter_end_measurement_string(unified_id):
        return '__perf_{id}.leaveCritical(__vs_{id});\n'.format(id=unified_id)

    @staticmethod
    def perf_section_start_string(unified_id,
                                  size,
                                  in_size,
                                  core_str="PAPI_thread_id()"):
        return (
            "__perf_store.markSectionStart(%d, (long long)%s, (long long)%s, %s);\n"
            % (unified_id, str(size), str(in_size), core_str))

    @staticmethod
    def perf_supersection_start_string(unified_id):
        return '__perf_store.markSuperSectionStart(%d);\n' % unified_id

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
            p = subprocess.Popen(
                "papi_avail -d -a | grep -E '^PAPI_'",
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


class PAPIUtils(object):
    _unique_counter = 0

    @staticmethod
    def is_papi_used(sdfg):
        # Checking for PAPI usage
        for node, _ in sdfg.all_nodes_recursive():
            if isinstance(
                    node, nodes.EntryNode
            ) and node.map.instrument == dace.InstrumentationType.PAPI_Counters:
                return True
            if hasattr(
                    node, 'instrument'
            ) and node.instrument == dace.InstrumentationType.PAPI_Counters:
                return True
        return False

    @staticmethod
    def reduce_iteration_count(begin, end, step, rparams: dict):
        # There are different rules when expanding depending on where the expand
        # should happen

        if isinstance(begin, int):
            start_syms = []
        else:
            start_syms = symbolic.symbols_in_sympy_expr(begin)

        if isinstance(end, int):
            end_syms = []
        else:
            end_syms = symbolic.symbols_in_sympy_expr(end)

        if isinstance(step, int):
            step_syms = []
        else:
            step_syms = symbolic.symbols_in_sympy_expr(step)

        def intersection(lista, listb):
            return [x for x in lista if x in listb]

        start_dyn_syms = intersection(start_syms, rparams.keys())
        end_dyn_syms = intersection(end_syms, rparams.keys())
        step_dyn_syms = intersection(step_syms, rparams.keys())

        def replace_func(element, dyn_syms, retparams):
            # Resolve all symbols using the retparams-dict

            for x in dyn_syms:
                target = sp.functions.Min(
                    retparams[x] * (retparams[x] - 1) / 2, 0)
                bstr = str(element)
                element = symbolic.pystr_to_symbolic(bstr)
                element = element.subs(
                    x, target)  # Add the classic sum formula; going upwards

                # To not have hidden elements that get added again later, we
                # also replace the values in the other itvars...
                for k, v in retparams.items():
                    newv = symbolic.pystr_to_symbolic(str(v))

                    tarsyms = symbolic.symbols_in_sympy_expr(target)
                    if x in map(str, tarsyms):
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
    def get_iteration_count(mapEntry: MapEntry, vars: dict):
        """
        Get the number of iterations for this map, allowing other variables
        as bounds.
        """

        _map = mapEntry.map
        _it = _map.params

        retparams = dict()
        for k, v in vars.items():
            retparams[k] = v

        # print("Params: " + str(_it))
        for i, r in enumerate(_map.range):
            begin, end, step = r

            end = end + 1  # end is inclusive, but we want it exclusive

            if isinstance(begin, symbolic.SymExpr):
                begin = begin.expr
            if isinstance(end, symbolic.SymExpr):
                end = end.expr
            if isinstance(step, symbolic.SymExpr):
                step = step.expr

            begin, end, step = PAPIUtils.reduce_iteration_count(
                begin, end, step, retparams)
            num = (end - begin) / step  # The count of iterations
            retparams[_it[i]] = num

        return retparams

    @staticmethod
    def all_maps(mapEntry: EntryNode, dfg: SubgraphView):
        children = [
            x for x in dfg.scope_dict(True)[mapEntry]
            if isinstance(x, EntryNode)
        ]

        sub = []
        for x in children:
            sub.extend(PAPIUtils.all_maps(x, dfg))

        children.extend(sub)
        return children

    @staticmethod
    def get_memlet_byte_size(sdfg, memlet: Memlet):
        pass
        memdata = sdfg.arrays[memlet.data]
        # For now, deal with arrays only
        if isinstance(memdata, Array):
            elems = [str(memdata.dtype.bytes)]
            if memlet.num_accesses is not None and memlet.num_accesses >= 0:
                elems.append(str(memlet.num_accesses))
            else:
                # Skip dynamic number of accesses
                pass

            return "(" + "*".join(elems) + ")"

        else:
            raise NotImplementedError(
                "Untreated data type: %s" % type(memdata).__name__)

    @staticmethod
    def get_out_memlet_costs(sdfg, state_id, node, dfg):
        scope_dict = sdfg.nodes()[state_id].scope_dict()

        out_costs = 0
        for edge in dfg.out_edges(node):
            _, uconn, v, _, memlet = edge
            dst_node = dfg.memlet_path(edge)[-1].dst

            # Target is neither a data nor a tasklet node
            if (isinstance(node, nodes.AccessNode)
                    and (not isinstance(dst_node, nodes.AccessNode)
                         and not isinstance(dst_node, nodes.CodeNode))):
                continue

            # Skip array->code (will be handled as a tasklet input)
            if isinstance(node, nodes.AccessNode) and isinstance(
                    v, nodes.CodeNode):
                continue

            # code->code (e.g., tasklet to tasklet)
            if isinstance(v, nodes.CodeNode):
                continue

            # If the memlet is not pointing to a data node (e.g. tasklet), then
            # the tasklet will take care of the copy
            if not isinstance(dst_node, nodes.AccessNode):
                continue
            # If the memlet is pointing into an array in an inner scope, then
            # the inner scope (i.e., the output array) must handle it
            if (scope_dict[node] != scope_dict[dst_node]
                    and scope_contains_scope(scope_dict, node, dst_node)):
                continue

            # Array to tasklet (path longer than 1, handled at tasklet entry)
            if node == dst_node:
                continue

            # Tasklet -> array
            if isinstance(node, nodes.CodeNode):
                if not uconn:
                    # This would normally raise a syntax error
                    return 0

                try:
                    positive_accesses = bool(memlet.num_accesses >= 0)
                except TypeError:
                    positive_accesses = False

                if memlet.subset.data_dims() == 0 and positive_accesses:

                    if memlet.wcr is not None:
                        # write_and_resolve
                        # We have to assume that every reduction costs 3
                        # accesses of the same size (read old, read new, write)
                        out_costs += 3 * symbolic.pystr_to_symbolic(
                            PAPIUtils.get_memlet_byte_size(sdfg, memlet))
                    else:
                        # This standard operation is already counted
                        out_costs += symbolic.pystr_to_symbolic(
                            PAPIUtils.get_memlet_byte_size(sdfg, memlet))
            # Dispatch array-to-array outgoing copies here
            elif isinstance(node, nodes.AccessNode):
                pass
        return out_costs

    @staticmethod
    def get_tasklet_byte_accesses(tasklet: nodes.CodeNode,
                                  dfg: dace.sdfg.ScopeSubgraphView, sdfg,
                                  state_id):
        """ Get the amount of bytes processed by `tasklet`. The formula is
            sum(inedges * size) + sum(outedges * size) """
        in_accum = []
        out_accum = []
        in_edges = dfg.in_edges(tasklet)

        for ie in in_edges:
            # type ie.data == Memlet
            # type ie.data.data == Data
            in_accum.append(PAPIUtils.get_memlet_byte_size(sdfg, ie.data))

        out_accum.append(
            str(PAPIUtils.get_out_memlet_costs(sdfg, state_id, tasklet, dfg)))

        # Merge (kept split to be able to change the behavior easily)
        full = in_accum
        full.extend(out_accum)

        return "(" + "+".join(full) + ")"

    @staticmethod
    def get_parents(outermost_node, node, sdfg, state_id):

        parent = None
        # Because dfg is only a subgraph view, it does not contain the entry
        # node for a given entry. This O(n) solution is suboptimal
        for state in sdfg.nodes():
            s_d = state.scope_dict(node_to_children=False)
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

        return PAPIUtils.get_parents(outermost_node, parent, sdfg,
                                     state_id) + [parent]

    @staticmethod
    def perf_enable_instrumentation_for(sdfg, node=None):
        return not sdfg.has_instrumented_parent()

    @staticmethod
    def get_memory_input_size(node, sdfg, dfg, state_id):
        curr_state = sdfg.nodes()[state_id]

        data_inputs = []
        for edge in curr_state.in_edges(node):
            # Accumulate over range size and get the amount of data accessed
            num_accesses = edge.data.num_accesses

            # TODO: It might be better to just take the source object size

            try:
                bytes_per_element = sdfg.arrays[edge.data.data].dtype.bytes
            except (KeyError, AttributeError):
                print("Failed to get bytes_per_element")
                bytes_per_element = 0
            data_inputs.append(bytes_per_element * num_accesses)

        input_size = reduce(operator.add, data_inputs, 0)

        input_size = symbolic.pystr_to_symbolic(input_size)

        used_symbols = symbolic.symbols_in_sympy_expr(input_size)
        defined_symbols = sdfg.symbols_defined_at(node)
        undefined_symbols = [
            x for x in used_symbols if not (x in defined_symbols)
        ]
        if len(undefined_symbols) > 0:
            # We cannot statically determine the size at this point
            print("Failed to determine size because of undefined symbols (\"" +
                  str(undefined_symbols) + "\") in \"" + str(input_size) +
                  "\", falling back to 0")
            input_size = 0

        input_size = sym2cpp(input_size)

        return input_size

    @staticmethod
    def accumulate_byte_movements_v2(outermost_node, node,
                                     dfg: dace.sdfg.ScopeSubgraphView, sdfg,
                                     state_id):

        itvars = dict()  # initialize an empty dict

        # First, get a list of children
        if isinstance(node, MapEntry):
            children = dfg.scope_dict(node_to_children=True)[node]
        else:
            children = []
        assert not (node in children)

        # If there still are children, descend recursively (dfs is fine here)
        if len(children) > 0:
            size = 0
            for x in children:
                size = size + PAPIUtils.accumulate_byte_movements_v2(
                    outermost_node, x, dfg, sdfg, state_id)

            return size
        else:
            if isinstance(node, MapExit):
                return 0  # We can ignore this.

            # If we reached the deepest node, get all parents
            parent_list = PAPIUtils.get_parents(outermost_node, node, sdfg,
                                                state_id)
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
                raise ValueError(
                    "Unexpected node"
                )  # A map entry should never be the innermost node
            elif isinstance(node, MapExit):
                return 0  # We can ignore this.
            elif isinstance(node, Tasklet):
                return itcount * symbolic.pystr_to_symbolic(
                    PAPIUtils.get_tasklet_byte_accesses(
                        node, dfg, sdfg, state_id))
            else:
                raise NotImplementedError
