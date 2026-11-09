# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
from dace import config, dtypes, registry
from dace.codegen import common
from dace.codegen.instrumentation.provider import InstrumentationProvider
from dace.codegen.prettycode import CodeIOStream
from dace.codegen.targets.cpp import mangle_dace_state_struct_name
from dace.sdfg import is_devicelevel_gpu, nodes
from dace.sdfg.scope import is_thread_block_level
from dace.sdfg.sdfg import SDFG
from dace.sdfg.state import AbstractControlFlowRegion, ControlFlowRegion, SDFGState


@registry.autoregister_params(type=dtypes.InstrumentationType.GPU_Events)
class GPUEventProvider(InstrumentationProvider):
    """Timing instrumentation that reports GPU/copy time using CUDA/HIP events.

    On code nodes and control flow regions inside GPU kernels (tasklets, nested SDFGs, and regions of nested SDFGs
    that the threads of a block run together, such as the code of an instrumented nested SDFG after inlining), every
    thread block that runs the element records an event: its start and end on the GPU's global timer (one time axis
    for all multiprocessors, aligned with the host's), the clock cycles of its multiprocessor in between, the block,
    and the multiprocessor. The events of a call are collected after it and added to the report, a thread of the trace
    per thread block (see ``dace/cuda/trace.cuh``). A barrier of the thread block precedes either measurement.
    Control flow regions in host code are measured with events on the GPU stream, as states are.
    """

    def __init__(self):
        self.backend = common.get_gpu_backend()
        # The instrumented nodes and regions in device code, and their index in the generated table
        self._device_elements: dict[nodes.Node | AbstractControlFlowRegion, int] = {}
        self._device_table: list[tuple[str, int, int, int]] = []  # Name and element IDs
        super().__init__()

    def writes_to_report(self) -> bool:
        return True

    def on_sdfg_begin(self, sdfg: SDFG, local_stream: CodeIOStream, global_stream: CodeIOStream, codegen) -> None:
        if self.backend == "cuda":
            header_name = "cuda_runtime.h"
        elif self.backend == "hip":
            header_name = "hip/hip_runtime.h"
        else:
            raise NameError(f'GPU backend "{self.backend}" not recognized')

        global_stream.write("#include <chrono>")
        global_stream.write(f"#include <{header_name}>")

        # For other file headers
        sdfg.append_global_code("\n#include <chrono>", None)
        sdfg.append_global_code(f"\n#include <{header_name}>", None)

        if sdfg.parent is not None:
            return
        for element, parent in sdfg.all_nodes_recursive():
            if isinstance(element, nodes.CodeNode) and isinstance(parent, SDFGState):
                where = (parent.sdfg, parent, element)  # Where in the SDFG the element runs
                ids = (parent.parent_graph.cfg_id, parent.block_id, parent.node_id(element))
            elif isinstance(element, AbstractControlFlowRegion) and element.sdfg.parent_nsdfg_node is not None:
                caller = element.sdfg.parent  # The region runs where the nested SDFG that contains it does
                where = (caller.sdfg, caller, element.sdfg.parent_nsdfg_node)
                ids = (element.cfg_id, -1, -1)
            else:
                continue
            if element.instrument != dtypes.InstrumentationType.GPU_Events or not is_devicelevel_gpu(*where):
                continue
            if not is_thread_block_level(*where):
                raise TypeError(
                    f'GPU event instrumentation of "{element.label}" in a kernel applies to code that the threads '
                    "of a block run together (outside thread-block maps, in a kernel with thread-block maps or a "
                    "consume scope)"
                )
            self._device_elements[element] = len(self._device_table)
            self._device_table.append((element.label, *ids))
        if not self._device_elements:
            return
        sdfg.append_global_code(self._device_trace_code(sdfg), "cuda")
        state = mangle_dace_state_struct_name(sdfg)
        global_stream.write(
            f"""
DACE_EXPORTED void __dace_gpu_trace_init_{sdfg.name}({state} *__state, unsigned long long capacity);
DACE_EXPORTED void __dace_gpu_trace_exit_{sdfg.name}({state} *__state);
DACE_EXPORTED void __dace_gpu_trace_begin_{sdfg.name}({state} *__state);
DACE_EXPORTED void __dace_gpu_trace_end_{sdfg.name}({state} *__state, const char *const *names, const int *ids,
                                                    int num_elements);""",
            sdfg,
        )
        local_stream.write(f"__dace_gpu_trace_begin_{sdfg.name}(__state);", sdfg)

    def on_sdfg_end(self, sdfg: SDFG, local_stream: CodeIOStream, global_stream: CodeIOStream) -> None:
        if sdfg.parent is not None or not self._device_elements:
            return
        names = ", ".join(f'"{name}"' for name, *_ in self._device_table)
        ids = ", ".join(", ".join(str(i) for i in element_ids) for _, *element_ids in self._device_table)
        local_stream.write(
            f"""{{
    // The events of the thread blocks that ran instrumented nodes in kernels
    static const char *const __dace_gpu_trace_names[] = {{{names}}};
    static const int __dace_gpu_trace_ids[] = {{{ids}}};
    __dace_gpu_trace_end_{sdfg.name}(__state, __dace_gpu_trace_names, __dace_gpu_trace_ids,
                                    {len(self._device_table)});
}}""",
            sdfg,
        )

    def on_sdfg_init_end(self, sdfg: SDFG, local_stream: CodeIOStream, global_stream: CodeIOStream) -> None:
        if sdfg.parent is None and self._device_elements:
            capacity = config.Config.get("instrumentation", "gpu_events", "max_device_events")
            local_stream.write(f"__dace_gpu_trace_init_{sdfg.name}(__state, {capacity}ull);", sdfg)

    def on_sdfg_exit_begin(self, sdfg: SDFG, local_stream: CodeIOStream, global_stream: CodeIOStream) -> None:
        if sdfg.parent is None and self._device_elements:
            local_stream.write(f"__dace_gpu_trace_exit_{sdfg.name}(__state);", sdfg)

    def _device_trace_code(self, sdfg: SDFG) -> str:
        """The device-side buffer of events, and the host functions that manage it (in the GPU code file)."""
        b = self.backend
        name = sdfg.name
        state = mangle_dace_state_struct_name(sdfg)
        symbol = "HIP_SYMBOL(__dace_gpu_trace)" if b == "hip" else "__dace_gpu_trace"
        if b == "hip":
            rate = """int rate_khz = 0;
    DACE_GPU_CHECK(hipDeviceGetAttribute(&rate_khz, hipDeviceAttributeWallClockRate, 0));
    __dace_gpu_trace_ns_per_tick = 1e6 / rate_khz;"""
        else:
            rate = "__dace_gpu_trace_ns_per_tick = 1.0;  // The global timer counts nanoseconds"
        return f"""
#include <vector>
#include <dace/cuda/trace.cuh>

// The events that thread blocks record (see the GPU_Events instrumentation)
static __device__ dace::perf::GPUTraceBuffer __dace_gpu_trace;
static dace::perf::GPUTraceEvent *__dace_gpu_trace_events = nullptr;
static unsigned long long __dace_gpu_trace_capacity = 0;
// The host time (in nanoseconds) at a reading of the global timer, and the timer's period
static unsigned long long __dace_gpu_trace_host_ns = 0, __dace_gpu_trace_ticks = 0;
static double __dace_gpu_trace_ns_per_tick = 1.0;

static unsigned long long __dace_gpu_trace_host_now_ns() {{
    return std::chrono::duration_cast<std::chrono::nanoseconds>(
               std::chrono::high_resolution_clock::now().time_since_epoch())
        .count();
}}

DACE_EXPORTED void __dace_gpu_trace_init_{name}({state} *__state, unsigned long long capacity) {{
    __dace_gpu_trace_capacity = capacity;
    DACE_GPU_CHECK({b}Malloc(&__dace_gpu_trace_events, capacity * sizeof(dace::perf::GPUTraceEvent)));
    dace::perf::GPUTraceBuffer buffer = {{__dace_gpu_trace_events, capacity, 0}};
    DACE_GPU_CHECK({b}MemcpyToSymbol({symbol}, &buffer, sizeof(buffer)));
    {rate}
    // Relate the global timer to the host's clock (to within the time of a kernel launch)
    uint64_t *ticks;
    DACE_GPU_CHECK({b}Malloc(&ticks, sizeof(uint64_t)));
    DACE_GPU_CHECK({b}DeviceSynchronize());
    unsigned long long before = __dace_gpu_trace_host_now_ns();
    dace::perf::gpu_trace_read_ticks<<<1, 1>>>(ticks);
    DACE_GPU_CHECK({b}DeviceSynchronize());
    unsigned long long after = __dace_gpu_trace_host_now_ns();
    DACE_GPU_CHECK({b}Memcpy(&__dace_gpu_trace_ticks, ticks, sizeof(uint64_t), {b}MemcpyDeviceToHost));
    DACE_GPU_CHECK({b}Free(ticks));
    __dace_gpu_trace_host_ns = before + (after - before) / 2;
}}

DACE_EXPORTED void __dace_gpu_trace_exit_{name}({state} *__state) {{
    DACE_GPU_CHECK({b}Free(__dace_gpu_trace_events));
    __dace_gpu_trace_events = nullptr;
}}

DACE_EXPORTED void __dace_gpu_trace_begin_{name}({state} *__state) {{
    unsigned long long zero = 0;
    DACE_GPU_CHECK({b}MemcpyToSymbol({symbol}, &zero, sizeof(zero), offsetof(dace::perf::GPUTraceBuffer, count)));
}}

DACE_EXPORTED void __dace_gpu_trace_end_{name}({state} *__state, const char *const *names, const int *ids,
                                                int num_elements) {{
    DACE_GPU_CHECK({b}DeviceSynchronize());
    dace::perf::GPUTraceBuffer buffer;
    DACE_GPU_CHECK({b}MemcpyFromSymbol(&buffer, {symbol}, sizeof(buffer)));
    unsigned long long stored = buffer.count < __dace_gpu_trace_capacity ? buffer.count : __dace_gpu_trace_capacity;
    std::vector<dace::perf::GPUTraceEvent> events(stored);
    DACE_GPU_CHECK({b}Memcpy(events.data(), __dace_gpu_trace_events, stored * sizeof(dace::perf::GPUTraceEvent),
                             {b}MemcpyDeviceToHost));
    for (const auto &e : events) {{
        if (e.element >= (uint32_t)num_elements)
            continue;
        long long since = (long long)(e.start - __dace_gpu_trace_ticks);
        unsigned long long start = __dace_gpu_trace_host_ns + (long long)(since * __dace_gpu_trace_ns_per_tick);
        unsigned long long duration = (unsigned long long)((e.end - e.start) * __dace_gpu_trace_ns_per_tick);
        const int *id = ids + 3 * e.element;
        __state->report.add_device_completion(names[e.element], start, duration, e.block, e.sm, e.cycles, id[0], id[1],
                                      id[2]);
    }}
    if (buffer.count > __dace_gpu_trace_capacity)
        printf("WARNING: %llu GPU instrumentation events did not fit in %llu (see the configuration entry "
               "instrumentation.gpu_events.max_device_events)\\n",
               buffer.count - __dace_gpu_trace_capacity, __dace_gpu_trace_capacity);
}}
"""

    @staticmethod
    def _device_begin(idstr: str) -> str:
        """The start of the measurement of an element in a kernel, by every thread block that runs it."""
        return f"""unsigned long long __dace_gpu_trace_start_{idstr} = 0, __dace_gpu_trace_cycles_{idstr} = 0;
__syncthreads();
if (dace::perf::gpu_trace_is_first_thread()) {{
    __dace_gpu_trace_start_{idstr} = dace::perf::gpu_trace_ticks();
    __dace_gpu_trace_cycles_{idstr} = clock64();
}}"""

    def _device_end(self, element: nodes.Node | AbstractControlFlowRegion, idstr: str) -> str:
        """The end of the measurement of an element in a kernel: the event of the thread block."""
        return f"""__syncthreads();
if (dace::perf::gpu_trace_is_first_thread())
    dace::perf::gpu_trace_record(__dace_gpu_trace, {self._device_elements[element]}, __dace_gpu_trace_start_{idstr},
                                 __dace_gpu_trace_cycles_{idstr});"""

    def _get_sobj(self, node: nodes.EntryNode | nodes.ExitNode):
        # Get object behind scope
        return node.map

    def _create_event(self, id):
        return f"""{self.backend}Event_t __dace_ev_{id};
{self.backend}EventCreate(&__dace_ev_{id});"""

    def _destroy_event(self, id):
        return f"{self.backend}EventDestroy(__dace_ev_{id});"

    def _record_event(self, id, stream):
        concurrent_streams = int(config.Config.get("compiler", "cuda", "max_concurrent_streams"))
        if concurrent_streams < 0 or stream == -1:
            streamstr = "nullptr"
        else:
            streamstr = f"__state->gpu_context->streams[{stream}]"
        return f"{self.backend}EventRecord(__dace_ev_{id}, {streamstr});"

    def _report(self, timer_name: str, cfg: ControlFlowRegion = None, state: SDFGState = None, node: nodes.Node = None):
        idstr = self._idstr(cfg, state, node)

        state_id = -1
        node_id = -1
        if state is not None:
            state_id = state.block_id
            if node is not None:
                node_id = state.node_id(node)

        return f"""float __dace_ms_{idstr} = -1.0f;
{self.backend}EventSynchronize(__dace_ev_e{idstr});
{self.backend}EventElapsedTime(&__dace_ms_{idstr}, __dace_ev_b{idstr}, __dace_ev_e{idstr});
int __dace_micros_{idstr} = (int) (__dace_ms_{idstr} * 1000.0);
unsigned long int __dace_ts_end_{idstr} = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now().time_since_epoch()).count();
unsigned long int __dace_ts_start_{idstr} = __dace_ts_end_{idstr} - __dace_micros_{idstr};
__state->report.add_completion("{timer_name}", "GPU", __dace_ts_start_{idstr}, __dace_ts_end_{idstr}, {cfg.cfg_id}, {state_id}, {node_id});"""

    # Code generation hooks
    def on_state_begin(
        self,
        sdfg: SDFG,
        cfg: ControlFlowRegion,
        state: SDFGState,
        local_stream: CodeIOStream,
        global_stream: CodeIOStream,
    ) -> None:
        state_id = state.parent_graph.node_id(state)
        # Create GPU events for each instrumented scope in the state
        for node in state.nodes():
            if isinstance(node, nodes.CodeNode) and node in self._device_elements:
                continue
            if isinstance(node, (nodes.CodeNode, nodes.EntryNode)):
                s = self._get_sobj(node) if isinstance(node, nodes.EntryNode) else node
                if s.instrument == dtypes.InstrumentationType.GPU_Events:
                    idstr = self._idstr(cfg, state, node)
                    local_stream.write(self._create_event("b" + idstr), cfg, state_id, node)
                    local_stream.write(self._create_event("e" + idstr), cfg, state_id, node)

        # Create and record a CUDA/HIP event for the entire state
        if state.instrument == dtypes.InstrumentationType.GPU_Events:
            idstr = "b" + self._idstr(cfg, state, None)
            local_stream.write(self._create_event(idstr), cfg, state_id)
            local_stream.write(self._record_event(idstr, 0), cfg, state_id)
            idstr = "e" + self._idstr(cfg, state, None)
            local_stream.write(self._create_event(idstr), cfg, state_id)

    def on_state_end(
        self,
        sdfg: SDFG,
        cfg: ControlFlowRegion,
        state: SDFGState,
        local_stream: CodeIOStream,
        global_stream: CodeIOStream,
    ) -> None:
        state_id = state.parent_graph.node_id(state)
        # Record and measure state stream event
        if state.instrument == dtypes.InstrumentationType.GPU_Events:
            idstr = self._idstr(cfg, state, None)
            local_stream.write(self._record_event("e" + idstr, 0), cfg, state_id)
            local_stream.write(self._report(f"State {state.label}", cfg, state), cfg, state_id)
            local_stream.write(self._destroy_event("b" + idstr), cfg, state_id)
            local_stream.write(self._destroy_event("e" + idstr), cfg, state_id)

        # Destroy CUDA/HIP events for scopes in the state
        for node in state.nodes():
            if isinstance(node, nodes.CodeNode) and node in self._device_elements:
                continue
            if isinstance(node, (nodes.CodeNode, nodes.EntryNode)):
                s = self._get_sobj(node) if isinstance(node, nodes.EntryNode) else node
                if s.instrument == dtypes.InstrumentationType.GPU_Events:
                    idstr = self._idstr(cfg, state, node)
                    local_stream.write(self._destroy_event("b" + idstr), cfg, state_id, node)
                    local_stream.write(self._destroy_event("e" + idstr), cfg, state_id, node)

    def on_region_begin(
        self,
        sdfg: SDFG,
        cfg: ControlFlowRegion,
        region: AbstractControlFlowRegion,
        local_stream: CodeIOStream,
        global_stream: CodeIOStream,
    ) -> None:
        idstr = self._idstr(region, None, None)
        if region in self._device_elements:
            local_stream.write(self._device_begin(idstr), region)
            return
        local_stream.write(self._create_event("b" + idstr), region)
        local_stream.write(self._create_event("e" + idstr), region)
        local_stream.write(self._record_event("b" + idstr, 0), region)

    def on_region_end(
        self,
        sdfg: SDFG,
        cfg: ControlFlowRegion,
        region: AbstractControlFlowRegion,
        local_stream: CodeIOStream,
        global_stream: CodeIOStream,
    ) -> None:
        idstr = self._idstr(region, None, None)
        if region in self._device_elements:
            local_stream.write(self._device_end(region, idstr), region)
            return
        local_stream.write(self._record_event("e" + idstr, 0), region)
        local_stream.write(self._report(f"{type(region).__name__} {region.label}", region), region)
        local_stream.write(self._destroy_event("b" + idstr), region)
        local_stream.write(self._destroy_event("e" + idstr), region)

    def on_scope_entry(
        self,
        sdfg: SDFG,
        cfg: ControlFlowRegion,
        state: SDFGState,
        node: nodes.EntryNode,
        outer_stream: CodeIOStream,
        inner_stream: CodeIOStream,
        global_stream: CodeIOStream,
    ) -> None:
        state_id = state.parent_graph.node_id(state)
        s = self._get_sobj(node)
        if s.instrument == dtypes.InstrumentationType.GPU_Events:
            if s.schedule != dtypes.ScheduleType.GPU_Device:
                raise TypeError("GPU Event instrumentation only applies to GPU_Device map scopes")

            idstr = "b" + self._idstr(cfg, state, node)
            stream = getattr(node, "_cuda_stream", -1)
            outer_stream.write(self._record_event(idstr, stream), cfg, state_id, node)

    def on_scope_exit(
        self,
        sdfg: SDFG,
        cfg: ControlFlowRegion,
        state: SDFGState,
        node: nodes.ExitNode,
        outer_stream: CodeIOStream,
        inner_stream: CodeIOStream,
        global_stream: CodeIOStream,
    ) -> None:
        state_id = state.parent_graph.node_id(state)
        entry_node = state.entry_node(node)
        s = self._get_sobj(node)
        if s.instrument == dtypes.InstrumentationType.GPU_Events:
            idstr = "e" + self._idstr(cfg, state, entry_node)
            stream = getattr(node, "_cuda_stream", -1)
            outer_stream.write(self._record_event(idstr, stream), cfg, state_id, node)
            outer_stream.write(
                self._report(f"{type(s).__name__} {s.label}", cfg, state, entry_node), cfg, state_id, node
            )

    def on_node_begin(
        self,
        sdfg: SDFG,
        cfg: ControlFlowRegion,
        state: SDFGState,
        node: nodes.Node,
        outer_stream: CodeIOStream,
        inner_stream: CodeIOStream,
        global_stream: CodeIOStream,
    ) -> None:
        if not isinstance(node, nodes.CodeNode) or node.instrument != dtypes.InstrumentationType.GPU_Events:
            return
        state_id = state.parent_graph.node_id(state)
        idstr = self._idstr(cfg, state, node)
        if node in self._device_elements:
            outer_stream.write(self._device_begin(idstr), cfg, state_id, node)
        elif not is_devicelevel_gpu(sdfg, state, node):
            stream = getattr(node, "_cuda_stream", -1)
            outer_stream.write(self._record_event("b" + idstr, stream), cfg, state_id, node)

    def on_node_end(
        self,
        sdfg: SDFG,
        cfg: ControlFlowRegion,
        state: SDFGState,
        node: nodes.Node,
        outer_stream: CodeIOStream,
        inner_stream: CodeIOStream,
        global_stream: CodeIOStream,
    ) -> None:
        if not isinstance(node, nodes.CodeNode) or node.instrument != dtypes.InstrumentationType.GPU_Events:
            return
        state_id = state.parent_graph.node_id(state)
        idstr = self._idstr(cfg, state, node)
        if node in self._device_elements:
            outer_stream.write(self._device_end(node, idstr), cfg, state_id, node)
        elif not is_devicelevel_gpu(sdfg, state, node):
            stream = getattr(node, "_cuda_stream", -1)
            outer_stream.write(self._record_event("e" + idstr, stream), cfg, state_id, node)
            outer_stream.write(
                self._report(f"{type(node).__name__} {node.label}", cfg, state, node), cfg, state_id, node
            )
