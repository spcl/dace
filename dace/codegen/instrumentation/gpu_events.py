# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
from dace import config, dtypes, registry
from dace.codegen import common
from dace.codegen.instrumentation.provider import InstrumentationProvider
from dace.codegen.prettycode import CodeIOStream
from dace.sdfg import is_devicelevel_gpu, nodes
from dace.sdfg.sdfg import SDFG
from dace.sdfg.state import ControlFlowRegion, SDFGState


@registry.autoregister_params(type=dtypes.InstrumentationType.GPU_Events)
class GPUEventProvider(InstrumentationProvider):
    """Timing instrumentation that reports GPU/copy time using CUDA/HIP events."""

    def __init__(self):
        self.backend = common.get_gpu_backend()
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

    def _get_sobj(self, node: nodes.EntryNode | nodes.ExitNode):
        # Get object behind scope
        if isinstance(node, (nodes.ConsumeEntry, nodes.ConsumeExit)):
            return node.consume
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
            if isinstance(node, (nodes.CodeNode, nodes.EntryNode)):
                s = self._get_sobj(node) if isinstance(node, nodes.EntryNode) else node
                if s.instrument == dtypes.InstrumentationType.GPU_Events:
                    idstr = self._idstr(cfg, state, node)
                    local_stream.write(self._destroy_event("b" + idstr), cfg, state_id, node)
                    local_stream.write(self._destroy_event("e" + idstr), cfg, state_id, node)

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
            stream = gpu_stream_of(node, state)
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
            stream = gpu_stream_of(node, state)
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
        if not isinstance(node, nodes.CodeNode) or is_devicelevel_gpu(sdfg, state, node):
            return
        # Only run for host nodes
        # TODO(later): Implement "clock64"-based GPU counters
        if node.instrument == dtypes.InstrumentationType.GPU_Events:
            state_id = state.parent_graph.node_id(state)
            idstr = "b" + self._idstr(cfg, state, node)
            stream = gpu_stream_of(node, state)
            outer_stream.write(self._record_event(idstr, stream), cfg, state_id, node)

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
        if not isinstance(node, nodes.Tasklet) or is_devicelevel_gpu(sdfg, state, node):
            return
        # Only run for host nodes
        # TODO(later): Implement "clock64"-based GPU counters
        if node.instrument == dtypes.InstrumentationType.GPU_Events:
            state_id = state.parent_graph.node_id(state)
            idstr = "e" + self._idstr(cfg, state, node)
            stream = gpu_stream_of(node, state)
            outer_stream.write(self._record_event(idstr, stream), cfg, state_id, node)
            outer_stream.write(
                self._report(f"{type(node).__name__} {node.label}", cfg, state, node), cfg, state_id, node
            )


def gpu_stream_of(node: nodes.Node, state: SDFGState) -> int:
    """The GPU stream a node runs on, or ``-1`` (recorded on the default stream).

    The experimental codegen schedules ``Node.gpu_stream_id`` and a map exit takes its entry's; the legacy
    codegen attaches ``_cuda_stream`` to the node dynamically.
    """
    if isinstance(node, nodes.MapExit):
        entry = state.entry_node(node)
        if entry is not None and entry.gpu_stream_id is not None:
            return entry.gpu_stream_id
    if node.gpu_stream_id is not None:
        return node.gpu_stream_id
    return getattr(node, "_cuda_stream", -1)
