# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
from typing import Union
from dace import config, dtypes, registry
from dace.codegen.prettycode import CodeIOStream
from dace.sdfg import nodes, is_devicelevel_gpu
from dace.codegen import common
from dace.codegen.instrumentation.provider import InstrumentationProvider
from dace.sdfg.sdfg import SDFG
from dace.sdfg.state import ControlFlowRegion, SDFGState


@registry.autoregister_params(type=dtypes.InstrumentationType.GPU_Events)
class GPUEventProvider(InstrumentationProvider):
    """ Timing instrumentation that reports GPU/copy time using CUDA/HIP events. """

    def __init__(self):
        self.backend = common.get_gpu_backend()
        super().__init__()

    def on_sdfg_begin(self, sdfg: SDFG, local_stream: CodeIOStream, global_stream: CodeIOStream, codegen) -> None:
        if self.backend == 'cuda':
            header_name = 'cuda_runtime.h'
        elif self.backend == 'hip':
            header_name = 'hip/hip_runtime.h'
        else:
            raise NameError('GPU backend "%s" not recognized' % self.backend)

        global_stream.write('#include <chrono>')
        global_stream.write('#include <%s>' % header_name)

        # For other file headers
        sdfg.append_global_code('\n#include <chrono>', None)
        sdfg.append_global_code('\n#include <%s>' % header_name, None)

    def _get_sobj(self, node: Union[nodes.EntryNode, nodes.ExitNode]):
        # Get object behind scope
        if hasattr(node, 'consume'):
            return node.consume
        else:
            return node.map

    def _create_event(self, id):
        return '''{backend}Event_t __dace_ev_{id};
{backend}EventCreate(&__dace_ev_{id});'''.format(id=id, backend=self.backend)

    def _destroy_event(self, id):
        return '{backend}EventDestroy(__dace_ev_{id});'.format(id=id, backend=self.backend)

    def _record_event(self, id, stream):
        concurrent_streams = int(config.Config.get('compiler', 'cuda', 'max_concurrent_streams'))
        if concurrent_streams < 0 or stream == -1:
            streamstr = 'nullptr'
        else:
            streamstr = f'__state->gpu_context->streams[{stream}]'
        return '%sEventRecord(__dace_ev_%s, %s);' % (self.backend, id, streamstr)

    def _report(self, timer_name: str, cfg: ControlFlowRegion = None, state: SDFGState = None, node: nodes.Node = None):
        idstr = self._idstr(cfg, state, node)

        state_id = -1
        node_id = -1
        if state is not None:
            state_id = state.block_id
            if node is not None:
                node_id = state.node_id(node)

        return '''float __dace_ms_{id} = -1.0f;
{backend}EventSynchronize(__dace_ev_e{id});
{backend}EventElapsedTime(&__dace_ms_{id}, __dace_ev_b{id}, __dace_ev_e{id});
int __dace_micros_{id} = (int) (__dace_ms_{id} * 1000.0);
unsigned long int __dace_ts_end_{id} = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now().time_since_epoch()).count();
unsigned long int __dace_ts_start_{id} = __dace_ts_end_{id} - __dace_micros_{id};
__state->report.add_completion("{timer_name}", "GPU", __dace_ts_start_{id}, __dace_ts_end_{id}, {cfg_id}, {state_id}, {node_id});'''.format(
            id=idstr,
            timer_name=timer_name,
            backend=self.backend,
            cfg_id=cfg.cfg_id,
            state_id=state_id,
            node_id=node_id)

    # Code generation hooks
    def on_state_begin(self, sdfg: SDFG, cfg: ControlFlowRegion, state: SDFGState, local_stream: CodeIOStream,
                       global_stream: CodeIOStream) -> None:
        state_id = state.parent_graph.node_id(state)
        # Create GPU events for each instrumented scope in the state
        for node in state.nodes():
            if isinstance(node, (nodes.CodeNode, nodes.EntryNode)):
                s = (self._get_sobj(node) if isinstance(node, nodes.EntryNode) else node)
                if s.instrument == dtypes.InstrumentationType.GPU_Events:
                    idstr = self._idstr(cfg, state, node)
                    local_stream.write(self._create_event('b' + idstr), cfg, state_id, node)
                    local_stream.write(self._create_event('e' + idstr), cfg, state_id, node)

        # Create and record a CUDA/HIP event for the entire state
        if state.instrument == dtypes.InstrumentationType.GPU_Events:
            idstr = 'b' + self._idstr(cfg, state, None)
            local_stream.write(self._create_event(idstr), cfg, state_id)
            local_stream.write(self._record_event(idstr, 0), cfg, state_id)
            idstr = 'e' + self._idstr(cfg, state, None)
            local_stream.write(self._create_event(idstr), cfg, state_id)

    def on_state_end(self, sdfg: SDFG, cfg: ControlFlowRegion, state: SDFGState, local_stream: CodeIOStream,
                     global_stream: CodeIOStream) -> None:
        state_id = state.parent_graph.node_id(state)
        # Record and measure state stream event
        if state.instrument == dtypes.InstrumentationType.GPU_Events:
            idstr = self._idstr(cfg, state, None)
            local_stream.write(self._record_event('e' + idstr, 0), cfg, state_id)
            local_stream.write(self._report('State %s' % state.label, cfg, state), cfg, state_id)
            local_stream.write(self._destroy_event('b' + idstr), cfg, state_id)
            local_stream.write(self._destroy_event('e' + idstr), cfg, state_id)

        # Destroy CUDA/HIP events for scopes in the state
        for node in state.nodes():
            if isinstance(node, (nodes.CodeNode, nodes.EntryNode)):
                s = (self._get_sobj(node) if isinstance(node, nodes.EntryNode) else node)
                if s.instrument == dtypes.InstrumentationType.GPU_Events:
                    idstr = self._idstr(cfg, state, node)
                    local_stream.write(self._destroy_event('b' + idstr), cfg, state_id, node)
                    local_stream.write(self._destroy_event('e' + idstr), cfg, state_id, node)

    def on_scope_entry(self, sdfg: SDFG, cfg: ControlFlowRegion, state: SDFGState, node: nodes.EntryNode,
                       outer_stream: CodeIOStream, inner_stream: CodeIOStream, global_stream: CodeIOStream) -> None:
        state_id = state.parent_graph.node_id(state)
        s = self._get_sobj(node)
        if s.instrument == dtypes.InstrumentationType.GPU_Events:
            if s.schedule != dtypes.ScheduleType.GPU_Device:
                raise TypeError('GPU Event instrumentation only applies to '
                                'GPU_Device map scopes')

            idstr = 'b' + self._idstr(cfg, state, node)
            stream = getattr(node, '_cuda_stream', -1)
            outer_stream.write(self._record_event(idstr, stream), cfg, state_id, node)

    def on_scope_exit(self, sdfg: SDFG, cfg: ControlFlowRegion, state: SDFGState, node: nodes.ExitNode,
                      outer_stream: CodeIOStream, inner_stream: CodeIOStream, global_stream: CodeIOStream) -> None:
        state_id = state.parent_graph.node_id(state)
        entry_node = state.entry_node(node)
        s = self._get_sobj(node)
        if s.instrument == dtypes.InstrumentationType.GPU_Events:
            idstr = 'e' + self._idstr(cfg, state, entry_node)
            stream = getattr(node, '_cuda_stream', -1)
            outer_stream.write(self._record_event(idstr, stream), cfg, state_id, node)
            outer_stream.write(self._report('%s %s' % (type(s).__name__, s.label), cfg, state, entry_node), cfg,
                               state_id, node)

    def on_node_begin(self, sdfg: SDFG, cfg: ControlFlowRegion, state: SDFGState, node: nodes.Node,
                      outer_stream: CodeIOStream, inner_stream: CodeIOStream, global_stream: CodeIOStream) -> None:
        if (not isinstance(node, nodes.CodeNode) or is_devicelevel_gpu(sdfg, state, node)):
            return
        # Only run for host nodes
        # TODO(later): Implement "clock64"-based GPU counters
        if node.instrument == dtypes.InstrumentationType.GPU_Events:
            state_id = state.parent_graph.node_id(state)
            idstr = 'b' + self._idstr(cfg, state, node)
            stream = getattr(node, '_cuda_stream', -1)
            outer_stream.write(self._record_event(idstr, stream), cfg, state_id, node)

    def on_node_end(self, sdfg: SDFG, cfg: ControlFlowRegion, state: SDFGState, node: nodes.Node,
                    outer_stream: CodeIOStream, inner_stream: CodeIOStream, global_stream: CodeIOStream) -> None:
        if (not isinstance(node, nodes.Tasklet) or is_devicelevel_gpu(sdfg, state, node)):
            return
        # Only run for host nodes
        # TODO(later): Implement "clock64"-based GPU counters
        if node.instrument == dtypes.InstrumentationType.GPU_Events:
            state_id = state.parent_graph.node_id(state)
            idstr = 'e' + self._idstr(cfg, state, node)
            stream = getattr(node, '_cuda_stream', -1)
            outer_stream.write(self._record_event(idstr, stream), cfg, state_id, node)
            outer_stream.write(self._report('%s %s' % (type(node).__name__, node.label), cfg, state, node), cfg,
                               state_id, node)
