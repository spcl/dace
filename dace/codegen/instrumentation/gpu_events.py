# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
from dace import config, dtypes, registry
from dace.sdfg import nodes, is_devicelevel_gpu, get_gpulevel_node_location
from dace.codegen.instrumentation.provider import InstrumentationProvider


@registry.autoregister_params(type=dtypes.InstrumentationType.GPU_Events)
class GPUEventProvider(InstrumentationProvider):
    """ Timing instrumentation that reports GPU/copy time using CUDA/HIP events. """
    def __init__(self):
        self.backend = config.Config.get('compiler', 'cuda', 'backend')
        super().__init__()
        self.debug = 1 if config.Config.get('compiler',
                                            'build_type') == 'Debug' else 0
        self.concurrent_streams = int(
            config.Config.get('compiler', 'cuda', 'max_concurrent_streams'))

    def on_sdfg_begin(self, sdfg, local_stream, global_stream):
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

    def _create_event(self, id, gpu_id):
        return f'''{self.backend}SetDevice({gpu_id});
        {self.backend}Event_t __dace_ev_{id};
        {self.backend}EventCreate(&__dace_ev_{id});'''

    def _destroy_event(self, id, gpu_id):
        return f'''{self.backend}SetDevice({gpu_id});
        {self.backend}EventDestroy(__dace_ev_{id});'''

    def _record_event(self, id, stream, gpu_id):
        if self.concurrent_streams < 0 or stream == -1:
            streamstr = 'nullptr'
        else:
            streamstr = f'__state->gpu_context->at({gpu_id}).streams[{stream}]'
        return f'{self.backend}EventRecord(__dace_ev_{id}, {streamstr});'

    def _report(self,
                timer_name: str,
                sdfg=None,
                state=None,
                node=None,
                idstr=None,
                gpu_id=-1):
        idstr = self._idstr(sdfg, state, node) if idstr is None else idstr

        state_id = -1
        node_id = -1
        if state is not None:
            state_id = sdfg.node_id(state)
            if node is not None:
                node_id = state.node_id(node)

        return f'''float __dace_ms_{idstr} = -1.0f;
{self.backend}EventSynchronize(__dace_ev_e{idstr});
{self.backend}EventElapsedTime(&__dace_ms_{idstr}, __dace_ev_b{idstr}, __dace_ev_e{idstr});
int __dace_micros_{idstr} = (int) (__dace_ms_{idstr} * 1000.0);
unsigned long int __dace_ts_end_{idstr} = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now().time_since_epoch()).count();
unsigned long int __dace_ts_start_{idstr} = __dace_ts_end_{idstr} - __dace_micros_{idstr};
__state->report.add_completion("{timer_name}", "GPU", __dace_ts_start_{idstr}, __dace_ts_end_{idstr}, {sdfg.sdfg_id}, {state_id}, {node_id}, {gpu_id});'''

    # Code generation hooks
    def on_state_begin(self, sdfg, state, local_stream, global_stream=None):
        state_id = sdfg.node_id(state)
        gpu_ids = set()
        # Create GPU events for each instrumented scope in the state
        for node in state.nodes():
            gpu_id = get_gpulevel_node_location(sdfg, state, node)
            if gpu_id is not None:
                gpu_ids.add(gpu_id)

            if isinstance(node, (nodes.CodeNode, nodes.EntryNode)):
                if node.instrument == dtypes.InstrumentationType.GPU_Events:
                    idstr = self._idstr(sdfg, state, node)
                    if self.debug:
                        local_stream.write(
                            f"// state begin: {node}: {gpu_id}\n", sdfg,
                            state_id, node)
                    local_stream.write(self._create_event('b' + idstr, gpu_id),
                                       sdfg, state_id, node)
                    local_stream.write(self._create_event('e' + idstr, gpu_id),
                                       sdfg, state_id, node)

        # Create and record a CUDA/HIP event for the entire state
        if state.instrument == dtypes.InstrumentationType.GPU_Events:
            for gpu_id in gpu_ids:
                if self.debug:
                    local_stream.write(f"\n// state begin: {node}: {gpu_id}\n",
                                       sdfg, state_id, node)
                idstr = self._idstr(sdfg, state, None) + f'__{gpu_id}'
                local_stream.write(self._create_event('b' + idstr, gpu_id),
                                   sdfg, state_id)
                local_stream.write(self._record_event('b' + idstr, 0, gpu_id),
                                   sdfg, state_id)
                local_stream.write(self._create_event('e' + idstr, gpu_id),
                                   sdfg, state_id)

    def on_state_end(self, sdfg, state, local_stream, global_stream=None):
        state_id = sdfg.node_id(state)
        gpu_ids = set()
        for node in state.nodes():
            gpu_id = get_gpulevel_node_location(sdfg, state, node)
            if gpu_id is not None:
                gpu_ids.add(gpu_id)
        # Record and measure state stream event
        if state.instrument == dtypes.InstrumentationType.GPU_Events:
            for gpu_id in gpu_ids:
                if self.debug:
                    local_stream.write(f"\n// state end: {node}: {gpu_id}\n",
                                       sdfg, state_id, node)
                idstr = self._idstr(sdfg, state, None) + f'__{gpu_id}'
                local_stream.write(self._record_event('e' + idstr, 0, gpu_id),
                                   sdfg, state_id)
                local_stream.write(
                    self._report(f'State {state.label}',
                                 sdfg,
                                 state,
                                 idstr=idstr,
                                 gpu_id=gpu_id), sdfg, state_id)
                local_stream.write(self._destroy_event('b' + idstr, gpu_id),
                                   sdfg, state_id)
                local_stream.write(self._destroy_event('e' + idstr, gpu_id),
                                   sdfg, state_id)

        # Destroy CUDA/HIP events for scopes in the state
        for node in state.nodes():
            if isinstance(node, nodes.CodeNode):
                # if isinstance(node, nodes.ExitNode):
                #     self.on_scope_exit(sdfg, state, node, local_stream)
                # if isinstance(node, nodes.Tasklet):
                #     self.on_node_end(sdfg, state, node, local_stream)
                gpu_id = get_gpulevel_node_location(sdfg, state, node)
                if node.instrument == dtypes.InstrumentationType.GPU_Events:
                    if self.debug:
                        local_stream.write(
                            f"\n// state end: {node}: {gpu_id}\n", sdfg,
                            state_id, node)
                    idstr = self._idstr(sdfg, state, node)
                    local_stream.write(self._destroy_event('b' + idstr, gpu_id),
                                       sdfg, state_id, node)
                    local_stream.write(self._destroy_event('e' + idstr, gpu_id),
                                       sdfg, state_id, node)
            if isinstance(node, nodes.EntryNode):
                gpu_id = get_gpulevel_node_location(sdfg, state, node)
                map_or_consume = node.consume if hasattr(
                    node, 'consume') else getattr(node, 'map', None)
                if map_or_consume.instrument == dtypes.InstrumentationType.GPU_Events:
                    idstr = self._idstr(sdfg, state, node)
                    if self.debug:
                        local_stream.write(
                            f"\n// state end: {node}: {gpu_id}\n", sdfg,
                            state_id, node)
                    local_stream.write(self._destroy_event('b' + idstr, gpu_id),
                                       sdfg, state_id, node)
                    local_stream.write(self._destroy_event('e' + idstr, gpu_id),
                                       sdfg, state_id, node)

    def on_scope_entry(self,
                       sdfg,
                       state,
                       node,
                       outer_stream,
                       inner_stream=None,
                       global_stream=None):
        state_id = sdfg.node_id(state)
        if node.instrument == dtypes.InstrumentationType.GPU_Events:
            if node.schedule != dtypes.ScheduleType.GPU_Device:
                raise TypeError('GPU Event instrumentation only applies to '
                                'GPU_Device map scopes')
            idstr = 'b' + self._idstr(sdfg, state, node)
            gpu_id = node.location['gpu']
            stream = node._cuda_stream[gpu_id]
            if self.debug:
                outer_stream.write(f"\n// scope entry: {node}: {gpu_id}\n",
                                   sdfg, state_id, node)
            outer_stream.write(self._record_event(idstr, stream, gpu_id), sdfg,
                               state_id, node)

    def on_scope_exit(self,
                      sdfg,
                      state,
                      node,
                      outer_stream,
                      inner_stream=None,
                      global_stream=None):
        state_id = sdfg.node_id(state)
        entry_node = state.entry_node(node)
        map_or_consume = node.consume if hasattr(node, 'consume') else node.map
        if map_or_consume.instrument == dtypes.InstrumentationType.GPU_Events:
            idstr = 'e' + self._idstr(sdfg, state, entry_node)
            gpu_id = map_or_consume.location['gpu']
            stream = entry_node._cuda_stream[gpu_id]
            if self.debug:
                outer_stream.write(f"\n// scope exit: {node}: {gpu_id}\n", sdfg,
                                   state_id, node)
            outer_stream.write(self._record_event(idstr, stream, gpu_id), sdfg,
                               state_id, node)
            outer_stream.write(
                self._report(f'{type(map_or_consume).__name__} {node.label}',
                             sdfg,
                             state,
                             entry_node,
                             gpu_id=gpu_id), sdfg, state_id, node)

    def on_node_begin(self,
                      sdfg,
                      state,
                      node,
                      outer_stream,
                      inner_stream=None,
                      global_stream=None):
        if (not isinstance(node, nodes.CodeNode)
                or is_devicelevel_gpu(sdfg, state, node)):
            return
        # Only run for host nodes
        # TODO(later): Implement "clock64"-based GPU counters
        if node.instrument == dtypes.InstrumentationType.GPU_Events:
            try:
                gpu_id = node.location['gpu']
            except KeyError:
                return
            state_id = sdfg.node_id(state)
            idstr = 'b' + self._idstr(sdfg, state, node)
            stream = node._cuda_stream[gpu_id]
            if self.debug:
                outer_stream.write(f"\n// node begin: {node}: {gpu_id}\n", sdfg,
                                   state_id, node)
            outer_stream.write(self._record_event(idstr, stream, gpu_id), sdfg,
                               state_id, node)

    def on_node_end(self,
                    sdfg,
                    state,
                    node,
                    outer_stream,
                    inner_stream=None,
                    global_stream=None):
        if (not isinstance(node, nodes.Tasklet)
                or is_devicelevel_gpu(sdfg, state, node)):
            return
        # Only run for host nodes
        # TODO(later): Implement "clock64"-based GPU counters
        if node.instrument == dtypes.InstrumentationType.GPU_Events:
            try:
                gpu_id = node.location['gpu']
            except KeyError:
                return
            state_id = sdfg.node_id(state)
            idstr = 'e' + self._idstr(sdfg, state, node)
            stream = node._cuda_stream[gpu_id]
            if self.debug:
                outer_stream.write(f"\n// node end: {node}: {gpu_id}\n", sdfg,
                                   state_id, node)
            outer_stream.write(self._record_event(idstr, stream, gpu_id), sdfg,
                               state_id, node)
            outer_stream.write(
                self._report(f'{type(node).__name__} {node.label}',
                             sdfg,
                             state,
                             node,
                             gpu_id=gpu_id), sdfg, state_id, node)
