from dace import dtypes, registry
from dace.sdfg import nodes, is_devicelevel_gpu
from dace.codegen.instrumentation.provider import InstrumentationProvider


@registry.autoregister_params(type=dtypes.InstrumentationType.CUDA_Events)
class CUDAEventProvider(InstrumentationProvider):
    """ Timing instrumentation that reports GPU/copy time using CUDA events. """
    def on_sdfg_begin(self, sdfg, local_stream, global_stream):
        global_stream.write('#include <cuda_runtime.h>')

        # For other file headers
        sdfg.append_global_code('\n#include <cuda_runtime.h>', None)

    def _get_sobj(self, node):
        # Get object behind scope
        if hasattr(node, 'consume'):
            return node.consume
        else:
            return node.map

    def _create_event(self, id):
        return '''cudaEvent_t __dace_ev_{id};
cudaEventCreate(&__dace_ev_{id});'''.format(id=id)

    def _destroy_event(self, id):
        return 'cudaEventDestroy(__dace_ev_%s);' % id

    def _record_event(self, id, stream):
        return 'cudaEventRecord(__dace_ev_%s, dace::cuda::__streams[%d]);' % (
            id, stream)

    def _report(self, timer_name: str, sdfg=None, state=None, node=None):
        idstr = self._idstr(sdfg, state, node)

        return '''float __dace_ms_{id} = -1.0f;
cudaEventSynchronize(__dace_ev_e{id});
cudaEventElapsedTime(&__dace_ms_{id}, __dace_ev_b{id}, __dace_ev_e{id});
dace::perf::report.add("cudaev_{timer_name}", __dace_ms_{id});'''.format(
            id=idstr, timer_name=timer_name)

    # Code generation hooks
    def on_state_begin(self, sdfg, state, local_stream, global_stream):
        state_id = sdfg.node_id(state)
        # Create CUDA events for each instrumented scope in the state
        for node in state.nodes():
            if isinstance(node, (nodes.CodeNode, nodes.EntryNode)):
                s = (self._get_sobj(node)
                     if isinstance(node, nodes.EntryNode) else node)
                if s.instrument == dtypes.InstrumentationType.CUDA_Events:
                    idstr = self._idstr(sdfg, state, node)
                    local_stream.write(self._create_event('b' + idstr), sdfg,
                                       state_id, node)
                    local_stream.write(self._create_event('e' + idstr), sdfg,
                                       state_id, node)

        # Create and record a CUDA event for the entire state
        if state.instrument == dtypes.InstrumentationType.CUDA_Events:
            idstr = 'b' + self._idstr(sdfg, state, None)
            local_stream.write(self._create_event(idstr), sdfg, state_id)
            local_stream.write(self._record_event(idstr, 0), sdfg, state_id)
            idstr = 'e' + self._idstr(sdfg, state, None)
            local_stream.write(self._create_event(idstr), sdfg, state_id)

    def on_state_end(self, sdfg, state, local_stream, global_stream):
        state_id = sdfg.node_id(state)
        # Record and measure state stream event
        if state.instrument == dtypes.InstrumentationType.CUDA_Events:
            idstr = self._idstr(sdfg, state, None)
            local_stream.write(self._record_event('e' + idstr, 0), sdfg,
                               state_id)
            local_stream.write(
                self._report('State %s' % state.label, sdfg, state), sdfg,
                state_id)
            local_stream.write(self._destroy_event('b' + idstr), sdfg, state_id)
            local_stream.write(self._destroy_event('e' + idstr), sdfg, state_id)

        # Destroy CUDA events for scopes in the state
        for node in state.nodes():
            if isinstance(node, (nodes.CodeNode, nodes.EntryNode)):
                s = (self._get_sobj(node)
                     if isinstance(node, nodes.EntryNode) else node)
                if s.instrument == dtypes.InstrumentationType.CUDA_Events:
                    idstr = self._idstr(sdfg, state, node)
                    local_stream.write(self._destroy_event('b' + idstr), sdfg,
                                       state_id, node)
                    local_stream.write(self._destroy_event('e' + idstr), sdfg,
                                       state_id, node)

    def on_scope_entry(self, sdfg, state, node, outer_stream, inner_stream,
                       global_stream):
        state_id = sdfg.node_id(state)
        s = self._get_sobj(node)
        if s.instrument == dtypes.InstrumentationType.CUDA_Events:
            if s.schedule != dtypes.ScheduleType.GPU_Device:
                raise TypeError('CUDA Event instrumentation only applies to '
                                'GPU_Device map scopes')

            idstr = 'b' + self._idstr(sdfg, state, node)
            outer_stream.write(self._record_event(idstr, node._cuda_stream),
                               sdfg, state_id, node)

    def on_scope_exit(self, sdfg, state, node, outer_stream, inner_stream,
                      global_stream):
        state_id = sdfg.node_id(state)
        entry_node = state.entry_node(node)
        s = self._get_sobj(node)
        if s.instrument == dtypes.InstrumentationType.CUDA_Events:
            idstr = 'e' + self._idstr(sdfg, state, entry_node)
            outer_stream.write(self._record_event(idstr, node._cuda_stream),
                               sdfg, state_id, node)
            outer_stream.write(
                self._report('%s %s' % (type(s).__name__, s.label), sdfg, state,
                             entry_node), sdfg, state_id, node)

    def on_node_begin(self, sdfg, state, node, outer_stream, inner_stream,
                      global_stream):
        if (not isinstance(node, nodes.CodeNode)
                or is_devicelevel_gpu(sdfg, state, node)):
            return
        # Only run for host nodes
        # TODO(later): Implement "clock64"-based GPU counters
        if node.instrument == dtypes.InstrumentationType.CUDA_Events:
            state_id = sdfg.node_id(state)
            idstr = 'b' + self._idstr(sdfg, state, node)
            outer_stream.write(self._record_event(idstr, node._cuda_stream),
                               sdfg, state_id, node)

    def on_node_end(self, sdfg, state, node, outer_stream, inner_stream,
                    global_stream):
        if (not isinstance(node, nodes.Tasklet)
                or is_devicelevel_gpu(sdfg, state, node)):
            return
        # Only run for host nodes
        # TODO(later): Implement "clock64"-based GPU counters
        if node.instrument == dtypes.InstrumentationType.CUDA_Events:
            state_id = sdfg.node_id(state)
            idstr = 'e' + self._idstr(sdfg, state, node)
            outer_stream.write(self._record_event(idstr, node._cuda_stream),
                               sdfg, state_id, node)
            outer_stream.write(
                self._report('%s %s' % (type(node).__name__, node.label), sdfg,
                             state, node), sdfg, state_id, node)
