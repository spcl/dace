from dace import dtypes
from dace.codegen.instrumentation.provider import InstrumentationProvider
from dace.codegen.prettycode import CodeIOStream


class TimerProvider(InstrumentationProvider):
    """ Timing instrumentation that prints wall-clock time directly after
        timed execution is complete. """

    def on_sdfg_begin(self, sdfg, local_stream, global_stream):
        global_stream.write('#include <chrono>')

        # For other file headers
        if len(sdfg.global_code) == 0:
            sdfg.set_global_code('#include <chrono>')
        else:
            sdfg.set_global_code(sdfg.global_code + '\n#include <chrono>')

    def _idstr(self, sdfg, state, node):
        if state is not None:
            if node is not None:
                node = state.node_id(node)
            else:
                node = ''
            state = sdfg.node_id(state)
        else:
            state = ''
        return str(state) + '_' + str(node)

    def on_tbegin(self, stream: CodeIOStream, sdfg=None, state=None,
                  node=None):
        idstr = self._idstr(sdfg, state, node)

        stream.write(
            'auto __dace_tbegin_%s = std::chrono::high_resolution_clock::now();'
            % idstr)

    def on_tend(self,
                timer_name: str,
                stream: CodeIOStream,
                sdfg=None,
                state=None,
                node=None):
        idstr = self._idstr(sdfg, state, node)

        stream.write(
            '''auto __dace_tend_{id} = std::chrono::high_resolution_clock::now();
std::chrono::duration<double, std::milli> __dace_tdiff_{id} = __dace_tend_{id} - __dace_tbegin_{id};
printf("{timer_name}: %lf ms\\n", __dace_tdiff_{id}.count());'''.format(
                timer_name=timer_name, id=idstr))

    # Code generation hooks
    def on_state_begin(self, sdfg, state, local_stream, global_stream):
        if state.instrument == dtypes.InstrumentationType.Timer:
            self.on_tbegin(local_stream, sdfg, state)

    def on_state_end(self, sdfg, state, local_stream, global_stream):
        if state.instrument == dtypes.InstrumentationType.Timer:
            self.on_tend('State %s' % state.label, local_stream, sdfg, state)

    def _get_sobj(self, node):
        # Get object behind scope
        if hasattr(node, 'consume'):
            return node.consume
        else:
            return node.map

    def on_scope_entry(self, sdfg, state, node, outer_stream, inner_stream,
                       global_stream):
        s = self._get_sobj(node)
        if s.instrument == dtypes.InstrumentationType.Timer:
            self.on_tbegin(outer_stream, sdfg, state, node)

    def on_scope_exit(self, sdfg, state, node, outer_stream, inner_stream,
                      global_stream):
        entry_node = state.entry_node(node)
        s = self._get_sobj(node)
        if s.instrument == dtypes.InstrumentationType.Timer:
            self.on_tend('%s %s' % (type(s).__name__, s.label), outer_stream,
                         sdfg, state, entry_node)
