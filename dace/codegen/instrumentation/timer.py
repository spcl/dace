from dace import types
from dace.codegen.instrumentation.provider import InstrumentationProvider
from dace.codegen.prettycode import CodeIOStream


class TimerProvider(InstrumentationProvider):
    """ Timing instrumentation that prints wall-clock time directly after
        timed execution is complete. """

    def on_sdfg_begin(self, sdfg, local_stream, global_stream):
        if isinstance(sdfg.global_code, dict):
            gc = sdfg.global_code['code_or_block']
        elif isinstance(sdfg.global_code, str):
            gc = sdfg.global_code
        else:
            gc = ''

        sdfg.set_global_code(gc + '\n#include <chrono>')

    def on_tbegin(self, stream: CodeIOStream, sdfg=None, state=None,
                  node=None):
        if state is not None:
            if node is not None:
                node = state.node_id(node)
            state = sdfg.node_id(state)

        stream.write(
            'auto __dace_tbegin = std::chrono::high_resolution_clock::now();')

    def on_tend(self,
                timer_name: str,
                stream: CodeIOStream,
                sdfg=None,
                state=None,
                node=None):
        if state is not None:
            if node is not None:
                node = state.node_id(node)
            state = sdfg.node_id(state)

        stream.write('''
auto __dace_tend = std::chrono::high_resolution_clock::now();
std::chrono::duration<double, std::milli> __dace_tdiff = __dace_tend - __dace_tbegin;
printf("{timer_name}: %lf ms\\n", __dace_tdiff.count());'''
                     .format(timer_name=timer_name))

    # Code generation hooks
    def on_state_begin(self, sdfg, state, local_stream, global_stream):
        if state.instrument == types.InstrumentationType.Timer:
            self.on_tbegin(local_stream, sdfg, state)

    def on_state_end(self, sdfg, state, local_stream, global_stream):
        if state.instrument == types.InstrumentationType.Timer:
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
        if s.instrument == types.InstrumentationType.Timer:
            self.on_tbegin(outer_stream, sdfg, state, node)

    def on_scope_exit(self, sdfg, state, node, outer_stream, inner_stream,
                      global_stream):
        s = self._get_sobj(node)
        if s.instrument == types.InstrumentationType.Timer:
            self.on_tend('%s %s' % (type(s).__name__, s.label), outer_stream,
                         sdfg, state, node)
