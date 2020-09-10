# Copyright 2019-2020 ETH Zurich and the DaCe authors. All rights reserved.
from dace import dtypes, registry
from dace.sdfg.nodes import CodeNode
from dace.codegen.instrumentation.provider import InstrumentationProvider
from dace.codegen.prettycode import CodeIOStream


@registry.autoregister_params(type=dtypes.InstrumentationType.Timer)
class TimerProvider(InstrumentationProvider):
    """ Timing instrumentation that reports wall-clock time directly after
        timed execution is complete. """
    def on_sdfg_begin(self, sdfg, local_stream, global_stream):
        global_stream.write('#include <chrono>')

        # For other file headers
        sdfg.append_global_code('\n#include <chrono>', None)

    def on_tbegin(self, stream: CodeIOStream, sdfg=None, state=None, node=None):
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
dace::perf::report.add("timer_{timer_name}", __dace_tdiff_{id}.count());'''.
            format(timer_name=timer_name, id=idstr))

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

    def on_node_begin(self, sdfg, state, node, outer_stream, inner_stream,
                      global_stream):
        if not isinstance(node, CodeNode):
            return
        if node.instrument == dtypes.InstrumentationType.Timer:
            self.on_tbegin(outer_stream, sdfg, state, node)

    def on_node_end(self, sdfg, state, node, outer_stream, inner_stream,
                    global_stream):
        if not isinstance(node, CodeNode):
            return
        if node.instrument == dtypes.InstrumentationType.Timer:
            idstr = self._idstr(sdfg, state, node)
            self.on_tend('%s %s' % (type(node).__name__, idstr), outer_stream,
                         sdfg, state, node)
