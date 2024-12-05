# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
from dace import dtypes, registry
from dace.sdfg.nodes import CodeNode
from dace.codegen.instrumentation.provider import InstrumentationProvider
from dace.codegen.prettycode import CodeIOStream


@registry.autoregister_params(type=dtypes.InstrumentationType.Timer)
class TimerProvider(InstrumentationProvider):
    """ Timing instrumentation that reports wall-clock time directly after
        timed execution is complete. """
    def on_sdfg_begin(self, sdfg, local_stream, global_stream, codegen):
        global_stream.write('#include <chrono>')

        # For other file headers
        sdfg.append_global_code('\n#include <chrono>', None)

        if sdfg.instrument == dtypes.InstrumentationType.Timer:
            self.on_tbegin(local_stream, sdfg, sdfg)

    def on_sdfg_end(self, sdfg, local_stream, global_stream):
        if sdfg.instrument == dtypes.InstrumentationType.Timer:
            self.on_tend('SDFG %s' % sdfg.name, local_stream, sdfg, sdfg)

    def on_tbegin(self, stream: CodeIOStream, sdfg=None, cfg=None, state=None, node=None):
        idstr = self._idstr(cfg, state, node)

        stream.write('auto __dace_tbegin_%s = std::chrono::high_resolution_clock::now();' % idstr)

    def on_tend(self, timer_name: str, stream: CodeIOStream, sdfg=None, cfg=None, state=None, node=None):
        idstr = self._idstr(cfg, state, node)

        state_id = -1
        node_id = -1
        if state is not None:
            state_id = state.block_id
            if node is not None:
                node_id = state.node_id(node)

        stream.write('''auto __dace_tend_{id} = std::chrono::high_resolution_clock::now();
unsigned long int __dace_ts_start_{id} = std::chrono::duration_cast<std::chrono::microseconds>(__dace_tbegin_{id}.time_since_epoch()).count();
unsigned long int __dace_ts_end_{id} = std::chrono::duration_cast<std::chrono::microseconds>(__dace_tend_{id}.time_since_epoch()).count();
__state->report.add_completion("{timer_name}", "Timer", __dace_ts_start_{id}, __dace_ts_end_{id}, {cfg_id}, {state_id}, {node_id});'''
                     .format(timer_name=timer_name, id=idstr, cfg_id=cfg.cfg_id, state_id=state_id, node_id=node_id))

    # Code generation hooks
    def on_state_begin(self, sdfg, cfg, state, local_stream, global_stream):
        if state.instrument == dtypes.InstrumentationType.Timer:
            self.on_tbegin(local_stream, sdfg, cfg, state)

    def on_state_end(self, sdfg, cfg, state, local_stream, global_stream):
        if state.instrument == dtypes.InstrumentationType.Timer:
            self.on_tend('State %s' % state.label, local_stream, sdfg, cfg, state)

    def _get_sobj(self, node):
        # Get object behind scope
        if hasattr(node, 'consume'):
            return node.consume
        else:
            return node.map

    def on_scope_entry(self, sdfg, cfg, state, node, outer_stream, inner_stream, global_stream):
        s = self._get_sobj(node)
        if s.instrument == dtypes.InstrumentationType.Timer:
            self.on_tbegin(outer_stream, sdfg, cfg, state, node)

    def on_scope_exit(self, sdfg, cfg, state, node, outer_stream, inner_stream, global_stream):
        entry_node = state.entry_node(node)
        s = self._get_sobj(node)
        if s.instrument == dtypes.InstrumentationType.Timer:
            self.on_tend('%s %s' % (type(s).__name__, s.label), outer_stream, sdfg, cfg, state, entry_node)

    def on_node_begin(self, sdfg, cfg, state, node, outer_stream, inner_stream, global_stream):
        if not isinstance(node, CodeNode):
            return
        if node.instrument == dtypes.InstrumentationType.Timer:
            self.on_tbegin(outer_stream, sdfg, cfg, state, node)

    def on_node_end(self, sdfg, cfg, state, node, outer_stream, inner_stream, global_stream):
        if not isinstance(node, CodeNode):
            return
        if node.instrument == dtypes.InstrumentationType.Timer:
            idstr = self._idstr(cfg, state, node)
            self.on_tend('%s %s' % (type(node).__name__, idstr), outer_stream, sdfg, cfg, state, node)
