# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Bracket an SDFG with a steady-clock timer that reports its wall-clock runtime in nanoseconds."""
from typing import Any, Dict, Optional

from dace import Memlet, dtypes, properties
from dace.sdfg.sdfg import SDFG, InterstateEdge
from dace.sdfg.state import SDFGState
from dace.transformation import pass_pipeline as ppl, transformation

NOW_NS = ('std::chrono::duration_cast<std::chrono::nanoseconds>('
          'std::chrono::steady_clock::now().time_since_epoch()).count()')


@properties.make_properties
@transformation.explicit_cf_compatible
class InstrumentWithTimer(ppl.Pass):
    """
    Adds a new start state and a new sink state, each holding one side-effecting C++ tasklet that samples
    ``std::chrono::steady_clock``. The first stores the start instant in the transient ``time_start``, the second
    writes ``now - time_start`` into the new non-transient ``time_ns``, which joins the SDFG signature.

    Both tasklets read and write, so neither is a pure sink or source that dead-code elimination may drop, but only
    the second one writes the reported result.
    """

    CATEGORY: str = 'Helper'

    def modifies(self) -> ppl.Modifies:
        return ppl.Modifies.Descriptors | ppl.Modifies.States

    def should_reapply(self, modified: ppl.Modifies) -> bool:
        return False

    def wire(self, state: SDFGState, src: str, dst: str, code: str, code_global: str) -> None:
        """Read ``src``, write ``dst``, through one side-effecting C++ tasklet in ``state``."""
        tasklet = state.add_tasklet(state.label, {'__in': dtypes.int64}, {'__out': dtypes.int64},
                                    code,
                                    dtypes.Language.CPP,
                                    code_global=code_global,
                                    side_effects=True)
        state.add_edge(state.add_read(src), None, tasklet, '__in', Memlet(data=src, subset='0'))
        state.add_edge(tasklet, '__out', state.add_write(dst), None, Memlet(data=dst, subset='0'))

    def apply_pass(self, sdfg: SDFG, _: Dict[str, Any]) -> Optional[str]:
        """
        :param sdfg: The SDFG to instrument in-place.
        :return: Name of the descriptor carrying the elapsed time, or None if the SDFG is already instrumented.
        """
        if 'time_ns' in sdfg.arrays:  # A second timer would only measure the first one, so refuse.
            return None
        sdfg.add_array('time_ns', (1, ), dtypes.int64, transient=False)
        sdfg.add_array('time_start', (1, ), dtypes.int64, transient=True)
        begin = sdfg.add_state_before(sdfg.start_state, 'timer_begin', is_start_block=True)
        sinks = [n for n in sdfg.nodes() if sdfg.out_degree(n) == 0]
        end = sdfg.add_state('timer_end')
        for sink in sinks:
            sdfg.add_edge(sink, end, InterstateEdge())
        self.wire(begin, 'time_ns', 'time_start', f'__out = {NOW_NS};', '#include <chrono>')
        self.wire(end, 'time_start', 'time_ns', f'__out = {NOW_NS} - __in;', '')
        return 'time_ns'
