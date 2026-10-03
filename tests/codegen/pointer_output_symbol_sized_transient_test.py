# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
""" A pointer-typed tasklet output into a transient sized by an assigned symbol, written in branches. """
import numpy as np

import dace
from dace.sdfg.state import ConditionalBlock, ControlFlowRegion


def _writer(region: ControlFlowRegion, label: str, value: float) -> None:
    state = region.add_state(label, is_start_block=True)
    tasklet = state.add_tasklet(label, {}, {'o': dace.pointer(dace.float64)},
                                f'for (int i = 0; i < M; ++i) o[i] = {value};',
                                language=dace.Language.CPP)
    state.add_edge(tasklet, 'o', state.add_write('T'), None, dace.Memlet('T[0:M]'))


def test_pointer_output_into_transient_allocated_in_a_dominating_state():
    """``T`` is sized by ``M``, assigned on an edge, so it is declared at SDFG scope and allocated in the state that
    dominates its accesses. That state's scope ends before the branches are generated; the pointer output connector
    in each branch must still find ``T``'s declaration."""
    sdfg = dace.SDFG('pointer_output_symbol_sized')
    sdfg.add_symbol('M', dace.int64)
    sdfg.add_array('out', [4], dace.float64)
    sdfg.add_scalar('c', dace.int32)
    sdfg.add_transient('T', [dace.symbol('M')], dace.float64)

    start = sdfg.add_state('start', is_start_block=True)
    pre = sdfg.add_state('pre')
    sdfg.add_edge(start, pre, dace.InterstateEdge(assignments={'M': '4'}))
    branches = ConditionalBlock('pick')
    sdfg.add_node(branches)
    sdfg.add_edge(pre, branches, dace.InterstateEdge())
    then_body = ControlFlowRegion('then', sdfg=sdfg)
    else_body = ControlFlowRegion('else', sdfg=sdfg)
    branches.add_branch(dace.properties.CodeBlock('c > 0'), then_body)
    branches.add_branch(None, else_body)
    _writer(then_body, 'write_one', 1.0)
    _writer(else_body, 'write_two', 2.0)
    read = sdfg.add_state('read')
    sdfg.add_edge(branches, read, dace.InterstateEdge())
    read.add_nedge(read.add_read('T'), read.add_write('out'), dace.Memlet('T[0:4]'))

    out = np.zeros(4)
    sdfg(c=np.int32(1), out=out)
    assert np.all(out == 1.0)
    sdfg(c=np.int32(0), out=out)
    assert np.all(out == 2.0)


if __name__ == '__main__':
    test_pointer_output_into_transient_allocated_in_a_dominating_state()
