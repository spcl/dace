# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""``_find_unconditionally_executed_states`` must return only ``SDFGState``s.

It derives its result from the top-level CFG's dominators, and since control-flow regions those
dominators are ``ControlFlowBlock``s -- ``LoopRegion`` and ``ConditionalBlock`` included. Callers
index the returned blocks as dataflow graphs (``state.in_edges(access_node)``), but a region's node
set holds blocks rather than AccessNodes, so handing one back raised
``KeyError: AccessNode (...)``. Real CloudSC graphs hit this: the pass returned 15 SDFGStates
alongside 5 ConditionalBlocks and a LoopRegion.
"""
import dace
from dace.properties import CodeBlock
from dace.sdfg.state import ConditionalBlock, ControlFlowRegion, SDFGState
from dace.sdfg.analysis.writeset_underapproximation import (UnderapproximateWrites,
                                                            _find_unconditionally_executed_states)

N = 8


def build_conditional_with_nested_sdfg() -> dace.SDFG:
    """A ConditionalBlock at top level, with a nested SDFG that writes an array inside it."""
    inner = dace.SDFG('inner')
    inner.add_array('ia', [N], dace.float64)
    istate = inner.add_state('iwrite', is_start_block=True)
    itask = istate.add_tasklet('set', {}, {'o'}, 'o = 1.0')
    istate.add_edge(itask, 'o', istate.add_access('ia'), None, dace.Memlet('ia[0]'))

    sdfg = dace.SDFG('cond_with_nsdfg')
    sdfg.add_array('a', [N], dace.float64)
    sdfg.add_symbol('flag', dace.int32)

    entry = sdfg.add_state('entry', is_start_block=True)
    cb = ConditionalBlock('cb')
    sdfg.add_node(cb)
    sdfg.add_edge(entry, cb, dace.InterstateEdge())

    branch = ControlFlowRegion('branch', sdfg=sdfg)
    bstate = branch.add_state('bstate', is_start_block=True)
    nsdfg = bstate.add_nested_sdfg(inner, {}, {'ia'})
    bstate.add_edge(nsdfg, 'ia', bstate.add_access('a'), None, dace.Memlet('a[0:8]'))
    cb.add_branch(CodeBlock('flag > 0'), branch)

    tail = sdfg.add_state('tail')
    sdfg.add_edge(cb, tail, dace.InterstateEdge())
    sdfg.validate()
    return sdfg


def test_only_sdfg_states_are_returned():
    sdfg = build_conditional_with_nested_sdfg()
    blocks = _find_unconditionally_executed_states(sdfg)
    offenders = [type(b).__name__ for b in blocks if not isinstance(b, SDFGState)]
    assert not offenders, offenders


def test_underapproximate_writes_handles_regions():
    """The regression: this raised KeyError when a region reached the AccessNode loop."""
    sdfg = build_conditional_with_nested_sdfg()
    result = UnderapproximateWrites().apply_pass(sdfg, {})
    assert result is not None


if __name__ == '__main__':
    test_only_sdfg_states_are_returned()
    test_underapproximate_writes_handles_regions()
