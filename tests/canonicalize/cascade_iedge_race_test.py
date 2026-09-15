# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""``CascadeInterstateEdgeAssignmentsUp`` must not land an assignment on an edge it races."""

import dace
from dace.sdfg.sdfg import InterstateEdge
from dace.sdfg.state import LoopRegion
from dace.transformation.passes.canonicalize.cascade_iedge_assignments_up import (
    CascadeInterstateEdgeAssignmentsUp, )


def build_gather_loop() -> dace.SDFG:
    """TSVC s353 shape: an indirect read hoisted next to the loop's own counter update.

    The assignments on one interstate edge are unordered -- they all read the state on entry --
    so an edge carrying both ``ip_index = ip[i]`` and ``i = i + 1`` is rejected by validation as
    a race. Every legality guard in the pass examines the child's interior and predecessors;
    none looked at what the destination edge already assigns.
    """
    sdfg = dace.SDFG('s353_gather')
    sdfg.add_array('ip', [16], dace.int64)
    sdfg.add_array('a', [16], dace.float64)
    sdfg.add_symbol('ip_index', dace.int64)
    sdfg.add_symbol('i', dace.int64)

    outer = LoopRegion('outer', 'k < 4', 'k', 'k = 0', 'k = k + 1')
    sdfg.add_node(outer, is_start_block=True)

    body = outer.add_state('body', is_start_block=True)
    tail = outer.add_state('tail')
    # The counter update and the indirect read share one edge only if the pass puts them there.
    outer.add_edge(body, tail, InterstateEdge(assignments={'i': 'i + 1'}))
    inner = outer.add_state('inner')
    outer.add_edge(tail, inner, InterstateEdge(assignments={'ip_index': 'ip[i]'}))
    return sdfg


def assignment_map(sdfg: dace.SDFG) -> dict[tuple[str, str], dict[str, str]]:
    """Every interstate edge's assignments, keyed by the block labels it connects."""
    return {(e.src.label, e.dst.label): dict(e.data.assignments) for e in sdfg.all_interstate_edges()}


def test_cascade_refuses_to_create_an_interstate_race():
    sdfg = build_gather_loop()
    sdfg.validate()
    before = assignment_map(sdfg)

    applied = CascadeInterstateEdgeAssignmentsUp().apply_pass(sdfg, {})

    # ``validate`` already rejects an edge whose assignment reads a symbol a sibling assignment on
    # that same edge writes (``dace/sdfg/validation.py``), so re-deriving that rule here would only
    # copy the production check. What it does NOT say is that the pass declined: the contract is
    # that ``ip_index`` stayed where it was instead of joining ``i = i + 1``.
    sdfg.validate()
    assert applied is None, f'the pass rewrote a graph whose only candidate edge races: {applied}'
    assert assignment_map(sdfg) == before, 'the refused cascade still moved an assignment'
    assert before[('body', 'tail')] == {'i': 'i + 1'}
    assert before[('tail', 'inner')] == {'ip_index': 'ip[i]'}


if __name__ == '__main__':
    test_cascade_refuses_to_create_an_interstate_race()
