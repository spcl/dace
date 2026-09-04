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


def test_cascade_refuses_to_create_an_interstate_race():
    sdfg = build_gather_loop()
    sdfg.validate()

    CascadeInterstateEdgeAssignmentsUp().apply_pass(sdfg, {})

    # Whatever the pass chose to do, the result must still be a valid SDFG: no edge may carry an
    # assignment whose rhs reads a symbol another assignment on that same edge writes.
    sdfg.validate()
    for edge in sdfg.all_interstate_edges():
        assigned = set(edge.data.assignments.keys())
        for name, rhs in edge.data.assignments.items():
            reads = set(dace.symbolic.free_symbols_and_functions(rhs))
            racing = (reads & assigned) - {name}
            assert not racing, f'edge assigns {name} = {rhs} while also writing {racing}'


if __name__ == '__main__':
    test_cascade_refuses_to_create_an_interstate_race()
