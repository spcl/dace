# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
""" Tests that a symbol replacement leaves name-typed slots holding strings. ``safe_replace``
hands its callback symbolic values, so a consumer that writes one straight into a dictionary key
or a ``Property(dtype=str)`` produces an SDFG that no longer serializes. This is the path
``SDFGState.add_nested_sdfg`` takes when it applies a nested SDFG's symbol mapping. """

import dace
from dace import symbolic
from dace.sdfg.state import LoopRegion


def test_interstate_assignment_keys_stay_strings():
    """A sympy object used as an ``assignments`` key breaks ``to_json``."""
    sdfg = dace.SDFG('assignment_keys')
    sdfg.add_symbol('i', dace.int64)
    start = sdfg.add_state('start', is_start_block=True)
    end = sdfg.add_state('end')
    sdfg.add_edge(start, end, dace.InterstateEdge(assignments={'i': '1'}))

    symbolic.safe_replace({'i': 'k'}, lambda m: sdfg.replace_dict(m))

    edge = sdfg.edges()[0]
    assert all(isinstance(key, str) for key in edge.data.assignments), edge.data.assignments
    assert 'k' in edge.data.assignments, edge.data.assignments
    sdfg.to_json()


def test_loop_variable_stays_a_string():
    """``LoopRegion.loop_variable`` is a ``Property(dtype=str)`` and rejects a symbol."""
    sdfg = dace.SDFG('loop_variable_type')
    sdfg.add_symbol('i', dace.int64)
    loop = LoopRegion('loop', condition_expr='i < 10', loop_var='i', initialize_expr='i = 0', update_expr='i = i + 1')
    sdfg.add_node(loop, is_start_block=True)
    loop.add_state('body', is_start_block=True)

    symbolic.safe_replace({'i': 'k'}, lambda m: sdfg.replace_dict(m))

    assert isinstance(loop.loop_variable, str), type(loop.loop_variable)
    assert loop.loop_variable == 'k', loop.loop_variable
    sdfg.to_json()


if __name__ == '__main__':
    test_interstate_assignment_keys_stay_strings()
    test_loop_variable_stays_a_string()
