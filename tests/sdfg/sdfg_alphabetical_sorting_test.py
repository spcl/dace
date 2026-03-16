import random

import dace
from dace.sdfg.utils import get_deterministic_node_key, get_deterministic_edge_key

# Enable alphabetical sorting for all tests in this module.
dace.Config.set("compiler", "sdfg_alphabetical_sorting", value=True)


def _scramble_dict_in_place(d):
    """Helper to randomize dictionary insertion order without changing its type."""
    if not d:
        return
    keys = list(d.keys())
    random.shuffle(keys)
    items = [(k, d[k]) for k in keys]
    d.clear()
    d.update(items)


def _scramble_sdfg(sdfg):
    """Deeply scramble all internal dictionaries of an SDFG to simulate non-determinism."""
    # Scramble top-level metadata
    _scramble_dict_in_place(sdfg._arrays)
    if hasattr(sdfg, 'symbols') and sdfg.symbols:
        _scramble_dict_in_place(sdfg.symbols)
    if hasattr(sdfg, 'constants_prop') and sdfg.constants_prop:
        _scramble_dict_in_place(sdfg.constants_prop)

    # Scramble state machine level
    _scramble_dict_in_place(sdfg._nodes)
    _scramble_dict_in_place(sdfg._edges)

    # Scramble dataflow inside each state
    for state in sdfg.nodes():
        _scramble_dict_in_place(state._nodes)
        _scramble_dict_in_place(state._edges)

        for node, (in_edges, out_edges) in state._nodes.items():
            _scramble_dict_in_place(in_edges)
            _scramble_dict_in_place(out_edges)

        # Recurse into nested SDFGs
        for node in state.nodes():
            if hasattr(node, 'sdfg') and node.sdfg is not None:
                _scramble_sdfg(node.sdfg)


def _snapshot_order(sdfg):
    """Capture the current iteration order of all internal dictionaries as a hashable snapshot."""
    result = []

    # Metadata
    result.append(('arrays', tuple(sdfg._arrays.keys())))

    # State machine
    state_node_keys = tuple(get_deterministic_node_key(n, graph=sdfg) for n in sdfg._nodes.keys())
    result.append(('sdfg_nodes', state_node_keys))

    # Dataflow per state
    for i, state in enumerate(sdfg.nodes()):
        node_keys = tuple(get_deterministic_node_key(n, graph=state) for n in state._nodes.keys())
        edge_keys = tuple(get_deterministic_edge_key(state._edges[k], graph=state) for k in state._edges.keys())
        result.append((f'state_{i}_nodes', node_keys))
        result.append((f'state_{i}_edges', edge_keys))

        for j, (node, (in_edges, out_edges)) in enumerate(state._nodes.items()):
            in_keys = tuple(get_deterministic_edge_key(in_edges[k], graph=state) for k in in_edges.keys())
            out_keys = tuple(get_deterministic_edge_key(out_edges[k], graph=state) for k in out_edges.keys())
            result.append((f'state_{i}_node_{j}_in', in_keys))
            result.append((f'state_{i}_node_{j}_out', out_keys))

    return tuple(result)


def _build_test_sdfg():
    """Build a simple SDFG with enough structure to exercise all sorting paths."""
    sdfg = dace.SDFG('deterministic_test')
    sdfg.add_array('A', [10], dace.float64)
    sdfg.add_array('B', [10], dace.float64)
    sdfg.add_scalar('s', dace.float64, transient=True)

    state = sdfg.add_state('state0')
    a = state.add_read('A')
    b = state.add_write('B')
    tasklet = state.add_tasklet('compute', {'a'}, {'b'}, 'b = a + 1')
    state.add_edge(a, None, tasklet, 'a', dace.Memlet.from_array('A', sdfg.arrays['A']))
    state.add_edge(tasklet, 'b', b, None, dace.Memlet.from_array('B', sdfg.arrays['B']))

    return sdfg


def test_sdfg_alphabetical_sorting_basic():
    """
    Tests that the SDFG and its internal states can be forced into a strictly
    deterministic topological order, regardless of dictionary insertion history.
    """
    sdfg = _build_test_sdfg()
    state = sdfg.nodes()[0]

    # Scramble everything
    _scramble_sdfg(sdfg)

    # Apply the canonicalizer
    sdfg.sort_sdfg_alphabetically()

    # Assert that graph nodes are sorted
    node_keys = list(state._nodes.keys())
    expected_node_keys = sorted(node_keys, key=lambda n: get_deterministic_node_key(n, graph=state))
    assert node_keys == expected_node_keys, "Graph nodes were not deterministically sorted!"

    # Assert that graph edges are sorted
    edge_keys = list(state._edges.keys())
    expected_edge_keys = sorted(edge_keys, key=lambda k: get_deterministic_edge_key(state._edges[k], graph=state))
    assert edge_keys == expected_edge_keys, "Graph edges were not deterministically sorted!"

    # Assert that metadata dicts are sorted
    array_keys = list(sdfg._arrays.keys())
    assert array_keys == sorted(array_keys), "SDFG arrays were not alphabetically sorted!"

    # Assert that per-node adjacency lists are sorted
    for node, (in_edges, out_edges) in state._nodes.items():
        in_keys = list(in_edges.keys())
        expected_in = sorted(in_keys, key=lambda k: get_deterministic_edge_key(in_edges[k], graph=state))
        assert in_keys == expected_in, f"In-edges for {node} were not deterministically sorted!"

        out_keys = list(out_edges.keys())
        expected_out = sorted(out_keys, key=lambda k: get_deterministic_edge_key(out_edges[k], graph=state))
        assert out_keys == expected_out, f"Out-edges for {node} were not deterministically sorted!"


def test_sdfg_alphabetical_sorting_rebuild_nx():
    """
    Tests that when rebuild_nx=True, the NetworkX backend matches
    the sorted DaCe dictionary order.
    """
    sdfg = _build_test_sdfg()
    state = sdfg.nodes()[0]

    _scramble_sdfg(sdfg)

    # Sort with NX rebuild enabled
    sdfg.sort_sdfg_alphabetically(rebuild_nx=True)

    # The NX node order must match the _nodes dict order
    nx_nodes = list(state._nx.nodes())
    dace_nodes = list(state._nodes.keys())
    assert nx_nodes == dace_nodes, ("NetworkX node order does not match sorted DaCe _nodes dict order!")


def test_sdfg_alphabetical_sorting_stability():
    """
    Tests that regardless of the initial scrambling, the sort always
    produces the same canonical order. Runs multiple scramble+sort
    cycles with different random seeds.
    """
    reference_snapshot = None

    for seed in range(10):
        sdfg = _build_test_sdfg()

        random.seed(seed)
        _scramble_sdfg(sdfg)

        sdfg.sort_sdfg_alphabetically()

        snapshot = _snapshot_order(sdfg)

        if reference_snapshot is None:
            reference_snapshot = snapshot
        else:
            assert snapshot == reference_snapshot, (f"Sort produced different order with seed={seed}! "
                                                    f"Expected:\n{reference_snapshot}\nGot:\n{snapshot}")


def test_sdfg_alphabetical_sorting_idempotency():
    """
    Tests that sorting an already-sorted SDFG produces the same result,
    i.e., the operation is idempotent.
    """
    sdfg = _build_test_sdfg()

    _scramble_sdfg(sdfg)

    # Sort once
    sdfg.sort_sdfg_alphabetically()
    snapshot_first = _snapshot_order(sdfg)

    # Sort again
    sdfg.sort_sdfg_alphabetically()
    snapshot_second = _snapshot_order(sdfg)

    assert snapshot_first == snapshot_second, ("Sorting is not idempotent! Second sort produced a different order.")


def _build_multistate_test_sdfg():
    """Build an SDFG with multiple states and interstate edges to exercise
    ControlFlowRegion sorting paths."""
    sdfg = dace.SDFG('multistate_test')
    sdfg.add_array('A', [10], dace.float64)
    sdfg.add_array('B', [10], dace.float64)
    sdfg.add_scalar('s', dace.float64, transient=True)

    state0 = sdfg.add_state('init')
    state1 = sdfg.add_state('compute')
    state2 = sdfg.add_state('finalize')

    # Add interstate edges
    sdfg.add_edge(state0, state1, dace.InterstateEdge())
    sdfg.add_edge(state1, state2, dace.InterstateEdge())

    # Add dataflow in the compute state
    a = state1.add_read('A')
    b = state1.add_write('B')
    tasklet = state1.add_tasklet('work', {'a'}, {'b'}, 'b = a * 2')
    state1.add_edge(a, None, tasklet, 'a', dace.Memlet.from_array('A', sdfg.arrays['A']))
    state1.add_edge(tasklet, 'b', b, None, dace.Memlet.from_array('B', sdfg.arrays['B']))

    return sdfg


def test_sdfg_alphabetical_sorting_multistate():
    """
    Tests that an SDFG with multiple states and interstate edges is
    sorted correctly at both the state machine and dataflow levels.
    Exercises the all_control_flow_regions() / all_states() paths.
    """
    reference_snapshot = None

    for seed in range(10):
        sdfg = _build_multistate_test_sdfg()

        random.seed(seed)
        _scramble_sdfg(sdfg)

        sdfg.sort_sdfg_alphabetically()

        snapshot = _snapshot_order(sdfg)

        if reference_snapshot is None:
            reference_snapshot = snapshot
        else:
            assert snapshot == reference_snapshot, (f"Multi-state sort produced different order with seed={seed}! "
                                                    f"Expected:\n{reference_snapshot}\nGot:\n{snapshot}")


if __name__ == "__main__":
    test_sdfg_alphabetical_sorting_basic()
    test_sdfg_alphabetical_sorting_rebuild_nx()
    test_sdfg_alphabetical_sorting_stability()
    test_sdfg_alphabetical_sorting_idempotency()
    test_sdfg_alphabetical_sorting_multistate()
