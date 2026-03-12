import dace
import random
from dace.sdfg.utils import get_deterministic_node_key, get_deterministic_edge_key


def test_sdfg_alphabetical_sorting():
    """
    Tests that the SDFG and its internal states can be forced into a strictly
    deterministic topological order, regardless of dictionary insertion history.
    """
    # 1. Create a simple SDFG
    sdfg = dace.SDFG('deterministic_test')
    sdfg.add_array('A', [10], dace.float64)
    sdfg.add_array('B', [10], dace.float64)

    state = sdfg.add_state('state0')
    a = state.add_read('A')
    b = state.add_write('B')
    tasklet = state.add_tasklet('compute', {'a'}, {'b'}, 'b = a + 1')

    state.add_edge(a, None, tasklet, 'a', dace.Memlet.from_array('A', sdfg.arrays['A']))
    state.add_edge(tasklet, 'b', b, None, dace.Memlet.from_array('B', sdfg.arrays['B']))

    # 2. Intentionally scramble the internal dictionaries to simulate non-determinism

    def scramble_dict_in_place(d):
        """Helper to randomize dictionary insertion order without changing its type."""
        if not d: return
        keys = list(d.keys())
        random.shuffle(keys)
        for k in keys:
            # Pop and re-insert to shuffle the underlying Python 3.7+ insertion order
            d[k] = d.pop(k)

    # Scramble top-level arrays (Safely mutating the NestedDict in-place)
    scramble_dict_in_place(sdfg._arrays)

    # Scramble state nodes
    scramble_dict_in_place(state._nodes)

    # Scramble master edges
    scramble_dict_in_place(state._edges)

    # Scramble nested adjacency lists (in_edges / out_edges)
    for node, (in_edges, out_edges) in state._nodes.items():
        scramble_dict_in_place(in_edges)
        scramble_dict_in_place(out_edges)

    # 3. Apply the canonicalizer
    sdfg.sort_sdfg_alphabetically()

    # 4. Assert that the underlying dictionaries are now strictly ordered
    node_keys = list(state._nodes.keys())
    expected_node_keys = sorted(node_keys, key=get_deterministic_node_key)

    edge_keys = list(state._edges.keys())
    expected_edge_keys = sorted(edge_keys, key=lambda k: get_deterministic_edge_key(state._edges[k]))

    assert node_keys == expected_node_keys, "Graph nodes were not deterministically sorted!"
    assert edge_keys == expected_edge_keys, "Graph edges were not deterministically sorted!"

    # Ensure networkx backend was also rebuilt deterministically
    nx_nodes = list(state._nx.nodes())
    assert nx_nodes == expected_node_keys, "NetworkX nodes do not match DaCe dict order!"


if __name__ == "__main__":
    test_sdfg_alphabetical_sorting()
