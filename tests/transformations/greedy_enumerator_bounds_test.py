# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
""" Cost bounds for ``GreedyEnumerator`` on map-heavy states.

    ``auto_optimize.greedy_fuse`` drives ``GreedyEnumerator`` over every state,
    so the enumerator must stay linear in the number of maps. A powerset or
    any other superlinear candidate scheme is unusable on real kernels: a
    single loop body in npbench ``cavity_flow`` carries 56 outermost maps.

    These tests pin the contract on a state whose maps are all mutually
    adjacent (worst case for the greedy BFS, because every map is a neighbour
    of every other one):

    * enumeration terminates,
    * the yielded groups partition the maps -- disjoint and complete,
    * the condition function is called at most once per map,
    * the enumerator allocates a bounded amount of memory.

    The call count is the load-bearing assertion. It is what separates the
    greedy scan (N-1 calls) from any scheme that materializes candidate sets.
    The allocation ceiling is the backstop: a combinatorial enumerator can
    keep the call count low by batching, but it cannot hold 2**N candidate
    subgraphs in a few megabytes.
"""
import tracemalloc
from typing import List, Tuple

import pytest

import dace
from dace.sdfg import nodes
from dace.sdfg.graph import SubgraphView
from dace.transformation.auto.auto_optimize import greedy_fuse
from dace.transformation.estimator.enumeration import GreedyEnumerator
from dace.transformation.subgraph.composite import CompositeFusion

N_MAPS = 48

# Measured peak for N_MAPS on this enumerator is well under 1 MB. The ceiling
# leaves an order of magnitude of headroom for interpreter and cache noise
# while staying far below anything that materializes candidate subgraphs.
MAX_PEAK_BYTES = 8 * 1024 * 1024


def _clique_state_sdfg(num_maps: int) -> dace.SDFG:
    """ ``num_maps`` maps reading a shared input and writing private outputs.

        Sharing the input makes every map adjacent to every other one in the
        enumerator's adjacency list, which is the densest topology it can see.
    """
    sdfg = dace.SDFG(f'greedy_enum_clique_{num_maps}')
    sdfg.add_array('A', [64], dace.float64)
    state = sdfg.add_state('main')
    read = state.add_access('A')
    for i in range(num_maps):
        sdfg.add_array(f'B{i}', [64], dace.float64)
        write = state.add_access(f'B{i}')
        state.add_mapped_tasklet(f'm{i}', {'j': '0:64'}, {'inp': dace.Memlet('A[j]')},
                                 f'out = inp + {i}', {'out': dace.Memlet(f'B{i}[j]')},
                                 external_edges=True,
                                 input_nodes={'A': read},
                                 output_nodes={f'B{i}': write})
    return sdfg


def _enumerate(sdfg: dace.SDFG, measure: bool = False) -> Tuple[List[Tuple[nodes.MapEntry, ...]], int, int]:
    """ Runs the enumerator over the single state of ``sdfg``.

        :return: The yielded groups, the number of condition-function calls and
                 the peak bytes allocated during enumeration (0 unless ``measure``).
    """
    state = sdfg.states()[0]
    condition = CompositeFusion()
    condition.setup_match(SubgraphView(state, state.nodes()))
    condition.allow_tiling = False
    condition.expansion_split = False

    calls = [0]

    def counting_condition(inner_sdfg: dace.SDFG, subgraph: SubgraphView) -> bool:
        calls[0] += 1
        return condition.can_be_applied(inner_sdfg, subgraph)

    def run() -> List[Tuple[nodes.MapEntry, ...]]:
        enumerator = GreedyEnumerator(sdfg,
                                      state,
                                      SubgraphView(state, state.nodes()),
                                      condition_function=counting_condition)
        return list(enumerator)

    if not measure:
        return run(), calls[0], 0

    tracemalloc.start()
    try:
        groups = run()
        peak = tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()
    return groups, calls[0], peak


def test_greedy_enumerator_is_linear_in_map_count():
    # Warm up the symbolic and descriptor caches so the traced peak measures the
    # enumerator rather than first-use allocations elsewhere in dace.
    _enumerate(_clique_state_sdfg(4))

    sdfg = _clique_state_sdfg(N_MAPS)
    groups, calls, peak = _enumerate(sdfg, measure=True)

    # The groups must partition the maps: every map exactly once, no duplicates.
    covered = [me for group in groups for me in group]
    assert len(covered) == N_MAPS
    assert len(set(id(me) for me in covered)) == N_MAPS
    assert len(groups) <= N_MAPS

    # Greedy scan: one condition check per map that is offered to a non-empty set.
    # The bound is deliberately loose (2x) so it fails only on a complexity
    # regression, not on a different-but-still-linear traversal order.
    assert calls <= 2 * N_MAPS, f'condition function called {calls} times for {N_MAPS} maps'
    assert peak <= MAX_PEAK_BYTES, f'enumeration of {N_MAPS} maps peaked at {peak / 1e6:.2f} MB'


def test_greedy_enumerator_scales_linearly():
    """ Doubling the map count must not more than double the cost. """
    _enumerate(_clique_state_sdfg(4))

    _, small_calls, small_peak = _enumerate(_clique_state_sdfg(N_MAPS // 2), measure=True)
    _, large_calls, large_peak = _enumerate(_clique_state_sdfg(N_MAPS), measure=True)

    assert large_calls <= 2 * small_calls + 2, \
        f'{small_calls} calls for {N_MAPS // 2} maps but {large_calls} for {N_MAPS}'
    # Allow a quadratic slack on memory: the adjacency list of a clique is
    # inherently O(N**2). Anything combinatorial blows straight through this.
    assert large_peak <= 4 * small_peak + MAX_PEAK_BYTES // 32, \
        f'{small_peak / 1e6:.2f} MB for {N_MAPS // 2} maps but {large_peak / 1e6:.2f} MB for {N_MAPS}'


def test_greedy_fuse_terminates_on_map_heavy_state():
    """ End-to-end guard: the driver completes and leaves a valid, acyclic SDFG. """
    sdfg = _clique_state_sdfg(N_MAPS)
    greedy_fuse(sdfg, validate_all=False)
    sdfg.validate()

    for state in sdfg.states():
        assert not list(state.find_cycles()), f'greedy_fuse left a cyclic state {state.label}'

    remaining = sum(1 for node, _ in sdfg.all_nodes_recursive() if isinstance(node, nodes.MapEntry))
    assert remaining <= N_MAPS


if __name__ == '__main__':
    raise SystemExit(pytest.main([__file__, '-v']))
