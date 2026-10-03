# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""The unroll's local state fusion runs its region to a fixpoint, and only that region.

``ShortLoopUnroll`` clones a loop body per iteration and then re-merges the clones through
``parallelization_prep._local_state_fusion``. Two properties of that helper decide the map count
downstream and neither had a test.

* It RESTARTS the region's edge scan after every fusion, so a chain of cloned bodies collapses
  within one call. Fusing a disjoint matching instead settles on a coarser partition -- state
  fusion is not confluent -- and on CloudSC that left the unrolled LU bodies sharing two extra
  ``zqlhs_index`` copies, which ``PrivatizeScalars`` could then not privatize and ``LoopToMap``
  refused 30 loops over: 314 lifted maps down to 282. That CloudSC divergence needs nested
  unrolled regions and does not reduce to a synthetic graph, so what is pinned here is the
  fixpoint the restart reaches, not the non-confluence itself.
* It fuses inside the region it is handed and nowhere else. Interstate matching ignores
  ``apply_transformations``' ``states=`` filter, which is the reason the helper drives the pairs
  itself, so a sibling region must come back with every state it started with.

The helper is reached through the module rather than imported by name, so a falsification harness
can swap it without the tests holding a stale binding.
"""
import numpy as np

import dace
from dace.sdfg import nodes
from dace.sdfg.state import LoopRegion
from dace.transformation.passes import parallelization_prep

#: Length of the scratch chain. Long enough that a scan which stops at its first fusion leaves a
#: different graph than one that restarts.
CHAIN_LENGTH = 5

#: Every tasklet the chain builds, by label. Fusion moves nodes between states; it creates and
#: destroys none, so this list is what must still be there afterwards.
CHAIN_TASKLETS = ['bump0', 'bump1', 'bump2', 'bump3', 'bump4', 'scale0', 'scale1', 'scale2', 'scale3', 'scale4']


def scratch_chain_sdfg() -> dace.SDFG:
    """``out[i] = A[i] * 2 + 1`` split over one state per ``i``, each staging through its own scalar.

    The shape an unrolled straight-line body leaves behind: consecutive states that share no
    transient, so every adjacent pair is fusable and the whole chain is one state's worth of work.
    """
    sdfg = dace.SDFG('scratch_chain')
    sdfg.add_array('A', [CHAIN_LENGTH], dace.float64)
    sdfg.add_array('out', [CHAIN_LENGTH], dace.float64)
    previous: dace.SDFGState | None = None
    for i in range(CHAIN_LENGTH):
        sdfg.add_scalar(f'acc{i}', dace.float64, transient=True)
        state = sdfg.add_state(f's{i}', is_start_block=(i == 0))
        if previous is not None:
            sdfg.add_edge(previous, state, dace.InterstateEdge())
        scale = state.add_tasklet(f'scale{i}', {'a'}, {'o'}, 'o = a * 2.0')
        state.add_edge(state.add_access('A'), None, scale, 'a', dace.Memlet(f'A[{i}]'))
        staged = state.add_access(f'acc{i}')
        state.add_edge(scale, 'o', staged, None, dace.Memlet(f'acc{i}[0]'))
        bump = state.add_tasklet(f'bump{i}', {'a'}, {'o'}, 'o = a + 1.0')
        state.add_edge(staged, None, bump, 'a', dace.Memlet(f'acc{i}[0]'))
        state.add_edge(bump, 'o', state.add_access('out'), None, dace.Memlet(f'out[{i}]'))
        previous = state
    sdfg.validate()
    return sdfg


def two_sibling_regions_sdfg() -> tuple[dace.SDFG, LoopRegion, LoopRegion]:
    """Two sequential ``LoopRegion``s, each holding a three-state fusable chain of its own."""
    sdfg = dace.SDFG('two_regions')
    sdfg.add_array('A', [3], dace.float64)
    regions: list[LoopRegion] = []
    previous_region: LoopRegion | None = None
    for r in range(2):
        region = LoopRegion(f'r{r}', 'k < 4', 'k', 'k = 0', 'k = k + 1')
        sdfg.add_node(region, is_start_block=(r == 0))
        if previous_region is not None:
            sdfg.add_edge(previous_region, region, dace.InterstateEdge())
        previous_state: dace.SDFGState | None = None
        for i in range(3):
            scalar = f'v{r * 3 + i}'
            sdfg.add_scalar(scalar, dace.float64, transient=True)
            state = region.add_state(f'r{r}s{i}', is_start_block=(i == 0))
            if previous_state is not None:
                region.add_edge(previous_state, state, dace.InterstateEdge())
            tasklet = state.add_tasklet(f't{r}{i}', {'a'}, {'o'}, 'o = a + 1.0')
            state.add_edge(state.add_access('A'), None, tasklet, 'a', dace.Memlet(f'A[{i}]'))
            state.add_edge(tasklet, 'o', state.add_access(scalar), None, dace.Memlet(f'{scalar}[0]'))
            previous_state = state
        regions.append(region)
        previous_region = region
    sdfg.validate()
    return sdfg, regions[0], regions[1]


def tasklet_labels(sdfg: dace.SDFG) -> list[str]:
    return sorted(node.label for state in sdfg.all_states() for node in state.nodes()
                  if isinstance(node, nodes.Tasklet))


def test_a_chain_of_staged_scratch_states_collapses_within_one_fusion_call() -> None:
    sdfg = scratch_chain_sdfg()

    fused = parallelization_prep._local_state_fusion(sdfg, sdfg)

    assert fused == 4
    assert len(list(sdfg.all_states())) == 1
    assert tasklet_labels(sdfg) == CHAIN_TASKLETS
    sdfg.validate()

    a = np.arange(1.0, CHAIN_LENGTH + 1.0)
    out = np.full(CHAIN_LENGTH, -9.0)
    sdfg(A=a, out=out)
    assert np.array_equal(out, np.array([3.0, 5.0, 7.0, 9.0, 11.0]))


def test_fusing_one_region_leaves_its_sibling_regions_states_alone() -> None:
    sdfg, first, second = two_sibling_regions_sdfg()

    fused = parallelization_prep._local_state_fusion(sdfg, first)

    assert fused == 2
    assert len(list(first.all_states())) == 1
    assert len(list(second.all_states())) == 3
    assert tasklet_labels(sdfg) == ['t00', 't01', 't02', 't10', 't11', 't12']
    sdfg.validate()
