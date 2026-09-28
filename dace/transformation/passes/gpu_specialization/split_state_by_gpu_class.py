# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Split mixed-class states into a chain of class-pure CPU / GPU / CPU states.

When a state entangles CPU and GPU work, :class:`AutoSingleStreamGPUScheduler` would fall back to
:class:`NaiveGPUStreamScheduler`. This preprocess pass rearranges such states into class-pure ones
when the structure allows: independent CPU WCCs and the CPU prefixes of mixed ``[CPU?, GPU, CPU?]``
WCCs lift into a new predecessor state; CPU suffixes are left trailing.

The "after" lift reuses :func:`dace.transformation.helpers.state_fission` (which only lifts into a
*predecessor*): lifting the GPU middle out leaves the original state holding just the downstream CPU
suffix. Genuinely interleaved patterns (``GPU -> CPU -> GPU``, cycles, ``NodeKind.MIXED`` interior nodes
like a mixed NestedSDFG) are refused and fall through to the naive strategy.
"""
from typing import Dict, List, Optional, Set, Tuple

from dace import SDFG, SDFGState
from dace.sdfg import nodes
from dace.sdfg.graph import SubgraphView
from dace.sdfg.utils import dfs_topological_sort
from dace.transformation import pass_pipeline as ppl, transformation
from dace.transformation.helpers import state_fission
from ordered_set import OrderedSet
from dace.transformation.passes.gpu_specialization.gpu_stream_scheduling import (classify_node, fold_kinds, NodeKind)
from dace.transformation.passes.gpu_specialization.helpers.gpu_helpers import (is_stream_wiring_applied,
                                                                               weakly_connected_node_sets)

#: A mixed WCC as ``(cpu_prefix, gpu_middle, cpu_suffix)``.
Chain = Tuple[List[nodes.Node], List[nodes.Node], List[nodes.Node]]

#: The class sequences a mixed WCC may have, with the band index of each chain part (-1 for none).
CHAIN_SHAPES = {
    (NodeKind.CPU, NodeKind.GPU): (0, 1, -1),
    (NodeKind.GPU, NodeKind.CPU): (-1, 0, 1),
    (NodeKind.CPU, NodeKind.GPU, NodeKind.CPU): (0, 1, 2),
}


def wcc_kind(wcc: Set[nodes.Node], sdfg: SDFG, state: SDFGState) -> NodeKind:
    return fold_kinds(classify_node(n, sdfg, state) for n in wcc)


def group_into_bands(order: List[nodes.Node], kinds: Dict[nodes.Node, NodeKind]) -> List[list]:
    """Group consecutive same-class nodes into ``[kind, nodes]`` bands. A NEUTRAL node joins the current
    band; a NEUTRAL-only band takes the class of the next classed node."""
    bands: List[list] = []
    for n in order:
        k = kinds[n]
        if bands and (k in (NodeKind.NEUTRAL, bands[-1][0]) or bands[-1][0] == NodeKind.NEUTRAL):
            if bands[-1][0] == NodeKind.NEUTRAL:
                bands[-1][0] = k
            bands[-1][1].append(n)
        else:
            bands.append([k, [n]])
    return bands


def chain_bands(wcc: Set[nodes.Node], sdfg: SDFG, state: SDFGState) -> Optional[Chain]:
    """Topologically partition a mixed WCC into ``(cpu_prefix, gpu_middle, cpu_suffix)``.

    Returns ``None`` when the WCC contains a ``MIXED`` interior node, is purely one class, or its
    topo order alternates more than once per side (``GPU -> CPU -> GPU`` or worse). NEUTRAL nodes
    (AccessNodes / MapExits) attach to the adjacent band and get duplicated at the cut by
    :func:`state_fission`.
    """
    kinds: Dict[nodes.Node, NodeKind] = {n: classify_node(n, sdfg, state) for n in wcc}
    if NodeKind.MIXED in kinds.values():
        return None
    roots = [n for n in wcc if all(e.src not in wcc for e in state.in_edges(n))]
    order = list(dfs_topological_sort(SubgraphView(state, list(wcc)), sources=roots))
    bands = [b for b in group_into_bands(order, kinds) if b[0] != NodeKind.NEUTRAL]
    shape = CHAIN_SHAPES.get(tuple(b[0] for b in bands))
    if shape is None:
        return None
    return tuple(bands[i][1] if i >= 0 else [] for i in shape)


def lift(state: SDFGState, to_lift: OrderedSet, label: str) -> None:
    # ``allow_isolated_nodes=False``: isolated nodes go with the lifted part rather than stay behind.
    if to_lift:
        state_fission(SubgraphView(state, list(to_lift)), label=label, allow_isolated_nodes=False)


@transformation.explicit_cf_compatible
class SplitStateByGPUClass(ppl.Pass):
    """Lift CPU work out of mixed-class states to before / after the GPU work.

    Up to two :func:`state_fission` calls per state: one lifts pure-CPU WCCs and mixed-WCC CPU
    prefixes into a new predecessor state; the second (when any chain has a CPU suffix) lifts the
    GPU work out so the original state is left holding only the suffix, now downstream of the GPU.
    """

    def modifies(self) -> ppl.Modifies:
        return ppl.Modifies.CFG | ppl.Modifies.States

    def should_reapply(self, modified: ppl.Modifies) -> bool:
        return False

    def apply_pass(self, sdfg: SDFG, _: Dict) -> Optional[Dict[str, int]]:
        # Skip when the stream pipeline has already run: the SDFG carries ``gpu_streams`` (and
        # consumers carry ``gpu_stream_id``), so a second split would corrupt the wired structure.
        if is_stream_wiring_applied(sdfg):
            return None
        # Only root-level states: ``state_fission`` works on dataflow, and the scheduler classifies
        # loops, conditionals and nested SDFGs as a whole.
        states_split = sum(1 for block in list(sdfg.nodes())
                           if isinstance(block, SDFGState) and self.split_one_state(block, sdfg))
        return {'states_split': states_split} if states_split else None

    @staticmethod
    def split_one_state(state: SDFGState, sdfg: SDFG) -> bool:
        # Order as fission sees it: pure WCCs first, then the chain parts.
        cpu_wccs, prefixes = OrderedSet(), OrderedSet()
        gpu_wccs, middles = OrderedSet(), OrderedSet()
        has_suffix = False
        for wcc in weakly_connected_node_sets(state):
            kind = wcc_kind(wcc, sdfg, state)
            if kind == NodeKind.MIXED:
                # Every mixed WCC must decompose as [CPU?, GPU, CPU?]; otherwise refuse this state.
                chain = chain_bands(wcc, sdfg, state)
                if chain is None:
                    return False
                prefixes.update(chain[0])
                middles.update(chain[1])
                has_suffix = has_suffix or bool(chain[2])
            elif kind == NodeKind.CPU:
                cpu_wccs.update(wcc)
            elif kind == NodeKind.GPU:
                gpu_wccs.update(wcc)
        before, gpu = cpu_wccs | prefixes, gpu_wccs | middles
        # Nothing to split without GPU work, or without CPU work around it.
        if not gpu or not (before or has_suffix):
            return False
        lift(state, before, f"{state.label}_cpu_before")
        # With a CPU suffix, lifting the GPU work leaves the suffix behind as the trailing state.
        if has_suffix:
            lift(state, gpu, f"{state.label}_gpu_middle")
        return True
