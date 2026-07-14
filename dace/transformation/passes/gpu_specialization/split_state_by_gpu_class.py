# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Split mixed-class states into a chain of class-pure CPU / GPU / CPU states.

When a state entangles CPU and GPU work, :class:`AutoSingleStreamGPUScheduler` would fall back to
:class:`NaiveGPUStreamScheduler`. This preprocess pass rearranges such states into class-pure ones
when the structure allows: independent CPU WCCs and the CPU prefixes of mixed ``[CPU?, GPU, CPU?]``
WCCs lift into a new predecessor state; CPU suffixes are left trailing.

The "after" lift reuses :func:`dace.transformation.helpers.state_fission` (which only lifts into a
*predecessor*): lifting the GPU middle out leaves the original state holding just the downstream CPU
suffix. Genuinely interleaved patterns (``GPU -> CPU -> GPU``, cycles, ``_Kind.MIXED`` interior nodes
like a mixed NestedSDFG) are refused and fall through to the naive strategy.
"""
from typing import Dict, List, Optional, Set, Tuple

from dace import SDFG, SDFGState
from dace.sdfg import nodes
from dace.sdfg.graph import SubgraphView
from dace.sdfg.utils import dfs_topological_sort
from dace.transformation import pass_pipeline as ppl, transformation
from dace.transformation.helpers import state_fission
from dace.transformation.passes.gpu_specialization.gpu_stream_scheduling import (_classify_node, _fold_kinds, _Kind)
from dace.transformation.passes.gpu_specialization.helpers.gpu_helpers import (is_stream_wiring_applied,
                                                                               weakly_connected_node_sets)


def _wcc_kind(wcc: Set[nodes.Node], sdfg: SDFG, state: SDFGState) -> _Kind:
    """Fold ``_classify_node`` over ``wcc``'s nodes to get the component-level kind."""
    return _fold_kinds(_classify_node(n, sdfg, state) for n in wcc)


def _chain_bands(wcc: Set[nodes.Node], sdfg: SDFG,
                 state: SDFGState) -> Optional[Tuple[List[nodes.Node], List[nodes.Node], List[nodes.Node]]]:
    """Topologically partition a mixed WCC into ``(cpu_prefix, gpu_middle, cpu_suffix)``.

    Returns ``None`` when the WCC contains a ``MIXED`` interior node, is purely one class, or its
    topo order alternates more than once per side (``GPU -> CPU -> GPU`` or worse). NEUTRAL nodes
    (AccessNodes / MapExits) attach to the adjacent band and get duplicated at the cut by
    :func:`state_fission`.
    """
    kinds: Dict[nodes.Node, _Kind] = {n: _classify_node(n, sdfg, state) for n in wcc}
    if any(k == _Kind.MIXED for k in kinds.values()):
        return None
    if not any(k == _Kind.CPU for k in kinds.values()) or not any(k == _Kind.GPU for k in kinds.values()):
        return None

    # Roots = WCC nodes with no in-edges within the WCC.
    roots = [n for n in wcc if all(e.src not in wcc for e in state.in_edges(n))]
    subgraph = SubgraphView(state, list(wcc))
    order = list(dfs_topological_sort(subgraph, sources=roots))

    # Group consecutive same-class nodes into bands. NEUTRAL attaches to the current band; a
    # non-neutral node that disagrees either promotes a not-yet-classed NEUTRAL band or opens a
    # new band.
    bands: List[List] = []  # each entry: [kind, [nodes]]
    for n in order:
        k = kinds[n]
        if k == _Kind.NEUTRAL:
            if bands:
                bands[-1][1].append(n)
            else:
                bands.append([_Kind.NEUTRAL, [n]])
            continue
        if not bands:
            bands.append([k, [n]])
            continue
        if bands[-1][0] == _Kind.NEUTRAL:
            bands[-1][0] = k
            bands[-1][1].append(n)
        elif bands[-1][0] == k:
            bands[-1][1].append(n)
        else:
            bands.append([k, [n]])

    non_neutral_kinds = [b[0] for b in bands if b[0] != _Kind.NEUTRAL]
    if non_neutral_kinds == [_Kind.CPU, _Kind.GPU]:
        return bands[0][1], bands[1][1], []
    if non_neutral_kinds == [_Kind.GPU, _Kind.CPU]:
        return [], bands[0][1], bands[1][1]
    if non_neutral_kinds == [_Kind.CPU, _Kind.GPU, _Kind.CPU]:
        return bands[0][1], bands[1][1], bands[2][1]
    return None


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
        # Only split root-level ``SDFGState`` blocks. Other top-level kinds (``LoopRegion``,
        # ``ConditionalBlock``, inner ``NestedSDFG`` output) are opaque: ``state_fission`` works on
        # dataflow nodes, and the scheduler classifies these blocks as a whole, so their GPU/CPU
        # boundary is handled at their parent iedges rather than by lifting inner subgraphs.
        states_split = 0
        for block in list(sdfg.nodes()):
            if not isinstance(block, SDFGState):
                continue
            if self._split_one_state(block, sdfg):
                states_split += 1
        return {'states_split': states_split} if states_split else None

    @staticmethod
    def _split_one_state(state: SDFGState, sdfg: SDFG) -> bool:
        wccs = weakly_connected_node_sets(state)
        if not wccs:
            return False

        kinds = [_wcc_kind(wcc, sdfg, state) for wcc in wccs]

        cpu_wccs = [w for w, k in zip(wccs, kinds) if k == _Kind.CPU]
        gpu_wccs = [w for w, k in zip(wccs, kinds) if k == _Kind.GPU]
        mixed_wccs = [w for w, k in zip(wccs, kinds) if k == _Kind.MIXED]

        # Every mixed WCC must decompose as [CPU?, GPU, CPU?]; otherwise refuse this state.
        chains: List[Tuple[List[nodes.Node], List[nodes.Node], List[nodes.Node]]] = []
        for wcc in mixed_wccs:
            bands = _chain_bands(wcc, sdfg, state)
            if bands is None:
                return False
            chains.append(bands)

        # "Lift to before" set: pure-CPU WCCs + chain prefixes.
        before_nodes: Set[nodes.Node] = set()
        for wcc in cpu_wccs:
            before_nodes.update(wcc)
        for prefix, _middle, _suffix in chains:
            before_nodes.update(prefix)

        # No GPU work at all: nothing to schedule.
        if not gpu_wccs and not any(middle for _p, middle, _s in chains):
            return False
        # No CPU work to move (no pure CPU WCCs, no chain prefix/suffix): already pure-GPU + NEUTRAL.
        has_suffix = any(suffix for _p, _m, suffix in chains)
        if not before_nodes and not has_suffix:
            return False

        # First fission: lift everything that lands before the GPU work. ``allow_isolated_nodes=False``
        # because ``state_fission`` otherwise leaves isolated nodes behind; we want them in the prefix.
        if before_nodes:
            state_fission(SubgraphView(state, list(before_nodes)),
                          label=f"{state.label}_cpu_before",
                          allow_isolated_nodes=False)

        # Second fission: when a chain has a CPU suffix, lift the GPU work (pure-GPU WCCs + each
        # chain's GPU middle) out so the suffix is left behind as the trailing state.
        if has_suffix:
            gpu_to_lift: Set[nodes.Node] = set()
            for wcc in gpu_wccs:
                gpu_to_lift.update(wcc)
            for _prefix, middle, _suffix in chains:
                gpu_to_lift.update(middle)
            if gpu_to_lift:
                state_fission(SubgraphView(state, list(gpu_to_lift)),
                              label=f"{state.label}_gpu_middle",
                              allow_isolated_nodes=False)

        return True
