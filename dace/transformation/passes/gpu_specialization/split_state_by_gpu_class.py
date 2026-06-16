# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Split mixed-class states into a chain of CPU / GPU / CPU pure states.

The :class:`AutoSingleStreamGPUScheduler` strategy classifies every top-level dataflow node as
CPU / GPU / NEUTRAL / MIXED and runs a single-stream pipeline only when every node lands in a
pure bucket. When the user-written graph has CPU and GPU work entangled in the same state
(scalar host init feeding a kernel, GPU output feeding a host finalize, two independent
components running side by side, ...), the scheduler would otherwise fall back to
:class:`NaiveGPUStreamScheduler`. This preprocess transformation rearranges those states into a
chain of class-pure ones whenever the structure allows it -- specifically the patterns

* one state with multiple independent WCCs, some CPU and some GPU -- lift the CPU components
  into a new predecessor state;
* one mixed WCC whose topological band structure is ``[CPU?, GPU, CPU?]`` -- lift the CPU
  prefix to before, the CPU suffix to after, keep the GPU middle in place;
* mixed states that combine the two -- both lifts are applied, with the prefix-bound set being
  the union of independent CPU WCCs and chain prefixes, and the suffix-bound set being the
  chain suffixes.

DaCe ships :func:`dace.transformation.helpers.state_fission`, which lifts a subgraph into a new
*predecessor* state. The "after" direction is built from the same primitive: after lifting the
GPU middle out of the original state, the original state holds only what was downstream of the
GPU middle -- which is exactly the CPU suffix the user wanted to live after the kernel.

Genuinely interleaved patterns (``GPU -> CPU -> GPU``, cycles, or a `_Kind.MIXED` interior node
like a mixed NestedSDFG) are refused; those states fall through to the scheduler, which falls
back to the naive strategy.
"""
from typing import Dict, List, Optional, Set, Tuple

from dace import SDFG, SDFGState
from dace.sdfg import nodes
from dace.sdfg.graph import SubgraphView
from dace.sdfg.utils import dfs_topological_sort
from dace.transformation import pass_pipeline as ppl, transformation
from dace.transformation.helpers import state_fission
from dace.transformation.passes.gpu_specialization.gpu_stream_scheduling import (_classify_node, _fold_kinds, _Kind)
from dace.transformation.passes.gpu_specialization.helpers.gpu_helpers import weakly_connected_node_sets


def _weakly_connected_components(state: SDFGState) -> List[Set[nodes.Node]]:
    """Weakly connected components of ``state``'s dataflow (delegates to the shared
    :func:`~...helpers.gpu_helpers.weakly_connected_node_sets`)."""
    return weakly_connected_node_sets(state)


def _wcc_kind(wcc: Set[nodes.Node], sdfg: SDFG, state: SDFGState) -> _Kind:
    """Fold ``_classify_node`` over ``wcc``'s nodes to get the component-level kind."""
    return _fold_kinds(_classify_node(n, sdfg, state) for n in wcc)


def _chain_bands(wcc: Set[nodes.Node], sdfg: SDFG,
                 state: SDFGState) -> Optional[Tuple[List[nodes.Node], List[nodes.Node], List[nodes.Node]]]:
    """Topologically partition a mixed WCC into ``(cpu_prefix, gpu_middle, cpu_suffix)``.

    Returns ``None`` when the WCC contains a ``MIXED`` interior node, when it's purely one
    class, or when the topological order produces more than one CPU/GPU alternation per side
    (i.e. ``GPU -> CPU -> GPU`` or worse). NEUTRAL nodes (AccessNodes / MapExits) attach to
    whichever band the topo order places them next to; they get duplicated at the cut by
    :func:`state_fission`.
    """
    kinds: Dict[nodes.Node, _Kind] = {n: _classify_node(n, sdfg, state) for n in wcc}
    if any(k == _Kind.MIXED for k in kinds.values()):
        return None
    if not any(k == _Kind.CPU for k in kinds.values()) or not any(k == _Kind.GPU for k in kinds.values()):
        return None

    # Topologically sort the WCC restricted to its source nodes (those with no in-edges within
    # the WCC).
    roots = [n for n in wcc if all(e.src not in wcc for e in state.in_edges(n))]
    subgraph = SubgraphView(state, list(wcc))
    order = list(dfs_topological_sort(subgraph, sources=roots))

    # Walk topo order; group consecutive same-class nodes into bands. NEUTRAL attaches to the
    # current band; when the next non-neutral node disagrees with the band's class, we either
    # promote a NEUTRAL-only band to that class (the band hasn't acquired a class yet) or open
    # a new band.
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

    Walks every state. For each state with both CPU and GPU work,
    classifies WCCs and partitions mixed WCCs into ``[CPU?, GPU, CPU?]`` bands. Then up to two
    :func:`state_fission` calls rewrite the state: one lifts the union of pure-CPU WCCs and
    mixed-WCC CPU prefixes into a new predecessor state, the second (when any chain has a CPU
    suffix) lifts the GPU work so the original state is left holding only the suffix -- i.e.
    the suffix is now downstream of the GPU work, matching the user's intent.

    States with genuinely interleaved patterns are left unchanged; the scheduler falls back to
    Naive for those.
    """

    def modifies(self) -> ppl.Modifies:
        return ppl.Modifies.CFG | ppl.Modifies.States

    def should_reapply(self, modified: ppl.Modifies) -> bool:
        return False

    def apply_pass(self, sdfg: SDFG, _: Dict) -> Optional[Dict[str, int]]:
        # Skip when the stream pipeline has already run on this SDFG -- the SDFG already
        # carries ``gpu_streams`` (and consumers carry ``gpu_stream_id``), so a second split
        # would corrupt the now-wired structure.
        from dace.transformation.passes.gpu_specialization.helpers.gpu_helpers import is_stream_wiring_applied
        if is_stream_wiring_applied(sdfg):
            return None
        # Only attempt to split root-level ``SDFGState`` blocks. Other top-level block kinds
        # (``LoopRegion``, ``ConditionalBlock``, anything yielded by an inner ``NestedSDFG``)
        # are opaque to this pass: ``state_fission`` operates on dataflow nodes, and the
        # AutoSingleStreamGPUScheduler classifies these other blocks as a whole, so the
        # GPU/CPU boundary is handled at their parent iedges rather than by lifting inner
        # subgraphs.
        states_split = 0
        for block in list(sdfg.nodes()):
            if not isinstance(block, SDFGState):
                continue
            if self._split_one_state(block, sdfg):
                states_split += 1
        return {'states_split': states_split} if states_split else None

    @staticmethod
    def _split_one_state(state: SDFGState, sdfg: SDFG) -> bool:
        wccs = _weakly_connected_components(state)
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

        # Build the "lift to before" set: pure-CPU WCCs + chain prefixes.
        before_nodes: Set[nodes.Node] = set()
        for wcc in cpu_wccs:
            before_nodes.update(wcc)
        for prefix, _middle, _suffix in chains:
            before_nodes.update(prefix)

        # If there is no GPU work at all, there's nothing to schedule -- no-op.
        if not gpu_wccs and not any(middle for _p, middle, _s in chains):
            return False
        # If there is no CPU work to move (no pure CPU WCCs and no chain prefix or suffix),
        # the state is already pure-GPU + NEUTRAL -- no-op.
        has_suffix = any(suffix for _p, _m, suffix in chains)
        if not before_nodes and not has_suffix:
            return False

        # First fission: lift everything that should land before the GPU work.
        # ``allow_isolated_nodes=False`` -- ``state_fission`` keeps isolated nodes in the
        # original state by default; we want them moved into the new prefix state.
        if before_nodes:
            state_fission(SubgraphView(state, list(before_nodes)),
                          label=f"{state.label}_cpu_before",
                          allow_isolated_nodes=False)

        # Second fission: if any chain has a CPU suffix, lift the GPU work out of the original
        # state so the suffix is left as the (now-trailing) state. We collect: pure-GPU WCCs
        # plus each chain's GPU middle.
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
