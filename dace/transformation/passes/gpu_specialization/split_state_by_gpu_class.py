# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Split mixed-class states into class-pure CPU / GPU / CPU states, so that a single-stream schedule applies.

Independent CPU components and the CPU prefixes of mixed ``[CPU?, GPU, CPU?]`` components lift into a new
predecessor state; lifting the GPU middle out leaves the CPU suffix in the original state. Interleaved patterns
(``GPU -> CPU -> GPU``, cycles, ``MIXED`` interior nodes) are refused.
"""

from typing import Dict, List, Optional, Set, Tuple, Type, Union

from dace import SDFG, SDFGState
from dace.sdfg import nodes
from dace.sdfg.graph import SubgraphView
from dace.sdfg.utils import dfs_topological_sort
from dace.transformation import pass_pipeline as ppl, transformation
from dace.transformation.helpers import state_fission
from dace.transformation.passes.insert_explicit_copies import InsertExplicitCopies
from dace.ordered import OrderedSet
from dace.transformation.passes.gpu_specialization.gpu_stream_scheduling import classify_node, fold_kinds, NodeKind
from dace.transformation.passes.gpu_specialization.helpers.gpu_helpers import (
    is_stream_wiring_applied,
    weakly_connected_node_sets,
)

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
    """Group consecutive same-class nodes into ``[kind, nodes]`` bands; a neutral node joins the current band."""
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
    """Partition a mixed component into ``(cpu_prefix, gpu_middle, cpu_suffix)``, or ``None`` if it has a ``MIXED``
    node, one class only or more alternations (``state_fission`` duplicates the neutral nodes at the cut)."""
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
    if to_lift:
        state_fission(SubgraphView(state, list(to_lift)), label=label, allow_isolated_nodes=False)


@transformation.explicit_cf_compatible
class SplitStateByGPUClass(ppl.Pass):
    """Lift CPU work out of mixed-class root states to before and after the GPU work, with up to two fissions."""

    def modifies(self) -> ppl.Modifies:
        return ppl.Modifies.CFG | ppl.Modifies.States

    def should_reapply(self, modified: ppl.Modifies) -> bool:
        return False

    def depends_on(self) -> List[Union[Type[ppl.Pass], ppl.Pass]]:
        # A copy is classified GPU only once lifted to a ``CopyLibraryNode``.
        return [InsertExplicitCopies]

    def apply_pass(self, sdfg: SDFG, _: Dict) -> Optional[Dict[str, int]]:
        # A wired SDFG must not be split again.
        if is_stream_wiring_applied(sdfg):
            return None
        # Root states only: the scheduler classifies loops, conditionals and nested SDFGs as a whole.
        states_split = sum(
            1 for block in list(sdfg.nodes()) if isinstance(block, SDFGState) and self.split_one_state(block, sdfg)
        )
        return {"states_split": states_split} if states_split else None

    @staticmethod
    def split_one_state(state: SDFGState, sdfg: SDFG) -> bool:
        cpu_wccs, prefixes = OrderedSet(), OrderedSet()
        gpu_wccs, middles = OrderedSet(), OrderedSet()
        has_suffix = False
        for wcc in weakly_connected_node_sets(state):
            kind = wcc_kind(wcc, sdfg, state)
            if kind == NodeKind.MIXED:
                # A mixed component that does not decompose as [CPU?, GPU, CPU?] refuses the state.
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
        if not gpu or not (before or has_suffix):
            return False
        lift(state, before, f"{state.label}_cpu_before")
        if has_suffix:
            lift(state, gpu, f"{state.label}_gpu_middle")
        return True
