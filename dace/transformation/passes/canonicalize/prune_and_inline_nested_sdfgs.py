# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Prune dead connectors and inline single-state nested SDFGs, in the order the matcher applied them.

The traversal half of ``PatternApplyOnceEverywhere([PruneConnectors(), InlineSDFG()])``, which restarts
its enumeration after every application: a graph collapse per state and both predicates per nested SDFG
over the prefix it already refused (CloudSC ``reduction_to_wcr_map``: 339 inlines, 43488 probes of each
transformation, 110772 collapses). This pass walks the same order -- regions preorder, a region's states
before its children, in a state every ``PruneConnectors`` probe before any ``InlineSDFG`` probe -- and
applies the first accepted candidate, so the application sequence is the matcher's.

It skips only refusals nothing could have changed. Both predicates read the candidate's state, its nested
SDFG's contents and its owner's descriptors of names that state uses. An application in state ``T``
rewrites ``T`` (any state split off it is new) and thereby the contents of ``T``'s SDFG, which is what the
state holding that SDFG's own node reads; refusals on ``T`` and its ancestors are dropped.
"""
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from dace import SDFG
from dace.config import Config
from dace.sdfg import nodes
from dace.sdfg.state import AbstractControlFlowRegion, ControlFlowRegion, SDFGState
from dace.transformation import pass_pipeline as ppl, transformation
from dace.transformation.dataflow import PruneConnectors
from dace.transformation.interstate import InlineSDFG

#: A transformation and the pattern node its single-node expression binds.
Probe = Tuple[transformation.SingleStateTransformation, transformation.PatternNode]

#: An accepted candidate: the transformation, the state it applies in, and the nested SDFG node.
Candidate = Tuple[transformation.SingleStateTransformation, SDFGState, nodes.NestedSDFG]


def ancestor_blocks(state: SDFGState) -> List[Any]:
    """``state``'s enclosing regions and states up to the root, crossing nested-SDFG boundaries."""
    chain: List[Any] = []
    block: Any = state.parent_graph
    while block is not None:
        chain.append(block)
        block = block.parent if isinstance(block, SDFG) else block.parent_graph
    return chain


@dataclass(slots=True)
class Walk:
    """One fixpoint's probes and the refusals still known to hold."""
    probes: List[Probe]
    #: States whose every candidate was refused, with their nested SDFG nodes in node order.
    clean_states: Dict[SDFGState, List[nodes.NestedSDFG]] = field(default_factory=dict)
    #: Regions with no accepted candidate in their states or child regions.
    clean_regions: Dict[ControlFlowRegion, None] = field(default_factory=dict)

    def first_accepted(self, region: ControlFlowRegion) -> Optional[Candidate]:
        if region in self.clean_regions:
            return None
        blocks = region.nodes()
        for block in blocks:
            if isinstance(block, SDFGState) and block not in self.clean_states:
                candidate = self.first_accepted_in_state(block)
                if candidate is not None:
                    return candidate
        for block in blocks:
            if isinstance(block, SDFGState):
                for node in self.clean_states[block]:
                    # ``all_control_flow_regions`` recurses on ``if node.sdfg:``, false for a node-less SDFG.
                    if node.sdfg is None or node.sdfg.number_of_nodes() == 0:
                        continue
                    candidate = self.first_accepted(node.sdfg)
                    if candidate is not None:
                        return candidate
            elif isinstance(block, AbstractControlFlowRegion):
                candidate = self.first_accepted(block)
                if candidate is not None:
                    return candidate
        self.clean_regions[region] = None
        return None

    def first_accepted_in_state(self, state: SDFGState) -> Optional[Candidate]:
        nested = [node for node in state.nodes() if isinstance(node, nodes.NestedSDFG)]
        for xform, pattern_node in self.probes:
            for node in nested:
                if accepts(xform, pattern_node, state, node):
                    return xform, state, node
        self.clean_states[state] = nested
        return None

    def forget(self, blocks: List[Any]) -> None:
        for block in blocks:
            self.clean_regions.pop(block, None)
            self.clean_states.pop(block, None)


def accepts(xform: transformation.SingleStateTransformation, pattern_node: transformation.PatternNode, state: SDFGState,
            node: nodes.NestedSDFG) -> bool:
    owner = state.sdfg
    # Bound by node OBJECT: ``PatternNode.__get__`` returns a non-int as-is, so no node index is resolved.
    xform.setup_match(owner, 0, -1, {pattern_node: node}, 0, override=True)
    try:
        return xform.can_be_applied(state, 0, owner, permissive=False)
    except Exception as exception:
        if Config.get_bool('optimizer', 'match_exception'):
            raise
        print(f'WARNING: {type(xform).__name__}::can_be_applied triggered a '
              f'{type(exception).__name__} exception: {exception}')
        return False


@transformation.explicit_cf_compatible
class PruneAndInlineNestedSDFGs(ppl.Pass):
    """Apply ``PruneConnectors`` and ``InlineSDFG`` at every nested SDFG they accept, to a fixpoint."""
    CATEGORY: str = 'Canonicalization'

    def modifies(self) -> ppl.Modifies:
        return PruneConnectors().modifies() | InlineSDFG().modifies()

    def should_reapply(self, modified: ppl.Modifies) -> bool:
        return PruneConnectors().should_reapply(modified) or InlineSDFG().should_reapply(modified)

    def depends_on(self):
        return {}

    def apply_pass(self, sdfg: SDFG, pipeline_results: Dict[str, Any]) -> Optional[int]:
        """Prune and inline until neither transformation accepts a nested SDFG of ``sdfg``.

        :param sdfg: The SDFG to transform in place.
        :param pipeline_results: Unused; neither transformation reads pipeline results.
        :returns: Number of applications, or ``None`` if none.
        """
        walk = Walk([(PruneConnectors(), PruneConnectors.nsdfg), (InlineSDFG(), InlineSDFG.nested_sdfg)])
        applied = 0
        candidate = walk.first_accepted(sdfg)
        while candidate is not None:
            xform, state, node = candidate
            dirty = ancestor_blocks(state)
            dirty.append(state)
            xform.permissive = False
            xform.apply(state, state.sdfg)
            applied += 1
            walk.forget(dirty)
            candidate = walk.first_accepted(sdfg)
        # No validation here: the pipeline's ``validate`` / ``validate_all`` own it, as they did for the wrapper, and
        # validation is not read-only (``Fill.validate`` drops an unwired value connector).
        return applied or None
