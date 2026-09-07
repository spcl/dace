# Copyright 2019-2022 ETH Zurich and the DaCe authors. All rights reserved.
"""
Contains implementations of SDFG inlining and state fusion passes.
"""

from dataclasses import dataclass
from typing import Any, Dict, Optional

from dace import SDFG, properties
from dace.sdfg import nodes
from dace.sdfg.state import ConditionalBlock, FunctionCallRegion, LoopRegion, NamedRegion
from dace.sdfg.utils import fuse_states, inline_control_flow_regions, inline_sdfgs
from dace.transformation import pass_pipeline as ppl
from dace.transformation.transformation import explicit_cf_compatible


@dataclass(unsafe_hash=True)
@properties.make_properties
@explicit_cf_compatible
class LinearStateFusion(ppl.Pass):
    """Fuse states by walking each control-flow region's chain forward, once.

    :func:`~dace.sdfg.utils.fuse_states` scans a region's edges and, after fusing ``(u, v)``, puts
    BOTH endpoints in a skip set -- so the merged block is not retried until the next full scan. A
    chain of N states therefore costs about log(N) scans of every edge in the region.

    Within a region the blocks are overwhelmingly a line graph, and a line can be collapsed in one
    walk: fuse ``(i, i+1)``, then try the block that survived against ``i+2``, and keep going. A
    refusal advances to the next pair instead of restarting. Chains that are not lines -- a block
    with several successors, or a successor reachable from elsewhere -- simply end the walk, so
    branching regions cost one visit per block and no rescan.

    Which block survives a fusion is decided by ``StateFusion.apply``: it deletes the FIRST state
    when that one is empty and the SECOND in every other case.
    """

    CATEGORY: str = 'Simplification'

    permissive = properties.Property(dtype=bool, default=False, desc='If True, ignores some race condition checks.')

    def should_reapply(self, modified: ppl.Modifies) -> bool:
        return modified & (ppl.Modifies.States | ppl.Modifies.InterstateEdges)

    def modifies(self) -> ppl.Modifies:
        return ppl.Modifies.States

    def fuse_pair(self, cfg, sd, first, second) -> bool:
        """Fuse ``first`` into ``second`` (or the reverse) if the transformation allows it."""
        from dace.sdfg.state import SDFGState
        from dace.transformation.interstate import BlockFusion, StateFusionExtended
        if isinstance(first, SDFGState) and isinstance(second, SDFGState):
            xform = StateFusionExtended()
            candidate = {StateFusionExtended.first_state: first, StateFusionExtended.second_state: second}
        else:
            xform = BlockFusion()
            candidate = {BlockFusion.first_block: first, BlockFusion.second_block: second}
        xform.setup_match(cfg, cfg.cfg_id, -1, candidate, 0, override=True)
        if not xform.can_be_applied(cfg, 0, sd, permissive=self.permissive):
            return False
        xform.apply(cfg, sd)
        return True

    def apply_pass(self, sdfg: SDFG, _: Dict[str, Any]) -> Optional[int]:
        """Fuse every fusible pair of blocks, walking each region's chains forward.

        :param sdfg: The SDFG to transform.
        :returns: The number of fusions applied, or ``None`` if none were.
        """
        from dace.sdfg.state import SDFGState
        fused = 0
        for sd in sdfg.all_sdfgs_recursive():
            for cfg in sd.all_control_flow_regions():
                seen = set()
                for block in list(cfg.nodes()):
                    if block in seen:
                        continue
                    current = block
                    while current is not None and current not in seen:
                        seen.add(current)
                        successors = [e.dst for e in cfg.out_edges(current)]
                        # Not a line here: leave it, rather than guessing an order for a branch.
                        if len(successors) != 1 or cfg.in_degree(successors[0]) != 1:
                            break
                        following = successors[0]
                        if following is current:
                            break
                        # Ask before applying: ``apply`` deletes the first block only when it is
                        # empty, so this is what says which one the walk continues from.
                        first_empty = isinstance(current, SDFGState) and current.is_empty()
                        if self.fuse_pair(cfg, sd, current, following):
                            fused += 1
                            # The merged block against i+2, without rescanning the region.
                            current = following if first_empty else current
                            seen.discard(current)
                        else:
                            current = following
        return fused or None

    def report(self, pass_retval: int) -> str:
        return f'Fused {pass_retval} states.'


@dataclass(unsafe_hash=True)
@properties.make_properties
@explicit_cf_compatible
class FuseStates(ppl.Pass):
    """
    Fuses all possible states of an SDFG (and all sub-SDFGs).
    """

    CATEGORY: str = 'Simplification'

    permissive = properties.Property(dtype=bool, default=False, desc='If True, ignores some race condition checks.')
    progress = properties.Property(dtype=bool,
                                   default=None,
                                   allow_none=True,
                                   desc='Whether to print progress, or None for default (print after 5 seconds).')

    def should_reapply(self, modified: ppl.Modifies) -> bool:
        return modified & (ppl.Modifies.States | ppl.Modifies.InterstateEdges)

    def modifies(self) -> ppl.Modifies:
        return ppl.Modifies.States

    def apply_pass(self, sdfg: SDFG, _: Dict[str, Any]) -> Optional[int]:
        """
        Fuses all possible states of an SDFG (and all sub-SDFGs).

        :param sdfg: The SDFG to transform.

        :return: The total number of states fused, or None if did not apply.
        """
        fused = fuse_states(sdfg, self.permissive, self.progress)
        return fused or None

    def report(self, pass_retval: int) -> str:
        return f'Fused {pass_retval} states.'


@dataclass(unsafe_hash=True)
@properties.make_properties
@explicit_cf_compatible
class InlineSDFGs(ppl.Pass):
    """
    Inlines all possible nested SDFGs (and sub-SDFGs).
    """

    CATEGORY: str = 'Simplification'

    permissive = properties.Property(dtype=bool, default=False, desc='If True, ignores some checks on inlining.')
    progress = properties.Property(dtype=bool,
                                   default=None,
                                   allow_none=True,
                                   desc='Whether to print progress, or None for default (print after 5 seconds).')
    multistate = properties.Property(dtype=bool, default=True, desc='If True, include multi-state inlining.')

    def should_reapply(self, modified: ppl.Modifies) -> bool:
        return modified & (ppl.Modifies.NestedSDFGs | ppl.Modifies.States)

    def modifies(self) -> ppl.Modifies:
        return ppl.Modifies.States | ppl.Modifies.NestedSDFGs

    def apply_pass(self, sdfg: SDFG, _: Dict[str, Any]) -> Optional[int]:
        """
        Inlines all possible nested SDFGs (and all sub-SDFGs).

        :param sdfg: The SDFG to transform.

        :return: The total number of states fused, or None if did not apply.
        """
        inlined = inline_sdfgs(sdfg, self.permissive, self.progress, self.multistate)
        return inlined or None

    def report(self, pass_retval: int) -> str:
        return f'Inlined {pass_retval} SDFGs.'


@dataclass(unsafe_hash=True)
@properties.make_properties
@explicit_cf_compatible
class InlineControlFlowRegions(ppl.Pass):
    """
    Inlines all control flow regions.
    """

    CATEGORY: str = 'Simplification'

    progress = properties.Property(dtype=bool,
                                   default=None,
                                   allow_none=True,
                                   desc='Whether to print progress, or None for default (print after 5 seconds).')

    no_inline_loops = properties.Property(dtype=bool, default=True, desc='Whether to prevent inlining loops.')
    no_inline_conditional = properties.Property(dtype=bool,
                                                default=True,
                                                desc='Whether to prevent inlining conditional blocks.')
    no_inline_function_call_regions = properties.Property(dtype=bool,
                                                          default=True,
                                                          desc='Whether to prevent inlining function call regions.')
    no_inline_named_regions = properties.Property(dtype=bool,
                                                  default=True,
                                                  desc='Whether to prevent inlining named control flow regions.')

    def should_reapply(self, modified: ppl.Modifies) -> bool:
        return modified & (ppl.Modifies.NestedSDFGs | ppl.Modifies.States)

    def modifies(self) -> ppl.Modifies:
        return ppl.Modifies.States | ppl.Modifies.NestedSDFGs

    def apply_pass(self, sdfg: SDFG, _: Dict[str, Any]) -> Optional[int]:
        """
        Inlines all possible nested SDFGs (and all sub-SDFGs).

        :param sdfg: The SDFG to transform.

        :return: The total number of states fused, or None if did not apply.
        """
        ignore_region_types = []
        if self.no_inline_loops:
            ignore_region_types.append(LoopRegion)
        if self.no_inline_conditional:
            ignore_region_types.append(ConditionalBlock)
        if self.no_inline_named_regions:
            ignore_region_types.append(NamedRegion)
        if self.no_inline_function_call_regions:
            ignore_region_types.append(FunctionCallRegion)
        if len(ignore_region_types) < 1:
            ignore_region_types = None

        inlined = 0
        while True:
            inlined_in_iteration = inline_control_flow_regions(sdfg, None, ignore_region_types, self.progress)
            if inlined_in_iteration < 1:
                break
            inlined += inlined_in_iteration

        if inlined:
            sdfg.reset_cfg_list()
            return inlined
        return None

    def report(self, pass_retval: int) -> str:
        return f'Inlined {pass_retval} regions.'


@dataclass(unsafe_hash=True)
@properties.make_properties
@explicit_cf_compatible
class FixNestedSDFGReferences(ppl.Pass):
    """
    Fixes nested SDFG references to parent state/SDFG/node
    """

    CATEGORY: str = 'Cleanup'

    def should_reapply(self, modified: ppl.Modifies) -> bool:
        return modified & (ppl.Modifies.States | ppl.Modifies.NestedSDFGs)

    def modifies(self) -> ppl.Modifies:
        return ppl.Modifies.NestedSDFGs

    def apply_pass(self, sdfg: SDFG, _: Dict[str, Any]) -> Optional[int]:
        modified = 0
        for node, state in sdfg.all_nodes_recursive():
            if not isinstance(node, nodes.NestedSDFG) or node.sdfg is None:
                continue
            was_modified = False
            if node.sdfg.parent_nsdfg_node is not node:
                was_modified = True
                node.sdfg.parent_nsdfg_node = node
            if node.sdfg.parent is not state:
                was_modified = True
                node.sdfg.parent = state
            if node.sdfg.parent_sdfg is not state.parent:
                was_modified = True
                node.sdfg.parent_sdfg = state.parent

            if was_modified:
                modified += 1

        return modified or None

    def report(self, pass_retval: int) -> str:
        return f'Fixed {pass_retval} nested SDFG references.'
