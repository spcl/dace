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
