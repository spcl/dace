# Copyright 2019-2025 ETH Zurich and the DaCe authors. All rights reserved.

from typing import Any, Dict, Optional, Set

from dace import properties
from dace.sdfg.sdfg import SDFG
from dace.sdfg.state import AbstractControlFlowRegion, ConditionalBlock
from dace.transformation import pass_pipeline as ppl
from dace.transformation import transformation


@properties.make_properties
@transformation.explicit_cf_compatible
class RegionBoundaryStates(ppl.Pass):
    """
    Brackets a control flow region with an empty state in its parent region when its size symbols demand it.

    Allocation is only emitted inside a state. Without the leading state, data sized by a symbol that the region's
    incoming edge assigns is allocated at the region's predecessor, before the symbol is defined. The trailing state
    gives the matching deallocation a place to go.

    Only regions whose incoming assignments feed a transient's size are bracketed. Bracketing every region instead
    triples the state count of a program that never sizes anything from an interstate assignment.
    """

    CATEGORY: str = 'Helper'

    def modifies(self) -> ppl.Modifies:
        return ppl.Modifies.CFG

    def should_reapply(self, modified: ppl.Modifies) -> bool:
        # Reapplying would bracket the brackets, so the pass runs once.
        return False

    def apply_pass(self, sdfg: SDFG, _: Dict[str, Any]) -> Optional[int]:
        """
        :param sdfg: The SDFG to modify in-place.
        :return: Number of states inserted, or None if unchanged.
        """
        # Sizes are collected across the whole tree, not per SDFG: a nested transient can be sized by a
        # symbol an enclosing region assigns and passes down through symbol_mapping, and reading only the
        # owning SDFG's arrays leaves that region unbracketed. Sharing a name that needs no boundary only
        # costs two empty states.
        sized_by: Set[str] = {
            str(s)
            for nested in sdfg.all_sdfgs_recursive()
            for desc in nested.arrays.values() if desc.transient for s in desc.free_symbols
        }
        if not sized_by:
            return None

        inserted = 0
        # Branches of a conditional are entered without an inter-state edge, so they need no boundary.
        regions = [
            cfg for cfg in sdfg.all_control_flow_regions(recursive=True) if not isinstance(cfg, ConditionalBlock)
        ]
        for cfg in regions:
            for node in list(cfg.nodes()):
                if not isinstance(node, AbstractControlFlowRegion):
                    continue
                # The two sides answer different questions. A leading state is where an allocation goes,
                # so it is needed when the region's incoming edge assigns a symbol a size reads. A
                # trailing state is where the matching deallocation goes, so it is needed when the region
                # ends its scope: without a state after it, the free has nowhere to be emitted and is
                # silently dropped.
                assigned = {name for e in cfg.in_edges(node) for name in e.data.assignments}
                if assigned & sized_by:
                    cfg.add_state_before(node, is_start_block=node is cfg.start_block)
                    inserted += 1
                if (assigned & sized_by) or not cfg.out_edges(node):
                    cfg.add_state_after(node)
                    inserted += 1
        return inserted or None
