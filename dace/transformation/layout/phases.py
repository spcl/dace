# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Program phases for layout-aware MPI: the top-level control-flow blocks, each an opaque phase.

Unlike ``line_graph`` (flat states only, used by the layout DP), ``program_phases`` treats every top-level
block as one phase and does NOT recurse -- a ``LoopRegion``/``ConditionalBlock`` is a single opaque phase
(its body is scanned for work but not split into sub-phases). Layout is invariant within a phase (relayout
happens only at phase boundaries), so any MPI op in a phase communicates in that phase's one layout.
Sub-phasing (splitting a loop into several layout phases) is a deferred TODO.
"""
from dataclasses import dataclass
from typing import List

from dace import SDFG, SDFGState
from dace.sdfg.state import ControlFlowBlock


@dataclass
class Phase:
    """One top-level control-flow block treated as an opaque layout phase."""
    block: ControlFlowBlock
    index: int

    def states(self) -> List[SDFGState]:
        """The states this phase contains (itself if a plain state; the whole body if a region -- not sub-phased)."""
        if isinstance(self.block, SDFGState):
            return [self.block]
        return list(self.block.all_states())


def program_phases(sdfg: SDFG) -> List[Phase]:
    """The top-level control-flow blocks of ``sdfg``, one :class:`Phase` each (NOT recursed). Order is the
    SDFG's block order; the MPI pass handles each point-to-point op independently, so ordering is cosmetic."""
    return [Phase(block=block, index=i) for i, block in enumerate(sdfg.nodes())]
