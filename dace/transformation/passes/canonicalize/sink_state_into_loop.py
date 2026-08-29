# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Sink a state sitting between two loops into the second loop, so the loops become adjacent.

``FuseLoops`` matches a two-node path graph, so ``loop1 -> state -> loop2`` produces no candidate
pair at all and the loops are never offered for fusion. When the intervening state can be
*replicated* per iteration without changing what the program computes, sinking it into the second
loop restores adjacency and lets ``LoopFusion`` do its job.

Replication is only value-preserving when re-running the state is idempotent:
  * the second loop must not write anything the state reads -- otherwise later replicas see
    different inputs than the original single execution did,
  * the second loop must not write anything the state writes (WAW), and the state must not
    accumulate -- neither the WCR spelling nor a container it both reads and writes,
  * the second loop must run at least once, or the state would never execute at all.
Everything the state reads from the first loop is fine: the state stays after it.
"""
from typing import Any, Dict, Optional, Tuple

import sympy
from dace.ordered import OrderedSet

from dace import SDFG, Memlet, symbolic
from dace.sdfg.sdfg import InterstateEdge
from dace.sdfg.state import ControlFlowRegion, LoopRegion, SDFGState
from dace.transformation import pass_pipeline as ppl, transformation
from dace.transformation.passes.analysis import loop_analysis
from dace.transformation.passes.loop_to_scan import _fuse_body_states as fuse_body_states


@transformation.explicit_cf_compatible
class SinkStateIntoLoop(ppl.Pass):
    """Move a replicable state between two loops into the second loop."""
    CATEGORY: str = 'Canonicalization'

    def modifies(self) -> ppl.Modifies:
        return ppl.Modifies.CFG | ppl.Modifies.States | ppl.Modifies.Edges

    def should_reapply(self, modified: ppl.Modifies) -> bool:
        return False

    def depends_on(self):
        return OrderedSet()

    def apply_pass(self, sdfg: SDFG, _pipeline_results: Dict[str, Any]) -> Optional[int]:
        """Sink every replicable between-loops state in ``sdfg`` and its nested SDFGs.

        :param sdfg: The SDFG to transform in place.
        :returns: Number of states sunk, or ``None`` if none.
        """
        sunk = 0
        for sd in sdfg.all_sdfgs_recursive():
            changed = True
            while changed:
                changed = False
                for cfg in list(sd.all_control_flow_regions(recursive=True)):
                    if self.sink_one(cfg):
                        sunk += 1
                        changed = True
                        break
        return sunk or None

    @staticmethod
    def sink_one(cfg: ControlFlowRegion) -> bool:
        """Find and sink one replicable ``loop -> state -> loop`` middle state inside ``cfg``."""
        for state in cfg.nodes():
            if not isinstance(state, SDFGState):
                continue
            in_edges, out_edges = cfg.in_edges(state), cfg.out_edges(state)
            if len(in_edges) != 1 or len(out_edges) != 1:
                continue
            first, second = in_edges[0].src, out_edges[0].dst
            if not isinstance(first, LoopRegion) or not isinstance(second, LoopRegion):
                continue
            if not SinkStateIntoLoop.can_replicate(state, second):
                continue
            if in_edges[0].data.assignments or out_edges[0].data.assignments:
                continue
            if not in_edges[0].data.is_unconditional() or not out_edges[0].data.is_unconditional():
                continue

            body_start = second.start_block
            cfg.remove_node(state)
            cfg.add_edge(first, second, InterstateEdge())
            second.add_node(state, is_start_block=True)
            second.add_edge(state, body_start, InterstateEdge())
            # A two-state body is not fusable, so the point of sinking is lost unless the state
            #  merges into the body it now precedes. Whole-SDFG state fusion cannot reach inside a
            #  LoopRegion, so use the body-local merge that already exists for that reason.
            fuse_body_states(second)
            return True
        return False

    @staticmethod
    def can_replicate(state: SDFGState, loop: LoopRegion) -> bool:
        """Whether re-running ``state`` once per iteration of ``loop`` computes the same thing."""
        if not SinkStateIntoLoop.runs_at_least_once(loop):
            return False
        if SinkStateIntoLoop.accumulates(state):
            return False

        reads, writes = SinkStateIntoLoop.access_sets(state)
        # Read-modify-write: replica k would read what replica k-1 wrote.
        if reads & writes:
            return False
        loop_writes: OrderedSet[str] = OrderedSet()
        for block in loop.all_states():
            loop_writes |= SinkStateIntoLoop.access_sets(block)[1]
        # A replica must observe the same inputs, and must not race the loop's own writes.
        return not (loop_writes & (reads | writes))

    @staticmethod
    def accumulates(state: SDFGState) -> bool:
        """Whether ``state`` carries a WCR anywhere -- nested SDFGs too, whose boundary memlet
        need not carry the ``wcr`` the body writes with."""
        return any(isinstance(e.data, Memlet) and e.data.wcr is not None for e, _ in state.all_edges_recursive())

    @staticmethod
    def access_sets(state: SDFGState) -> Tuple[OrderedSet[str], OrderedSet[str]]:
        reads, writes = state.read_and_write_sets()
        return OrderedSet(reads), OrderedSet(writes)

    @staticmethod
    def runs_at_least_once(loop: LoopRegion) -> bool:
        """Whether ``loop`` provably performs at least one iteration."""
        start = loop_analysis.get_init_assignment(loop)
        end = loop_analysis.get_loop_end(loop)
        stride = loop_analysis.get_loop_stride(loop)
        if start is None or end is None or stride is None:
            return False
        span = symbolic.simplify(end - start) if stride > 0 else symbolic.simplify(start - end)
        return sympy.simplify(span >= 0) == sympy.true


__all__ = ['SinkStateIntoLoop']
