# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Turn the in-place updates ``X[s] = X[s] + r`` a refused loop carries into ``X[s] += r`` (WCR), so it maps.

lavamd's box loop accumulates each box's forces into the box's own slice of ``fv``::

    for l in range(n_boxes):
        fv[box_offsets[l] : box_offsets[l] + ppb, k] += forces(l)

The slice start is data, so ``LoopToMap`` cannot prove two boxes write apart and keeps the loop sequential. The
update is an accumulation, though: as ``fv[...] += forces`` it is the same arithmetic whichever boxes overlap, and a
loop whose only conflicting writes are WCR ones is a map (the WCR stays atomic across iterations). ``AugAssignToWCR``
leaves such an update alone inside its own map -- every map index addresses one element, so the map has no conflict
of its own -- which is right until an enclosing loop is to be lifted.

Only the container a refused loop names is rewritten, and only its in-place updates by a free tasklet or one in a
top-level map; a WCR
that does not let the loop map after all is conflict-free and ``RevertNonReductionWCR`` restores it.
"""

from typing import Any

from dace import SDFG
from dace.sdfg import nodes
from dace.sdfg.state import LoopRegion, SDFGState
from dace.transformation import pass_pipeline as ppl
from dace.transformation.dataflow.wcr_conversion import AugAssignToWCR
from dace.transformation.interstate.loop_to_map import LoopToMap
from dace.transformation.passes.canonicalize.eliminate_copy_back import EliminateCopyBack


def refused_container(sdfg: SDFG, loop: LoopRegion) -> str | None:
    """The container whose non-affine write keeps ``loop`` from mapping, or ``None`` if it maps or refuses else."""
    probe = LoopToMap()
    probe.setup_match(sdfg, loop.parent_graph.cfg_id, -1, {LoopToMap.loop: loop.parent_graph.node_id(loop)}, 0, override=True)
    if probe.can_be_applied(loop.parent_graph, 0, sdfg):
        return None
    reason = probe.last_refusal_reason or ""
    return probe.last_refusal_container if "not uniquely indexed" in reason else None


def in_place_update(sdfg: SDFG, state: SDFGState, name: str) -> AugAssignToWCR | None:
    """One matched ``X -> [map ->] tasklet [-> map] -> X`` update of ``name`` in ``state`` that may become WCR."""
    for tasklet in [n for n in state.nodes() if isinstance(n, nodes.Tasklet)]:
        entry = state.entry_node(tasklet)
        if entry is not None and state.entry_node(entry) is not None:
            continue
        ins = [e for e in state.in_edges(tasklet) if not e.data.is_empty() and e.data.data == name]
        outs = [e for e in state.out_edges(tasklet) if not e.data.is_empty() and e.data.data == name]
        if len(ins) != 1 or len(outs) != 1:
            continue
        source = state.memlet_path(ins[0])[0].src
        target = state.memlet_path(outs[0])[-1].dst
        if not (isinstance(source, nodes.AccessNode) and isinstance(target, nodes.AccessNode)):
            continue
        # A free tasklet (``f[j] = f[j] - d`` inside a pair loop) or one in a top-level map
        match = {
            AugAssignToWCR.input: state.node_id(source),
            AugAssignToWCR.tasklet: state.node_id(tasklet),
            AugAssignToWCR.output: state.node_id(target),
        }
        expr = 0
        if entry is not None:
            match[AugAssignToWCR.map_entry] = state.node_id(entry)
            match[AugAssignToWCR.map_exit] = state.node_id(state.exit_node(entry))
            expr = 1
        xform = AugAssignToWCR()
        xform.setup_match(sdfg, state.parent_graph.cfg_id, state.block_id, match, expr)
        if xform.can_be_applied(state, expr, sdfg, permissive=True):
            return xform
    return None


class CarriedAccumulationToWCR(ppl.Pass):
    """WCR for the in-place accumulations that alone keep a loop from mapping (``EliminateCopyBack`` first)."""

    CATEGORY: str = "Canonicalization"

    def modifies(self) -> ppl.Modifies:
        return ppl.Modifies.Nodes | ppl.Modifies.Edges | ppl.Modifies.Memlets

    def should_reapply(self, modified: ppl.Modifies) -> bool:
        return False

    def apply_pass(self, sdfg: SDFG, _pipeline_results: dict[str, Any]) -> int | None:
        """:returns: how many updates became WCR, or ``None`` if none."""
        EliminateCopyBack().apply_pass(sdfg, {})
        converted = 0
        for owner in sdfg.all_sdfgs_recursive():
            for loop in [r for r in owner.all_control_flow_regions() if isinstance(r, LoopRegion)]:
                # Until the loop maps or nothing is left to rewrite: an applied rewrite renumbers its state.
                while (name := refused_container(owner, loop)) is not None:
                    applied = False
                    for state in [b for b in loop.all_control_flow_blocks() if isinstance(b, SDFGState)]:
                        xform = in_place_update(owner, state, name)
                        if xform is not None:
                            xform.apply(state, owner)
                            converted += 1
                            applied = True
                    if not applied:
                        break
        return converted or None
