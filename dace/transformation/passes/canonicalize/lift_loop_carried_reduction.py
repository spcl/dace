# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Lift a loop-carried in-place array reduction to a write-conflict resolution."""

import ast
from typing import Any

from dace import SDFG, nodes, properties, symbolic
from dace.sdfg import SDFGState
from dace.sdfg.state import LoopRegion
from dace.transformation import pass_pipeline as ppl
from dace.transformation.dataflow.wcr_conversion import AugAssignToWCR
from dace.transformation.passes.analysis import loop_analysis
from dace.transformation.passes.loop_to_reduce import map_tasklet_augassign_candidates, only_read_modify_written
from dace.transformation.transformation import explicit_cf_compatible


def _copy_input_connector(tasklet: nodes.Tasklet) -> str | None:
    """The input connector of the pure copy ``__out = __inp`` the tasklet is, else ``None``."""
    if tasklet.language.name != "Python" or len(tasklet.code.code) != 1:
        return None
    stmt = tasklet.code.code[0]
    if not isinstance(stmt, ast.Assign) or len(stmt.targets) != 1:
        return None
    if isinstance(stmt.value, ast.Name) and stmt.value.id in tasklet.in_connectors:
        return stmt.value.id
    return None


def bounds_are_concrete(loop: LoopRegion) -> bool:
    """Whether the trip count of ``loop`` is a compile-time integer: see the module docstring for why a symbolic extent
    is refused."""
    bounds = (
        loop_analysis.get_init_assignment(loop),
        loop_analysis.get_loop_end(loop),
        loop_analysis.get_loop_stride(loop),
    )
    return all(bound is not None and not symbolic.pystr_to_symbolic(bound).free_symbols for bound in bounds)


def carried_update(loop: LoopRegion) -> tuple[SDFGState, AugAssignToWCR] | None:
    """The first update in a map of ``loop``, over a concrete range, of an element no iteration of ``loop`` moves, whose
    accumulator nothing else reads: the read-back it drops is what a recurrence needs."""
    itervar = loop.loop_variable
    for state in loop.states():
        for tasklet in [n for n in state.nodes() if isinstance(n, nodes.Tasklet)]:
            entry = state.entry_node(tasklet)
            if not isinstance(entry, nodes.MapEntry) or entry.map.range.free_symbols:
                continue
            for expr, binding in map_tasklet_augassign_candidates(state, entry, tasklet):
                source, target = binding[AugAssignToWCR.input], binding[AugAssignToWCR.output]
                written = state.edges_between(binding[AugAssignToWCR.map_exit], target)
                if source.data != target.data or any(
                    e.data.wcr is not None or itervar in {str(s) for s in e.data.subset.free_symbols} for e in written
                ):
                    continue
                xform = AugAssignToWCR()
                xform.setup_match(state.sdfg, state.parent_graph.cfg_id, state.block_id, binding, expr, override=True)
                if xform.can_be_applied(state, expr, state.sdfg, permissive=True) and only_read_modify_written(
                    loop, target.data
                ):
                    return state, xform
    return None


@properties.make_properties
@explicit_cf_compatible
class LiftLoopCarriedReduction(ppl.Pass):
    """Lift loop-carried in-place array reductions to WCR writes so the enclosing
    loop can be parallelized by ``LoopToMap``."""

    CATEGORY: str = "Canonicalization"

    def modifies(self) -> ppl.Modifies:
        return ppl.Modifies.Nodes | ppl.Modifies.Memlets

    def should_reapply(self, modified: ppl.Modifies) -> bool:
        return False

    def __init__(self, only: list | None = None):
        super().__init__()
        # When given, only these loops are candidates (the segments a split left behind).
        self._only = only

    def apply_pass(self, sdfg: SDFG, _pipeline_results: dict[str, Any]) -> int | None:
        lifted = 0
        only = None if self._only is None else {id(loop) for loop in self._only}
        if only is not None and not only:
            return None
        for sd in sdfg.all_sdfgs_recursive():
            for loop in [n for n in sd.all_control_flow_regions() if isinstance(n, LoopRegion)]:
                if loop.pinned_sequential or not loop.loop_variable or (only is not None and id(loop) not in only):
                    continue
                if not bounds_are_concrete(loop):
                    continue
                # A rewrite may fission its state, so the updates are found afresh after each
                while (update := carried_update(loop)) is not None:
                    state, xform = update
                    xform.apply(state, state.sdfg)
                    lifted += 1
        return lifted or None


__all__ = ["LiftLoopCarriedReduction"]
