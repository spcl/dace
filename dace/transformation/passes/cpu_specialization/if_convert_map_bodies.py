# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""If-convert the per-element ``if``/``else`` in a parallel CPU map's body into ``ITE`` selects.

A data-dependent branch per element (TSVC ``s2710``: ``if a[i] > b[i]: ... else: ...``) mispredicts
about half the time on random data and keeps the compiler from vectorizing the body; on one MI300A
quadrant that left the canonical form at 0.81x of a vectorized baseline. As selects, every arm is
evaluated and the result chosen per element, which the compiler vectorizes.

The rewrite reuses the vectorizer's branch lowering (``SameWriteSetIfElseToITECFG`` then
``BranchNormalization``) on the map's nested body. Evaluating both arms and writing back the old value
where an arm does not write is only sound when every arm reads and writes nothing but the map's own
element (the map parameters, unshifted) or element 0 of a container, and writes only containers the body
also reads; anything else -- a gather, a shifted read an index guard protects, a library node, a loop --
leaves the body as it was. A body the lowering cannot flatten completely is left untouched too.

Selects also make every arm's loads unconditional, so the conversion only pays when it adds no memory stream:
every container an arm of a top-level conditional reads or writes must be touched on every path through it (or
outside it). s2710's arms both read ``d`` and ``e``, and convert. TSVC ``s442``'s ``if k == 1 .. elif k == 4``
reads ``b``, ``c``, ``d`` or ``e`` on one path each, and has a path through none of them: as selects every element
streamed all four and stored ``a`` (40.9 ms against 29.2 ms branchy at preset L), so it keeps its branches.
"""

import ast
import copy
from typing import Any

from dace import SDFG, dtypes, symbolic
from dace import data as dt
from dace.sdfg import nodes
from dace.sdfg.state import ConditionalBlock, LoopRegion, SDFGState
from dace.transformation import pass_pipeline as ppl
from dace.transformation.passes.vectorization.branch_normalization import BranchNormalization
from dace.transformation.passes.vectorization.demote_data_reading_interstate_symbols import (
    DemoteDataReadingInterstateSymbols,
)
from dace.transformation.passes.vectorization.lower_interstate_conditional_assignments_to_tasklets import (
    LowerInterstateConditionalAssignmentsToTasklets,
)
from dace.transformation.passes.vectorization.same_write_set_if_else_to_ite_cfg import SameWriteSetIfElseToITECFG


def own_element(subset, lanes: set[str]) -> bool:
    """Whether ``subset`` names one element at the unshifted lanes, or element 0 of every dimension."""
    names = []
    for begin, end, step in subset.ndrange():
        if symbolic.simplify(end - begin) != 0:
            return False
        text = str(begin)
        if text in lanes:
            names.append(text)
        elif begin != 0:
            return False
    return not names or sorted(names) == sorted(lanes)


def subscripts_own_element(text: str, lanes: set[str]) -> bool:
    """Whether every subscript in the expression ``text`` indexes with an unshifted lane or 0."""
    for node in ast.walk(ast.parse(text, mode="eval")):
        if isinstance(node, ast.Subscript):
            index = node.slice.elts if isinstance(node.slice, ast.Tuple) else [node.slice]
            for dim in index:
                if not (isinstance(dim, ast.Name) and dim.id in lanes) and not (
                    isinstance(dim, ast.Constant) and dim.value == 0
                ):
                    return False
    return True


def arms(block: ConditionalBlock) -> list:
    return [branch for condition, branch in block.branches]


def arm_states(block: ConditionalBlock) -> list[SDFGState]:
    return [s for branch in arms(block) for s in branch.all_control_flow_blocks() if isinstance(s, SDFGState)]


def touched(region, body: SDFG, lanes: set[str]) -> set[str]:
    """The non-transient containers the states of ``region`` (a state or a region) stream: read or write at the lanes'
    element. Element 0 of a container (``x[0]``) is one value every element shares, not a stream."""
    states = [region] if isinstance(region, SDFGState) else list(region.all_control_flow_blocks())
    return {
        edge.data.data
        for state in states
        if isinstance(state, SDFGState)
        for edge in state.edges()
        if not edge.data.is_empty()
        and not body.arrays[edge.data.data].transient
        and {str(s) for s in edge.data.subset.free_symbols} & lanes
    }


def adds_no_streams(body: SDFG, lanes: set[str]) -> bool:
    """Whether selecting between the arms of ``body``'s top-level conditionals touches no container a branch would
    not: what any path touches, every path touches, unless the body touches it outside the conditionals too."""
    outside = set().union(*(touched(block, body, lanes) for block in body.nodes() if isinstance(block, SDFGState)))
    for block in body.nodes():
        if not isinstance(block, ConditionalBlock):
            continue
        paths = [touched(branch, body, lanes) for branch in arms(block)]
        if block.branches[-1][0] is not None:  # no else: the elements no arm takes touch nothing
            paths.append(set())
        if set().union(*paths) - set.intersection(*paths) - outside:
            return False
    return True


def convertible(body: SDFG, lanes: set[str], inputs: set[str]) -> bool:
    """Whether every arm of every conditional in ``body`` touches only the lanes' own element."""
    if any(isinstance(r, LoopRegion) for r in body.all_control_flow_regions()):
        return False
    blocks = [b for b in body.all_control_flow_blocks() if isinstance(b, ConditionalBlock)]
    if not blocks or not adds_no_streams(body, lanes):
        return False
    for block in blocks:
        for branch in arms(block):
            # An arm's own symbol binding read from data could be a gather its guard protects.
            if not all(
                subscripts_own_element(rhs, lanes)
                for e in branch.all_interstate_edges()
                for rhs in e.data.assignments.values()
            ):
                return False
        for state in arm_states(block):
            for node in state.nodes():
                if isinstance(node, (nodes.LibraryNode, nodes.NestedSDFG, nodes.EntryNode)):
                    return False
            for edge in state.edges():
                if edge.data.is_empty() or body.arrays[edge.data.data].transient:
                    continue
                if not own_element(edge.data.subset, lanes):
                    return False
                if isinstance(edge.dst, nodes.AccessNode) and edge.data.data not in inputs:
                    return False
    return True


def if_convert(body: SDFG) -> SDFG | None:
    """A copy of ``body`` with every conditional lowered to ITE selects, or ``None`` if one remains."""
    candidate = copy.deepcopy(body)
    try:
        for step in (
            DemoteDataReadingInterstateSymbols(),
            LowerInterstateConditionalAssignmentsToTasklets(),
            SameWriteSetIfElseToITECFG(),
            BranchNormalization(),
        ):
            step.apply_pass(candidate, {})
    except NotImplementedError:
        return None
    if any(isinstance(b, ConditionalBlock) for b in candidate.all_control_flow_blocks()):
        return None
    return candidate


class IfConvertParallelMapBodies(ppl.Pass):
    """Lower the per-element conditionals of every parallel CPU map's nested body to ITE selects."""

    CATEGORY: str = "Device Specialization"

    def modifies(self) -> ppl.Modifies:
        return ppl.Modifies.Nodes | ppl.Modifies.Descriptors | ppl.Modifies.CFG

    def should_reapply(self, modified: ppl.Modifies) -> bool:
        return False

    def depends_on(self):
        return set()

    def apply_pass(self, sdfg: SDFG, pipeline_results: dict[str, Any]) -> int | None:
        """:returns: how many map bodies were converted, or ``None`` if none were."""
        converted = 0
        for state in sdfg.states():
            scopes = state.scope_dict()
            for node in state.nodes():
                if not isinstance(node, nodes.NestedSDFG) or node.sdfg is None:
                    continue
                entry = scopes[node]
                if not isinstance(entry, nodes.MapEntry) or entry.map.schedule != dtypes.ScheduleType.CPU_Multicore:
                    continue
                lanes = {inner for inner, outer in node.symbol_mapping.items() if str(outer) in entry.map.params}
                if len(lanes) != len(entry.map.params):
                    continue
                inputs = {name for name in node.in_connectors if not isinstance(node.sdfg.arrays[name], dt.Stream)}
                if not convertible(node.sdfg, lanes, inputs):
                    continue
                lowered = if_convert(node.sdfg)
                if lowered is None:
                    continue
                lowered.parent = state
                lowered.parent_sdfg = sdfg
                lowered.parent_nsdfg_node = node
                node.sdfg = lowered
                converted += 1
        return converted or None
