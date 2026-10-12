# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Keep as symbols only what iteration domains and data accesses are written in; demote the rest to scalars.

A symbol is the IR's statement that a value shapes the graph: an array extent, a memlet subset, a map range, a loop's
bounds. A value an interstate edge assigns that does none of this (a counter such as ICON's ``clip_count = clip_count
+ 1``, a guard, a scale factor) is data spelled as a symbol. Loop analyses cannot see it as data flow, so a loop that
merely updates it reads as carrying a symbol and stays sequential; as a scalar it is an ordinary write the reduction
and privatization passes handle.

A value only conditions read stays a symbol when it is read from data the SDFG also writes (s275's
``g = aa[0, i]; if g > 0`` over the ``aa`` recurrence): as a scalar its read joins the dataflow of those writes and
blurs their dependences. A guard read from data nothing writes (CloudSC's ``iphase[jm]``) demotes, so the loops it
guards still fuse.
"""

import ast
import dataclasses
from collections.abc import Callable, Iterable
from typing import Any

from ordered_set import OrderedSet

from dace import SDFG, InterstateEdge, dtypes, properties
from dace.memlet import Memlet
from dace.sdfg import nodes
from dace.sdfg import utils as sdutil
from dace.sdfg.graph import MultiConnectorEdge
from dace.sdfg.state import AbstractControlFlowRegion, ConditionalBlock, LoopRegion
from dace.transformation import pass_pipeline as ppl
from dace.transformation import transformation


@dataclasses.dataclass(slots=True)
class StructuralSymbols:
    """:func:`~dace.sdfg.utils.structural_symbols` of one SDFG, kept current across add-only rewrites.

    ``names`` is the symbol set; the other fields record what it already covers, so
    :func:`extend_structural_symbols` scans only what a rewrite added. A demotion adds a scalar, states, access
    nodes, tasklets and edges and removes or edits nothing the scan reads, so the extended set equals a rebuild.
    """

    names: set[str] = dataclasses.field(default_factory=set)
    arrays: dict[str, None] = dataclasses.field(default_factory=dict)
    regions: dict[AbstractControlFlowRegion, None] = dataclasses.field(default_factory=dict)
    map_entries: dict[nodes.MapEntry, None] = dataclasses.field(default_factory=dict)
    nested: dict[nodes.NestedSDFG, None] = dataclasses.field(default_factory=dict)
    tasklets: dict[nodes.Tasklet, None] = dataclasses.field(default_factory=dict)
    edges: dict[MultiConnectorEdge[Memlet], None] = dataclasses.field(default_factory=dict)
    interstate: dict[InterstateEdge, None] = dataclasses.field(default_factory=dict)


def extend_structural_symbols(sd: SDFG, found: StructuralSymbols) -> None:
    """Add to ``found`` the structural uses of every descriptor, region, map and edge of ``sd`` it has not seen.

    Mirrors :func:`~dace.sdfg.utils.structural_symbols` at the same depth; from an empty ``found`` it is that scan.
    """
    for name, desc in sd.arrays.items():
        if name not in found.arrays:
            found.arrays[name] = None
            found.names.update(str(s) for s in desc.free_symbols)
    for region in sd.all_control_flow_regions():
        if region not in found.regions:
            found.regions[region] = None
            if isinstance(region, LoopRegion):
                found.names.update(sdutil.loop_domain_symbols(region))
            if isinstance(region, (LoopRegion, ConditionalBlock)):
                for code in region.get_meta_codeblocks():
                    found.names.update(sdutil.subscript_index_symbols(code.code, sd.arrays))
    for iedge in sd.all_interstate_edges():
        if iedge.data not in found.interstate:
            found.interstate[iedge.data] = None
            found.names.update(sdutil.subscript_index_symbols(iedge.data.condition.code, sd.arrays))
            for value in iedge.data.assignments.values():
                found.names.update(sdutil.subscript_index_symbols(ast.parse(value), sd.arrays))
    for state in sd.states():
        for node in state.nodes():
            if isinstance(node, nodes.MapEntry) and node not in found.map_entries:
                found.map_entries[node] = None
                found.names.update(str(s) for s in node.map.range.free_symbols)
            elif isinstance(node, nodes.NestedSDFG) and node not in found.nested:
                found.nested[node] = None
                found.names.update(sdutil.nested_mapping_symbols(node))
            elif isinstance(node, nodes.Tasklet) and node not in found.tasklets:
                found.tasklets[node] = None
                if node.code.language is dtypes.Language.Python:
                    connectors = node.in_connectors.keys() | node.out_connectors.keys()
                    found.names.update(sdutil.subscript_index_symbols(node.code.code, connectors))
        for edge in state.edges():
            if edge not in found.edges:
                found.edges[edge] = None
                found.names.update(str(s) for s in edge.data.free_symbols)


def guard_symbols(sd: SDFG) -> set[str]:
    """The symbols ``sd`` reads only in conditions (a branch, a loop condition, an interstate edge's condition) and
    assigns from a container ``sd`` also writes. No interstate assignment and no tasklet reads them, so none is a
    counter updated from itself."""
    in_condition: set[str] = set()
    elsewhere: set[str] = set()
    sources: dict[str, set[str]] = {}
    for region in sd.all_control_flow_regions():
        if isinstance(region, ConditionalBlock):
            for code in region.get_meta_codeblocks():
                in_condition.update(code.get_free_symbols())
        elif isinstance(region, LoopRegion) and region.loop_condition is not None:
            in_condition.update(region.loop_condition.get_free_symbols())
    for edge in sd.all_interstate_edges():
        in_condition.update(edge.data.condition.get_free_symbols())
        for name, value in edge.data.assignments.items():
            read = {node.id for node in ast.walk(ast.parse(value)) if isinstance(node, ast.Name)}
            elsewhere |= read
            sources.setdefault(name, set()).update(read & sd.arrays.keys())
    written: set[str] = set()
    for state in sd.states():
        for node in state.nodes():
            if isinstance(node, nodes.Tasklet):
                elsewhere.update(node.free_symbols)
            elif isinstance(node, nodes.AccessNode) and state.in_degree(node) > 0:
                written.add(node.data)
    return {name for name in in_condition - elsewhere if sources.get(name, set()) & written}


def demote_symbols(sdfg: SDFG, candidates: Callable[[SDFG], Iterable[str]]) -> int:
    """Demote to a transient scalar each non-guard symbol ``candidates`` names in an SDFG of ``sdfg``, unless the graph's
    structure is built from it (an array shape, a memlet subset, a map range, a loop's iteration domain, a nested
    SDFG's mapping) or it cannot become a transient.

    :returns: How many symbols were demoted.
    """
    demoted = 0
    for sd in sdfg.all_sdfgs_recursive():
        # Both gates answer from a whole-SDFG scan, asked once per unmutated span. A demotion only adds to what
        # the structural scan reads, so that set is extended over the additions; ``free_symbols`` is
        # flow-sensitive (a moved read can flip another name), so it is rebuilt, and only once a candidate
        # passes the cheap structural refusal.
        structural: StructuralSymbols | None = None
        structural_stale = False
        free_syms: set[str] | None = None
        guards: set[str] | None = None
        for name in candidates(sd):
            if name not in sd.symbols:
                continue
            if guards is None:
                # Demoting another name rewrites only that name's reads, so the set holds for the whole span
                guards = guard_symbols(sd)
            if name in guards:
                continue
            if structural is None:
                structural = StructuralSymbols()
                extend_structural_symbols(sd, structural)
            elif structural_stale:
                extend_structural_symbols(sd, structural)
            structural_stale = False
            # A symbol the graph's own structure is built from (a loop bound, a nested SDFG's
            # mapping) is not data and must stay a symbol; demoting it would rewrite control
            # flow rather than a read.
            if sdutil.symbol_carries_graph_structure(sd, name, structural=structural.names):
                continue
            if free_syms is None:
                # Consulted only for a top-level SDFG; a nested one reads its parent's mapping.
                free_syms = sd.free_symbols if sd.parent_nsdfg_node is None else set()
            if not sdutil.symbol_demotes_to_transient_scalar(sd, name, free_symbols=free_syms):
                continue
            # The symbol's OWN dtype: the arrays it reads carry it, and every consumer already
            # reads the symbol at the type it was declared with.
            sdutil.demote_symbol_to_scalar(sd, name, sd.symbols[name], None, free_symbols=free_syms)
            demoted += 1
            structural_stale = True
            # ``free_syms`` stays valid: the demoted symbol was defined inside ``sd``, so it was
            # never free, and replacing its reads by a scalar frees no other candidate.
    return demoted


def assigned_symbols(sd: SDFG) -> OrderedSet[str]:
    """Every symbol ``sd``'s own interstate edges assign, in insertion order (the demotion adds arrays to ``sd``)."""
    return OrderedSet(name.strip() for edge in sd.all_interstate_edges() for name in edge.data.assignments)


@properties.make_properties
@transformation.explicit_cf_compatible
class DemoteNonStructuralSymbols(ppl.Pass):
    """Demote every interstate-assigned symbol no iteration domain or data access is written in to a scalar."""

    CATEGORY: str = "Canonicalization"

    def modifies(self) -> ppl.Modifies:
        return ppl.Modifies.Everything

    def should_reapply(self, modified: ppl.Modifies) -> bool:
        return False

    def depends_on(self) -> list[type[ppl.Pass] | ppl.Pass]:
        return []

    def apply_pass(self, sdfg: SDFG, _pipeline_results: dict[str, Any]) -> int | None:
        """:returns: how many symbols were demoted, or ``None`` when none was."""
        return demote_symbols(sdfg, assigned_symbols) or None

    def report(self, pass_retval: int) -> str:
        return f"Demoted {pass_retval} non-structural symbols to scalars."
