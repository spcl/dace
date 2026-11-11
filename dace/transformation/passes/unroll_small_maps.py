# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Unroll a map over a small constant domain whose body holds large parallel work.

``for jm in 0:5 { zexplicit = 0; for (b, l) in nblocks x klon {...}; ... }`` is a parallel map whose own
iterations are few: shared out as they stand, five threads walk the whole ``nblocks x klon`` planes. Unrolled,
the large maps in its body are top-level, each one shared out (or launched) over its own range. The canonical
form keeps the map; a target specialization unrolls it before it decides what runs in parallel.
"""

from typing import Any

import sympy

from dace import SDFG, symbolic
from dace.sdfg import nodes
from dace.sdfg.state import SDFGState, SymbolResolver
from dace.transformation import helpers as xfh
from dace.transformation import pass_pipeline as ppl
from dace.transformation.dataflow.map_expansion import MapExpansion
from dace.transformation.dataflow.map_for_loop import MapToForLoop
from dace.transformation.interstate.loop_unroll import LoopUnroll


def constant_trips(entry: nodes.MapEntry) -> int | None:
    """How many iterations ``entry``'s map runs, when every bound is a literal."""
    trips = 1
    for begin, end, step in entry.map.range.ranges:
        count = sympy.sympify((end - begin) // step + 1)
        if not count.is_Integer:
            return None
        trips *= max(int(count), 0)
    return trips


def holds_large_map(state: SDFGState, entry: nodes.MapEntry, resolver: SymbolResolver) -> bool:
    """Whether a map in ``entry``'s scope, nested SDFGs included, spans an axis that is not small."""
    for node in state.scope_subgraph(entry, include_entry=False, include_exit=False).nodes():
        if isinstance(node, nodes.MapEntry) and not all(xfh.small_map_axes(state, node, resolver)):
            return True
        if isinstance(node, nodes.NestedSDFG):
            for inner, owner in node.sdfg.all_nodes_recursive():
                if isinstance(inner, nodes.MapEntry) and not all(xfh.small_map_axes(owner, inner, resolver)):
                    return True
    return False


def outside_every_map(sdfg: SDFG) -> bool:
    """Whether no map scope encloses ``sdfg`` (its nesting chain included)."""
    while sdfg.parent_nsdfg_node is not None:
        if sdfg.parent.entry_node(sdfg.parent_nsdfg_node) is not None:
            return False
        sdfg = sdfg.parent_sdfg
    return True


def small_maps(sdfg: SDFG, resolver: SymbolResolver) -> list[tuple[SDFGState, nodes.MapEntry]]:
    """The top-level maps outside every map scope with a small constant domain and large work in their body."""
    found = []
    for sd in sdfg.all_sdfgs_recursive():
        if not outside_every_map(sd):
            continue
        for state in sd.states():
            for node in state.scope_children()[None]:
                if not isinstance(node, nodes.MapEntry):
                    continue
                trips = constant_trips(node)
                if (
                    trips is not None
                    and trips > 1
                    and symbolic.is_small_extent(sympy.Integer(trips), resolver.facts_at(state, node))
                    and holds_large_map(state, node, resolver)
                ):
                    found.append((state, node))
    return found


def unroll(state: SDFGState, entry: nodes.MapEntry) -> bool:
    """Unroll the outer axis of ``entry``'s map, a loop (:class:`MapToForLoop`) that is then unrolled; the axes an
    expansion leaves inside are top-level maps of the unrolled copies. Whether anything changed."""
    sdfg = state.sdfg
    if len(entry.map.params) > 1:
        MapExpansion.apply_to(sdfg, map_entry=entry, verify=False, save=False)
    to_loop = MapToForLoop()
    to_loop.setup_match(sdfg, state.parent_graph.cfg_id, state.block_id, {MapToForLoop.map_entry: entry}, 0, override=True)
    if not to_loop.can_be_applied(state, 0, sdfg):
        return False
    to_loop.apply(state, sdfg)
    loop = to_loop.loop_region
    if LoopUnroll.can_be_applied_to(loop.sdfg, loop=loop):
        LoopUnroll.apply_to(loop.sdfg, loop=loop, verify=False, save=False, annotate=False)
    return True


class UnrollSmallMaps(ppl.Pass):
    """Unroll every map outside a map scope whose constant domain is small and whose body holds large maps."""

    CATEGORY: str = "Device Specialization"

    def modifies(self) -> ppl.Modifies:
        return ppl.Modifies.States | ppl.Modifies.CFG | ppl.Modifies.Nodes | ppl.Modifies.Edges | ppl.Modifies.Descriptors

    def should_reapply(self, modified: ppl.Modifies) -> bool:
        return False

    def apply_pass(self, sdfg: SDFG, pipeline_results: dict[str, Any]) -> int | None:
        """Unroll until no such map is left: an expanded map's inner axes become top-level maps in turn.

        :param sdfg: the SDFG, in place.
        :param pipeline_results: unused.
        :returns: how many maps were unrolled, or ``None`` if none were.
        """
        unrolled, refused = 0, set()
        while True:
            found = [(state, entry) for state, entry in small_maps(sdfg, SymbolResolver()) if entry not in refused]
            if not found:
                return unrolled or None
            # One at a time: an unroll moves the states the others were found in
            state, entry = found[0]
            if unroll(state, entry):
                unrolled += 1
                sdfg.reset_cfg_list()
            else:
                refused.add(entry)
