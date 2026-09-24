# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Scalar-localize an array-slot reduction accumulator so the tile widener can widen it.

The widener folds a map-body reduction into per-lane partials plus one ``TileReduce`` only when
the accumulator is a scalar variable. A map-exit WCR into one element of a multi-element array
(``s[3] += a[i] * b[i]``) has none, so the vectorizer would refuse. This pass rewrites it to a
private scalar accumulator seeded from the slot, with a writeback afterwards, via
:func:`~dace.transformation.passes.canonicalize.privatize_reduction_accumulator.privatize_reduction_accumulator`;
seed and writeback stay in the outer scope so the map body remains one dataflow state.

Fires only when it enables widening: a genuine multi-element array slot, an associative op
(``+`` / ``*`` / ``min`` / ``max``), a loop-invariant slot, on an innermost unit-step map. A
recurrence that reads the accumulator in the body is not a map-exit WCR and never matches; the
rewrite is value-preserving, also for zero iterations.
"""
from typing import Any

from dace import SDFG, data
from dace.dtypes import ReductionType
from dace.frontend.operations import detect_reduction_type
from dace.memlet import Memlet
from dace.sdfg import SDFGState, nodes
from dace.sdfg.graph import MultiConnectorEdge
from dace.transformation import pass_pipeline as ppl
from dace.transformation import transformation as xf
from dace.transformation.passes.canonicalize.privatize_reduction_accumulator import (
    privatize_reduction_accumulator, )
from dace.transformation.passes.vectorization.utils.map_predicates import map_body_nodes

#: Reduction ops the tile widener + ``TileReduce`` fold. A ``ReductionType.Custom`` WCR
#: (non-associative ``-`` / ``/``) is not a foldable reduction, so it is never rewritten.
_FOLDABLE_OPS = (ReductionType.Sum, ReductionType.Product, ReductionType.Min, ReductionType.Max)


@xf.explicit_cf_compatible
class PrepareReductionForWidening(ppl.Pass):
    """Scalar-localize array-slot WCR reductions that would otherwise block the tile widener.

    A gated front-end over :func:`privatize_reduction_accumulator`: it restricts the rewrite to the
    array-slot reductions that make the vectorizer bail (see the module docstring), leaving every
    already-widenable or non-reduction shape untouched.
    """

    def modifies(self) -> ppl.Modifies:
        return (ppl.Modifies.CFG | ppl.Modifies.Nodes | ppl.Modifies.Memlets | ppl.Modifies.States
                | ppl.Modifies.Descriptors)

    def should_reapply(self, modified: ppl.Modifies) -> bool:
        return bool(modified & ppl.Modifies.CFG)

    def apply_pass(self, sdfg: SDFG, _: dict[str, Any]) -> int | None:
        """Scalar-localize every array-slot WCR reduction that gates widening.

        :param sdfg: The SDFG to transform in place (recursively over all states).
        :returns: The number of reductions rewritten, or ``None`` if none.
        """
        count = 0
        for state in list(sdfg.all_states()):
            for map_exit in [n for n in state.nodes() if isinstance(n, nodes.MapExit)]:
                if not self._map_is_widening_candidate(state, map_exit):
                    continue
                for iedge in list(state.in_edges(map_exit)):
                    if not self._is_array_slot_reduction(state, map_exit, iedge):
                        continue
                    if privatize_reduction_accumulator(state, map_exit, iedge):
                        count += 1
        return count or None

    @staticmethod
    def _map_is_widening_candidate(state: SDFGState, map_exit: nodes.MapExit) -> bool:
        # True iff ``map_exit``'s map is a tile-widening candidate: innermost (no nested map in its scope) and unit-step
        # on every dim (the shape the widener strides to W).
        map_entry = state.entry_node(map_exit)
        if map_entry is None:
            return False
        # Scope membership, not ``all_nodes_between``: an emptied walk holds no MapEntry, so an
        # OUTER map would read as innermost and be privatized as a widening candidate.
        between = map_body_nodes(state, map_entry)
        if any(isinstance(n, nodes.MapEntry) for n in between):
            return False
        return all(str(step) == "1" for _, _, step in map_entry.map.range)

    def _is_array_slot_reduction(self, state: SDFGState, map_exit: nodes.MapExit,
                                 iedge: MultiConnectorEdge[Memlet]) -> bool:
        # True iff ``iedge`` is a foldable WCR reduction into a genuine multi-element array slot -- the shape the
        # widener bails on and this pass privatizes.
        if iedge.data is None or iedge.data.wcr is None:
            return False
        in_conn = iedge.dst_conn
        if not in_conn or not in_conn.startswith("IN_"):
            return False
        out_edges = [e for e in state.out_edges(map_exit) if e.src_conn == "OUT_" + in_conn[3:]]
        if len(out_edges) != 1:
            return False
        arr_node = out_edges[0].dst
        if not isinstance(arr_node, nodes.AccessNode):
            return False
        desc = state.sdfg.arrays.get(arr_node.data)
        # A genuine multi-element array slot only: a Scalar / length-1 accumulator already widens.
        if desc is None or isinstance(desc, data.Scalar) or desc.total_size == 1:
            return False
        if detect_reduction_type(iedge.data.wcr) not in _FOLDABLE_OPS:
            return False
        write_subset = iedge.data.subset
        if write_subset is None or write_subset.num_elements() != 1:
            return False
        map_param_set = set(state.entry_node(map_exit).map.params)
        if any(s in map_param_set for s in (str(x) for x in write_subset.free_symbols)):
            return False
        # A read of the accumulator inside the map scope would make this a cross-iteration
        # recurrence, not a pure reduction -- refuse (the WCR shape normally precludes it, but guard
        # explicitly so an aliased read is never mis-rewritten).
        map_entry = state.entry_node(map_exit)
        if any(e.data is not None and e.data.data == arr_node.data for e in state.out_edges(map_entry)):
            return False
        return True
