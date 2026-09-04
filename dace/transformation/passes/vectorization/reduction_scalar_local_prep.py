# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Pre-vectorization pass: scalar-localize an array-slot reduction accumulator so the SIMD/tile
widener can widen it.

The K-lane tile widener folds a map-body reduction into K per-lane partial accumulators + one
horizontal ``TileReduce`` ONLY when the accumulator is a scalar-local: a plain scalar VARIABLE the
body reduces into, with the cross-tile reduction on the protected ``NestedSDFG -> AccessNode
-[wcr]-> MapExit -> acc`` boundary (the shape ``sum_1d`` / ``dot_1d`` already vectorize through).
When the reduction target is instead an ARRAY SLOT -- a map-exit WCR writing a single element of a
multi-element array ``arr[c]`` (a dot product into a fixed output slot, ``s[3] += a[i] * b[i]``) --
there is no scalar VARIABLE to give K per-lane copies of, the ``no loose WCR in the map body``
precondition fires, and the vectorizer BAILS.

This pass rewrites such an array-slot WCR reduction into a private transient SCALAR accumulator,
seeded unconditionally from the original slot, with an unconditional writeback afterwards --
delegating the actual rewrite to :func:`~dace.transformation.passes.canonicalize.
privatize_reduction_accumulator.privatize_reduction_accumulator` (the same array-slot -> scalar
machinery the CPU WCR codegen relies on). After the rewrite the reduction target is a scalar the
widener folds to per-lane partial sums + one horizontal ``TileReduce``; the seed / writeback stay at
the OUTER scope (a plain init state before + writeback state after the map state), so the map body
itself remains a single-state dataflow chain the walker can widen.

Only fires when it ENABLES widening (an "only if it helps" oracle -- no-ops are left alone):

* a GENUINE multi-element array slot -- a plain scalar / length-1 accumulator already widens, so is
  skipped (rewriting it would perturb a currently-working path);
* an ASSOCIATIVE reduction op (``+`` / ``*`` / ``min`` / ``max``); a non-associative ``-`` / ``/``
  WCR is not a foldable reduction and is left untouched;
* a LOOP-INVARIANT single-element slot -- a slot indexed by the map param is an indexed scatter, not
  a scalar fold, and is skipped;
* on an INNERMOST, unit-step map -- the tile-widening candidate; a non-innermost or strided map is
  not what the widener tiles.

A genuine cross-iteration recurrence (the accumulator read back in the body to compute the
increment) is NOT expressed as a map-exit WCR -- it is a read+write pair -- so it is never matched,
and the rewrite is value-preserving (the scalar seed is the slot's pre-map value; the writeback is
unconditional, hence correct for zero map iterations too).
"""
from typing import Optional

from dace import SDFG, data
from dace.dtypes import ReductionType
from dace.frontend.operations import detect_reduction_type
from dace.sdfg import SDFGState, nodes
from dace.transformation import pass_pipeline as ppl
from dace.transformation import transformation as xf
from dace.transformation.passes.canonicalize.privatize_reduction_accumulator import (
    privatize_reduction_accumulator, )

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

    def apply_pass(self, sdfg: SDFG, _) -> Optional[int]:
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
        """True iff ``map_exit``'s map is a tile-widening candidate: innermost (no nested map in its
        scope) and unit-step on every dim (the shape the widener strides to W).

        The widener only tiles innermost unit-step maps, so an array-slot reduction on any other map
        is not something widening would reach -- leave it alone (the "only if it helps" gate)."""
        map_entry = state.entry_node(map_exit)
        if map_entry is None:
            return False
        between = state.all_nodes_between(map_entry, map_exit) or set()
        if any(isinstance(n, nodes.MapEntry) for n in between):
            return False
        return all(str(step) == "1" for _, _, step in map_entry.map.range)

    def _is_array_slot_reduction(self, state: SDFGState, map_exit: nodes.MapExit, iedge) -> bool:
        """True iff ``iedge`` is a foldable WCR reduction into a genuine multi-element array slot --
        the shape the widener bails on and this pass privatizes.

        Guards (each leaves the reduction untouched when it fails):

        * ``iedge`` carries a WCR and enters through an ``IN_*`` connector with exactly one matching
          ``OUT_*`` boundary edge to an :class:`AccessNode`;
        * the target descriptor is a GENUINE multi-element ``Array`` (a ``Scalar`` / length-1
          accumulator already widens, so is skipped);
        * the WCR op is associative / foldable (``+`` / ``*`` / ``min`` / ``max``);
        * the write slot is a single element independent of the map params (a loop-invariant scalar
          fold, not a per-iteration indexed scatter);
        * the accumulator array is not read inside the map scope (a read-back would be a genuine
          recurrence, not a WCR reduction -- must not be rewritten).
        """
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
