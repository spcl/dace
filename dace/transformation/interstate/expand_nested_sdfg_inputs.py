# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Widen a top-level NSDFG's in/out subsets to the full outer arrays so
:class:`~dace.transformation.interstate.multistate_inline.InlineMultistateSDFG`
can subsequently inline it.

Background
----------

``InlineMultistateSDFG.can_be_applied`` refuses NSDFGs whose in/out edges
do not read/write the full outer array, because the corresponding
``apply()`` does not perform the dimension-offsetting required to correct
inner memlets after the inline (there is a literal ``TODO: Modify
memlets by offsetting`` in its body). This refusal is a correctness
gate: bypassing it would inline the body and rename the inner connector
``IN_a`` to the outer array ``a`` without adjusting the per-iteration
offset baked into inner memlets, so ``IN_a[i]`` would land on ``a[i]``
instead of ``a[ii + i]``.

This transformation performs the missing offset adjustment up-front:

1. For each in/out edge of the NSDFG node:

   * Compute the per-axis offset (the lower bound of the original
     narrowed subset, e.g. ``ii`` for ``a[ii:ii+6, 0:M]``).
   * Replace the inner array descriptor with one that mirrors the OUTER
     descriptor's shape / strides.
   * Widen the outer-side subset to the full outer-array range.
   * Walk every inner memlet that references this connector's array
     and add the offset to its subset coordinates.

2. After this transformation runs, every NSDFG edge satisfies the
   strict full-array check and ``InlineMultistateSDFG.apply()`` produces
   a correct inlined SDFG.

Refusal criteria
----------------

* The NSDFG is inside a Map scope (``state.entry_node(nsdfg) is not
  None``). Per-iteration narrowing is intentional inside a Map, so
  widening it would lose the parallelism contract.
* The narrowed-to-full expansion would change the rank of the inner
  descriptor (axis-collapse case, e.g. ``a[0:1, 0:M]`` whose inner
  descriptor was simplified to a 1-D ``[M]`` array). The transformation
  refuses rather than guess how to re-promote the missing axis.
* The outer array doesn't exist in the parent SDFG (orphan descriptor).
"""
import copy
from typing import List, Optional, Set

from dace import SDFG, dtypes, subsets, symbolic
from dace.sdfg import SDFGState, nodes
from dace.sdfg import utils as sdutil
from dace.sdfg.state import LoopRegion
from dace.transformation import transformation


def _resolve_outer_symbol_type(sym_name: str, sdfg: SDFG, default=None):
    """Walk up the SDFG nesting + the parent CFR tree to locate the
    type ``sym_name`` was declared with. Falls back to ``default`` (or
    ``dace.int64`` if ``default`` is ``None``)."""
    if default is None:
        default = dtypes.int64
    cur = sdfg
    while cur is not None:
        if sym_name in cur.symbols:
            return cur.symbols[sym_name]
        cur = getattr(cur, 'parent_sdfg', None)
    # Try LoopRegion loop-variables in the original SDFG (the loop var
    # carries the type via its enclosing SDFG's symbol table).
    for cfg in sdfg.all_control_flow_regions():
        if isinstance(cfg, LoopRegion) and cfg.loop_variable == sym_name:
            owner = getattr(cfg, 'sdfg', None)
            if owner is not None and sym_name in owner.symbols:
                return owner.symbols[sym_name]
    return default


def _full_subset(sdfg: SDFG, arr_name: str) -> subsets.Range:
    return subsets.Range.from_array(sdfg.arrays[arr_name])


def _has_narrowed_edge(state: SDFGState, nsdfg: nodes.NestedSDFG, sdfg: SDFG) -> bool:
    for edge in (*state.in_edges(nsdfg), *state.out_edges(nsdfg)):
        if edge.data is None or edge.data.data is None:
            continue
        if edge.data.data not in sdfg.arrays:
            continue
        if edge.data.subset != _full_subset(sdfg, edge.data.data):
            return True
    return False


def _connector_for(state: SDFGState, nsdfg: nodes.NestedSDFG, edge) -> Optional[str]:
    if edge in state.in_edges(nsdfg):
        return edge.dst_conn
    return edge.src_conn


def _inner_descriptor_collapses_dim(outer_shape, inner_shape) -> bool:
    """Return True if widening the inner shape to ``outer_shape`` would
    change the inner descriptor's rank (i.e. the inner shape lost an
    axis when the outer subset had a size-1 dimension)."""
    return len(outer_shape) != len(inner_shape)


def _offset_inner_memlets(nsdfg: SDFG, inner_arr_name: str, offsets: List) -> None:
    """For every memlet inside ``nsdfg`` whose ``data`` field is
    ``inner_arr_name``, shift its subset coordinates by ``offsets``
    (one entry per axis). Walks every state in the NSDFG, including
    nested control-flow regions."""
    from dace.subsets import Range

    if all(o == 0 for o in offsets):
        return  # nothing to offset

    for state in nsdfg.all_states():
        for edge in state.edges():
            mm = edge.data
            if mm is None or mm.data != inner_arr_name:
                continue
            for sub in (mm.subset, mm.other_subset):
                if sub is None:
                    continue
                # Only shift Range subsets here; the rare non-Range subsets
                # (e.g. Indices) are extended by ``offset`` below.
                if isinstance(sub, Range):
                    if len(sub.ranges) != len(offsets):
                        continue  # rank mismatch -- caller should have refused
                    new_ranges = []
                    for (lo, hi, stp), off in zip(sub.ranges, offsets):
                        new_ranges.append((lo + off, hi + off, stp))
                    sub.ranges = new_ranges


def _rename_nsdfg_connector(state: SDFGState, nsdfg_node: nodes.NestedSDFG, old: str, new: str) -> None:
    """Rename ``old`` to ``new`` on both the ``NestedSDFG`` connector
    sets and on every edge that targets that connector. Used after
    inner-array rename so the outer-side connector name still matches
    the inner array name."""
    if old in nsdfg_node.in_connectors:
        nsdfg_node.in_connectors[new] = nsdfg_node.in_connectors.pop(old)
    if old in nsdfg_node.out_connectors:
        nsdfg_node.out_connectors[new] = nsdfg_node.out_connectors.pop(old)
    for edge in state.in_edges(nsdfg_node):
        if edge.dst_conn == old:
            edge.dst_conn = new
    for edge in state.out_edges(nsdfg_node):
        if edge.src_conn == old:
            edge.src_conn = new


class ExpandNestedSDFGInputs(transformation.SingleStateTransformation):
    """Pre-processor for :class:`InlineMultistateSDFG`: widen narrowed
    NSDFG in/out memlets to full-array subsets, reshape inner descriptors
    to match, and offset inner memlets accordingly.

    Handles both top-level NSDFGs and NSDFGs nested inside a Map scope.
    For the in-map case the caller is expected to have first widened the
    parent map's IN/OUT memlets to full arrays (via
    ``propagate_full_array_subsets_through_map``) so the
    MapEntry-to-NSDFG connector already carries the full extent; the
    per-iteration tile offset is then captured from the original
    narrowed subset and threaded onto every inner memlet referencing
    that connector.
    """

    nested_sdfg = transformation.PatternNode(nodes.NestedSDFG)

    @classmethod
    def expressions(cls):
        return [sdutil.node_path_graph(cls.nested_sdfg)]

    @staticmethod
    def annotates_memlets():
        return True

    def can_be_applied(self, state: SDFGState, expr_index, sdfg: SDFG, permissive=False) -> bool:
        nsdfg_node = self.nested_sdfg
        if nsdfg_node.no_inline:
            return False
        return True

    def apply(self, state: SDFGState, sdfg: SDFG) -> None:
        nsdfg_node = self.nested_sdfg
        inner = nsdfg_node.sdfg

        # Symbols introduced by the offsets that need to be propagated
        # from the outer scope into the NSDFG (added to
        # ``symbol_mapping`` + the inner ``symbols`` table so the inner
        # memlet references validate).
        introduced_symbols: Set[str] = set()

        # Track which inner array names we have already widened. When a
        # connector name is used for BOTH an in-edge and an out-edge
        # (the same outer array is read AND written by the NSDFG;
        # ``MapToForLoop`` does this when the kernel mutates the array
        # in-place, e.g. ``A[i,j,k+1] = A[i,j,k] + A[i,j,k-1]``), the
        # inner array is shared. Without de-duplication we would offset
        # every inner memlet referencing that array TWICE, corrupting
        # numerics.
        processed_inner_arrays: Set[str] = set()

        for edge in (*state.in_edges(nsdfg_node), *state.out_edges(nsdfg_node)):
            if edge.data is None or edge.data.data is None:
                continue
            outer_arr_name = edge.data.data
            outer_arr = sdfg.arrays[outer_arr_name]
            conn = _connector_for(state, nsdfg_node, edge)
            if conn is None or conn not in inner.arrays:
                continue
            inner_arr = inner.arrays[conn]
            old_sub = edge.data.subset
            full_sub = _full_subset(sdfg, outer_arr_name)
            if old_sub == full_sub:
                processed_inner_arrays.add(conn)
                continue  # already full; nothing to do
            if conn in processed_inner_arrays:
                # The same inner array was already widened via another
                # edge (typically an in-edge whose connector also serves
                # as an out-edge). Only widen the EDGE subset; the inner
                # descriptor + inner memlets are already in their
                # widened, offset form.
                edge.data.subset = full_sub
                continue

            # Per-axis lower-bound offset = the per-iteration shift the
            # inliner needs to add to every inner memlet.
            offsets = [lo for (lo, _hi, _stp) in old_sub.ranges]

            # Collect any non-trivial symbols in those offsets so we can
            # propagate them through symbol_mapping below.
            for off in offsets:
                try:
                    free = symbolic.pystr_to_symbolic(str(off)).free_symbols
                except Exception:
                    free = set()
                for s in free:
                    introduced_symbols.add(str(s))

            # Reshape inner descriptor to match the outer one. Preserve
            # transient flag and storage class of the existing inner
            # descriptor (the outer descriptor's transient flag may
            # differ, e.g. the outer arg is non-transient).
            new_inner = copy.deepcopy(outer_arr)
            new_inner.transient = inner_arr.transient
            new_inner.storage = inner_arr.storage
            inner.arrays[conn] = new_inner

            # Widen the outer subset to the full array range.
            edge.data.subset = full_sub

            # Offset every inner memlet that references this connector.
            _offset_inner_memlets(inner, conn, offsets)
            processed_inner_arrays.add(conn)

            # Connector-name unification: rename the inner array (and
            # the corresponding NSDFG connector) to match the outer
            # array name so downstream inlining doesn't have to do the
            # rename. ``inner.replace_dict`` handles arrays + every
            # memlet reference + interstate-edge subst in one shot.
            # Skip if the outer name is already taken by another inner
            # array (rare; would require a separate disambiguation pass).
            if conn != outer_arr_name and outer_arr_name not in inner.arrays:
                inner.replace_dict({conn: outer_arr_name})
                _rename_nsdfg_connector(state, nsdfg_node, conn, outer_arr_name)
                processed_inner_arrays.discard(conn)
                processed_inner_arrays.add(outer_arr_name)

        # Propagate any offset symbols not already passed in via
        # symbol_mapping. Identity binding is the right default --
        # outer ``ii`` becomes inner ``ii``. The validation step will
        # complain on the next pass if the symbol isn't defined at the
        # outer scope where the NSDFG sits; that's a real bug in the
        # caller and we want it surfaced.
        #
        # Symbol-type preservation: when a symbol isn't already in the
        # inner ``symbols`` table we resolve its type by walking the
        # outer SDFG ancestry (parent_sdfg chain) + LoopRegion loop
        # variables. Defaulting to ``int64`` silently would mismatch a
        # caller that declared the symbol with another integer width.
        for sym_name in introduced_symbols:
            if sym_name in nsdfg_node.symbol_mapping:
                continue
            if sym_name in inner.arrays:
                continue  # an array name -- not a symbol, skip
            if sym_name in inner.symbols:
                # Symbol exists in inner but not mapped from outer -- bind it.
                nsdfg_node.symbol_mapping[sym_name] = symbolic.pystr_to_symbolic(sym_name)
                continue
            # Symbol is new to both inner and the mapping. Resolve its
            # type from the outer scope and copy it through.
            outer_type = _resolve_outer_symbol_type(sym_name, sdfg)
            inner.add_symbol(sym_name, outer_type)
            nsdfg_node.symbol_mapping[sym_name] = symbolic.pystr_to_symbolic(sym_name)
