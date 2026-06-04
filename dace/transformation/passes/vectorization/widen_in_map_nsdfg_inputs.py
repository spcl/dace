# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Widen the memlet path between a parent-map source/sink access node and
an in-map :class:`~dace.sdfg.nodes.NestedSDFG` to the full outer-array
extent, restore inner descriptors to the full source rank, and
uncollapse every inner memlet referencing the connector so the per-iter
offset (and per-tile W-range on tile-var-bearing dims) lives on each
inner memlet.

Algorithm
---------

For each in/out edge of the NSDFG (symmetric for out-edges):

1. Walk :meth:`SDFGState.memlet_path` up (down) through every nested
   :class:`~dace.sdfg.nodes.MapEntry` (resp. ``MapExit``) to the source
   (sink) :class:`~dace.sdfg.nodes.AccessNode`.
2. Snapshot the original narrowed outer subset (the edge directly
   touching the NSDFG). For each source dim it gives one of:

   * a single-point ``[expr : expr]`` -- the NSDFG collapsed this dim
     in the inner descriptor; ``expr`` is the per-iter offset.
   * a non-trivial range ``[lo : hi]`` -- the inner descriptor retains
     this dim; ``lo`` is the per-iter offset for this dim.

3. Widen every memlet on the path to the full outer-array extent and
   replace the inner-connector descriptor with a copy of the source
   descriptor.
4. Build a source-dim -> inner-dim map: the kth non-point dim of the
   narrowed subset maps to inner-dim ``k`` of the original inner
   descriptor (the inner descriptor's rank equals the count of non-point
   dims in the narrowed subset; the convention follows DaCe's
   automatic size-1-axis collapse when descriptor shapes are derived
   from subsets).
5. Rewrite every inner memlet referencing the connector to a
   source-rank subset:

   * for each point dim of the narrowed subset, emit ``[offset[d] :
     offset[d] + W - 1]`` if ``offset[d]`` references a tile iter-var
     (a per-tile range), else ``[offset[d] : offset[d]]``;
   * for each non-point dim of the narrowed subset, emit the inner
     memlet's existing subset on the corresponding inner-dim, shifted by
     ``offset[d]``; if that shifted range collapses to a tile-var point
     it is also grown to a per-tile range.

6. Add any newly-referenced free symbols (typically map iter-vars) to
   ``nsdfg_node.symbol_mapping`` + the inner ``symbols`` table so the
   rewritten memlets validate.

The rewrite is unconditional: every connector that has both an
:class:`AccessNode` endpoint and a tile-var reference somewhere on the
narrowed subset or in the inner memlets is widened. The downstream
classify / promote pipeline then sees uniformly-full-array connectors
regardless of source access pattern.
"""
import copy
from typing import Dict, List, Optional, Set, Tuple

from dace import SDFG, symbolic
from dace.sdfg import SDFGState, nodes
from dace.subsets import Range
from dace.transformation.interstate.expand_nested_sdfg_inputs import (
    _full_subset,
    _resolve_outer_symbol_type,
)


def _path_endpoint(state: SDFGState, edge, *, walk_up: bool) -> Optional[nodes.AccessNode]:
    """Return the :class:`AccessNode` at the outer end of the memlet
    path containing ``edge`` (``walk_up=True`` -> source / path head;
    ``walk_up=False`` -> sink / path tail)."""
    try:
        path = state.memlet_path(edge)
    except Exception:
        return None
    if not path:
        return None
    node = path[0].src if walk_up else path[-1].dst
    return node if isinstance(node, nodes.AccessNode) else None


def _capture_introduced_symbols(exprs) -> Set[str]:
    """Return the set of free symbols referenced by any expression in
    ``exprs`` (best-effort; expressions that fail to sympify contribute
    nothing)."""
    introduced: Set[str] = set()
    for e in exprs:
        try:
            sym = symbolic.pystr_to_symbolic(str(e))
        except Exception:
            continue
        for s in sym.free_symbols:
            introduced.add(str(s))
    return introduced


def _propagate_symbols(nsdfg_node: nodes.NestedSDFG, inner: SDFG, sdfg: SDFG, introduced: Set[str]) -> None:
    """Add any newly-referenced symbols to the NSDFG's ``symbol_mapping``
    + the inner ``symbols`` table so the rewritten inner memlets
    validate. Resolves each symbol's dtype from the outer SDFG ancestry."""
    for sym_name in introduced:
        if sym_name in nsdfg_node.symbol_mapping:
            continue
        if sym_name in inner.arrays:
            continue
        if sym_name in inner.symbols:
            nsdfg_node.symbol_mapping[sym_name] = symbolic.pystr_to_symbolic(sym_name)
            continue
        outer_type = _resolve_outer_symbol_type(sym_name, sdfg)
        inner.add_symbol(sym_name, outer_type)
        nsdfg_node.symbol_mapping[sym_name] = symbolic.pystr_to_symbolic(sym_name)


def _tilevar_in_expr(expr, tile_widths: Dict[str, int]) -> Optional[str]:
    """Return a tile iter-var name appearing as a free symbol in
    ``expr``, or ``None`` if none does."""
    try:
        sym = symbolic.pystr_to_symbolic(str(expr))
    except Exception:
        return None
    free = {str(s) for s in sym.free_symbols}
    for name in tile_widths:
        if name in free:
            return name
    return None


def _is_point(lo, hi) -> bool:
    """Whether ``[lo : hi]`` is a single-element range (``hi == lo``)."""
    try:
        return symbolic.simplify(hi - lo) == 0
    except Exception:
        return False


def _build_outer_to_inner_dim_map(narrowed_ranges, inner_rank: int) -> List[Optional[int]]:
    """Map each source-dim of the narrowed outer subset to the
    corresponding inner-dim, ``None`` for source dims the inner
    descriptor collapsed.

    Two layouts are recognised:

    * Full-rank inner: ``len(narrowed) == inner_rank`` -- identity map
      (no collapse).
    * Collapsed inner: ``inner_rank == count(non-point dims)`` -- the
      kth non-point source dim maps to inner-dim k; point dims map to
      ``None``.

    Any other shape returns ``None`` for every dim (the inner subset
    contributes nothing; offsets supply all values).
    """
    n_src = len(narrowed_ranges)
    if inner_rank == n_src:
        return list(range(n_src))
    point_flags = [_is_point(lo, hi) for (lo, hi, _stp) in narrowed_ranges]
    n_ranged = sum(1 for p in point_flags if not p)
    mapping: List[Optional[int]] = [None] * n_src
    if inner_rank == n_ranged:
        inner_d = 0
        for d, is_pt in enumerate(point_flags):
            if not is_pt:
                mapping[d] = inner_d
                inner_d += 1
        return mapping
    return [None] * n_src


def _rewrite_inner_memlet_subset(sub: Range, narrowed_ranges, dim_map: List[Optional[int]],
                                 tile_widths: Dict[str, int]) -> Optional[Range]:
    """Rebuild a source-rank :class:`Range` for one inner memlet:

    * For each source dim ``d`` with ``dim_map[d] is None`` (collapsed
      in the inner descriptor): emit ``[offset[d] : offset[d] + W - 1]``
      if ``offset[d]`` references a tile iter-var, else ``[offset[d] :
      offset[d]]``.
    * For each source dim ``d`` with ``dim_map[d] == inner_d``: take the
      inner subset's range on dim ``inner_d`` and shift it by
      ``offset[d]``; if the shifted range collapses to a tile-var point
      it is grown to a per-tile range too.

    Returns the new :class:`Range`, or ``None`` if rewriting is not
    well-defined for this subset (e.g. inner subset rank mismatches the
    dim-map count of non-``None`` entries)."""
    inner_dims_needed = sum(1 for m in dim_map if m is not None)
    if len(sub.ranges) != inner_dims_needed:
        return None

    new_ranges: List[Tuple] = []
    for d, m in enumerate(dim_map):
        offset = narrowed_ranges[d][0]
        if m is None:
            var = _tilevar_in_expr(offset, tile_widths)
            if var is not None:
                new_ranges.append((offset, offset + tile_widths[var] - 1, 1))
            else:
                new_ranges.append((offset, offset, 1))
        else:
            (lo_i, hi_i, stp_i) = sub.ranges[m]
            new_lo = lo_i + offset
            new_hi = hi_i + offset
            if _is_point(new_lo, new_hi):
                var = _tilevar_in_expr(new_lo, tile_widths)
                if var is not None:
                    new_hi = new_lo + tile_widths[var] - 1
            new_ranges.append((new_lo, new_hi, stp_i))
    return Range(new_ranges)


def _process_connector(sdfg: SDFG, state: SDFGState, nsdfg_node: nodes.NestedSDFG, edge, *, walk_up: bool,
                       processed: Set[str], tile_widths: Dict[str, int]) -> Set[str]:
    """Widen one in/out connector of the NSDFG: walk the memlet path to
    the source/sink AccessNode, widen every edge to the full source
    subset, restore the inner descriptor to the source shape, and
    rewrite every inner memlet referencing the connector.

    Returns the set of free symbols the original narrowed subset
    referenced (caller propagates them into ``nsdfg_node.symbol_mapping``)."""
    if edge.data is None or edge.data.data is None:
        return set()
    conn = edge.dst_conn if walk_up else edge.src_conn
    if conn is None:
        return set()
    inner = nsdfg_node.sdfg
    if conn not in inner.arrays:
        return set()

    endpoint = _path_endpoint(state, edge, walk_up=walk_up)
    if endpoint is None:
        return set()
    source_data = endpoint.data
    if source_data not in sdfg.arrays:
        return set()
    source_arr = sdfg.arrays[source_data]

    # Snapshot the narrowed-side subset BEFORE widening; this carries
    # the per-iter offset for every source dim.
    narrowed_ranges = list(edge.data.subset.ranges)
    if len(narrowed_ranges) != len(source_arr.shape):
        # Defensive: rank of the narrowed outer subset must match source
        # rank. (DaCe builds the narrowed memlet against the source
        # descriptor so this normally holds; a violation usually means
        # an earlier pass left the memlet in an inconsistent state.)
        return set()

    if conn in processed:
        # Already processed via the in/out twin; only widen this edge's
        # subset along the path.
        full_sub = _full_subset(sdfg, source_data)
        for e in state.memlet_path(edge):
            if e.data is None:
                continue
            e.data.data = source_data
            e.data.subset = copy.deepcopy(full_sub)
        return set()

    # Build the source-dim -> inner-dim correspondence BEFORE we replace
    # the inner descriptor (which would otherwise lose the collapsed-dim
    # information).
    inner_arr = inner.arrays[conn]
    dim_map = _build_outer_to_inner_dim_map(narrowed_ranges, len(inner_arr.shape))

    introduced = _capture_introduced_symbols(lo for (lo, _hi, _stp) in narrowed_ranges)

    # Widen every edge on the path to the full source extent.
    full_sub = _full_subset(sdfg, source_data)
    for e in state.memlet_path(edge):
        if e.data is None:
            continue
        e.data.data = source_data
        e.data.subset = copy.deepcopy(full_sub)

    # Replace the inner connector descriptor with a copy of the source
    # descriptor (preserving the inner descriptor's transient/storage
    # attributes -- the outer descriptor's transient flag may differ).
    new_inner = copy.deepcopy(source_arr)
    new_inner.transient = inner_arr.transient
    new_inner.storage = inner_arr.storage
    inner.arrays[conn] = new_inner

    # Rewrite every inner memlet referencing ``conn`` to a source-rank
    # subset using the dim map + per-iter offsets + per-tile expansion.
    for istate in inner.all_states():
        for ie in istate.edges():
            mm = ie.data
            if mm is None or mm.data != conn:
                continue
            for attr in ("subset", "other_subset"):
                sub = getattr(mm, attr)
                if sub is None or not isinstance(sub, Range):
                    continue
                rebuilt = _rewrite_inner_memlet_subset(sub, narrowed_ranges, dim_map, tile_widths)
                if rebuilt is not None:
                    setattr(mm, attr, rebuilt)

    processed.add(conn)
    return introduced


def widen_in_map_nsdfg_inputs(sdfg: SDFG, state: SDFGState, nsdfg_node: nodes.NestedSDFG,
                              tile_widths: Dict[str, int]) -> bool:
    """Widen every in/out memlet path between ``nsdfg_node`` and its
    outermost source/sink :class:`AccessNode` to the full outer-array
    extent. Restores the inner connector descriptors to the full source
    rank and rewrites every inner memlet referencing the connector to
    a source-rank subset using the per-iter offsets captured from the
    original narrowed path. Inner-memlet dims whose offset references
    a tile iter-var (any key of ``tile_widths``) are grown from a point
    to a per-tile W-range.

    :param sdfg: The parent SDFG that owns ``state``.
    :param state: The state containing ``nsdfg_node``.
    :param nsdfg_node: The NestedSDFG node to widen.
    :param tile_widths: Mapping ``iter_var_name -> tile_width`` from
        :class:`TileDimSpec`. Used to decide which dims become per-tile
        ranges in the inner memlet rewrite.

    :returns: ``True`` if any connector was rewritten."""
    if state.entry_node(nsdfg_node) is None:
        # Top-level NSDFG -- :class:`ExpandNestedSDFGInputs` handles it.
        return False

    inner = nsdfg_node.sdfg
    processed: Set[str] = set()
    introduced: Set[str] = set()
    changed = False

    for edge in list(state.in_edges(nsdfg_node)):
        before = len(processed)
        introduced.update(_process_connector(sdfg, state, nsdfg_node, edge, walk_up=True,
                                             processed=processed, tile_widths=tile_widths))
        if len(processed) > before:
            changed = True

    for edge in list(state.out_edges(nsdfg_node)):
        before = len(processed)
        introduced.update(_process_connector(sdfg, state, nsdfg_node, edge, walk_up=False,
                                             processed=processed, tile_widths=tile_widths))
        if len(processed) > before:
            changed = True

    if introduced:
        _propagate_symbols(nsdfg_node, inner, sdfg, introduced)

    return changed
