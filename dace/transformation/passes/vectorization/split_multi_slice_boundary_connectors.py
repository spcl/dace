# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Split a body-NSDFG boundary connector with a non-tiled multi-slice dim
into per-slice connectors.

Runs BEFORE :class:`PromoteNSDFGBodyToTiles`. When the propagated outer
memlet of an inner-map body NestedSDFG carries an extent > 1 in a dim
that is NOT a tile iter-var (the heat3d-style stencil pattern where only
the inner ``j, k`` are tiled and the outer ``i`` carries adjacent point
reads at ``i-1, i, i+1``), Promote's :meth:`_widen_boundary_connectors`
refuses with ``non-tiled dim of extent > 1 — a multi-slice access``.
This pass pre-materialises the slices: for each distinct point along the
multi-slice dim that the inner body reads, it emits a fresh connector
of extent 1 in that dim, wires a per-slice outer memlet from the source
access node, and redirects the inner reads at that point to the new
connector. The original union connector is removed; Promote then sees
clean per-slice connectors with extent-1 non-tiled dims and processes
them via the existing degenerate-dim path.
"""
from typing import Dict, List, Optional, Set, Tuple

import dace
from dace import properties, subsets
from dace.sdfg import nodes
from dace.transformation import pass_pipeline as ppl, transformation
from dace.sdfg.nodes import MapEntry
from dace.transformation.passes.vectorization.utils.symbolic_polymorphism import free_symbol_names


@properties.make_properties
@transformation.explicit_cf_compatible
class SplitMultiSliceBoundaryConnectors(ppl.Pass):
    """Pre-emit pass: split non-tiled multi-slice body-NSDFG connectors into
    per-slice connectors. See module docstring for the contract."""

    CATEGORY: str = "Vectorization"

    widths = properties.ListProperty(element_type=int, default=[8], desc="Tile widths.")

    def __init__(self, widths: Tuple[int, ...]):
        super().__init__()
        self.widths = list(widths)

    def modifies(self) -> ppl.Modifies:
        return (ppl.Modifies.Edges | ppl.Modifies.AccessNodes | ppl.Modifies.Memlets | ppl.Modifies.Nodes)

    def should_reapply(self, modified: ppl.Modifies) -> bool:
        return False

    def apply_pass(self, sdfg: dace.SDFG, pipeline_results) -> Optional[int]:
        """Split per-NSDFG; return count of connector splits, or ``None``.

        :param sdfg: Top-level SDFG to walk.
        :param pipeline_results: Reads ``"MarkTileDims"`` when present so the
            tile iter-vars match the spec the orchestrator decided on.
        :returns: Number of connector splits across all body NSDFGs.
        """
        specs = None
        if pipeline_results and "MarkTileDims" in pipeline_results:
            specs = pipeline_results["MarkTileDims"]
        count = 0
        # Collect (parent_state, map_entry, nsdfg_node, spec) tuples; mutating
        # while iterating all_nodes_recursive is unsafe.
        targets: List[Tuple[dace.SDFGState, MapEntry, nodes.NestedSDFG, object]] = []
        for n, g in list(sdfg.all_nodes_recursive()):
            if not isinstance(n, MapEntry) or not isinstance(g, dace.SDFGState):
                continue
            spec = specs.get(n) if specs is not None else None
            if spec is None:
                continue
            # Find a single body NSDFG inside this map scope.
            scope_nodes = g.all_nodes_between(n, g.exit_node(n))
            ns_nodes = [x for x in scope_nodes if isinstance(x, nodes.NestedSDFG)]
            if len(ns_nodes) != 1:
                continue
            targets.append((g, n, ns_nodes[0], spec))
        for state, map_entry, nsdfg_node, spec in targets:
            count += self._split_nsdfg(state, map_entry, nsdfg_node, spec)
        return count or None

    def _split_nsdfg(self, parent_state: dace.SDFGState, map_entry: MapEntry,
                     nsdfg_node: nodes.NestedSDFG, spec) -> int:
        """Split every multi-slice in-connector of ``nsdfg_node``; return count."""
        tile_vars: Set[str] = set(spec.iter_vars)
        count = 0
        for conn in list(nsdfg_node.in_connectors):
            if self._try_split_connector(parent_state, map_entry, nsdfg_node, conn, tile_vars):
                count += 1
        return count

    def _try_split_connector(self, parent_state: dace.SDFGState, map_entry: MapEntry,
                             nsdfg_node: nodes.NestedSDFG, conn: str, tile_vars: Set[str]) -> bool:
        """Detect + split one connector; return True iff a split happened."""
        outer_edges = [e for e in parent_state.in_edges(nsdfg_node) if e.dst_conn == conn]
        if len(outer_edges) != 1:
            return False
        oe = outer_edges[0]
        if oe.data is None or oe.data.subset is None:
            return False
        subset = oe.data.subset
        # Find the first non-tiled dim with extent > 1.
        multi_slice_dim: Optional[int] = None
        for d, (b, e, _s) in enumerate(subset):
            # Tiled dim: the begin contains a tile iter-var.
            b_syms = free_symbol_names(b)
            if b_syms & tile_vars:
                continue
            try:
                ext_expr = dace.symbolic.simplify(e - b)
                ext = int(ext_expr) + 1
            except (TypeError, ValueError):
                continue
            if ext > 1:
                multi_slice_dim = d
                break
        if multi_slice_dim is None:
            return False
        b_ms, e_ms, _ = subset[multi_slice_dim]
        try:
            slice_count = int(dace.symbolic.simplify(e_ms - b_ms)) + 1
        except (TypeError, ValueError):
            return False
        # Find the distinct integer offsets along the multi-slice dim used by
        # the inner reads (e.g., ``{0, 1, 2}`` for a 3-point stencil at i-1/i/i+1
        # after Promote's outer-subset rewrite normalises the origin to ``b_ms``).
        offsets = self._collect_inner_slice_offsets(nsdfg_node, conn, multi_slice_dim, slice_count)
        if not offsets:
            return False
        inner = nsdfg_node.sdfg
        conn_arr = inner.arrays.get(conn)
        if conn_arr is None or not isinstance(conn_arr, dace.data.Array):
            return False
        # Build per-slice connectors. New shape FULLY DROPS the multi-slice
        # dim (going from N dims to N-1) so Promote's existing degenerate
        # path handles them without needing the leading-singleton extension.
        # Outer memlet: 3D source ``A[i+off:i+off+1, j:j+3, k:k+3]`` reads
        # into the 2D per-slice connector via DaCe's standard outer-subset-
        # to-lower-rank-connector view.
        slice_names: Dict[int, str] = {}
        dtype = conn_arr.dtype
        orig_strides = list(conn_arr.strides)
        new_strides = [s for i, s in enumerate(orig_strides) if i != multi_slice_dim]
        for off in sorted(offsets):
            new_name = self._fresh_array_name(inner, f"{conn}_slice_{off}")
            slice_names[off] = new_name
            new_shape = [s for i, s in enumerate(conn_arr.shape) if i != multi_slice_dim]
            inner.add_array(
                new_name,
                new_shape,
                dtype,
                strides=new_strides,
                storage=conn_arr.storage,
                transient=False,
            )
            nsdfg_node.add_in_connector(new_name)
            # Outer wiring: same source AccessNode + MapEntry pair as ``oe``;
            # per-slice memlet picks the single slab ``b_ms + off``.
            new_subset = self._per_slice_subset(subset, multi_slice_dim, off)
            data_name = oe.data.data
            new_memlet = dace.Memlet(data=data_name, subset=new_subset)
            parent_state.add_edge(oe.src, oe.src_conn, nsdfg_node, new_name, new_memlet)
        # Walk inner states; rewire each access of ``conn`` to the
        # appropriate per-slice connector.
        self._rewire_inner_reads(nsdfg_node, conn, multi_slice_dim, slice_names)
        # Drop the old union connector + outer edge.
        parent_state.remove_edge(oe)
        nsdfg_node.remove_in_connector(conn)
        # Remove the inner connector array if no inner read references it.
        if conn in inner.arrays and not self._inner_uses_array(inner, conn):
            inner.remove_data(conn, validate=False)
        return True

    def _collect_inner_slice_offsets(self, nsdfg_node: nodes.NestedSDFG, conn: str, dim: int,
                                     slice_count: int) -> Set[int]:
        """Walk inner states; collect the distinct integer offsets the inner
        reads use along ``dim`` of ``conn``. Returns the empty set on any
        non-integer / out-of-range offset (the connector is not split)."""
        inner = nsdfg_node.sdfg
        offsets: Set[int] = set()
        for st in inner.all_states():
            for e in st.edges():
                if e.data is None or e.data.data != conn:
                    continue
                if e.data.subset is None or len(e.data.subset) <= dim:
                    return set()
                b, e_, _ = e.data.subset[dim]
                try:
                    diff = dace.symbolic.simplify(b - e_)
                    if diff != 0:
                        return set()  # not a point — multi-element read along this dim
                    off = int(dace.symbolic.simplify(b))
                except (TypeError, ValueError):
                    return set()
                if off < 0 or off >= slice_count:
                    return set()
                offsets.add(off)
        return offsets

    def _per_slice_subset(self, subset: subsets.Range, dim: int, off: int) -> subsets.Range:
        """Outer per-slice subset: shrink ``dim`` to ``[b_ms + off : b_ms + off]``."""
        ranges = list(subset.ranges)
        b_ms, _e_ms, _s_ms = ranges[dim]
        ranges[dim] = (b_ms + off, b_ms + off, 1)
        return subsets.Range(ranges)

    def _rewire_inner_reads(self, nsdfg_node: nodes.NestedSDFG, conn: str, dim: int,
                            slice_names: Dict[int, str]) -> None:
        """Redirect each inner read of ``conn`` at point ``off`` along ``dim``
        to ``slice_names[off]`` with the multi-slice dim collapsed to ``[0:1]``.
        AccessNodes for ``conn`` are renamed in place; their out-edge memlets
        are rewritten.
        """
        inner = nsdfg_node.sdfg
        for st in inner.all_states():
            # Group access nodes of ``conn`` by the offset their out-edges read.
            for an in list(st.data_nodes()):
                if an.data != conn:
                    continue
                # All out-edges of this access should read the same offset (the
                # collector enforced this). If multiple offsets share one access
                # node, fall back to per-edge rewrite below.
                offsets_per_edge: List[Tuple[dace.sdfg.graph.Edge, int]] = []
                for e in list(st.out_edges(an)):
                    if e.data is None or e.data.subset is None:
                        continue
                    b, _, _ = e.data.subset[dim]
                    try:
                        off = int(dace.symbolic.simplify(b))
                    except (TypeError, ValueError):
                        off = -1
                    if off in slice_names:
                        offsets_per_edge.append((e, off))
                if not offsets_per_edge:
                    continue
                # If every out-edge has the same offset, rename the access node;
                # otherwise add fresh per-slice access nodes per edge.
                unique_offs = {off for _, off in offsets_per_edge}
                if len(unique_offs) == 1:
                    only_off = next(iter(unique_offs))
                    new_name = slice_names[only_off]
                    an.data = new_name
                    for e, _ in offsets_per_edge:
                        new_subset = self._inner_per_slice_subset(e.data.subset, dim)
                        e.data = dace.Memlet(data=new_name, subset=new_subset)
                else:
                    for e, off in offsets_per_edge:
                        new_name = slice_names[off]
                        new_an = st.add_access(new_name)
                        new_subset = self._inner_per_slice_subset(e.data.subset, dim)
                        st.add_edge(new_an, None, e.dst, e.dst_conn,
                                    dace.Memlet(data=new_name, subset=new_subset))
                        st.remove_edge(e)
                    if st.degree(an) == 0:
                        st.remove_node(an)

    def _inner_per_slice_subset(self, subset: subsets.Range, dim: int) -> subsets.Range:
        """Inner per-slice subset: DROP ``dim`` entirely (the new connector
        has rank N-1 since the multi-slice dim is squeezed out)."""
        ranges = [r for i, r in enumerate(subset.ranges) if i != dim]
        return subsets.Range(ranges)

    def _inner_uses_array(self, inner: dace.SDFG, name: str) -> bool:
        """True iff any inner state still references ``name`` as data."""
        for st in inner.all_states():
            for n in st.data_nodes():
                if n.data == name:
                    return True
            for e in st.edges():
                if e.data is not None and e.data.data == name:
                    return True
        return False

    def _fresh_array_name(self, sdfg: dace.SDFG, base: str) -> str:
        name = base
        idx = 0
        while name in sdfg.arrays:
            idx += 1
            name = f"{base}_{idx}"
        return name
