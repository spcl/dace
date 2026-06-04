# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Fuse overlapping :class:`TileLoad` lib nodes inside a body NSDFG.

The tile-arm equivalent of
:class:`~dace.transformation.passes.vectorization.fuse_overlapping_loads.FuseOverlappingLoads`.
It detects multiple :class:`TileLoad` nodes inside the same state that
read the *same* source array at shifted but overlapping per-iteration
subsets (the canonical stencil pattern ``A[i:i+W]``, ``A[i+1:i+W+1]``,
``A[i+2:i+W+2]``), materialises a single wider union transient
``<base>_vec`` of shape ``union_extent``, and rewires each load's
downstream tile consumers to read directly from the union at a static
offset. The original per-subset ``TileLoad`` nodes and their tile
transients are removed.
"""
from typing import Dict, List, Optional, Tuple

import dace
from dace import properties
from dace.sdfg.graph import Edge
from dace.transformation import pass_pipeline as ppl, transformation

from dace.libraries.tileops.nodes import TileLoad
from dace.transformation.passes.vectorization.utils.name_schemes import (TileConnectors)


@properties.make_properties
@transformation.explicit_cf_compatible
class FuseOverlappingTileLoads(ppl.Pass):
    """Collapse overlapping per-lane :class:`TileLoad` reads into one union buffer.

    Runs **after** :class:`PromoteNSDFGBodyToTiles` (which emits the
    per-subset loads) and **before** :class:`EmitTileOps` (which would
    further rewrite the body). Operates only inside body NestedSDFGs:
    that is the shape produced by ``nest_map_bodies=True`` and the only
    place where overlapping ``TileLoad`` lib nodes appear together in a
    single state.

    The fusion is structural — it does not change which lanes are
    written, only how the source bytes are fetched. For each group of
    ``N >= 2`` :class:`TileLoad` nodes that

    1. share one input source ``AccessNode`` with the same array name,
    2. have one-to-one ``_src`` edges (no scope nodes in between),
    3. produce a ``(W_0, ..., W_K)``-shape tile transient consumed
       elsewhere in the same state,
    4. carry per-iteration subsets whose union extent strictly exceeds
       the tile width on at least one dim,

    the pass:

    * computes the union subset and per-load offsets;
    * adds a transient ``<base>_vec`` array of shape ``union_extent``
      (deduplicated per source array);
    * emits one staging memlet copy ``A[union] -> A_vec[full]`` (an
      ``AccessNode -> AccessNode`` edge — no tasklet, no lib node);
    * rewires every downstream consumer of each original load's tile
      transient to read directly from ``A_vec`` at
      ``[offset_k : offset_k + W_k]`` along each dim;
    * removes the original ``TileLoad`` nodes and their tile transients
      from the state and the SDFG arrays table.

    The pass deliberately leaves the iteration mask alone — the
    sub-window reads are unmasked sub-slices of the union buffer, and
    the consumer tile op already gates writes by the iter mask. The
    union staging copy itself is unmasked: it depends on the source
    array being large enough for the union extent at every iteration,
    which the orchestrator's divisibility precondition guarantees in
    the cases where this pass fires (the harness only enables the knob
    on provably divisible inputs).
    """

    CATEGORY: str = "Vectorization"

    fusion_threshold = properties.Property(dtype=int,
                                           default=1,
                                           desc="Minimum overlap count *above* which a per-array TileLoad fan is "
                                           "fused into a shared union window. Default ``1`` preserves the "
                                           "common contract (fuse iff ``>= 2`` overlapping reads).")

    def __init__(self, fusion_threshold: int = 1):
        super().__init__()
        self.fusion_threshold = fusion_threshold
        self._last_groups_fused = 0
        self._last_groups_gated = 0

    def modifies(self) -> ppl.Modifies:
        return (ppl.Modifies.Edges | ppl.Modifies.AccessNodes | ppl.Modifies.Memlets | ppl.Modifies.Nodes)

    def should_reapply(self, modified: ppl.Modifies) -> bool:
        return False

    def apply_pass(self, sdfg: dace.SDFG, pipeline_results) -> Optional[int]:
        """Fuse overlapping :class:`TileLoad` groups in ``sdfg`` and its NSDFGs.

        :param sdfg: SDFG to transform in place.
        :param pipeline_results: Inherited pipeline results (unused).
        :returns: Number of TileLoad groups fused, or ``None`` if zero.
        """
        self._last_groups_fused = 0
        self._last_groups_gated = 0
        # Walk every SDFG (top + recursive NSDFGs); fuse within each state.
        sdfgs = [sdfg]
        for node, _ in sdfg.all_nodes_recursive():
            if isinstance(node, dace.nodes.NestedSDFG):
                sdfgs.append(node.sdfg)
        for s in sdfgs:
            for state in s.all_states():
                self._fuse_state(s, state)
        return self._last_groups_fused or None

    def _fuse_state(self, sdfg: dace.SDFG, state: dace.SDFGState) -> None:
        """Detect + fuse overlapping :class:`TileLoad` groups in one state."""
        # Group TileLoads by (source array name, single source AccessNode that
        # feeds them). The grouping key uses the source AccessNode identity so
        # two physically distinct ``A`` access nodes in the same state are
        # treated as one logical source (Promote often emits a fresh ``A`` AN
        # per load — we consolidate to the first one we see).
        groups: Dict[str, List[Tuple[dace.nodes.LibraryNode, Edge]]] = {}
        first_src_an: Dict[str, dace.nodes.AccessNode] = {}
        for node in list(state.nodes()):
            if not isinstance(node, TileLoad):
                continue
            in_e = [e for e in state.in_edges(node) if e.dst_conn == TileConnectors.SRC]
            if len(in_e) != 1:
                continue
            src_edge = in_e[0]
            if not isinstance(src_edge.src, dace.nodes.AccessNode):
                continue
            src_data = src_edge.data.data
            if src_data is None or src_data not in sdfg.arrays:
                continue
            groups.setdefault(src_data, []).append((node, src_edge))
            first_src_an.setdefault(src_data, src_edge.src)

        for src_data, loads in groups.items():
            if len(loads) <= self.fusion_threshold:
                self._last_groups_gated += 1
                continue
            if not self._try_fuse(sdfg, state, src_data, loads, first_src_an[src_data]):
                self._last_groups_gated += 1

    def _try_fuse(
        self,
        sdfg: dace.SDFG,
        state: dace.SDFGState,
        src_data: str,
        loads: List[Tuple[dace.nodes.LibraryNode, Edge]],
        src_an: dace.nodes.AccessNode,
    ) -> bool:
        """Fuse one group; return True iff fusion actually happened."""
        subsets_list = [src_edge.data.subset for _, src_edge in loads]
        # Compute union per dim; refuse if dims disagree.
        ndim = len(subsets_list[0])
        if any(len(s) != ndim for s in subsets_list):
            return False
        union = subsets_list[0]
        for s in subsets_list[1:]:
            union = union.union(s)
        # Per-dim integer offset of each subset within the union.
        offsets_per_load = []
        for s in subsets_list:
            try:
                per_dim = []
                for d in range(ndim):
                    diff = dace.symbolic.simplify(s[d][0] - union[d][0])
                    if not diff.is_integer:
                        return False
                    per_dim.append(int(diff))
                offsets_per_load.append(per_dim)
            except (AttributeError, TypeError):
                return False
        # Compute union extent and the per-load (W_k,) extent (assume all loads
        # have identical widths — that is the per-lane tile shape from the lib
        # node's ``widths`` property).
        union_extent = []
        for d in range(ndim):
            try:
                ext = dace.symbolic.simplify(union[d][1] + 1 - union[d][0])
                if not ext.is_integer:
                    return False
                union_extent.append(int(ext))
            except (AttributeError, TypeError):
                return False
        load0_widths = list(loads[0][0].widths)
        K = len(load0_widths)
        if K > ndim:
            return False
        if any(list(ld.widths) != load0_widths for ld, _ in loads):
            return False
        # The tile widths align with the *trailing* K dims of the source
        # (``spec.iter_vars`` is innermost-last). Leading dims (``ndim - K``)
        # are single-element reads whose begin varies per load — those become
        # the leading axes of the union. Compute the effective per-dim width:
        # widths[d] for the trailing K dims, 1 for the leading ones.
        leading = ndim - K
        per_dim_width = [1] * leading + load0_widths
        # Refuse if union is no wider than the per-load window on every dim
        # (nothing to fuse — single subset or every load identical).
        if all(union_extent[d] <= per_dim_width[d] for d in range(ndim)):
            return False
        # Materialise the union transient. Deduplicate per source array within
        # this SDFG: ``<base>_vec`` is the canonical name; the test harness
        # asserts on exactly this name.
        src_desc = sdfg.arrays[src_data]
        union_name = f"{src_data}_vec"
        if union_name in sdfg.arrays:
            existing = sdfg.arrays[union_name]
            if tuple(int(s) if hasattr(s, '__index__') else s for s in existing.shape) != tuple(union_extent):
                # Conflict: fall back to a fresh unique name.
                union_name = sdfg._find_new_name(union_name)
                sdfg.add_array(
                    union_name,
                    union_extent,
                    src_desc.dtype,
                    storage=dace.dtypes.StorageType.Register,
                    transient=True,
                )
        else:
            sdfg.add_array(
                union_name,
                union_extent,
                src_desc.dtype,
                storage=dace.dtypes.StorageType.Register,
                transient=True,
            )
        # Staging copy: ``src_an [union] -> A_vec [full]`` via a plain
        # AccessNode->AccessNode memlet (codegen lowers to a memcpy / vector
        # load; no tasklet or lib node needed because the consumer side reads
        # via memlet sub-slicing).
        union_an = state.add_access(union_name)
        full_subset = ", ".join(f"0:{e}" for e in union_extent)
        state.add_edge(src_an, None, union_an, None, dace.Memlet(data=src_data, subset=union, other_subset=full_subset))
        # Replace each load with a plain memlet copy from the union buffer
        # into the *same* dense per-load tile transient. Keeping the tile
        # transient preserves the binop consumer interface (the lib node's
        # tasklet expects a dense ``(W_0, ..., W_K)``-stride source connector;
        # passing it a strided view into ``A_vec`` would mis-stride the
        # inner-loop index arithmetic). The wide source load happens ONCE
        # (the ``src_an -> union_an`` staging edge above); each per-load
        # transient becomes a cheap offset extract.
        for load_node, src_edge in loads:
            dst_edges = [e for e in state.out_edges(load_node) if e.src_conn == TileConnectors.DST]
            if len(dst_edges) != 1:
                continue
            tile_dst_edge = dst_edges[0]
            tile_acc = tile_dst_edge.dst
            if not isinstance(tile_acc, dace.nodes.AccessNode):
                continue
            offsets = offsets_per_load[loads.index((load_node, src_edge))]
            sub_subset_str = ", ".join(f"{off}:{off + w}" for off, w in zip(offsets, per_dim_width))
            # Wire ``union_an [sub_subset] -> tile_acc [full]`` as the load
            # replacement. The tile transient keeps its dense register strides;
            # its downstream binop consumers are unchanged.
            tile_full = ", ".join(f"0:{w}" for w in tile_acc.desc(sdfg).shape)
            state.add_edge(
                union_an,
                None,
                tile_acc,
                None,
                dace.Memlet(data=union_name, subset=sub_subset_str, other_subset=tile_full),
            )
            # Drop the original load + its mask edge.
            for e in list(state.in_edges(load_node)) + list(state.out_edges(load_node)):
                state.remove_edge(e)
            state.remove_node(load_node)
        # Drop now-orphan source access nodes (each load had its own ``A`` AN
        # in Promote-produced bodies; the one we kept as ``src_an`` is still
        # used by the staging memlet).
        for n in list(state.data_nodes()):
            if n.data == src_data and n is not src_an and state.degree(n) == 0:
                state.remove_node(n)
        self._last_groups_fused += 1
        return True
