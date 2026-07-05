# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Splice a ``TileReduce`` to fold a widened reduction accumulator tile to a scalar.

The ``reduce_at_output`` path keeps a nested reduction in place through tiling: the
frontend/``NormalizeNestedReduction`` shape is a per-lane *seeded addend* transient
(``_nnr_priv``) accumulated inside the innermost body, copied back to the body's
scalar output, which then exits via the ``AccessNode ─[wcr]→ MapExit`` reduction
chain (the "scalar-out" contract). When the walker widens that body, ``_nnr_priv``
becomes a tile ``[0:W]`` of ``W`` per-lane partials while the copyback still writes
the scalar output -- a ``tile → scalar`` copy that is invalid on its own.

This pass replaces that copy with a ``TileReduce(op)`` that folds the tile to the
scalar. The fold lives *inside* the body NSDFG (where the copyback is), matching
the "``nsdfg → tasklet → AccessNode`` with the tasklet inside the NSDFG" form of
the scalar-out contract; the scalar then exits via the untouched WCR chain, which
accumulates the per-tile partials across tiles. The reduction op is the body's
accumulation op (the ``TileBinop`` that writes ``_nnr_priv`` -- identical to the
fold op for an associative reduction).

General: it keys off the *structure* (a plain ``tile[0:W] → scalar`` copy inside a
``reduce_at_output`` map body whose tile has an associative accumulator), not any
kernel-specific name, so every widened nested reduction folds the same way.
"""
from typing import Optional, Tuple

import dace
from dace.memlet import Memlet
from dace.sdfg import SDFG, SDFGState, nodes
from dace.transformation import pass_pipeline as ppl
from dace.transformation.passes.vectorization.mark_reduce_at_output import REDUCE_AT_OUTPUT_MARKER

#: ``TileBinop`` op token -> ``TileReduce`` op token. Only associative folds a
#: horizontal reduction is defined for; a non-associative accumulator is left as-is.
_ACCUM_TO_FOLD = {"+": "+", "*": "*", "min": "min", "max": "max"}


def _enclosed_by_reduce_at_output_map(sd: SDFG) -> bool:
    """True iff ``sd`` sits (at any nesting depth) inside a ``reduce_at_output``-tagged
    map's body. Walks up EVERY nesting level -- the reduction body can be several
    NSDFGs deep (azimint: ``kernel_46_8`` inside ``tile_main_body`` inside the tagged
    ``tile_main`` map), so a single-level ``parent_nsdfg_node`` check misses it.
    """
    from dace.sdfg.nodes import MapEntry
    while sd.parent_nsdfg_node is not None:
        node = sd.parent_nsdfg_node
        st = sd.parent
        if st is None:
            return False
        scope = st.scope_dict()
        entry = scope.get(node)
        while entry is not None:
            if isinstance(entry, MapEntry) and REDUCE_AT_OUTPUT_MARKER in entry.map.label:
                return True
            entry = scope.get(entry)
        sd = st.sdfg
    return False


def _is_scalar_shape(desc) -> bool:
    """True iff ``desc`` is a single-element descriptor (Scalar or all-ones Array)."""
    if isinstance(desc, dace.data.Scalar):
        return True
    if not isinstance(desc, dace.data.Array):
        return False
    try:
        return all(bool(dace.symbolic.simplify(s - 1) == 0) for s in desc.shape)
    except Exception:  # noqa: BLE001
        return False


def _accumulation_op(body_sdfg: SDFG, tile_name: str) -> Optional[str]:
    """The associative op that accumulates into ``tile_name`` (== the fold op).

    The widened seeded-addend accumulator is produced by a ``TileBinop`` (the
    ``_nnr_priv = _nnr_priv <op> addend`` aug-assign after tiling); its op is the
    reduction op. Returns a :data:`_ACCUM_TO_FOLD` value, or ``None`` if no
    associative producer is found.
    """
    from dace.libraries.tileops import TileBinop
    for st in body_sdfg.all_states():
        for n in st.nodes():
            if not isinstance(n, TileBinop) or n.op not in _ACCUM_TO_FOLD:
                continue
            for e in st.out_edges(n):
                w = e.data.data if e.data is not None else None
                if w is None:
                    continue
                # Match on the shared accumulator base name: a masked reduction writes the
                # aug-assign into a guarded ``_then_<acc>`` intermediate before the select
                # lands it in ``<acc>``, so require containment either way (not equality).
                if w == tile_name or tile_name in w or w in tile_name:
                    return _ACCUM_TO_FOLD[n.op]
    return None


class SpliceReductionTileFold(ppl.Pass):
    """Fold every widened ``reduce_at_output`` accumulator tile to its scalar output
    via a spliced :class:`~dace.libraries.tileops.TileReduce`.

    :param widths: The per-dim tile widths (innermost-last); the fold is over all
        tile dims (``axis=None``) so the accumulator tile collapses to a scalar.
    :param target_isa: ISA stamped on the emitted ``TileReduce`` (matches the
        orchestrator's other tile nodes).
    """

    CATEGORY: str = "Vectorization"

    def __init__(self, widths: Tuple[int, ...], target_isa: str = "SCALAR") -> None:
        super().__init__()
        self._widths = tuple(widths)
        self._target_isa = target_isa

    def modifies(self) -> ppl.Modifies:
        return ppl.Modifies.Nodes | ppl.Modifies.Edges | ppl.Modifies.Memlets

    def should_reapply(self, modified: ppl.Modifies) -> bool:
        return False

    def depends_on(self):
        return set()

    def _fold_edge(self, state: SDFGState, sd: SDFG, edge) -> bool:
        """Replace a ``tile[0:W] → scalar`` reduction copyback edge with a TileReduce."""
        from dace.libraries.tileops import TileReduce
        src, dst = edge.src, edge.dst
        if not (isinstance(src, nodes.AccessNode) and isinstance(dst, nodes.AccessNode)):
            return False
        if edge.data is None or edge.data.wcr is not None:
            return False
        src_desc = sd.arrays.get(src.data)
        dst_desc = sd.arrays.get(dst.data)
        if src_desc is None or dst_desc is None:
            return False
        # Source must be a full tile (shape == widths); destination a scalar sink.
        if tuple(str(s) for s in src_desc.shape) != tuple(str(w) for w in self._widths):
            return False
        if not _is_scalar_shape(dst_desc):
            return False
        op = _accumulation_op(sd, src.data)
        if op is None:
            return False
        # Splice: src(tile) -> TileReduce(op, axis=None) -> dst(scalar).
        red = TileReduce(f"tile_reduce_{dst.data}", widths=list(self._widths), op=op, axis=None)
        red.target_isa = self._target_isa
        state.add_node(red)
        tile_sub = ", ".join(f"0:{w}" for w in self._widths)
        state.add_edge(src, None, red, "_src", Memlet(data=src.data, subset=tile_sub))
        state.add_edge(red, "_dst", dst, None, Memlet(data=dst.data, subset="0"))
        state.remove_edge(edge)
        return True

    def apply_pass(self, sdfg: SDFG, pipeline_results) -> Optional[int]:
        count = 0
        for sd in sdfg.all_sdfgs_recursive():
            # Only fold inside a body NSDFG whose enclosing map is reduce_at_output-tagged
            # (the only place the seeded-addend-tile → scalar copyback is a reduction fold).
            if not _enclosed_by_reduce_at_output_map(sd):
                continue
            for state in sd.states():
                for edge in list(state.edges()):
                    if self._fold_edge(state, sd, edge):
                        count += 1
        # The per-tile fold above turns each widened accumulator tile into one scalar
        # partial. Cross-tile accumulation is NOT done here: the untouched
        # ``scalar ─[wcr]→ MapExit`` reduction chain the fold feeds carries the partials
        # across tiles -- codegen lowers that map-exit WCR to an OMP/atomic reduction. No
        # explicit loop-carry is threaded through the map (a map cannot express one; the
        # WCR already IS the carry).
        return count or None
