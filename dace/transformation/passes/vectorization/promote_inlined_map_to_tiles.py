# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""``PromoteInlinedMapToTiles`` — K=2 outer-state port of the descent (slice 1).

The body-NSDFG descent (:class:`PromoteNSDFGBodyToTiles`) widens
boundary connectors + lowers every internal tasklet to a tile lib node
in place inside a body NestedSDFG. The K=2 multi-dim path the user
asked for instead inlines the body NSDFG into the outer Map scope
first, then performs the equivalent rewrites directly on the outer
state. This module is the in-progress port; it lands in 5 slices per
``PROMOTE_INLINED_MAP_TO_TILES_PLAN.md``.

This file ships **slice 1**: the pass scaffold + the body-scalar
widening step. After this pass runs, every transient whose access
nodes are entirely within the tile-tagged Map's scope and whose
descriptor is either a :class:`Scalar` or an :class:`Array` of shape
``(1,)`` / ``(1, 1, ...)`` is widened to ``Array(shape=widths,
dtype=orig_dtype, transient=True, storage=Register)``, and every
memlet that references it has its subset rewritten from the length-1
form to the full tile subset ``[0:W_0, ..., 0:W_{K-1}]``.

The widened SDFG is **temporarily uncompilable** — internal tasklets
still have scalar bodies (``_out = _in1 + _in2``) but their memlets
now describe full-tile reads / writes. Slice 2 closes the loop by
rewriting the tasklets to :class:`TileBinop` / :class:`TileUnop` lib
nodes that consume the widened shape. Unit tests on this slice assert
the descriptor + memlet rewrite in isolation; runtime / numerical
parity comes with slice 4.
"""
import copy
from typing import Any, Dict, List, Optional, Set, Tuple

import dace
from dace import data, nodes, subsets
from dace.sdfg import SDFG
from dace.sdfg.state import SDFGState
from dace.transformation import pass_pipeline as ppl, transformation

from dace.transformation.passes.vectorization.utils.tile_dims import TileDimSpec


def _scope_subgraph_nodes(state: SDFGState, map_entry: nodes.MapEntry) -> Set:
    """Return the set of nodes inside ``map_entry``'s scope (entry / exit excluded).

    Mirrors ``state.scope_subgraph(map_entry).nodes()`` but returns a
    set for fast membership tests during the widening sweep.
    """
    return set(state.scope_subgraph(map_entry).nodes())


def _is_scalar_shaped(arr: data.Data) -> bool:
    """True iff ``arr`` is a :class:`Scalar` or an :class:`Array` whose every
    dim is length 1.

    The user's contract: "(length 1 arrays are equivalent to scalars and
    might appear)". Both shapes are widened by this pass.
    """
    if isinstance(arr, data.Scalar):
        return True
    if isinstance(arr, data.Array) and tuple(arr.shape) == tuple(1 for _ in arr.shape):
        return True
    return False


def _collect_body_scalar_transients(state: SDFGState, map_entry: nodes.MapEntry, scope_nodes: Set) -> List[str]:
    """Return the names of scalar / length-1 transients whose access nodes are
    entirely inside ``map_entry``'s scope.

    A transient is body-scoped iff every :class:`AccessNode` for it in
    ``state.sdfg.all_sdfgs_recursive()`` is one of ``scope_nodes``. A
    transient that has any access node outside the scope (another state,
    another map, a nested SDFG) is left untouched -- widening it would
    change a non-tile read elsewhere.

    :param state: State holding the map.
    :param map_entry: Tile-tagged map entry.
    :param scope_nodes: Set of nodes inside the map scope (entry / exit
        excluded), produced by :func:`_scope_subgraph_nodes`.
    :returns: Sorted list of qualifying transient data names.
    """
    sdfg = state.sdfg
    # Collect every AN inside the scope first.
    in_scope_ans = {n.data: n for n in scope_nodes if isinstance(n, nodes.AccessNode)}
    candidates = {
        name
        for name, an in in_scope_ans.items()
        if name in sdfg.arrays and sdfg.arrays[name].transient and _is_scalar_shaped(sdfg.arrays[name])
    }
    # Drop any candidate that ALSO appears outside the scope.
    for nsdfg in sdfg.all_sdfgs_recursive():
        for st in nsdfg.states():
            for n in st.nodes():
                if not isinstance(n, nodes.AccessNode):
                    continue
                if n.data not in candidates:
                    continue
                if st is state and n in scope_nodes:
                    continue
                # Out-of-scope reference -> not body-scoped.
                candidates.discard(n.data)
    return sorted(candidates)


def _widen_descriptor(sdfg: SDFG, name: str, widths: Tuple[int, ...]) -> None:
    """In-place widen ``sdfg.arrays[name]`` to ``Array(shape=widths, ...)``.

    The original dtype is preserved; storage forces ``Register`` so the
    widened tile stays in registers per the lib-node contract.
    """
    orig = sdfg.arrays[name]
    dtype = orig.dtype
    sdfg.remove_data(name, validate=False)
    sdfg.add_array(name,
                   shape=widths,
                   dtype=dtype,
                   transient=True,
                   storage=dace.dtypes.StorageType.Register,
                   find_new_name=False)


def _widen_memlet(memlet: dace.Memlet, name: str, widths: Tuple[int, ...]) -> None:
    """Rewrite ``memlet``'s subset (and ``other_subset``) referencing ``name``
    from the length-1 form to the full tile subset.

    The rewriter looks at ``memlet.data`` vs ``name``: when they match,
    ``subset`` is the length-1 side and gets widened; ``other_subset``
    (when present) is the OTHER side and is left untouched. When
    ``memlet.data != name`` and ``other_subset is not None`` and
    references ``name``, ``other_subset`` is the length-1 side and gets
    widened.

    The full tile subset is ``[0:W_0, ..., 0:W_{K-1}]``, matching the
    contract :class:`TileLoad` / :class:`TileStore` / :class:`TileBinop`
    use.
    """
    tile_ranges = [(0, w - 1, 1) for w in widths]
    if memlet.data == name and memlet.subset is not None:
        memlet.subset = subsets.Range(tile_ranges)
    elif memlet.other_subset is not None and memlet.data != name:
        # The other-subset side references ``name``. Widen it.
        memlet.other_subset = subsets.Range(tile_ranges)


def _rewrite_memlets_in_scope(state: SDFGState, scope_nodes: Set, name: str, widths: Tuple[int, ...]) -> int:
    """Rewrite every memlet in ``scope_nodes`` that references ``name``.

    Walks every edge with at least one endpoint in ``scope_nodes`` and
    updates the memlet in place. Returns the number of memlets rewritten.
    """
    rewritten = 0
    for edge in state.edges():
        if edge.src not in scope_nodes and edge.dst not in scope_nodes:
            continue
        if edge.data is None:
            continue
        touches = (edge.data.data == name) or (edge.data.other_subset is not None and
                                               ((isinstance(edge.src, nodes.AccessNode) and edge.src.data == name) or
                                                (isinstance(edge.dst, nodes.AccessNode) and edge.dst.data == name)))
        if not touches:
            continue
        _widen_memlet(edge.data, name, widths)
        rewritten += 1
    return rewritten


def widen_body_scalars_to_tile(state: SDFGState, map_entry: nodes.MapEntry, spec: TileDimSpec) -> Dict[str, int]:
    """Widen every body scalar transient in ``map_entry``'s scope to tile shape.

    Returns a dict mapping ``widened_name -> memlets_rewritten`` so
    callers can verify the rewrite count in tests.

    :param state: State holding the tile-tagged map.
    :param map_entry: Tile-tagged map entry; ``map.tile_spec`` is read
        by the caller to populate ``spec``.
    :param spec: Tile spec describing iter-vars + widths.
    :returns: Per-name rewrite count.
    """
    widths = tuple(spec.widths)
    scope_nodes = _scope_subgraph_nodes(state, map_entry)
    out: Dict[str, int] = {}
    for name in _collect_body_scalar_transients(state, map_entry, scope_nodes):
        _widen_descriptor(state.sdfg, name, widths)
        out[name] = _rewrite_memlets_in_scope(state, scope_nodes, name, widths)
    return out


@transformation.explicit_cf_compatible
class PromoteInlinedMapToTiles(ppl.Pass):
    """Walk every tile-tagged map and widen its body scalars to tile shape.

    Slice 1 of the K=2 outer-state port. Subsequent slices add the
    tasklet rewrite (slice 2), constant-store + merge (slice 3), and
    the orchestrator wiring + integration test (slice 4).
    """

    CATEGORY: str = "Vectorization"

    def modifies(self) -> ppl.Modifies:
        """Modifies descriptors + memlets."""
        return ppl.Modifies.Descriptors | ppl.Modifies.Memlets

    def should_reapply(self, modified: ppl.Modifies) -> bool:
        """Single sweep is enough."""
        return False

    def depends_on(self) -> Set[type]:
        """Standalone pass."""
        return set()

    def apply_pass(self, sdfg: SDFG, pipeline_results: Dict[str, Any]) -> Optional[int]:
        """Sweep every tile-tagged map and apply :func:`widen_body_scalars_to_tile`.

        Reads the ``{MapEntry: TileDimSpec}`` dict from
        ``pipeline_results["MarkTileDims"]`` (matches the descent's
        convention). Skips maps whose scope contains a
        :class:`NestedSDFG` -- those go through the body-NSDFG descent.

        :param sdfg: Top-level SDFG.
        :param pipeline_results: Reads ``"MarkTileDims"`` when present;
            the pass is a no-op when the key is missing.
        :returns: Total memlets rewritten across all maps, or ``None``
            when zero.
        """
        if not pipeline_results or "MarkTileDims" not in pipeline_results:
            return None
        specs: Dict[nodes.MapEntry, TileDimSpec] = pipeline_results["MarkTileDims"]
        total = 0
        for n, g in list(sdfg.all_nodes_recursive()):
            if not isinstance(n, nodes.MapEntry) or not isinstance(g, SDFGState):
                continue
            spec = specs.get(n)
            if spec is None:
                continue
            scope_nodes = _scope_subgraph_nodes(g, n)
            if any(isinstance(sn, nodes.NestedSDFG) for sn in scope_nodes):
                continue
            counts = widen_body_scalars_to_tile(g, n, spec)
            total += sum(counts.values())
        return total if total > 0 else None
