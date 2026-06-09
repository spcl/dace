# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Inside-body staging pass (design section 3 + section 3.3).

For each tile-tagged Map's body NSDFG, walks every non-transient
AccessNode and stages each per-tile-subset access through a fresh
transient. The shape of the staged transient is picked from the per-
dim lattice (section 4):

* CONSTANT on every tile dim
    -> ``Scalar`` (or length-1 Array) transient. Direct AN -> AN scalar
       copy (no tasklet); the lib node downstream consumes it via the
       ``Scalar`` operand kind with hardware splat.

* At least one dim in {LINEAR, AFFINE, REPLICATE, MODULAR}
    -> ``(K_0, ..., K_{K-1})`` Array transient. AN -> TileLoad ->
       transient (read) / transient -> TileStore -> AN (write).

* Any dim is GATHER
    -> same tile-shape transient; the TileLoad / TileStore carries
       ``gather_dims=...`` + variable-shape ``_idx_<d>`` connectors
       fed by PreparePerLaneIndices.

This is the inside-body counterpart of the staging design
(STAGE_GLOBAL_THROUGH_SCALARS_SPEC.md): the outer global accesses now
flow through staged transients first, and the lib nodes are emitted at
the boundary inside the body. No non-transient AccessNode survives in
the middle of the body's dataflow.

Status (incremental landing):

* Step 1 (this commit): module-level helper :func:`stage_constant_access`
  that handles the CONSTANT-only case (direct AN -> AN scalar copy).
  Pass scaffold with a stub ``apply_pass`` that documents the contract.
  Unit tests for the helper.
* Step 2 (follow-up): :func:`stage_tile_access` for the
  LINEAR / AFFINE / REPLICATE / MODULAR case with TileLoad emission.
* Step 3 (follow-up): GATHER case + integration with
  :class:`PreparePerLaneIndices`.
* Step 4 (follow-up): Pass walker drives the helpers across every
  body NSDFG.
"""
from typing import Any, Dict, Optional, Tuple

from dace import dtypes, properties
from dace.libraries.tileops import TileLoad, TileStore
from dace.memlet import Memlet
from dace.sdfg import SDFG
from dace.sdfg.nodes import AccessNode, MapEntry, NestedSDFG
from dace.sdfg.state import SDFGState
from dace.transformation import pass_pipeline as ppl, transformation
from dace.transformation.passes.vectorization.prepare_per_lane_indices import materialise_per_lane_index_tile
from dace.transformation.passes.vectorization.utils.map_predicates import is_innermost_map
from dace.transformation.passes.vectorization.utils.subsets import an_side_subset
from dace.transformation.passes.vectorization.utils.tile_access import PerDimKind, classify_tile_access


def stage_constant_access(state: SDFGState, an: AccessNode, name_hint: str = "constant_bridge") -> str:
    """Stage a CONSTANT-only access on ``an`` through a fresh ``Scalar`` transient.

    Mints a transient ``dace.data.Scalar`` of ``an``'s element dtype, adds it
    to ``state.sdfg.arrays``, and emits a direct AccessNode -> AccessNode edge
    from ``an`` to a new AccessNode wrapping the scalar -- **no copy tasklet
    between** per design section 3.6.

    Per design section 3.1: when every tile dim's per-dim kind is CONSTANT,
    the source value is loop-invariant across every tile lane. The lib node
    consuming the staged transient sees a ``Scalar`` operand kind and emits
    a hardware splat (AVX-512 ``_mm512_set1_pd``, SVE ``svdup_f64``, etc.).

    :param state: State holding ``an``.
    :param an: Non-transient AccessNode that's being staged.
    :param name_hint: Hint for the bridge transient name; uniquified via
        ``find_new_name=True``.
    :returns: Name of the staged transient (an AccessNode for it is added
        to ``state`` with the AN -> AN edge already wired).
    """
    sdfg = state.sdfg
    desc = sdfg.arrays[an.data]
    dtype = desc.dtype
    bridge_name, _ = sdfg.add_scalar(name_hint,
                                     dtype,
                                     transient=True,
                                     storage=dtypes.StorageType.Register,
                                     find_new_name=True)
    bridge_an = state.add_access(bridge_name)
    # Direct AN -> AN scalar copy per section 3.6: the memlet names the
    # source array on its data side; other_subset is omitted (full-shape
    # implicit), which the staged consumer's an_side_subset helper handles.
    # The 1-element subset on `an` side is naturally the loop-invariant
    # access (the classifier upstream guarantees this is CONSTANT-only).
    state.add_edge(an, None, bridge_an, None, Memlet(data=an.data, subset=f"0:1" if len(desc.shape) == 1 else None))
    return bridge_name


def stage_tile_access(state: SDFGState,
                      an: AccessNode,
                      widths: Tuple[int, ...],
                      src_subset: Memlet,
                      name_hint: str = "tile_bridge",
                      dim_strides: Optional[Tuple[int, ...]] = None,
                      replicate_factor_per_dim: Optional[Tuple[int, ...]] = None,
                      src_dims: Optional[Tuple[int, ...]] = None,
                      gather_dims: Tuple[int, ...] = (),
                      idx_sources: Optional[Dict[int, AccessNode]] = None) -> Tuple[str, "TileLoad"]:
    """Stage a tile-shaped access on ``an`` through a fresh `(widths,)` Array transient.

    Mints a transient ``Array(shape=widths, ...)`` of ``an``'s element
    dtype, adds a :class:`TileLoad` lib node between ``an`` and the new
    transient, and wires the source / destination memlets.

    Covers both the structured case (design section 3.1's LINEAR /
    AFFINE / REPLICATE / MODULAR; ``gather_dims`` empty) and the GATHER
    case (``gather_dims`` non-empty; ``idx_sources`` provides the per-
    dim ``_idx_<d>`` AccessNodes, typically materialised upstream by
    :func:`materialise_per_lane_index_tile`).

    :param state: State holding ``an``.
    :param an: Non-transient AccessNode being staged.
    :param widths: Tile widths ``(W_0, ..., W_{K-1})``.
    :param src_subset: The memlet attached to the
        ``an -> TileLoad._src`` edge (carries the per-tile region).
    :param name_hint: Hint for the transient name; uniquified.
    :param dim_strides: Per-dim stride coefficients forwarded to
        :class:`TileLoad`. Defaults to all 1s (LINEAR).
    :param replicate_factor_per_dim: Per-dim REPLICATE factors;
        defaults to all 1s.
    :param src_dims: Source-array dim permutation. Defaults to the
        ``TileLoad`` default (innermost ``K`` dims in order).
    :param gather_dims: Sorted tuple of tile dims that gather (empty
        for the structured case).
    :param idx_sources: ``{d: AccessNode}`` for each ``d in
        gather_dims``. Required when ``gather_dims`` is non-empty.
    :returns: ``(bridge_name, load_node)`` -- staged transient name +
        :class:`TileLoad` instance.
    :raises ValueError: When ``gather_dims`` is non-empty and
        ``set(gather_dims) != set(idx_sources)``.
    """
    if gather_dims and (idx_sources is None or set(gather_dims) != set(idx_sources)):
        raise ValueError(f"stage_tile_access: gather_dims {gather_dims!r} must match the keys of "
                         f"idx_sources {sorted(idx_sources) if idx_sources else None}")
    sdfg = state.sdfg
    desc = sdfg.arrays[an.data]
    dtype = desc.dtype
    bridge_name, _ = sdfg.add_array(name_hint,
                                    shape=widths,
                                    dtype=dtype,
                                    transient=True,
                                    storage=dtypes.StorageType.Register,
                                    find_new_name=True)
    bridge_an = state.add_access(bridge_name)
    load = TileLoad(name=f"load_{bridge_name}",
                    widths=widths,
                    dim_strides=dim_strides,
                    replicate_factor_per_dim=replicate_factor_per_dim,
                    src_dims=src_dims,
                    gather_dims=gather_dims)
    state.add_node(load)
    state.add_edge(an, None, load, "_src", src_subset)
    for d in gather_dims:
        idx_an = idx_sources[d]
        idx_desc = sdfg.arrays[idx_an.data]
        idx_subset = ", ".join(f"0:{s}" for s in idx_desc.shape)
        state.add_edge(idx_an, None, load, f"_idx_{d}", Memlet(f"{idx_an.data}[{idx_subset}]"))
    dst_subset_str = ", ".join(f"0:{w}" for w in widths)
    state.add_edge(load, "_dst", bridge_an, None, Memlet(f"{bridge_name}[{dst_subset_str}]"))
    return bridge_name, load


# Backwards-compat alias: callers that already wrote against ``stage_gather_access`` keep working.
# The unified ``stage_tile_access`` above handles both shapes via the optional ``gather_dims`` arg.
stage_gather_access = stage_tile_access


def stage_tile_store(state: SDFGState,
                     an: AccessNode,
                     widths: Tuple[int, ...],
                     dst_subset: Memlet,
                     name_hint: str = "tile_store_bridge",
                     dim_strides: Optional[Tuple[int, ...]] = None,
                     dst_dims: Optional[Tuple[int, ...]] = None,
                     gather_dims: Tuple[int, ...] = (),
                     idx_sources: Optional[Dict[int, AccessNode]] = None) -> Tuple[str, "TileStore"]:
    """Stage a tile-shaped WRITE to ``an`` through a fresh ``(widths,)`` Array transient.

    Symmetric to :func:`stage_tile_access` but for the destination side. Mints a transient
    ``Array(shape=widths, ...)`` of ``an``'s element dtype, adds a :class:`TileStore` lib
    node between the new transient and ``an``, and wires the source / destination memlets.

    :param state: State holding ``an``.
    :param an: Non-transient AccessNode being staged (destination).
    :param widths: Tile widths ``(W_0, ..., W_{K-1})``.
    :param dst_subset: The memlet attached to the ``TileStore._dst -> an`` edge (carries
        the per-tile window on the destination array).
    :param name_hint: Hint for the bridge transient name; uniquified.
    :param dim_strides: Per-dim stride coefficients forwarded to :class:`TileStore`.
    :param dst_dims: Destination-array dim permutation; defaults to the innermost ``K`` dims.
    :param gather_dims: Sorted tuple of tile dims that scatter (empty for the structured
        full-tile-window case).
    :param idx_sources: ``{d: AccessNode}`` for each ``d in gather_dims``. Required when
        ``gather_dims`` is non-empty.
    :returns: ``(bridge_name, store_node)`` -- staged transient name + :class:`TileStore`
        instance.
    :raises ValueError: When ``gather_dims`` is non-empty and ``set(gather_dims) !=
        set(idx_sources)``.
    """
    if gather_dims and (idx_sources is None or set(gather_dims) != set(idx_sources)):
        raise ValueError(f"stage_tile_store: gather_dims {gather_dims!r} must match the keys of "
                         f"idx_sources {sorted(idx_sources) if idx_sources else None}")
    sdfg = state.sdfg
    desc = sdfg.arrays[an.data]
    dtype = desc.dtype
    bridge_name, _ = sdfg.add_array(name_hint,
                                    shape=widths,
                                    dtype=dtype,
                                    transient=True,
                                    storage=dtypes.StorageType.Register,
                                    find_new_name=True)
    bridge_an = state.add_access(bridge_name)
    store = TileStore(name=f"store_{bridge_name}",
                      widths=widths,
                      dim_strides=dim_strides,
                      dst_dims=dst_dims,
                      gather_dims=gather_dims)
    state.add_node(store)
    src_subset_str = ", ".join(f"0:{w}" for w in widths)
    state.add_edge(bridge_an, None, store, "_src", Memlet(f"{bridge_name}[{src_subset_str}]"))
    for d in gather_dims:
        idx_an = idx_sources[d]
        idx_desc = sdfg.arrays[idx_an.data]
        idx_subset = ", ".join(f"0:{s}" for s in idx_desc.shape)
        state.add_edge(idx_an, None, store, f"_idx_{d}", Memlet(f"{idx_an.data}[{idx_subset}]"))
    state.add_edge(store, "_dst", an, None, dst_subset)
    return bridge_name, store


@properties.make_properties
@transformation.explicit_cf_compatible
class StageInsideBody(ppl.Pass):
    """Inside-body staging pass (design section 3.3).

    Walks every tile-tagged Map's body NSDFG. For each non-transient
    AccessNode inside that body, reads the AN-incident memlet's per-tile
    subset, classifies it via :func:`classify_tile_access`, and stages
    through a fresh transient sized per the lattice (section 4):

    * **CONSTANT-only on every tile dim** -- staged via
      :func:`stage_constant_access` (direct AN -> AN scalar copy).
    * **Any LINEAR / AFFINE / REPLICATE / MODULAR / GATHER dim** --
      DEFERRED (step 4b / 4c). Currently logged via the return count
      so the pipeline can observe coverage but the SDFG is left alone.

    Conservative by design: anything outside the CONSTANT case is a
    silent skip in this step. Subsequent G7 step 4b adds the Tile
    branch; 4c adds the Gather branch via :class:`PreparePerLaneIndices`
    integration.

    :ivar widths: Per-tile-dim widths ``(W_0, ..., W_{K-1})``, innermost-
        last. Matches the ``EmitTileOps`` / ``GenerateTileIterationMask``
        constructor pattern; the spec for each visited map is rebuilt by
        slicing the last ``K`` params off ``map_entry.map.params``.
    """

    CATEGORY: str = "Vectorization"

    widths = properties.Property(
        dtype=tuple,
        default=(8, ),
        desc="Per-dim tile widths, innermost-last; length in {1, 2, 3}.",
    )

    def __init__(self, widths: Tuple[int, ...] = (8, )) -> None:
        """Build the pass.

        :param widths: Per-tile-dim widths, innermost-last.
        :raises ValueError: If ``widths`` length is not in ``{1, 2, 3}``.
        """
        super().__init__()
        if not (1 <= len(widths) <= 3):
            raise ValueError(f"StageInsideBody: widths length {len(widths)} not in {{1, 2, 3}}")
        self.widths = tuple(widths)

    def modifies(self) -> ppl.Modifies:
        return (ppl.Modifies.AccessNodes | ppl.Modifies.Memlets | ppl.Modifies.Descriptors | ppl.Modifies.Tasklets
                | ppl.Modifies.Nodes)

    def should_reapply(self, modified: ppl.Modifies) -> bool:
        return False

    def depends_on(self):
        return set()

    def _body_nsdfgs(self, sdfg: SDFG):
        """Yield ``(state, nsdfg_node, map_entry)`` for every body NSDFG sitting
        directly inside an innermost map whose dim count >= ``K``.

        The tile-tagging contract (per the orchestrator pipeline):
        ``NestInnermostMapBodyIntoNSDFG`` nests every eligible map's body in a
        :class:`NestedSDFG` so the descent + classifier can see a clean
        single-NSDFG body. This walker walks every state and yields any
        innermost map whose body is exactly one NestedSDFG.
        """
        K = len(self.widths)
        for node, parent in sdfg.all_nodes_recursive():
            if not isinstance(node, MapEntry):
                continue
            if not isinstance(parent, SDFGState):
                continue
            try:
                if not is_innermost_map(parent, node):
                    continue
            except (StopIteration, ValueError):
                continue
            if len(node.map.params) < K:
                continue
            try:
                scope_nodes = parent.scope_subgraph(node, include_entry=False, include_exit=False).nodes()
            except (StopIteration, ValueError):
                continue
            nsdfgs = [n for n in scope_nodes if isinstance(n, NestedSDFG)]
            if len(nsdfgs) != 1:
                continue
            yield parent, nsdfgs[0], node

    def _stage_inner_body(self, state: SDFGState, inner_sdfg: SDFG, iter_vars: Tuple[str, ...]) -> int:
        """Walk every non-transient read AccessNode in ``inner_sdfg``'s states and
        dispatch each one to its lattice-appropriate stager (design section 3.1).

        Per-lattice dispatch:

        * Every tile dim CONSTANT
            -> :func:`stage_constant_access` (Scalar bridge, AN -> AN copy).
        * Mix of CONSTANT / LINEAR / AFFINE / REPLICATE / MODULAR (no GATHER)
            -> :func:`stage_tile_access` (TileLoad with empty ``gather_dims``).
        * Any GATHER dim
            -> DEFERRED to step 4c, which calls
              :func:`PreparePerLaneIndices` (G8 step 3) first to materialise
              the index tile(s), then stages via the gather branch.

        :returns: Number of staged ANs (>= 0).
        """
        staged = 0
        for inner_state in inner_sdfg.states():
            for an in list(inner_state.nodes()):
                if not isinstance(an, AccessNode):
                    continue
                desc = inner_sdfg.arrays.get(an.data)
                if desc is None or desc.transient:
                    continue
                # Capture ORIGINAL consumer edges before staging adds the bridge edge -- the
                # rewire below redirects each original consumer to read from the bridge AN
                # instead of the non-transient ``an``. Without this, the bridge transient
                # dangles and tasklets continue to read directly from the global array.
                pre_stage_out_edges = list(inner_state.out_edges(an))
                pre_stage_in_edges = list(inner_state.in_edges(an))
                # Write-side staging: when ``an`` has IN edges (something writes to it),
                # symmetrically insert a tile-shape bridge transient + TileStore. The producer
                # rewire below redirects the producer to write to the bridge, then the bridge
                # flows through TileStore to ``an``. First-slice scope: structured full-tile
                # writes only (the LINEAR / AFFINE / REPLICATE / MODULAR / mixed-with-CONSTANT
                # case). Gather / scatter writes and CONSTANT-only writes are not yet handled
                # here; they are deferred to subsequent slices.
                if pre_stage_in_edges and not pre_stage_out_edges:
                    try:
                        wsubset = an_side_subset(pre_stage_in_edges[0], an, inner_sdfg)
                    except Exception:  # noqa: BLE001
                        continue
                    wrecord = classify_tile_access(wsubset, iter_vars=iter_vars, inner_sdfg=inner_sdfg)
                    if not wrecord.per_dim_kind:
                        continue
                    wkinds = set(wrecord.per_dim_kind)
                    if PerDimKind.GATHER in wkinds or wkinds == {PerDimKind.CONSTANT}:
                        # Deferred: scatter writes (gather_dims set on TileStore) + single-scalar
                        # writes via a Scalar bridge. The TileStore non-full-tile-write lock
                        # (commit b9173e366) will fire at validate() time if reached.
                        continue
                    dst_subset_memlet = Memlet.from_memlet(pre_stage_in_edges[0].data)
                    dim_strides_w = tuple(s if s is not None else 1 for s in wrecord.dim_strides)
                    bridge_name, _ = stage_tile_store(inner_state,
                                                      an,
                                                      widths=tuple(self.widths),
                                                      dst_subset=dst_subset_memlet,
                                                      name_hint=f"{an.data}_tile_out",
                                                      dim_strides=dim_strides_w)
                    self._rewire_producers_to_bridge(inner_state, an, bridge_name, pre_stage_in_edges)
                    staged += 1
                    continue
                out_edges = pre_stage_out_edges
                if not out_edges:
                    continue
                try:
                    subset = an_side_subset(out_edges[0], an, inner_sdfg)
                except Exception:  # noqa: BLE001 -- helper may refuse on edge shapes outside scope.
                    continue
                record = classify_tile_access(subset, iter_vars=iter_vars, inner_sdfg=inner_sdfg)
                if not record.per_dim_kind:
                    continue
                kinds = set(record.per_dim_kind)
                if PerDimKind.GATHER in kinds:
                    # GATHER case: materialise the per-lane index tile per gather dim, then call
                    # ``stage_tile_access`` with ``gather_dims`` + ``idx_sources``. Per design
                    # section 9.2 ``gather_dims`` indexes source-array dim indices; the subset
                    # dim ``k`` corresponds to source dim ``k`` for the canonical (non-permuted)
                    # case. Multi-tile-dim deps: the per-lane index expression for source dim
                    # ``k`` is the subset's begin string, with EVERY tile iter-var substituted
                    # at the right lane position. ``materialise_per_lane_index_tile`` handles
                    # both forms (str + int for K_dep == 1; tuple form for K_dep >= 2).
                    K_tile = len(iter_vars)
                    gather_source_dims = tuple(k for k, kind in enumerate(record.per_dim_kind)
                                               if kind == PerDimKind.GATHER)
                    idx_sources: Dict[int, AccessNode] = {}
                    for k in gather_source_dims:
                        begin_str = str(subset.ranges[k][0])
                        # Pass all tile iter-vars + widths to the helper; it substitutes each in
                        # the expression and produces an index tile of shape ``widths`` rank K_tile.
                        # When the expression depends on a subset of iter-vars, the helper still
                        # produces a full-rank tile (the missing vars are broadcast in the body).
                        idx_name = materialise_per_lane_index_tile(
                            inner_state,
                            name_hint=f"_idx_{an.data}_{k}",
                            gather_expr=begin_str,
                            tile_iter_vars=iter_vars[0] if K_tile == 1 else iter_vars,
                            tile_widths=int(self.widths[0]) if K_tile == 1 else tuple(int(w) for w in self.widths),
                        )
                        idx_an = next(n for n in inner_state.nodes()
                                      if isinstance(n, AccessNode) and n.data == idx_name)
                        idx_sources[k] = idx_an
                    src_subset_memlet = Memlet.from_memlet(out_edges[0].data)
                    bridge_name, _ = stage_tile_access(inner_state,
                                                       an,
                                                       widths=tuple(self.widths),
                                                       src_subset=src_subset_memlet,
                                                       name_hint=f"{an.data}_gather",
                                                       gather_dims=gather_source_dims,
                                                       idx_sources=idx_sources)
                    self._rewire_consumers_to_bridge(inner_state, an, bridge_name, pre_stage_out_edges)
                    staged += 1
                    continue
                if kinds == {PerDimKind.CONSTANT}:
                    bridge_name = stage_constant_access(inner_state, an, name_hint=f"{an.data}_const")
                    self._rewire_consumers_to_bridge(inner_state, an, bridge_name, pre_stage_out_edges)
                    staged += 1
                else:
                    # Tile case (LINEAR / AFFINE / REPLICATE / MODULAR; possibly mixed with
                    # CONSTANT dims). Build a full-tile-shape src_subset memlet covering the
                    # access (the inner_state's out-edge already carries the per-tile region;
                    # we reuse its memlet so the lib node downstream sees the same subset).
                    src_subset_memlet = Memlet.from_memlet(out_edges[0].data)
                    dim_strides = tuple(s if s is not None else 1 for s in record.dim_strides)
                    replicate = tuple(r if r is not None else 1 for r in record.replicate_factor_per_dim)
                    bridge_name, _ = stage_tile_access(inner_state,
                                                       an,
                                                       widths=tuple(self.widths),
                                                       src_subset=src_subset_memlet,
                                                       name_hint=f"{an.data}_tile",
                                                       dim_strides=dim_strides,
                                                       replicate_factor_per_dim=replicate)
                    self._rewire_consumers_to_bridge(inner_state, an, bridge_name, pre_stage_out_edges)
                    staged += 1
        return staged

    def _rewire_producers_to_bridge(self, inner_state: SDFGState, original_an: AccessNode, bridge_name: str,
                                    original_in_edges) -> None:
        """Symmetric to :meth:`_rewire_consumers_to_bridge` for the write side.

        Redirect each producer edge that previously wrote into ``original_an`` to write
        into the bridge AccessNode named ``bridge_name``. The bridge then flows through
        :class:`TileStore` to ``original_an``. Skips edges that the staging helper just
        added (the new ``TileStore._dst -> original_an`` edge is staging plumbing, not
        a producer).
        """
        for old_edge in original_in_edges:
            # Skip the new TileStore._dst -> original_an edge that stage_tile_store added.
            if hasattr(old_edge.src, "label") and old_edge.src_conn == "_dst":
                continue
            bridge_an = inner_state.add_access(bridge_name)
            inner_state.add_edge(old_edge.src, old_edge.src_conn, bridge_an, old_edge.dst_conn,
                                 Memlet.from_memlet(old_edge.data))
            inner_state.remove_edge(old_edge)

    def _rewire_consumers_to_bridge(self, inner_state: SDFGState, original_an: AccessNode, bridge_name: str,
                                    original_out_edges) -> None:
        """Redirect each consumer edge that previously read from ``original_an`` to read from
        the bridge AccessNode named ``bridge_name``.

        The original edges in ``original_out_edges`` were captured BEFORE the staging helper
        added the new ``original_an -> bridge`` edge. We delete each original consumer edge
        and replace it with one whose source is a new AccessNode wrapping ``bridge_name``.
        Memlets are reused verbatim (subsets carry over); the consumer's ``dst_conn`` /
        downstream tasklet semantics stay intact.
        """
        for old_edge in original_out_edges:
            # Skip the edge that the staging helper just added (an -> tile bridge / scalar bridge);
            # that edge is part of the staging structure, not a consumer.
            if isinstance(old_edge.dst, AccessNode) and old_edge.dst.data == bridge_name:
                continue
            # Skip edges that go through a tile lib node (TileLoad / TileMaskGen / etc.) directly
            # from the staging helper -- those are also part of the staging plumbing.
            if hasattr(old_edge.dst, "label") and old_edge.dst_conn == "_src":
                continue
            bridge_an = inner_state.add_access(bridge_name)
            inner_state.add_edge(bridge_an, None, old_edge.dst, old_edge.dst_conn, Memlet.from_memlet(old_edge.data))
            inner_state.remove_edge(old_edge)

    def apply_pass(self, sdfg: SDFG, pipeline_results: Dict[str, Any]) -> Optional[int]:
        """Walk every tile-tagged body NSDFG; stage CONSTANT-only ANs.

        :param sdfg: Top-level SDFG.
        :param pipeline_results: Pipeline results (unused at this step).
        :returns: Number of ANs staged across the SDFG, or ``None`` if zero.
        """
        K = len(self.widths)
        total = 0
        for _state, nsdfg_node, map_entry in self._body_nsdfgs(sdfg):
            iter_vars = tuple(map_entry.map.params[-K:])
            total += self._stage_inner_body(_state, nsdfg_node.sdfg, iter_vars)
        return total or None
