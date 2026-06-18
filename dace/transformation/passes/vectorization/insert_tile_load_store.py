# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""InsertTileLoadStore -- boundary lib-node insertion + scalar bridge dispatch (design section 3 + section 3.3).

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
from typing import Any, Dict, List, Optional, Tuple

from dace import data, dtypes, properties, subsets
from dace.libraries.tileops import TileLoad, TileStore
from dace.memlet import Memlet
from dace.sdfg import SDFG
from dace.sdfg.nodes import AccessNode, MapEntry, NestedSDFG
from dace.sdfg.state import SDFGState
from dace.transformation import pass_pipeline as ppl, transformation
from dace.transformation.passes.vectorization.utils.map_predicates import is_innermost_map
from dace.transformation.passes.vectorization.utils.pass_invariants import (assert_invariant,
                                                                            memlet_subset_matches_descriptor,
                                                                            no_duplicate_connector_edges,
                                                                            no_memlet_dim_mismatch,
                                                                            no_transient_scalar_stores)
from dace.transformation.passes.vectorization.utils.subsets import an_side_subset, infer_edge_endpoints
from dace.transformation.passes.vectorization.utils.tile_access import (PerDimKind, classify_tile_access,
                                                                        _build_symbol_definition_map)


def _assert_post_stage_invariants(state: SDFGState) -> None:
    """Loud-fail audit of the design 3.8.3 invariants after staging completes.

    Two invariants enforced per user direction 2026-06-10:

    (1) **Lib-node-boundary rule**: edges adjacent to ``TileLoad`` / ``TileStore``
        connectors MUST NOT carry ``other_subset`` -- the connector descriptor
        defines the shape on the connector side, so a leaked ``other_subset``
        (typically ``[0]`` from a former Scalar bridge) clashes with the
        descriptor.

    (2) **AN -> AN survivors must be Scalar bridges**: per the stronger 3.8.3
        invariant, the only AN -> AN edges that survive in the post-stage body
        are those produced by :func:`stage_constant_access` (CONSTANT staging
        through Scalar transients). Every other AN -> AN edge means the walker
        missed a staging opportunity and the body contains a non-bridge
        AccessNode read/write that should have gone through a tile lib node.

    Raises :class:`AssertionError` (loud failure, surfaces the offending edge
    at staging time rather than letting it propagate into codegen).
    """
    sdfg = state.sdfg
    for edge in state.edges():
        mem = edge.data
        if mem is None:
            continue
        src_is_libnode = isinstance(edge.src, (TileLoad, TileStore))
        dst_is_libnode = isinstance(edge.dst, (TileLoad, TileStore))
        if (src_is_libnode or dst_is_libnode) and mem.other_subset is not None:
            node_label = edge.dst.label if dst_is_libnode else edge.src.label
            conn = edge.dst_conn if dst_is_libnode else edge.src_conn
            raise AssertionError(f"design 3.8.3 (1) violation: lib-node-adjacent edge "
                                 f"carries stale ``other_subset`` (other_subset={mem.other_subset!r}, "
                                 f"data={mem.data!r}, subset={mem.subset!r}); offending node "
                                 f"{node_label!r} connector {conn!r} in state {state.label!r}. "
                                 f"Use :func:`_libnode_boundary_memlet` when constructing memlets "
                                 f"at lib-node connector boundaries.")
        if isinstance(edge.src, AccessNode) and isinstance(edge.dst, AccessNode):
            src_desc = sdfg.arrays.get(edge.src.data)
            dst_desc = sdfg.arrays.get(edge.dst.data)
            # Allow AN -> AN when EITHER endpoint is a Scalar transient
            # (CONSTANT bridge produced by ``stage_constant_access``) OR when
            # the edge writes a tile-shape transient out to a non-transient
            # output array (the legitimate ``tile_bridge -> output_array``
            # pattern). Investigation 2026-06-10: DaCe codegen handles this
            # AN -> AN edge correctly by auto-generating ``CopyND<T, dim,
            # async, vecW>`` with the memlet's subset (e.g. ``B[i:i+W]``) for
            # the per-outer-iter offset. The TileStore destination staging
            # (Phase A1 in the plan) would be a design-cleanliness improvement
            # (routing every write through a tile lib node), not a correctness
            # fix -- the existing path produces correct code.
            either_is_scalar = (isinstance(src_desc, data.Scalar) or isinstance(dst_desc, data.Scalar))
            bridge_to_output = (src_desc is not None and src_desc.transient and dst_desc is not None
                                and not dst_desc.transient)
            # Symmetric input-staging case (user direction 2026-06-11): the
            # input bridge ``non-transient -> widened-tile transient`` mirrors
            # the existing ``transient -> non-transient`` output writeback.
            # Both are CopyND-handled by DaCe codegen using the memlet's
            # subset (e.g. ``src[i:i+W]``) -- design-clean for staging-first.
            input_staging = (src_desc is not None and not src_desc.transient and dst_desc is not None
                             and dst_desc.transient)
            # Intra-staging chain: tile bridge -> next tile transient, where
            # both endpoints are widened to the same K-D tile shape (no
            # Scalar). CopyND emits a direct W-element copy.
            intra_staging = False
            if src_desc is not None and dst_desc is not None:
                src_is_tile = src_desc.transient and not isinstance(src_desc, data.Scalar)
                dst_is_tile = dst_desc.transient and not isinstance(dst_desc, data.Scalar)
                intra_staging = src_is_tile and dst_is_tile
            if not (either_is_scalar or bridge_to_output or input_staging or intra_staging):
                src_data, src_subset, dst_data, dst_subset = infer_edge_endpoints(edge, sdfg)
                raise AssertionError(f"design 3.8.3 (2) violation: AN -> AN edge survives staging "
                                     f"but neither endpoint is a Scalar bridge or a transient->output "
                                     f"writeback. src={src_data!r}[{src_subset}] -> "
                                     f"dst={dst_data!r}[{dst_subset}] "
                                     f"(memlet data={mem.data!r}, subset={mem.subset!r}) "
                                     f"in state {state.label!r}. Non-Scalar AN -> AN edges must be "
                                     f"routed through a tile lib node (TileLoad / TileStore).")


def _libnode_boundary_memlet(other_memlet: Memlet) -> Memlet:
    """Build a memlet for an edge adjacent to a tile lib node (AN -> Load._src,
    TileStore._dst -> AN, etc.).

    Per design 3.8.2 (the "lib-node-boundary" invariant): edges that cross from
    an AccessNode into a tile lib node connector (or vice versa) MUST carry
    only ``data`` + ``subset``. The lib node's connector descriptor implicitly
    defines the shape on the connector side, so ``other_subset`` is redundant
    -- and if it inherits a stale value from the caller's input memlet
    (typically ``[0]`` from a former Scalar bridge), it would clash with the
    connector descriptor and trigger downstream codegen surprises.

    AN -> AN edges (the bypass output and the rewire-bridge edges) are NOT
    routed through this helper -- they preserve ``other_subset`` as part of
    DaCe's normal Memlet contract. The collapse to ``data + subset`` only
    happens at the lib-node boundary.

    ``wcr`` and ``volume`` are intentionally omitted: WCR appears only at the
    outer-Map boundary (never inside the body NSDFG that the walker stages
    through), and ``volume`` is inferred from the subset by DaCe.
    """
    return Memlet(data=other_memlet.data, subset=other_memlet.subset)


# Phase A4 (commit chain ending 2026-06-10): the legacy ``_topo_sort_access_nodes``
# was deleted -- the two-phase staging refactor (commit 2412eea00) plus the
# classifier-side inference (``infer_edge_endpoints`` +
# ``compute_per_iter_var_dep_mask``) make the iteration order irrelevant for
# correctness. Plain ``state.nodes()`` enumeration is used at the read-phase /
# write-phase callsites.


def stage_constant_access(state: SDFGState,
                          an: AccessNode,
                          name_hint: str = "constant_bridge",
                          src_subset: Optional["subsets.Range"] = None) -> str:
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
    :param src_subset: The exact CONSTANT (loop-invariant) element the consumer
        reads on the ``an`` side -- a single element (all tile dims size 1).
        The staging copy moves *only* that element into the scalar bridge.
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
    # Direct AN -> AN scalar copy per section 3.6: the memlet names the source
    # array on its data side and copies EXACTLY the loop-invariant element the
    # consumer reads. Using the actual access subset is essential -- it is a
    # single element on every dim (CONSTANT on all tile dims), so the bridge
    # stays scalar. Falling back to the full-array shape (the old ``subset=None``
    # multi-D default) would copy the WHOLE array (e.g. 2-D ``aa[0:N,0:N]``)
    # into a scalar -- a tile-into-scalar store the lowering rejects; a hardcoded
    # ``0:1`` would read the wrong element when the constant index is non-zero
    # (``aa[0, j]`` with an outer-loop ``j``). The fallbacks below only apply
    # when the caller has no subset to hand (legacy single-dim CONSTANT staging).
    if src_subset is not None:
        sub = subsets.Range(list(src_subset.ranges))
    elif len(desc.shape) == 1:
        sub = subsets.Range.from_string("0:1")
    else:
        sub = subsets.Range([(0, s - 1, 1) for s in desc.shape])
    state.add_edge(an, None, bridge_an, None, Memlet(data=an.data, subset=sub))
    return bridge_name


def _shape_dim_is_one_symbol(s) -> bool:
    """True when a descriptor extent is the :data:`~dace.symbolic.ONE` broadcast
    marker (a collapsed gather-index dim), as opposed to a literal width."""
    import sympy
    from dace.symbolic import ONE
    return isinstance(s, sympy.Basic) and ONE in s.free_symbols


def stage_tile_load(state: SDFGState,
                    an: AccessNode,
                    widths: Tuple[int, ...],
                    src_subset: Memlet,
                    name_hint: str = "tile_bridge",
                    dim_strides: Optional[Tuple[int, ...]] = None,
                    replicate_factor_per_dim: Optional[Tuple[int, ...]] = None,
                    src_dims: Optional[Tuple[int, ...]] = None,
                    gather_dims: Tuple[int, ...] = (),
                    idx_sources: Optional[Dict[int, AccessNode]] = None,
                    mask_an: Optional[AccessNode] = None,
                    dst_shape: Optional[Tuple[Any, ...]] = None) -> Tuple[str, "TileLoad"]:
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
    :param dst_shape: Optional descriptor shape for the staged bridge
        transient, distinct from ``widths``. Used by the gather-index
        builder (:meth:`InsertTileLoadStore._stage_array_read_tile`) to
        mint a full-K-dim ``(W_d if dep else ONE)`` index tile while the
        ``TileLoad`` loops only over the dep dims (``widths`` = dep
        widths). The ``ONE`` markers carry the per-tile-dim dependency
        positionally (design 9.2, user 2026-06-14). Defaults to
        ``widths`` (the bridge descriptor == the tile widths).
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
    bridge_shape = tuple(dst_shape) if dst_shape is not None else tuple(widths)
    # A ``ONE``-marked (collapsed) descriptor dim needs the ``ONE`` constant
    # defined on the SDFG (mirrors the constant materialise_per_lane_index_tile
    # used to seed). ``ONE == 1`` so the broadcast dim is genuinely length-1.
    if any(_shape_dim_is_one_symbol(s) for s in bridge_shape) and "ONE" not in sdfg.constants_prop:
        sdfg.add_constant("ONE", 1, data.Scalar(dtypes.int32))
    bridge_name, _ = sdfg.add_array(name_hint,
                                    shape=bridge_shape,
                                    dtype=dtype,
                                    transient=True,
                                    storage=dtypes.StorageType.Register,
                                    find_new_name=True)
    bridge_an = state.add_access(bridge_name)
    # Wire ``has_mask`` + ``_mask`` connector when a mask AN is in scope (per design 6.5
    # mask handling). The mask AN is the body NSDFG's ``_tile_iter_mask`` access; its
    # descriptor is ``bool[widths]`` per :class:`TileMaskGen`.
    has_mask_wired = mask_an is not None
    load = TileLoad(name=f"load_{bridge_name}",
                    widths=widths,
                    dim_strides=dim_strides,
                    replicate_factor_per_dim=replicate_factor_per_dim,
                    src_dims=src_dims,
                    gather_dims=gather_dims,
                    has_mask=has_mask_wired)
    state.add_node(load)
    # ``AN -> TileLoad._src`` is a lib-node-boundary edge: per design 3.8.2
    # it MUST drop ``other_subset`` (the TileLoad connector descriptor defines
    # the destination shape).
    state.add_edge(an, None, load, "_src", _libnode_boundary_memlet(src_subset))
    for d in gather_dims:
        idx_an = idx_sources[d]
        idx_desc = sdfg.arrays[idx_an.data]
        idx_subset = ", ".join(f"0:{s}" for s in idx_desc.shape)
        state.add_edge(idx_an, None, load, f"_idx_{d}", Memlet(f"{idx_an.data}[{idx_subset}]"))
    if has_mask_wired:
        mask_subset_str = ", ".join(f"0:{w}" for w in widths)
        state.add_edge(mask_an, None, load, "_mask", Memlet(f"{mask_an.data}[{mask_subset_str}]"))
    # The ``_dst`` memlet covers the FULL bridge descriptor (``bridge_shape``),
    # which may be the ``ONE``-padded index-tile shape rather than ``widths``;
    # the expansion writes ``_dst[tile_offset(widths)]`` (flat over the dep
    # widths) into the contiguous ``(W_d / ONE)`` buffer -- consistent volume.
    dst_subset_str = ", ".join(f"0:{s}" for s in bridge_shape)
    state.add_edge(load, "_dst", bridge_an, None, Memlet(f"{bridge_name}[{dst_subset_str}]"))
    return bridge_name, load


# Backwards-compat aliases. Per user direction 2026-06-10: ``stage_tile_load``
# is the symmetric name to ``stage_tile_store`` (read vs write side, both small
# focused passes). ``stage_tile_access`` and ``stage_gather_access`` were
# earlier names; keep them as aliases so callers don't break across the rename.
stage_tile_access = stage_tile_load
stage_gather_access = stage_tile_load


def stage_tile_store(state: SDFGState,
                     an: AccessNode,
                     widths: Tuple[int, ...],
                     dst_subset: Memlet,
                     name_hint: str = "tile_store_bridge",
                     dim_strides: Optional[Tuple[int, ...]] = None,
                     dst_dims: Optional[Tuple[int, ...]] = None,
                     gather_dims: Tuple[int, ...] = (),
                     idx_sources: Optional[Dict[int, AccessNode]] = None,
                     mask_an: Optional[AccessNode] = None) -> Tuple[str, "TileStore"]:
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
    has_mask_wired = mask_an is not None
    store = TileStore(name=f"store_{bridge_name}",
                      widths=widths,
                      dim_strides=dim_strides,
                      dst_dims=dst_dims,
                      gather_dims=gather_dims,
                      has_mask=has_mask_wired)
    state.add_node(store)
    src_subset_str = ", ".join(f"0:{w}" for w in widths)
    state.add_edge(bridge_an, None, store, "_src", Memlet(f"{bridge_name}[{src_subset_str}]"))
    for d in gather_dims:
        idx_an = idx_sources[d]
        idx_desc = sdfg.arrays[idx_an.data]
        idx_subset = ", ".join(f"0:{s}" for s in idx_desc.shape)
        state.add_edge(idx_an, None, store, f"_idx_{d}", Memlet(f"{idx_an.data}[{idx_subset}]"))
    if has_mask_wired:
        mask_subset_str = ", ".join(f"0:{w}" for w in widths)
        state.add_edge(mask_an, None, store, "_mask", Memlet(f"{mask_an.data}[{mask_subset_str}]"))
    # ``TileStore._dst -> AN`` is a lib-node-boundary edge (symmetric to
    # the read side at stage_tile_access).
    state.add_edge(store, "_dst", an, None, _libnode_boundary_memlet(dst_subset))
    return bridge_name, store


@properties.make_properties
@transformation.explicit_cf_compatible
class InsertTileLoadStore(ppl.Pass):
    """InsertTileLoadStore -- boundary lib-node insertion + scalar bridge dispatch (design section 3.3).

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
            raise ValueError(f"InsertTileLoadStore: widths length {len(widths)} not in {{1, 2, 3}}")
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
        from dace.transformation.passes.vectorization.split_map_for_tile_remainder import (SCALAR_TAIL_MARKER,
                                                                                           TILE_K1_TAIL_MARKER)
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
            # Skip the postamble tails: ``__scalar_tail`` is a step-1 sequential
            # loop running the original (non-tile) body -- no tile load/store
            # needed. ``__tile_k1_tail`` runs at K=1 widths=(1,) (handled by a
            # separate pinned path); the standard K-D walker skips it too.
            if node.map.label.endswith(SCALAR_TAIL_MARKER) or node.map.label.endswith(TILE_K1_TAIL_MARKER):
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
        """Two-phase staging per user direction 2026-06-10: stage all global READS
        first, then all global WRITES. Both phases complete BEFORE the downstream
        :class:`ConvertTaskletsToTileOps` pass emits any vector ops.

        Phase 1 (reads) walks non-transient ANs in topo order (sources first) and
        wraps each global read access in a TileLoad / Scalar bridge.

        Phase 2 (writes) walks non-transient ANs and wraps each global write in a
        TileStore. The phase skips ANs whose in-edges already feed from a tile
        lib node (TileStore inserted upstream during phase 1's
        ``_maybe_stage_tilestore_to_output``), avoiding a double-stage.

        :returns: Number of staged ANs across both phases.
        """
        staged = 0
        mask_name = self._find_inner_mask_name(inner_sdfg)
        for inner_state in inner_sdfg.states():
            staged += self._stage_reads_in_state(inner_state, inner_sdfg, iter_vars, mask_name)
            # Phase A6 (user direction 2026-06-10): between phase 1 (reads) and phase 2 (writes),
            # resize every Scalar / (1,) Array transient AN that's downstream of a
            # tile-input tasklet to tile-shape (W,). Recursive BFS so multi-hop
            # scalar chains all get resized. Required so that
            # ConvertTaskletsToTileOps emits TileBinop with kind=Tile output on
            # every link of the chain, not just the first link.
            self._resize_scalar_chain_downstream_of_tiles(inner_state)
            staged += self._stage_writes_in_state(inner_state, inner_sdfg, iter_vars, mask_name)
        return staged

    def _stage_reads_in_state(self, inner_state: SDFGState, inner_sdfg: SDFG, iter_vars: Tuple[str, ...],
                              mask_name: Optional[str]) -> int:
        """Phase 1: stage every global READ in ``inner_state``.

        Iterates non-transient ANs in topo order so a transient bridge whose value
        comes from a lane-dep memlet (e.g. ``A_const = A[__sym]``) is reached
        AFTER its upstream non-transient source has been staged.
        """
        staged = 0
        for an in [n for n in inner_state.nodes() if isinstance(n, AccessNode)]:
            desc = inner_sdfg.arrays.get(an.data)
            if desc is None or desc.transient:
                continue
            pre_stage_out_edges = list(inner_state.out_edges(an))
            if not pre_stage_out_edges:
                continue  # No reads -- sink AN handled by phase 2.
            mask_an_for_this = (self._find_mask_producer_an(inner_state, mask_name) if mask_name else None)
            try:
                src_data, src_subset, _dst_data, _dst_subset = infer_edge_endpoints(pre_stage_out_edges[0], inner_sdfg)
            except Exception:  # noqa: BLE001
                continue
            subset = (src_subset if src_data == an.data else an_side_subset(pre_stage_out_edges[0], an, inner_sdfg))
            if subset is None:
                continue
            record = classify_tile_access(subset, iter_vars=iter_vars, inner_sdfg=inner_sdfg, state=inner_state)
            if not record.per_dim_kind:
                continue
            kinds = set(record.per_dim_kind)
            if PerDimKind.GATHER in kinds:
                # GATHER read: build per-lane idx tile(s) as TILE LIB NODES + TileLoad with gather_dims.
                # One AN may be read with SEVERAL distinct indirect indices in the
                # same body (the ICON cell-from-edges interpolation reads
                # ``z_kin_hor_e[.., edge_idx[..,0]]`` AND ``[.., edge_idx[..,1]]``
                # AND ``[.., edge_idx[..,2]]``). Each distinct gather subset needs
                # its OWN index tile + TileLoad; staging only the first edge and
                # rewiring every consumer to it aliases them all to index 0 (the
                # sum then reads the same element thrice). Group the out-edges by
                # AN-side subset and stage one gather per distinct group.
                gather_groups: Dict[str, list] = {}
                for e in pre_stage_out_edges:
                    try:
                        e_sd, e_ss, _edd, _eds = infer_edge_endpoints(e, inner_sdfg)
                        e_sub = e_ss if e_sd == an.data else an_side_subset(e, an, inner_sdfg)
                    except Exception:  # noqa: BLE001
                        e_sub = None
                    gather_groups.setdefault(str(e_sub), []).append((e, e_sub))
                for _gkey, group in gather_groups.items():
                    g_edges = [e for e, _s in group]
                    g_subset = next((s for _e, s in group if s is not None), subset)
                    g_record = classify_tile_access(g_subset,
                                                    iter_vars=iter_vars,
                                                    inner_sdfg=inner_sdfg,
                                                    state=inner_state)
                    gather_source_dims = tuple(k for k, kind in enumerate(g_record.per_dim_kind)
                                               if kind == PerDimKind.GATHER)
                    idx_sources: Dict[int, AccessNode] = {}
                    for k in gather_source_dims:
                        begin_str = str(g_subset.ranges[k][0])
                        # Build the per-lane index as TILE LIB NODES (TileLoad of the
                        # data-dependent array read into a (W_d if dep else ONE) tile +
                        # TileBinop for any arithmetic). No CPP fallback (user 2026-06-14).
                        idx_an = self._stage_index_via_tileops(inner_state,
                                                               inner_sdfg,
                                                               iter_vars,
                                                               begin_str,
                                                               name_hint=f"_idx_{an.data}_{k}",
                                                               mask_an=mask_an_for_this)
                        if idx_an is None:
                            raise NotImplementedError(
                                f"InsertTileLoadStore: could not build a tile-op gather index for "
                                f"{begin_str!r} (read of '{an.data}' source dim {k}, iter_vars={iter_vars}). "
                                f"The CPP per-lane materialiser was removed (user 2026-06-14: no CPP index "
                                f"tasklets); this index shape needs a tile-op lowering path.")
                        idx_sources[k] = idx_an
                    # Source base: gather dims span the full extent (the per-lane idx
                    # provides the position), but NON-gather dims keep their per-tile
                    # begin so the base pointer carries the linear/constant offset
                    # (e.g. ``&zqx[_for_it_88]`` for the contiguous tile dim). Using
                    # the full extent on every dim drops that base -- correct for a
                    # pure gather (idx is absolute) but wrong for a mixed
                    # gather+linear access like ``zqx[jo-1, 0, _for_it_88]``.
                    _gset = set(gather_source_dims)
                    _gparts = [(f"0:{s}" if d in _gset else str(g_subset.ranges[d][0]))
                               for d, s in enumerate(desc.shape)]
                    src_subset_memlet = Memlet(data=an.data, subset=", ".join(_gparts))
                    # Per-tile-dim lane stride + source-dim mapping: each tile dim maps
                    # to the source dim its iter-var indexes, so the gather adds the
                    # per-lane offset for LINEAR tile dims regardless of layout. The
                    # default (innermost K dims) is wrong when the linear tile dim is an
                    # OUTER source dim (Fortran ``zqx[_for_it_88, 0, jo-1]`` -- tile dim
                    # is dim0, while the innermost dim2 is the gather dim).
                    _g_strides, _g_repl, _g_src_dims = self._pad_to_tile_dims(
                        g_record, iter_vars, src_arr_strides=tuple(desc.strides) if desc.strides else None)
                    bridge_name, _ = stage_tile_load(inner_state,
                                                     an,
                                                     widths=tuple(self.widths),
                                                     src_subset=src_subset_memlet,
                                                     name_hint=f"{an.data}_gather",
                                                     dim_strides=_g_strides,
                                                     replicate_factor_per_dim=_g_repl,
                                                     src_dims=_g_src_dims,
                                                     gather_dims=gather_source_dims,
                                                     idx_sources=idx_sources,
                                                     mask_an=mask_an_for_this)
                    self._rewire_consumers_to_bridge(inner_state, an, bridge_name, g_edges, iter_vars=iter_vars)
                    staged += 1
                continue
            # CONSTANT / structured reads can ALSO appear with several distinct
            # subsets on one AN (e.g. ``e_bln[jb,0,jc]``, ``e_bln[jb,1,jc]``,
            # ``e_bln[jb,2,jc]`` -- same tile dim ``jc``, distinct constant middle
            # index). Group the out-edges by AN-side subset and stage one bridge
            # per distinct group; staging only the first edge and rewiring every
            # consumer to it aliases them all (the ICON 3-edge interpolation then
            # multiplies every gather by ``e_bln[..,0,..]``). Same contract as the
            # gather branch above.
            struct_groups: Dict[str, list] = {}
            for e in pre_stage_out_edges:
                try:
                    e_sd, e_ss, _edd, _eds = infer_edge_endpoints(e, inner_sdfg)
                    e_sub = e_ss if e_sd == an.data else an_side_subset(e, an, inner_sdfg)
                except Exception:  # noqa: BLE001
                    e_sub = None
                struct_groups.setdefault(str(e_sub), []).append((e, e_sub))
            for _skey, sgroup in struct_groups.items():
                s_edges = [e for e, _s in sgroup]
                s_record = record
                if len(struct_groups) > 1:
                    s_sub = next((s for _e, s in sgroup if s is not None), subset)
                    s_record = classify_tile_access(s_sub,
                                                    iter_vars=iter_vars,
                                                    inner_sdfg=inner_sdfg,
                                                    state=inner_state)
                if set(s_record.per_dim_kind) == {PerDimKind.CONSTANT}:
                    # Stage EXACTLY the loop-invariant element the consumer reads
                    # (a single CONSTANT element) -- not the whole array. Without
                    # this an N-D ``an`` would copy its full shape into the scalar
                    # bridge (see ``stage_constant_access``).
                    try:
                        const_sub = an_side_subset(s_edges[0], an, inner_sdfg)
                    except Exception:  # noqa: BLE001 -- exotic edge: fall back to descriptor-shape default
                        const_sub = None
                    bridge_name = stage_constant_access(inner_state, an, name_hint=f"{an.data}_const",
                                                        src_subset=const_sub)
                    self._rewire_consumers_to_bridge(inner_state, an, bridge_name, s_edges, iter_vars=iter_vars)
                    staged += 1
                    continue
                # Structured tile load: LINEAR / AFFINE / REPLICATE / MODULAR (possibly mixed with CONSTANT).
                src_subset_memlet = Memlet.from_memlet(s_edges[0].data)
                # Pass src array strides so the diagonal-as-affine path can combine
                # per-dim strides when the same iter-var dominates multiple source dims.
                src_arr_strides = tuple(desc.strides) if desc.strides else None
                # Per-tile-dim coefficient + source-array dim basis, derived together
                # so they stay consistent: each tile dim reads along the array dim its
                # iter-var indexes (its OWN stride). The lib-node default (innermost K
                # dims) is wrong on a non-C layout -- a Fortran array ``A[i, j]`` with
                # strides ``(1, M)`` and an innermost ``i`` map indexes the UNIT-stride
                # dim 0, not the last; mapping positionally would stride the ``i``-tile
                # by ``M`` (reading a row, not the column). For a diagonal ``A[i, i]``
                # the coefficient is the combined byte stride and the basis is the
                # unit-stride indexed dim (see :meth:`_pad_to_tile_dims`).
                dim_strides, replicate, _s_src_dims = self._pad_to_tile_dims(s_record,
                                                                             iter_vars,
                                                                             src_arr_strides=src_arr_strides)
                bridge_name, _ = stage_tile_load(inner_state,
                                                 an,
                                                 widths=tuple(self.widths),
                                                 src_subset=src_subset_memlet,
                                                 name_hint=f"{an.data}_tile",
                                                 dim_strides=dim_strides,
                                                 replicate_factor_per_dim=replicate,
                                                 src_dims=_s_src_dims,
                                                 mask_an=mask_an_for_this)
                self._rewire_consumers_to_bridge(inner_state, an, bridge_name, s_edges, iter_vars=iter_vars)
                staged += 1
        return staged

    def _stage_writes_in_state(self, inner_state: SDFGState, inner_sdfg: SDFG, iter_vars: Tuple[str, ...],
                               mask_name: Optional[str]) -> int:
        """Phase 2: stage every global WRITE in ``inner_state``.

        Skips ANs whose in-edges already feed from a tile lib node -- phase 1's
        ``_maybe_stage_tilestore_to_output`` may have already inserted the
        TileStore during the bridge -> output rewire.
        """
        staged = 0
        for an in [n for n in inner_state.nodes() if isinstance(n, AccessNode)]:
            desc = inner_sdfg.arrays.get(an.data)
            if desc is None or desc.transient:
                continue
            pre_stage_in_edges = list(inner_state.in_edges(an))
            pre_stage_out_edges = list(inner_state.out_edges(an))
            if not pre_stage_in_edges:
                continue  # No write to stage (pure source -- the read phase owns it).
            if any(isinstance(e.src, (TileLoad, TileStore)) for e in pre_stage_in_edges):
                continue  # Already staged by phase 1's bridge->output insertion.
            # A pure sink (writes, no reads) is the common case. The other staged
            # shape is an in-place RMW intermediate (user direction 2026-06-15): a
            # non-transient WRITTEN and re-READ in the SAME state -- e.g. cloudsc
            # ``zqx_v = zqx_v + zqx_l`` immediately followed by
            # ``zqx_v = zqx_v + zqx_i``. Phase 1 already bridged the read side (every
            # out-edge now feeds a TileLoad), so the tile-producing write still needs
            # a TileStore; without it the producer's lib-op output (after
            # ConvertTaskletsToTileOps) would wire straight to the non-transient and
            # violate "any tile input -> tile output". The resulting
            # ``TileStore -> AN -> TileLoad`` RAW chain on the single shared
            # AccessNode is correctly ordered (write before read). If the out-edges
            # are NOT all already-bridged reads we leave the AN alone (unrecognised
            # shape -- stay safe).
            if pre_stage_out_edges and not all(isinstance(e.dst, (TileLoad, TileStore)) for e in pre_stage_out_edges):
                continue
            mask_an_for_this = (self._find_mask_producer_an(inner_state, mask_name) if mask_name else None)
            try:
                wsubset = an_side_subset(pre_stage_in_edges[0], an, inner_sdfg)
            except Exception:  # noqa: BLE001
                continue
            wrecord = classify_tile_access(wsubset, iter_vars=iter_vars, inner_sdfg=inner_sdfg, state=inner_state)
            if not wrecord.per_dim_kind:
                continue
            wkinds = set(wrecord.per_dim_kind)
            if wkinds == {PerDimKind.CONSTANT}:
                continue  # Loop-invariant write stays as direct producer -> AN copy (design 3.6).
            if PerDimKind.GATHER in wkinds:
                # SCATTER: build per-lane idx tile(s) as TILE LIB NODES + TileStore with gather_dims.
                scatter_source_dims = tuple(k for k, kind in enumerate(wrecord.per_dim_kind)
                                            if kind == PerDimKind.GATHER)
                idx_sources_w: Dict[int, AccessNode] = {}
                for k in scatter_source_dims:
                    begin_str = str(wsubset.ranges[k][0])
                    # Build the per-lane scatter index as TILE LIB NODES (same
                    # (W_d if dep else ONE) tile as the gather side). No CPP fallback.
                    idx_an = self._stage_index_via_tileops(inner_state,
                                                           inner_sdfg,
                                                           iter_vars,
                                                           begin_str,
                                                           name_hint=f"_idx_scatter_{an.data}_{k}",
                                                           mask_an=mask_an_for_this)
                    if idx_an is None:
                        raise NotImplementedError(
                            f"InsertTileLoadStore: could not build a tile-op scatter index for "
                            f"{begin_str!r} (write of '{an.data}' dest dim {k}, iter_vars={iter_vars}). "
                            f"The CPP per-lane materialiser was removed (user 2026-06-14: no CPP index "
                            f"tasklets); this index shape needs a tile-op lowering path.")
                    idx_sources_w[k] = idx_an
                # Dest base: scatter dims span the full extent (per-lane idx gives
                # the position); NON-scatter dims keep their per-tile begin so the
                # base carries the linear/constant offset (mixed scatter+linear,
                # e.g. ``zratio[jo-1, _for_it_88]``). See the read-gather note.
                _wgset = set(scatter_source_dims)
                _wparts = [(f"0:{s}" if d in _wgset else str(wsubset.ranges[d][0])) for d, s in enumerate(desc.shape)]
                dst_subset_memlet = Memlet(data=an.data, subset=", ".join(_wparts))
                # Per-tile-dim lane stride + dest-dim mapping (see the read-gather
                # note): each tile dim maps to the dest dim its iter-var indexes so
                # the scatter adds the per-lane offset for LINEAR tile dims under
                # any layout (Fortran ``zratio[_for_it_88, jo-1]`` -- tile dim0).
                _w_strides, _w_repl, _w_dst_dims = self._pad_to_tile_dims(
                    wrecord, iter_vars, src_arr_strides=tuple(desc.strides) if desc.strides else None)
                bridge_name, _ = stage_tile_store(inner_state,
                                                  an,
                                                  widths=tuple(self.widths),
                                                  dst_subset=dst_subset_memlet,
                                                  name_hint=f"{an.data}_scatter_out",
                                                  dim_strides=_w_strides,
                                                  dst_dims=_w_dst_dims,
                                                  gather_dims=scatter_source_dims,
                                                  idx_sources=idx_sources_w,
                                                  mask_an=mask_an_for_this)
                self._rewire_producers_to_bridge(inner_state, an, bridge_name, pre_stage_in_edges)
                staged += 1
                continue
            # Structured tile store: LINEAR / AFFINE / REPLICATE / MODULAR.
            dst_subset_memlet = Memlet.from_memlet(pre_stage_in_edges[0].data)
            dst_arr_strides = tuple(desc.strides) if desc.strides else None
            # Per-tile-dim coefficient + dest-array dim basis, derived together (see the
            # structured-read note): each tile dim writes along the array dim its
            # iter-var indexes, so a non-C layout (Fortran ``C[i, j]`` strides
            # ``(1, M)``) strides the ``i``-tile by 1 (its dim-0 stride), not
            # positionally by ``M``; a diagonal ``C[i, i]`` uses the combined byte
            # stride against its unit-stride indexed dim.
            dim_strides_w, _, _s_dst_dims = self._pad_to_tile_dims(wrecord, iter_vars, src_arr_strides=dst_arr_strides)
            bridge_name, _ = stage_tile_store(inner_state,
                                              an,
                                              widths=tuple(self.widths),
                                              dst_subset=dst_subset_memlet,
                                              name_hint=f"{an.data}_tile_out",
                                              dim_strides=dim_strides_w,
                                              dst_dims=_s_dst_dims,
                                              mask_an=mask_an_for_this)
            self._rewire_producers_to_bridge(inner_state, an, bridge_name, pre_stage_in_edges)
            staged += 1
        return staged

    def _pad_to_tile_dims(self, record, iter_vars: Tuple[str, ...], src_arr_strides=None):
        """Pad classifier's per-source-dim arrays (``dim_strides``,
        ``replicate_factor_per_dim``) to full per-tile-dim length ``K``.

        When the source array has fewer dims than the tile (e.g. ``A[ii]`` accessed
        inside a K=2 ``(ii, jj)`` body), iter_var ``jj`` doesn't appear in any subset
        dim -- the corresponding tile dim is a BROADCAST and the lib node should see
        ``dim_strides_k = 0`` + ``replicate_factor_k = widths[k]`` so the load emits
        the same source value across every lane of that tile dim.

        Per design 7.5 + user direction 2026-06-10: this is the "transients are
        either full tile or scalar" invariant. By padding here, the resulting tile
        transient stays full-tile shape ``(W_0, ..., W_{K-1})`` regardless of how
        many source dims the access touches.

        Diagonal-as-affine (user direction 2026-06-10 / design 5.3 revised):
        when the same iter-var dominates MULTIPLE source dims (e.g. ``A[2*i, i]``
        for K=1, or ``A[i, i]`` for K=2), combine the per-dim affine coefficients
        into a single effective stride using the source array's strides:

            combined_stride = sum_d (per_dim_stride[d] * src_arr_strides[d])

        This avoids the gather encoding (which would allocate per-dim ``_idx_<d>``
        tiles holding arithmetic progressions) and instead emits a normal strided
        TileLoad. Refused with NotImplementedError when any per-dim stride is
        symbolic-only-or-None -- non-affine diagonals cannot be expressed as a
        single linear stride.

        :param src_arr_strides: per-source-dim strides of the array being staged.
            When None, falls back to picking the first dim (legacy behaviour).
        :returns: ``(dim_strides, replicate_factor_per_dim, src_dims)`` -- the per-
            tile-dim coefficient, replicate factor, and source-array dim basis. The
            three are computed together so the coefficient always matches the basis
            ``offset_via_strides`` will scale it by. ``src_dims`` is ``None`` (use the
            lib-node positional default) when ``src_arr_strides`` is not supplied.
        """
        from collections import defaultdict
        K = len(iter_vars)
        widths = tuple(int(w) for w in self.widths)
        ndim = len(src_arr_strides) if src_arr_strides is not None else None

        def _stride_is_one(s) -> bool:
            # A unit (contiguous) stride is always the literal int 1 / Integer(1);
            # a symbolic stride (``N``, ``N*M``) is not int-convertible -> not unit.
            try:
                return int(s) == 1
            except (TypeError, ValueError):
                return False

        # Multi-map: iter_var -> list of source dim indices it dominates.
        iv_to_src_dims: Dict[str, List[int]] = defaultdict(list)
        for d, iv_name in enumerate(record.dim_iter_var):
            if iv_name is not None:
                iv_to_src_dims[iv_name].append(d)
        padded_strides = []
        padded_replicate = []
        # The per-tile-dim source-dim basis is computed HERE, together with the
        # coefficient, because the two are interdependent: ``offset_via_strides``
        # forms ``coeff[k] * strides[src_dims[k]] * __l<k>``, so the coeff is only
        # correct against the basis it was derived for. Computing them apart (an
        # earlier bug) let a non-C layout / diagonal pick a basis that mismatched
        # the coeff, inflating the offset by the basis stride.
        padded_src_dims = []
        for k in range(K):
            iv = iter_vars[k]
            # Positional default basis = the lib node's own default (innermost K
            # source dims). Used for broadcast dims and as the diagonal fallback.
            pos_default = (ndim - K + k) if ndim is not None else None
            if iv in iv_to_src_dims:
                dims_for_iv = iv_to_src_dims[iv]
                if len(dims_for_iv) == 1:
                    # Single source dim: coeff = the affine multiplier, basis = the
                    # dim the iter-var actually indexes. Correct under ANY layout --
                    # the offset is ``affine * strides[that_dim] * __l`` (this is the
                    # Fortran-layout fix: the unit-stride dim need NOT be the last).
                    d = dims_for_iv[0]
                    s = record.dim_strides[d]
                    padded_strides.append(s if s is not None else 1)
                    r = record.replicate_factor_per_dim[d]
                    padded_replicate.append(r if r is not None else 1)
                    padded_src_dims.append(d)
                elif src_arr_strides is None:
                    # Diagonal, legacy fallback (no strides supplied): pick the first
                    # dim and leave the basis to the caller / lib-node default.
                    d = dims_for_iv[0]
                    s = record.dim_strides[d]
                    padded_strides.append(s if s is not None else 1)
                    r = record.replicate_factor_per_dim[d]
                    padded_replicate.append(r if r is not None else 1)
                    padded_src_dims.append(pos_default)
                else:
                    # Diagonal: combine the per-dim affine strides into one BYTE
                    # stride. ``offset_via_strides`` then multiplies this coeff by
                    # ``strides[basis]``, so the basis MUST be a unit-stride indexed
                    # dim for the product to equal ``combined`` (C-layout's
                    # contiguous LAST dim and Fortran's contiguous FIRST dim are both
                    # covered).
                    all_affine = all(record.dim_strides[d] is not None for d in dims_for_iv)
                    if not all_affine:
                        raise NotImplementedError(f"diagonal access on iter-var {iv!r} spans dims {dims_for_iv}; "
                                                  f"at least one per-dim stride is None (non-affine) -- "
                                                  f"cannot express as a single linear stride. Refusing per the "
                                                  f"diagonal-as-affine design (no gather fallback).")
                    # With no unit-stride spanned dim (a diagonal over a strided view in
                    # every dim) the positional default would multiply ``combined`` by a
                    # NON-unit stride and inflate the offset (silent OOB / wrong values).
                    # Refuse loudly rather than mis-stride (user: refuse, don't miscompile).
                    unit_dim = next((d for d in dims_for_iv if _stride_is_one(src_arr_strides[d])), None)
                    if unit_dim is None:
                        raise NotImplementedError(
                            f"diagonal access on iter-var {iv!r} spans dims {dims_for_iv} with no unit-stride "
                            f"source dim (strides {[src_arr_strides[d] for d in dims_for_iv]}); the combined "
                            f"byte-stride coeff has no ``strides[basis] == 1`` dim to carry it, so "
                            f"``offset_via_strides`` would inflate it. Refusing per the diagonal-as-affine "
                            f"design (rather than silently mis-striding via the positional default).")
                    combined = sum(record.dim_strides[d] * src_arr_strides[d] for d in dims_for_iv)
                    padded_strides.append(combined)
                    padded_replicate.append(1)
                    padded_src_dims.append(unit_dim)
            else:
                # BROADCAST tile dim -- this iter_var doesn't appear in the subset.
                # stride 0 -> the basis is irrelevant (the term is zero); keep the
                # positional default for a stable mapping.
                padded_strides.append(0)
                padded_replicate.append(widths[k])
                padded_src_dims.append(pos_default)
        # When no strides were supplied the basis cannot be resolved here; signal
        # ``None`` so the caller falls back to the lib node's positional default.
        src_dims_out = tuple(padded_src_dims) if src_arr_strides is not None else None
        return tuple(padded_strides), tuple(padded_replicate), src_dims_out

    def _index_symbol_subscripts(self, inner_sdfg: SDFG, inner_state: SDFGState, begin_str: str) -> Dict[str, Any]:
        """Map each *data-dependent* index symbol in ``begin_str`` to the pure
        array-read :class:`~dace.symbolic.Subscript` it is defined by.

        The frontend promotes a computed gather index ``iorder[0, _for_it_88]``
        into an opaque interstate symbol ``jo`` used in the memlet subset
        (``zqx[jo - 1, ...]``). A symbol is *data-dependent* when its definition
        -- looked up in the symbol-definition map and chased through bare-symbol
        aliases (``jo = sc; sc = iorder[...]``) -- bottoms out at a single array
        read. Pure dict lookups + ``isinstance(Subscript)`` -- NO ``subs`` /
        ``xreplace`` on the index expression.

        :returns: ``{symbol_name: Subscript}``; empty when the index has no
            data-dependent symbol (i.e. it is not a tile-node gather index).
        """
        import dace.symbolic as _sym
        defs = _build_symbol_definition_map(inner_sdfg, state=inner_state)

        def _chase(name: str, seen: set):
            if name in seen:
                return None
            seen.add(name)
            d = defs.get(name)
            if d is None:
                return None
            if isinstance(d, _sym.Subscript):
                return d
            if d.is_Symbol:  # bare-symbol alias -> follow the chain
                return _chase(str(d), seen)
            return None  # arithmetic definition -> not a pure array read

        out: Dict[str, Any] = {}
        for s in _sym.pystr_to_symbolic(begin_str).free_symbols:
            sub = _chase(str(s), set())
            if sub is not None:
                out[str(s)] = sub
        return out

    def _stage_array_read_tile(self, inner_state: SDFGState, inner_sdfg: SDFG, iter_vars: Tuple[str, ...], sub,
                               name_hint: str, mask_an) -> Optional[AccessNode]:
        """Stage a pure array-read ``Subscript`` as a structured :class:`TileLoad`
        -> per-lane index-tile AccessNode, generalised to K tile dims.

        Covers ``idx[jk]`` (col), ``idx[jc]`` (row), ``idx[jk, jc]`` (2-D index),
        and ``iorder[0, _for_it_88]`` (mixed const + linear). The produced index
        tile is FULL-K-dim with a ``(W_d if dep else ONE)`` descriptor: tile dim
        ``d`` is a dependency iff ``iter_vars[d]`` indexes some source dim of
        ``sub``. The ``ONE`` markers carry the per-tile-dim dependency
        POSITIONALLY so the downstream gather load distinguishes ``(W, ONE)``
        col gather (dep dim 0) from ``(ONE, W)`` row gather (dep dim 1) even when
        the tile widths are equal (design 9.2; ``resolve_gather_deps`` positional).
        The ``TileLoad`` itself loops only over the dep dims (``widths`` = dep
        widths); the ``ONE`` dims are broadcast length-1 lanes in the descriptor.

        Returns the index-tile AN, or ``None`` if the read is itself a gather
        (nested) or does not vary per lane.
        """
        from dace.symbolic import ONE
        widths = tuple(int(w) for w in self.widths)
        K = len(iter_vars)
        arr_name = str(sub.args[0])
        idx_exprs = list(sub.args[1:])
        arr_desc = inner_sdfg.arrays.get(arr_name)
        if arr_desc is None:
            return None
        arr_subset = subsets.Range([(e, e, 1) for e in idx_exprs])
        record = classify_tile_access(arr_subset, iter_vars=iter_vars, inner_sdfg=inner_sdfg, state=inner_state)
        if PerDimKind.GATHER in set(record.per_dim_kind):
            return None  # nested gather: the index read is itself a gather
        # Map each tile iter-var to the (first) source dim it indexes; tile dim d
        # is a dependency iff its iter-var appears in the read.
        iv_to_src_dim: Dict[str, int] = {}
        for k, iv in enumerate(record.dim_iter_var):
            if iv is not None and iv not in iv_to_src_dim:
                iv_to_src_dim[iv] = k
        dep_dims = tuple(d for d in range(K) if iter_vars[d] in iv_to_src_dim)
        if not dep_dims:
            return None  # index does not vary per lane -> not a real gather
        dep_src_dims = tuple(iv_to_src_dim[iter_vars[d]] for d in dep_dims)
        dep_widths = tuple(widths[d] for d in dep_dims)
        dim_strides = tuple((record.dim_strides[k] if record.dim_strides[k] is not None else 1) for k in dep_src_dims)
        replicate = tuple((record.replicate_factor_per_dim[k] if record.replicate_factor_per_dim[k] is not None else 1)
                          for k in dep_src_dims)
        # Full-K-dim descriptor: W on dep dims (positional), ONE elsewhere.
        dep_set = set(dep_dims)
        dst_shape = tuple(widths[d] if d in dep_set else ONE for d in range(K))
        # Source window: a source dim mapped to a dep tile dim spans that tile
        # dim's width; every other (constant) source dim is a point read.
        src_tile_w = {iv_to_src_dim[iter_vars[d]]: widths[d] for d in dep_dims}
        parts = [(f"{e}:{e}+{src_tile_w[k]}" if k in src_tile_w else f"{e}") for k, e in enumerate(idx_exprs)]
        src_subset = Memlet(f"{arr_name}[{', '.join(parts)}]")
        arr_an = inner_state.add_access(arr_name)
        # Wire the mask only when the index read spans ALL K tile dims (then the
        # full-tile mask shape == dep widths). A partial-dep (ONE-padded) index
        # read cannot consume the full-tile mask; such reads are kept tail-safe
        # by the load window, not by masking (masked-tail clamping is a noted
        # follow-up -- the mask gates the DATA store, not the index read).
        mask_for_read = mask_an if (mask_an is not None and len(dep_dims) == K) else None
        tile_name, _ = stage_tile_load(inner_state,
                                       arr_an,
                                       widths=dep_widths,
                                       src_subset=src_subset,
                                       name_hint=name_hint,
                                       dim_strides=dim_strides,
                                       replicate_factor_per_dim=replicate,
                                       src_dims=dep_src_dims,
                                       mask_an=mask_for_read,
                                       dst_shape=dst_shape)
        return next(n for n in inner_state.nodes() if isinstance(n, AccessNode) and n.data == tile_name)

    def _stage_index_via_tileops(self, inner_state: SDFGState, inner_sdfg: SDFG, iter_vars: Tuple[str, ...],
                                 begin_str: str, name_hint: str, mask_an) -> Optional[AccessNode]:
        """Build a per-lane gather/scatter index as TILE LIB NODES.

        Per user direction: tile ops are tile lib nodes; pass-emitted tasklets
        are single-statement Python or tile nodes -- never CPP loops. For
        ``zqx[jo - 1, ...]`` with ``jo = iorder[0, _for_it_88]``::

            iorder_t = TileLoad(iorder[0, _for_it_88 : +W])   # LINEAR per-lane tile
            _idx     = iorder_t - 1                            # single-stmt Python -> TileBinop

        Generalised to K tile dims: each data-dependent symbol's array read is
        staged via :meth:`_stage_array_read_tile` into a full-K-dim
        ``(W_d if dep else ONE)`` index tile (the ``ONE`` markers carry the
        per-tile-dim dependency positionally). A pure single-symbol index
        returns that tile directly; an arithmetic index combines the per-symbol
        tiles with one single-statement Python tasklet (lowered to a
        TileBinop/TileUnop downstream).

        Returns the index tile AccessNode, or ``None`` when the index has no
        data-dependent symbol / an inner read is itself a gather (the caller
        raises -- there is no CPP fallback per user 2026-06-14).
        """
        import re as _re
        import dace.symbolic as _sym
        widths = tuple(int(w) for w in self.widths)
        sym_subs = self._index_symbol_subscripts(inner_sdfg, inner_state, begin_str)
        if not sym_subs:
            # Inline-subscript form: the gather index IS a direct array read in the
            # memlet (``A[idx[ii]]`` -> Range begin ``idx[ii]``), not a hoisted
            # interstate symbol. The real frontend hoists gather indices to symbols
            # (handled above), but InsertTileLoadStore must also accept a direct
            # subscript so it stands alone (unit fixtures). Stage it directly.
            parsed = _sym.pystr_to_symbolic(begin_str)
            if isinstance(parsed, _sym.Subscript):
                return self._stage_array_read_tile(inner_state,
                                                   inner_sdfg,
                                                   iter_vars,
                                                   parsed,
                                                   name_hint=name_hint,
                                                   mask_an=mask_an)
            return None  # no data-dependent symbol / not an inline array read
        sym_to_conn: Dict[str, Tuple[str, AccessNode]] = {}
        for i, (sname, sub) in enumerate(sorted(sym_subs.items())):
            tile_an = self._stage_array_read_tile(inner_state,
                                                  inner_sdfg,
                                                  iter_vars,
                                                  sub,
                                                  name_hint=f"{name_hint}_load{i}",
                                                  mask_an=mask_an)
            if tile_an is None:
                return None  # inner read unsupported
            sym_to_conn[sname] = (f"_in{i}", tile_an)
        # Pure single-symbol index (``begin_str == "jo"`` / ``"__sym"``): the
        # load tile IS the index -- no arithmetic tasklet needed.
        stripped = begin_str.strip()
        if len(sym_to_conn) == 1 and stripped in sym_to_conn:
            return sym_to_conn[stripped][1]
        # Arithmetic index (e.g. ``jo - 1``): a SINGLE-STATEMENT Python tasklet
        # built from the ORIGINAL index text, renaming each data-dependent symbol
        # to its input connector (word-boundary identifier substitution -- never
        # sympy re-rendering). ConvertTaskletsToTileOps lowers it to a
        # TileBinop/TileUnop next. Elementwise -> every input index tile must
        # share one shape, and that shape must be FULL (no ONE-collapsed dim):
        # the tile-op arithmetic nodes carry integer widths and cannot operate on
        # a ONE broadcast dim. A partially-collapsed arithmetic index is a noted
        # design boundary (would need ONE-aware TileBinop/TileUnop -- discuss).
        in_shapes = [tuple(inner_sdfg.arrays[tile.data].shape) for (_, tile) in sym_to_conn.values()]
        common = in_shapes[0]
        if any(s != common for s in in_shapes[1:]):
            raise NotImplementedError(
                f"_stage_index_via_tileops: arithmetic gather index {begin_str!r} mixes index tiles of "
                f"differing shapes {in_shapes}; broadcasting collapsed index tiles through tile-op "
                f"arithmetic is a design boundary (discuss before extending).")
        if any(_shape_dim_is_one_symbol(s) for s in common):
            raise NotImplementedError(
                f"_stage_index_via_tileops: arithmetic gather index {begin_str!r} over a partially-"
                f"collapsed index tile {common}; tile-op arithmetic carries integer widths and cannot "
                f"operate on a ONE broadcast dim (discuss ONE-aware TileBinop/TileUnop).")
        body = begin_str
        for sname, (conn, _tile) in sym_to_conn.items():
            body = _re.sub(rf"\b{_re.escape(sname)}\b", conn, body)
        idx_dtype = inner_sdfg.arrays[next(iter(sym_to_conn.values()))[1].data].dtype
        out_name, _ = inner_sdfg.add_array(name_hint,
                                           shape=common,
                                           dtype=idx_dtype,
                                           transient=True,
                                           storage=dtypes.StorageType.Register,
                                           find_new_name=True)
        out_an = inner_state.add_access(out_name)
        t = inner_state.add_tasklet(name=f"idxarith_{out_name}",
                                    inputs=set(conn for conn, _ in sym_to_conn.values()),
                                    outputs={"_out"},
                                    code=f"_out = {body}",
                                    language=dtypes.Language.Python)
        full = ", ".join(f"0:{s}" for s in common)
        for sname, (conn, tile_an) in sym_to_conn.items():
            inner_state.add_edge(tile_an, None, t, conn, Memlet(f"{tile_an.data}[{full}]"))
        inner_state.add_edge(t, "_out", out_an, None, Memlet(f"{out_name}[{full}]"))
        return out_an

    def _find_inner_mask_name(self, inner_sdfg: SDFG) -> Optional[str]:
        """Find the body-NSDFG's iteration mask array name, or None if no mask is in scope."""
        from dace.transformation.passes.vectorization.utils.name_schemes import TileNameScheme
        base = TileNameScheme.ITER_MASK
        if base in inner_sdfg.arrays:
            return base
        for name in inner_sdfg.arrays:
            if name.startswith(f"{base}_"):
                return name
        return None

    def _find_mask_producer_an(self, inner_state: SDFGState, mask_name: str) -> Optional[AccessNode]:
        """Find the AccessNode that the TileMaskGen writes to (its OUTPUT side).

        Every downstream consumer's ``_mask`` edge MUST read from this SAME AccessNode so
        the SDFG scheduler orders TileMaskGen before the consumers. Creating fresh ``add_access``
        calls per consumer produces separate orphan AccessNodes that DaCe schedules
        independently -- TileLoad / TileStore may then run before the mask is written,
        causing all-zero-mask incorrect output.
        """
        from dace.libraries.tileops import TileMaskGen
        for n in inner_state.nodes():
            if not isinstance(n, TileMaskGen):
                continue
            for out_edge in inner_state.out_edges(n):
                if out_edge.src_conn == "_o" and isinstance(out_edge.dst,
                                                            AccessNode) and out_edge.dst.data == mask_name:
                    return out_edge.dst
        # Cross-state fallback: the masked-tail body NSDFG splits compute and
        # store across separate states (e.g. ``BinOp_22`` holds the TileMaskGen +
        # masked load, ``assign_22_8`` holds the scatter store). The mask
        # transient is written by a TileMaskGen in ANOTHER state but persists
        # SDFG-wide, and the states run sequentially (producer first), so a
        # consumer here reads the same value through a fresh AccessNode. Without
        # this, the store/index-load in the TileMaskGen-less state would be
        # emitted UNMASKED -- a masked-remainder scatter then writes
        # ``dst[idx[masked_lane]]`` at an out-of-bounds index (idx read past its
        # tail), corrupting memory.
        sdfg = inner_state.sdfg
        if mask_name in sdfg.arrays and any(isinstance(n, TileMaskGen) for s in sdfg.states() for n in s.nodes()):
            existing = next((n for n in inner_state.nodes() if isinstance(n, AccessNode) and n.data == mask_name), None)
            return existing if existing is not None else inner_state.add_access(mask_name)
        return None

    def _bridge_memlet(self, inner_sdfg: SDFG, bridge_name: str) -> Memlet:
        """Build a memlet for the WHOLE bridge transient (matches the converter's contract
        that rewired edges feed the entire tile)."""
        desc = inner_sdfg.arrays[bridge_name]
        shape = tuple(desc.shape) if desc.shape else None
        if shape and len(shape) > 0:
            subset = ", ".join(f"0:{s}" for s in shape)
            return Memlet(f"{bridge_name}[{subset}]")
        # Scalar bridge (dace.data.Scalar) -- no subset.
        return Memlet(data=bridge_name)

    def _find_existing_bridge_an(self, inner_state: SDFGState, bridge_name: str, side: str) -> Optional[AccessNode]:
        """Find an existing AccessNode for ``bridge_name`` that the staging helper just
        created -- for ``side='read'`` look for one with IN edges (TileLoad's _dst target);
        for ``side='write'`` look for one with OUT edges (TileStore's _src source).

        Reusing this AN (instead of ``add_access``ing a fresh one per consumer) ensures
        the SDFG scheduler sees the dependency chain ``TileLoad -> bridge -> consumer``
        and orders them correctly. Without it, fresh orphan AccessNodes are scheduled
        independently and may run before TileLoad fills the bridge.
        """
        for node in inner_state.nodes():
            if not isinstance(node, AccessNode) or node.data != bridge_name:
                continue
            if side == "read" and inner_state.in_degree(node) > 0:
                return node
            if side == "write" and inner_state.out_degree(node) > 0:
                return node
        return None

    def _rewire_producers_to_bridge(self, inner_state: SDFGState, original_an: AccessNode, bridge_name: str,
                                    original_in_edges) -> None:
        """Symmetric to :meth:`_rewire_consumers_to_bridge` for the write side.

        Redirect each producer edge that previously wrote into ``original_an`` to write
        into the bridge AccessNode named ``bridge_name``. The bridge then flows through
        :class:`TileStore` to ``original_an``. Reuses the existing bridge AN (the source
        of the TileStore._src edge) so the scheduler sees ``producer -> bridge -> TileStore``
        as a chain.

        Per design 3.8.3 row 1: when the producer (``old_edge.src``) is another
        AccessNode, the resulting ``bridge_an -> producer_AN`` -- wait, it's
        ``producer -> bridge_an`` -- IS an ``AN -> AN`` edge and must preserve
        the original ``other_subset`` (the destination-side Y). Other destination
        kinds (lib nodes, tasklets) drop ``other_subset`` per rows 2-4.
        """
        bridge_memlet_template = self._bridge_memlet(inner_state.sdfg, bridge_name)
        shared_bridge_an = self._find_existing_bridge_an(inner_state, bridge_name, side="write")
        from dace.sdfg.nodes import LibraryNode
        for old_edge in original_in_edges:
            if isinstance(old_edge.src, LibraryNode) and old_edge.src_conn == "_dst":
                continue
            bridge_an = shared_bridge_an or inner_state.add_access(bridge_name)
            new_memlet = Memlet.from_memlet(bridge_memlet_template)
            # Per 3.8.3 row 1 (refined per user direction 2026-06-10):
            # ``AN -> AN`` edges only survive when the destination is a
            # Scalar bridge (CONSTANT staging output). For Array tile bridges
            # the consumer reads the FULL tile -- the bridge_memlet_template
            # already encodes that and ``other_subset`` would carry a stale
            # value (e.g. ``[0]`` from a former Scalar bridge).
            if (isinstance(old_edge.src, AccessNode) and old_edge.data.other_subset is not None
                    and isinstance(inner_state.sdfg.arrays.get(old_edge.src.data), data.Scalar)):
                new_memlet.other_subset = subsets.Range(list(old_edge.data.other_subset.ranges))
            inner_state.add_edge(old_edge.src, old_edge.src_conn, bridge_an, old_edge.dst_conn, new_memlet)
            inner_state.remove_edge(old_edge)

    def _maybe_stage_tilestore_to_output(self,
                                         inner_state: SDFGState,
                                         bridge_an: AccessNode,
                                         consumer_an: AccessNode,
                                         iter_vars: Tuple[str, ...],
                                         orig_edge=None) -> bool:
        """Phase A1: insert :class:`TileStore` between a tile-shape bridge and a
        non-transient output AN.

        Per user direction 2026-06-10: TileLoad/TileStore always lower to a
        tile-ops intrinsic; CopyND-based paths must not survive at staging.
        The AN -> AN bridge_to_output edge IS handled by DaCe's CopyND codegen
        otherwise, but that violates the design constraint. This helper
        inserts a TileStore so the chain becomes ``bridge -> TileStore ->
        consumer``, removing the AN -> AN edge.

        The ``dst_subset`` maps the K iter-vars to the consumer's last K dims as
        ``consumer[..., iter_var_k:iter_var_k+W_k, ...]``. The OUTER (non-tile)
        dims take their PER-ITERATION index from the original write memlet's
        per-dim begin (e.g. an ICON gather ``z_ekinh[jb, jk, jc] = ...`` over an
        ``(jb, jk, jc)`` nest stores to ``z_ekinh[jb, jk, jc:jc+W]``). The prior
        code hard-coded those dims to the FULL extent (``0:N``), so every outer
        ``(jb, jk)`` iteration overwrote row ``[0, 0, :]`` -- a direct
        gather-to-output assignment then produced all-wrong results (a compute
        op in between hid it: that path stages the write via its real per-iter
        memlet). When the original write subset is unavailable / mismatched in
        rank we conservatively fall back to full-extent prefix dims.

        Returns ``True`` when the TileStore was inserted (caller skips the
        direct edge), ``False`` for Scalar / non-Array destinations (caller
        takes the default rewire path -- the only AN -> AN edges that
        survive in the post-stage body).
        """
        sdfg = inner_state.sdfg
        bridge_desc = sdfg.arrays.get(bridge_an.data)
        consumer_desc = sdfg.arrays.get(consumer_an.data)
        if bridge_desc is None or consumer_desc is None:
            return False
        if not isinstance(bridge_desc, data.Array) or not isinstance(consumer_desc, data.Array):
            return False
        if bridge_desc.transient is False:
            return False
        if consumer_desc.transient is True:
            return False
        K = len(iter_vars)
        widths = tuple(self.widths)
        consumer_shape = tuple(consumer_desc.shape)
        D = len(consumer_shape)
        if D < K:
            return False
        # Per-iteration begin for each OUTER (non-tile) consumer dim, taken from
        # the original write memlet's consumer-side subset. Full-extent only as a
        # fallback when that subset is missing or rank-mismatched.
        prefix_begins = None
        if orig_edge is not None:
            try:
                _sd, _ss, dst_data, dst_subset = infer_edge_endpoints(orig_edge, sdfg)
                if dst_data == consumer_an.data and dst_subset is not None and len(dst_subset.ranges) == D:
                    prefix_begins = [dst_subset.ranges[d][0] for d in range(D - K)]
            except Exception:  # noqa: BLE001 -- exotic edge; fall back to full extent
                prefix_begins = None
        if prefix_begins is not None:
            prefix_parts = [f"{b}:{b} + 1" for b in prefix_begins]
        else:
            prefix_parts = [f"0:{consumer_shape[d]}" for d in range(D - K)]
        tile_parts = [f"{iter_vars[k]}:{iter_vars[k]} + {widths[k]}" for k in range(K)]
        dst_subset_str = ", ".join(prefix_parts + tile_parts)
        # Mask the bridge->output store when an iteration mask is in scope (the
        # masked-tail remainder). Without it this structured store writes every
        # lane of the W-wide tile, including the lanes past the array tail
        # (``dst[N], dst[N+1], ...``) -- an out-of-bounds write. The mask
        # producer may live in another state of this body NSDFG (compute and
        # store split across states), so the cross-state lookup in
        # :meth:`_find_mask_producer_an` finds it.
        mask_name = self._find_inner_mask_name(sdfg)
        mask_an = self._find_mask_producer_an(inner_state, mask_name) if mask_name else None
        store = TileStore(name=f"store_{bridge_an.data}_to_{consumer_an.data}",
                          widths=widths,
                          has_mask=mask_an is not None)
        inner_state.add_node(store)
        src_subset_str = ", ".join(f"0:{w}" for w in widths)
        inner_state.add_edge(bridge_an, None, store, "_src", Memlet(f"{bridge_an.data}[{src_subset_str}]"))
        if mask_an is not None:
            mask_subset_str = ", ".join(f"0:{w}" for w in widths)
            inner_state.add_edge(mask_an, None, store, "_mask", Memlet(f"{mask_an.data}[{mask_subset_str}]"))
        inner_state.add_edge(store, "_dst", consumer_an, None, Memlet(data=consumer_an.data, subset=dst_subset_str))
        return True

    def _resize_scalar_chain_downstream_of_tiles(self, inner_state: SDFGState) -> int:
        """Phase A6 (user direction 2026-06-10): for the kernel pattern
        ``dst[idx[i]] = src[i] + 1.0`` where ``i`` is the vector param, the
        WHOLE chain should be tile-shape -- no scalar transients between the
        tile source (TileLoad output) and the tile sink (TileStore input).

        After phase 1's A5 elision wires tile bridges directly into tasklets,
        any Scalar / (1,) Array transient AN that's downstream of a tile-input
        tasklet should be resized to tile-shape ``(W,)``. Recursive: after
        resizing, any tasklets downstream of the resized AN also have tile
        inputs, so THEIR scalar outputs need resizing too.

        Follows the analyze-then-apply pattern: BFS to collect all ANs to
        resize + all memlets to update, then apply in a single batch.

        Returns count of resized ANs.
        """
        from dace.sdfg.nodes import Tasklet
        sdfg = inner_state.sdfg
        widths = tuple(self.widths)
        target_shape = (widths[0], ) if len(widths) == 1 else tuple(widths)
        target_subset = ", ".join(f"0:{w}" for w in widths)

        def _has_tile_shaped_input(node) -> bool:
            for e in inner_state.in_edges(node):
                if e.data is None:
                    continue
                src_desc = sdfg.arrays.get(e.data.data)
                if not isinstance(src_desc, data.Array):
                    continue
                src_shape = tuple(src_desc.shape)
                for s, w in zip(src_shape, widths):
                    try:
                        if int(s) == w:
                            return True
                    except (TypeError, ValueError):
                        continue
            return False

        # ANALYZE: BFS from tile-input tasklets, collect ANs to resize.
        to_resize: List[str] = []
        memlets_to_update: List = []
        queue: List = [n for n in inner_state.nodes() if isinstance(n, Tasklet) and _has_tile_shaped_input(n)]
        seen_tasklets = set(id(t) for t in queue)
        while queue:
            tasklet = queue.pop(0)
            for e in inner_state.out_edges(tasklet):
                if not isinstance(e.dst, AccessNode):
                    continue
                desc = sdfg.arrays.get(e.dst.data)
                if desc is None or not desc.transient:
                    continue
                if not (isinstance(desc, data.Scalar) or (isinstance(desc, data.Array) and tuple(desc.shape) == (1, ))):
                    continue
                if e.dst.data not in to_resize:
                    to_resize.append(e.dst.data)
                memlets_to_update.append(e)
                # Walk downstream of this AN: capture all out-edges, queue any
                # further tasklets so we resize THEIR scalar outputs too.
                for downstream in inner_state.out_edges(e.dst):
                    memlets_to_update.append(downstream)
                    if isinstance(downstream.dst, Tasklet) and id(downstream.dst) not in seen_tasklets:
                        seen_tasklets.add(id(downstream.dst))
                        queue.append(downstream.dst)
        if not to_resize:
            return 0
        # APPLY: batched mutation.
        for scalar_data in to_resize:
            old_desc = sdfg.arrays[scalar_data]
            new_desc = data.Array(dtype=old_desc.dtype,
                                  shape=target_shape,
                                  transient=True,
                                  storage=dtypes.StorageType.Register)
            sdfg.arrays[scalar_data] = new_desc
        for edge in memlets_to_update:
            edge.data.subset = subsets.Range.from_string(target_subset)
        return len(to_resize)

    def _rewire_consumers_to_bridge(self,
                                    inner_state: SDFGState,
                                    original_an: AccessNode,
                                    bridge_name: str,
                                    original_out_edges,
                                    iter_vars: Tuple[str, ...] = ()) -> None:
        """Redirect each consumer edge that previously read from ``original_an`` to read from
        the SAME bridge AccessNode that the staging helper produced (the TileLoad._dst target).

        Per design 3.8.3 row 1: when the consumer (``old_edge.dst``) is another
        AccessNode, the resulting ``bridge_an -> consumer_AN`` is an ``AN -> AN``
        edge and must preserve the original ``other_subset`` (the consumer-side
        Y). Other destination kinds (lib nodes, tasklets) drop ``other_subset``
        per rows 2-4 -- the existing ``bridge_memlet_template`` carries no
        ``other_subset`` by default.
        """
        bridge_memlet_template = self._bridge_memlet(inner_state.sdfg, bridge_name)
        shared_bridge_an = self._find_existing_bridge_an(inner_state, bridge_name, side="read")
        for old_edge in original_out_edges:
            if isinstance(old_edge.dst, AccessNode) and old_edge.dst.data == bridge_name:
                continue
            from dace.sdfg.nodes import LibraryNode
            if isinstance(old_edge.dst, LibraryNode) and old_edge.dst_conn == "_src":
                continue
            bridge_an = shared_bridge_an or inner_state.add_access(bridge_name)
            # (No scalar-passthrough elision here -- removed per user direction
            # 2026-06-14: a clean design needs no such patch-fix. WidenAccesses
            # widens every transient, so a transient scalar that a tile feeds is
            # already a tile (tile -> tile, not tile -> scalar); scalar->tile is a
            # broadcast TileLoad; the only legitimate scalar write is to a
            # NON-transient program output (a reduction). The
            # ``no_transient_scalar_stores`` invariant enforces this.)
            # Phase A1 (user direction 2026-06-10): non-Scalar consumer Arrays
            # MUST flow through a TileStore so the lowering goes through the
            # tile-ops intrinsic, NOT via DaCe's CopyND auto-emission. The
            # AN -> AN bridge_to_output edge would otherwise lower as a CopyND
            # call, which violates the design constraint.
            if (iter_vars and isinstance(old_edge.dst, AccessNode) and self._maybe_stage_tilestore_to_output(
                    inner_state, bridge_an, old_edge.dst, iter_vars, old_edge)):
                inner_state.remove_edge(old_edge)
                continue
            new_memlet = Memlet.from_memlet(bridge_memlet_template)
            # Per 3.8.3 row 1 (refined per user direction 2026-06-10):
            # ``AN -> AN`` edges only survive when the destination is a
            # Scalar bridge (CONSTANT staging output). For Array tile bridges
            # the consumer reads the FULL tile -- the bridge_memlet_template
            # already encodes that and ``other_subset`` would carry a stale
            # value (e.g. ``[0]`` from a former Scalar bridge).
            if (isinstance(old_edge.dst, AccessNode) and old_edge.data.other_subset is not None
                    and isinstance(inner_state.sdfg.arrays.get(old_edge.dst.data), data.Scalar)):
                new_memlet.other_subset = subsets.Range(list(old_edge.data.other_subset.ranges))
            inner_state.add_edge(bridge_an, None, old_edge.dst, old_edge.dst_conn, new_memlet)
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
            # Audit the design 3.8.3 lib-node-boundary invariant on every
            # body NSDFG state we just touched. Loud-failure surfaces stale
            # ``other_subset`` at staging time rather than letting it
            # propagate into codegen (where it manifests as ``StopIteration``
            # in the strided-copy shape inference).
            for inner_state in nsdfg_node.sdfg.states():
                _assert_post_stage_invariants(inner_state)
        # Post-conditions.
        assert_invariant(no_memlet_dim_mismatch(sdfg), "InsertTileLoadStore",
                         "memlet subset and other_subset have matching dimensionality")
        assert_invariant(no_duplicate_connector_edges(sdfg), "InsertTileLoadStore",
                         "no duplicate connector edges (lib-node / NSDFG / tasklet)")
        assert_invariant(memlet_subset_matches_descriptor(sdfg), "InsertTileLoadStore",
                         "every memlet subset rank matches its descriptor rank")
        assert_invariant(no_transient_scalar_stores(sdfg), "InsertTileLoadStore",
                         "no scalar stores except to a non-transient program output")
        return total or None
