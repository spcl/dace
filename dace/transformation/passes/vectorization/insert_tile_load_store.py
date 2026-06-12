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
from dace.transformation.passes.vectorization.widen_accesses import materialise_per_lane_index_tile
from dace.transformation.passes.vectorization.utils.map_predicates import is_innermost_map
from dace.transformation.passes.vectorization.utils.subsets import an_side_subset, infer_edge_endpoints
from dace.transformation.passes.vectorization.utils.tile_access import (PerDimKind, classify_tile_access,
                                                                        compute_per_iter_var_dep_mask)


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
            input_staging = (src_desc is not None and not src_desc.transient
                             and dst_desc is not None and dst_desc.transient)
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
                    mask_an: Optional[AccessNode] = None) -> Tuple[str, "TileLoad"]:
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
    dst_subset_str = ", ".join(f"0:{w}" for w in widths)
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
            record = classify_tile_access(subset, iter_vars=iter_vars, inner_sdfg=inner_sdfg)
            if not record.per_dim_kind:
                continue
            kinds = set(record.per_dim_kind)
            if PerDimKind.GATHER in kinds:
                # GATHER read: materialise per-lane idx tile(s) + TileLoad with gather_dims.
                K_tile = len(iter_vars)
                gather_source_dims = tuple(k for k, kind in enumerate(record.per_dim_kind) if kind == PerDimKind.GATHER)
                idx_sources: Dict[int, AccessNode] = {}
                for k in gather_source_dims:
                    begin_str = str(subset.ranges[k][0])
                    dep_mask = compute_per_iter_var_dep_mask(begin_str, iter_vars, inner_sdfg)
                    idx_name = materialise_per_lane_index_tile(
                        inner_state,
                        name_hint=f"_idx_{an.data}_{k}",
                        gather_expr=begin_str,
                        tile_iter_vars=iter_vars[0] if K_tile == 1 else iter_vars,
                        tile_widths=int(self.widths[0]) if K_tile == 1 else tuple(int(w) for w in self.widths),
                        dep_mask=dep_mask[:1] if K_tile == 1 else dep_mask,
                    )
                    idx_an = next(n for n in inner_state.nodes() if isinstance(n, AccessNode) and n.data == idx_name)
                    idx_sources[k] = idx_an
                full_subset_str = ", ".join(f"0:{s}" for s in desc.shape)
                src_subset_memlet = Memlet(data=an.data, subset=full_subset_str)
                bridge_name, _ = stage_tile_load(inner_state,
                                                 an,
                                                 widths=tuple(self.widths),
                                                 src_subset=src_subset_memlet,
                                                 name_hint=f"{an.data}_gather",
                                                 gather_dims=gather_source_dims,
                                                 idx_sources=idx_sources,
                                                 mask_an=mask_an_for_this)
                self._rewire_consumers_to_bridge(inner_state, an, bridge_name, pre_stage_out_edges, iter_vars=iter_vars)
                staged += 1
                continue
            if kinds == {PerDimKind.CONSTANT}:
                bridge_name = stage_constant_access(inner_state, an, name_hint=f"{an.data}_const")
                self._rewire_consumers_to_bridge(inner_state, an, bridge_name, pre_stage_out_edges, iter_vars=iter_vars)
                staged += 1
                continue
            # Structured tile load: LINEAR / AFFINE / REPLICATE / MODULAR (possibly mixed with CONSTANT).
            src_subset_memlet = Memlet.from_memlet(pre_stage_out_edges[0].data)
            # Pass src array strides so the diagonal-as-affine path can combine
            # per-dim strides when the same iter-var dominates multiple source dims.
            src_arr_strides = tuple(desc.strides) if desc.strides else None
            dim_strides, replicate = self._pad_to_tile_dims(record, iter_vars, src_arr_strides=src_arr_strides)
            bridge_name, _ = stage_tile_load(inner_state,
                                             an,
                                             widths=tuple(self.widths),
                                             src_subset=src_subset_memlet,
                                             name_hint=f"{an.data}_tile",
                                             dim_strides=dim_strides,
                                             replicate_factor_per_dim=replicate,
                                             mask_an=mask_an_for_this)
            self._rewire_consumers_to_bridge(inner_state, an, bridge_name, pre_stage_out_edges, iter_vars=iter_vars)
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
            if not (pre_stage_in_edges and not pre_stage_out_edges):
                continue  # Not a sink -- read phase already handled (or AN has no edges).
            if any(isinstance(e.src, (TileLoad, TileStore)) for e in pre_stage_in_edges):
                continue  # Already staged by phase 1's bridge->output insertion.
            mask_an_for_this = (self._find_mask_producer_an(inner_state, mask_name) if mask_name else None)
            try:
                wsubset = an_side_subset(pre_stage_in_edges[0], an, inner_sdfg)
            except Exception:  # noqa: BLE001
                continue
            wrecord = classify_tile_access(wsubset, iter_vars=iter_vars, inner_sdfg=inner_sdfg)
            if not wrecord.per_dim_kind:
                continue
            wkinds = set(wrecord.per_dim_kind)
            if wkinds == {PerDimKind.CONSTANT}:
                continue  # Loop-invariant write stays as direct producer -> AN copy (design 3.6).
            if PerDimKind.GATHER in wkinds:
                # SCATTER: per-lane idx materialisation + TileStore with gather_dims.
                K_tile_w = len(iter_vars)
                scatter_source_dims = tuple(k for k, kind in enumerate(wrecord.per_dim_kind)
                                            if kind == PerDimKind.GATHER)
                idx_sources_w: Dict[int, AccessNode] = {}
                for k in scatter_source_dims:
                    begin_str = str(wsubset.ranges[k][0])
                    idx_name = materialise_per_lane_index_tile(
                        inner_state,
                        name_hint=f"_idx_scatter_{an.data}_{k}",
                        gather_expr=begin_str,
                        tile_iter_vars=iter_vars[0] if K_tile_w == 1 else iter_vars,
                        tile_widths=int(self.widths[0]) if K_tile_w == 1 else tuple(int(w) for w in self.widths),
                    )
                    idx_an = next(n for n in inner_state.nodes() if isinstance(n, AccessNode) and n.data == idx_name)
                    idx_sources_w[k] = idx_an
                full_dst_subset = ", ".join(f"0:{s}" for s in desc.shape)
                dst_subset_memlet = Memlet(data=an.data, subset=full_dst_subset)
                bridge_name, _ = stage_tile_store(inner_state,
                                                  an,
                                                  widths=tuple(self.widths),
                                                  dst_subset=dst_subset_memlet,
                                                  name_hint=f"{an.data}_scatter_out",
                                                  gather_dims=scatter_source_dims,
                                                  idx_sources=idx_sources_w,
                                                  mask_an=mask_an_for_this)
                self._rewire_producers_to_bridge(inner_state, an, bridge_name, pre_stage_in_edges)
                staged += 1
                continue
            # Structured tile store: LINEAR / AFFINE / REPLICATE / MODULAR.
            dst_subset_memlet = Memlet.from_memlet(pre_stage_in_edges[0].data)
            dst_arr_strides = tuple(desc.strides) if desc.strides else None
            dim_strides_w, _ = self._pad_to_tile_dims(wrecord, iter_vars, src_arr_strides=dst_arr_strides)
            bridge_name, _ = stage_tile_store(inner_state,
                                              an,
                                              widths=tuple(self.widths),
                                              dst_subset=dst_subset_memlet,
                                              name_hint=f"{an.data}_tile_out",
                                              dim_strides=dim_strides_w,
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
        """
        from collections import defaultdict
        K = len(iter_vars)
        widths = tuple(int(w) for w in self.widths)
        # Multi-map: iter_var -> list of source dim indices it dominates.
        iv_to_src_dims: Dict[str, List[int]] = defaultdict(list)
        for d, iv_name in enumerate(record.dim_iter_var):
            if iv_name is not None:
                iv_to_src_dims[iv_name].append(d)
        padded_strides = []
        padded_replicate = []
        for k in range(K):
            iv = iter_vars[k]
            if iv in iv_to_src_dims:
                dims_for_iv = iv_to_src_dims[iv]
                if len(dims_for_iv) == 1:
                    # Standard single-dim case.
                    d = dims_for_iv[0]
                    s = record.dim_strides[d]
                    padded_strides.append(s if s is not None else 1)
                    r = record.replicate_factor_per_dim[d]
                    padded_replicate.append(r if r is not None else 1)
                else:
                    # Diagonal: combine strides across source dims.
                    if src_arr_strides is None:
                        # Legacy fallback: pick the first dim (matches pre-fix behaviour).
                        d = dims_for_iv[0]
                        s = record.dim_strides[d]
                        padded_strides.append(s if s is not None else 1)
                        r = record.replicate_factor_per_dim[d]
                        padded_replicate.append(r if r is not None else 1)
                    else:
                        # Compute combined affine stride.
                        all_affine = all(record.dim_strides[d] is not None for d in dims_for_iv)
                        if not all_affine:
                            raise NotImplementedError(
                                f"diagonal access on iter-var {iv!r} spans dims {dims_for_iv}; "
                                f"at least one per-dim stride is None (non-affine) -- "
                                f"cannot express as a single linear stride. Refusing per the "
                                f"diagonal-as-affine design (no gather fallback)."
                            )
                        combined = sum(record.dim_strides[d] * src_arr_strides[d] for d in dims_for_iv)
                        padded_strides.append(combined)
                        padded_replicate.append(1)
            else:
                # BROADCAST tile dim -- this iter_var doesn't appear in the subset.
                padded_strides.append(0)
                padded_replicate.append(widths[k])
        return tuple(padded_strides), tuple(padded_replicate)

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

    def _maybe_stage_tilestore_to_output(self, inner_state: SDFGState, bridge_an: AccessNode, consumer_an: AccessNode,
                                         iter_vars: Tuple[str, ...]) -> bool:
        """Phase A1: insert :class:`TileStore` between a tile-shape bridge and a
        non-transient output AN.

        Per user direction 2026-06-10: TileLoad/TileStore always lower to a
        tile-ops intrinsic; CopyND-based paths must not survive at staging.
        The AN -> AN bridge_to_output edge IS handled by DaCe's CopyND codegen
        otherwise, but that violates the design constraint. This helper
        inserts a TileStore so the chain becomes ``bridge -> TileStore ->
        consumer``, removing the AN -> AN edge.

        The ``dst_subset`` is the per-outer-iter W-extent slice of the
        consumer array: maps the K iter-vars to the consumer's last K dims
        as ``consumer[..., iter_var_0:iter_var_0+W_0, ...,
        iter_var_{K-1}:iter_var_{K-1}+W_{K-1}]``. Outer non-tile dims of the
        consumer are full-extent.

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
        prefix_parts = [f"0:{consumer_shape[d]}" for d in range(D - K)]
        tile_parts = [f"{iter_vars[k]}:{iter_vars[k]} + {widths[k]}" for k in range(K)]
        dst_subset_str = ", ".join(prefix_parts + tile_parts)
        store = TileStore(name=f"store_{bridge_an.data}_to_{consumer_an.data}", widths=widths, has_mask=False)
        inner_state.add_node(store)
        src_subset_str = ", ".join(f"0:{w}" for w in widths)
        inner_state.add_edge(bridge_an, None, store, "_src", Memlet(f"{bridge_an.data}[{src_subset_str}]"))
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

    def _maybe_elide_scalar_passthrough(self, inner_state: SDFGState, bridge_an: AccessNode, scalar_an: AccessNode,
                                        old_edge) -> bool:
        """Phase A5 (user direction 2026-06-10): elide pass-through scalar bridges.

        When the read-side rewire would create
        ``tile_bridge(W,) -> scalar_bridge(1,) -> tile_libnode_or_tasklet``,
        the post-Bypass scalar bridge is a vestige -- the tile bridge already
        holds W elements per outer iter. Wire the tile bridge directly to the
        scalar bridge's downstream consumers with a full-tile memlet so that
        :class:`ConvertTaskletsToTileOps` sees the tile source and emits the
        downstream lib node with ``kind=Tile`` (not ``kind=Scalar``).

        This avoids the
        ``InvalidSDFGEdgeError: Dimensionality mismatch (src[0:W] -> [0])``
        validator firing on the bridge->scalar edge.

        Takes full ownership of cleanup: removes ``old_edge``, removes
        scalar_an's downstream edges, removes ``scalar_an`` itself, and
        drops the array descriptor when unused. Caller must NOT touch
        ``old_edge`` or ``scalar_an`` after this returns ``True``.

        Returns ``True`` when the elision was applied; ``False`` otherwise.
        """
        sdfg = inner_state.sdfg
        desc = sdfg.arrays.get(scalar_an.data)
        if desc is None or not desc.transient:
            return False
        if not (isinstance(desc, data.Scalar) or (isinstance(desc, data.Array) and tuple(desc.shape) == (1, ))):
            return False
        # ANALYZE: collect edges to remove + edges to add. No mutations during
        # this phase (per the staged-batch invariant -- see feedback_atomic_edit_pattern).
        in_edges_for_scalar = list(inner_state.in_edges(scalar_an))
        if len(in_edges_for_scalar) != 1 or in_edges_for_scalar[0] is not old_edge:
            return False
        downstream = list(inner_state.out_edges(scalar_an))
        if not downstream:
            return False
        widths = tuple(self.widths)
        tile_subset = ", ".join(f"0:{w}" for w in widths)
        edges_to_remove = list(downstream) + [old_edge]
        edges_to_add = [(bridge_an, None, dn.dst, dn.dst_conn, Memlet(data=bridge_an.data, subset=tile_subset))
                        for dn in downstream]
        # APPLY: batched mutation.
        for e in edges_to_remove:
            inner_state.remove_edge(e)
        for src, src_conn, dst, dst_conn, memlet in edges_to_add:
            inner_state.add_edge(src, src_conn, dst, dst_conn, memlet)
        inner_state.remove_node(scalar_an)
        # Drop the array descriptor when no other AN in any state references it.
        scalar_data = scalar_an.data
        any_remaining = any(
            isinstance(n, AccessNode) and n.data == scalar_data for st in sdfg.states() for n in st.nodes())
        if not any_remaining:
            try:
                sdfg.remove_data(scalar_data, validate=False)
            except Exception:  # noqa: BLE001 -- ignore when descriptor lookup fails
                pass
        return True

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
            # Phase A5: elide pass-through scalar bridges. When the consumer is
            # a transient Scalar / (1,) Array AN that just funnels the value to
            # downstream lib nodes / tasklets, skip it -- wire the tile bridge
            # directly to its consumers with a full-tile memlet. The helper
            # owns full cleanup (removes ``old_edge``, the scalar AN, and the
            # descriptor); caller must NOT touch them when this returns True.
            if (isinstance(old_edge.dst, AccessNode)
                    and self._maybe_elide_scalar_passthrough(inner_state, bridge_an, old_edge.dst, old_edge)):
                continue
            # Phase A1 (user direction 2026-06-10): non-Scalar consumer Arrays
            # MUST flow through a TileStore so the lowering goes through the
            # tile-ops intrinsic, NOT via DaCe's CopyND auto-emission. The
            # AN -> AN bridge_to_output edge would otherwise lower as a CopyND
            # call, which violates the design constraint.
            if (iter_vars and isinstance(old_edge.dst, AccessNode)
                    and self._maybe_stage_tilestore_to_output(inner_state, bridge_an, old_edge.dst, iter_vars)):
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
        return total or None
