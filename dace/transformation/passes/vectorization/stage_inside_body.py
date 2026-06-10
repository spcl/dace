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
from collections import deque
from typing import Any, Dict, List, Optional, Tuple

from dace import data, dtypes, properties, subsets
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
            # Allow AN -> AN only when at least one endpoint is a Scalar
            # transient (CONSTANT bridge produced by ``stage_constant_access``).
            both_scalar_or_one_scalar = (isinstance(src_desc, data.Scalar) or isinstance(dst_desc, data.Scalar))
            if not both_scalar_or_one_scalar:
                raise AssertionError(f"design 3.8.3 (2) violation: AN -> AN edge survives staging "
                                     f"but neither endpoint is a Scalar bridge (src={edge.src.data!r} "
                                     f"-> dst={edge.dst.data!r}, data={mem.data!r}, subset={mem.subset!r}) "
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


def _topo_sort_access_nodes(state: SDFGState) -> List[AccessNode]:
    """Topo-sort AccessNodes by intra-state dataflow predecessors.

    Sources (no AccessNode predecessor) come first; each AccessNode is processed
    only after every AccessNode that writes into it. This is the iteration order
    the walker's staging requires -- a transient bridge AccessNode (e.g. a Scalar
    sink of a gather like ``A_const = A[__sym]``) must be classified AFTER its
    upstream non-transient source ``A`` has been staged, otherwise the bridge's
    own subset (``[0]``, classified CONSTANT) would absorb the gather edge before
    the walker sees it.

    Cycles among AccessNodes (not expected in well-formed SDFGs) are tolerated:
    any AccessNode left out of the topological order after Kahn's algorithm is
    appended at the end.
    """
    ans = [n for n in state.nodes() if isinstance(n, AccessNode)]
    in_deg = {n: 0 for n in ans}
    for n in ans:
        for e in state.in_edges(n):
            if isinstance(e.src, AccessNode) and e.src in in_deg:
                in_deg[n] += 1
    queue = deque(n for n in ans if in_deg[n] == 0)
    sorted_ans: List[AccessNode] = []
    seen: set = set()
    while queue:
        cur = queue.popleft()
        if cur in seen:
            continue
        sorted_ans.append(cur)
        seen.add(cur)
        for e in state.out_edges(cur):
            if isinstance(e.dst, AccessNode) and e.dst in in_deg:
                in_deg[e.dst] -= 1
                if in_deg[e.dst] == 0:
                    queue.append(e.dst)
    for n in ans:
        if n not in seen:
            sorted_ans.append(n)
    return sorted_ans


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
        # The iteration mask (per design 6.5) lives INSIDE the body NSDFG as an
        # ``_tile_iter_mask`` AccessNode produced by :class:`TileMaskGen` (emitted by
        # :class:`GenerateTileIterationMask`). Find it once per body NSDFG; each staging
        # call gets a fresh AccessNode (a new ``inner_state.add_access`` of the same name)
        # so the read edge is unambiguous.
        mask_name = self._find_inner_mask_name(inner_sdfg)
        for inner_state in inner_sdfg.states():
            # Topo iteration: process AccessNodes in dataflow order (sources first).
            # A transient bridge whose value comes from a lane-dep memlet (e.g.
            # ``A_const = A[__sym]``) must be reached AFTER its upstream source has
            # been staged; otherwise the bridge's local subset (``A_const[0]``,
            # CONSTANT) absorbs the gather edge before the walker sees it.
            for an in _topo_sort_access_nodes(inner_state):
                desc = inner_sdfg.arrays.get(an.data)
                if desc is None or desc.transient:
                    continue
                # Capture ORIGINAL consumer edges before staging adds the bridge edge -- the
                # rewire below redirects each original consumer to read from the bridge AN
                # instead of the non-transient ``an``. Without this, the bridge transient
                # dangles and tasklets continue to read directly from the global array.
                pre_stage_out_edges = list(inner_state.out_edges(an))
                pre_stage_in_edges = list(inner_state.in_edges(an))
                # Reuse the SAME AccessNode that TileMaskGen writes to (per-state). Creating
                # fresh add_access() calls per consumer produces orphan AccessNodes that
                # DaCe's scheduler doesn't order after TileMaskGen, causing zero-mask bugs.
                mask_an_for_this = (self._find_mask_producer_an(inner_state, mask_name) if mask_name else None)
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
                    if wkinds == {PerDimKind.CONSTANT}:
                        # Per user direction 2026-06-09: CONSTANT-only writes stay as the
                        # original direct producer -> AN copy. Symmetric to the read side
                        # where CONSTANT-only reads stay as a direct AN -> Scalar copy
                        # (no TileLoad lib node). No transformation required; the writer's
                        # output already targets a single element (loop-invariant subset),
                        # and the existing producer edge handles it correctly.
                        continue
                    if PerDimKind.GATHER in wkinds:
                        # SCATTER case (symmetric to the read-side GATHER branch). Per
                        # design section 9.2 ``gather_dims`` indexes destination-array dim
                        # indices on TileStore; the subset dim ``k`` corresponds to dest
                        # dim ``k`` for the canonical (non-permuted) case. Materialise the
                        # per-lane index tile per scatter dim, then call ``stage_tile_store``
                        # with ``gather_dims`` + ``idx_sources``. Each lane writes its tile
                        # value to ``dst[base + _idx_<k>[lane]]``. The non-full-tile-write
                        # lock is exempt in scatter mode (commit b9173e366): the dest memlet
                        # legitimately covers the full destination array; per-lane addressing
                        # comes from ``_idx_<k>``.
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
                                tile_widths=int(self.widths[0]) if K_tile_w == 1 else tuple(
                                    int(w) for w in self.widths),
                            )
                            idx_an = next(n for n in inner_state.nodes()
                                          if isinstance(n, AccessNode) and n.data == idx_name)
                            idx_sources_w[k] = idx_an
                        # For scatter, ``_dst`` must be wired as the FULL destination
                        # array (mirror of the gather widening at line 541). DaCe
                        # codegen passes a length-1 connector by-value (``T _dst``),
                        # which is unindexable; the scatter expansion needs a
                        # pointer (``T*``) it can subscript by the per-lane indices.
                        # Widen the dst memlet to the full descriptor extent.
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
                    dst_subset_memlet = Memlet.from_memlet(pre_stage_in_edges[0].data)
                    dim_strides_w, _ = self._pad_to_tile_dims(wrecord, iter_vars)
                    bridge_name, _ = stage_tile_store(inner_state,
                                                      an,
                                                      widths=tuple(self.widths),
                                                      dst_subset=dst_subset_memlet,
                                                      name_hint=f"{an.data}_tile_out",
                                                      dim_strides=dim_strides_w,
                                                      mask_an=mask_an_for_this)
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
                    # For gather, ``_src`` must be wired as the FULL source array (not
                    # the original single-element gather subset). DaCe codegen passes a
                    # length-1 connector by-value (``T _src``), which is unindexable;
                    # the gather expansion needs a pointer (``T*``) it can subscript by
                    # the per-lane indices. Widen the src memlet to the full descriptor
                    # extent on every dim so the connector lowers as an array view.
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
                    dim_strides, replicate = self._pad_to_tile_dims(record, iter_vars)
                    bridge_name, _ = stage_tile_load(inner_state,
                                                     an,
                                                     widths=tuple(self.widths),
                                                     src_subset=src_subset_memlet,
                                                     name_hint=f"{an.data}_tile",
                                                     dim_strides=dim_strides,
                                                     replicate_factor_per_dim=replicate,
                                                     mask_an=mask_an_for_this)
                    self._rewire_consumers_to_bridge(inner_state, an, bridge_name, pre_stage_out_edges)
                    staged += 1
        return staged

    def _pad_to_tile_dims(self, record, iter_vars: Tuple[str, ...]):
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
        """
        K = len(iter_vars)
        widths = tuple(int(w) for w in self.widths)
        # Build reverse map: iter_var -> source dim index that this iter_var dominates.
        iv_to_src_dim: Dict[str, int] = {}
        for d, iv_name in enumerate(record.dim_iter_var):
            if iv_name is not None and iv_name not in iv_to_src_dim:
                iv_to_src_dim[iv_name] = d
        padded_strides = []
        padded_replicate = []
        for k in range(K):
            iv = iter_vars[k]
            if iv in iv_to_src_dim:
                d = iv_to_src_dim[iv]
                s = record.dim_strides[d]
                padded_strides.append(s if s is not None else 1)
                r = record.replicate_factor_per_dim[d]
                padded_replicate.append(r if r is not None else 1)
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
        shape = tuple(desc.shape) if hasattr(desc, "shape") and desc.shape else None
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
        for old_edge in original_in_edges:
            if hasattr(old_edge.src, "label") and old_edge.src_conn == "_dst":
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

    def _rewire_consumers_to_bridge(self, inner_state: SDFGState, original_an: AccessNode, bridge_name: str,
                                    original_out_edges) -> None:
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
            if hasattr(old_edge.dst, "label") and old_edge.dst_conn == "_src":
                continue
            bridge_an = shared_bridge_an or inner_state.add_access(bridge_name)
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
