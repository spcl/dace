# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""cuTile lowering passes: tileops-anchored schedule, storage, and
implementation stamping for the Python/cuTile backend.

"""

import warnings
from typing import Any, Dict, Iterator, List, Optional, Set, Tuple, Type

from dace import SDFG, data, dtypes, properties, transformation
from dace.sdfg import nodes
from dace.sdfg.state import SDFGState
from dace.transformation import pass_pipeline as ppl

#: Scope-dictionary cache type: per-state ``state.scope_dict()`` results.
_ScopeCache = Dict[SDFGState, Dict[nodes.Node, Optional[nodes.Node]]]

#: Maximum provable map volume (total trip count) for which a residual
#: ``GPU_Device`` map is demoted to ``Sequential`` silently.  Such tiny maps
#: are the intended scalar-control steps of sequential solvers (e.g.
#: ``cholesky`` / ``trisolv``); anything larger (or symbolic, hence not
#: provably tiny) compiles to a pathological per-element host loop over GPU
#: data and is diagnosed via :func:`_warn_or_raise`.
_TINY_DEMOTION_VOLUME = 4


def _tile_node_types() -> Tuple[Type[nodes.LibraryNode], ...]:
    """Return the tuple of all tileops library-node classes.

    Imported lazily to keep the transformation package importable without
    pulling in the tileops library at module-import time.

    :returns: :data:`dace.libraries.tileops.nodes.TILE_NODES`.
    """
    from dace.libraries.tileops.nodes import TILE_NODES

    return TILE_NODES


def _collect_tile_nodes(sdfg: SDFG) -> List[Tuple[nodes.LibraryNode, SDFGState]]:
    """Collect every tileops library node in ``sdfg``, recursively.

    :param sdfg: SDFG to search (NestedSDFGs are included).
    :returns: List of ``(lib_node, state)`` pairs, where ``state`` is the
        SDFGState that owns the node.
    """
    tile_types = _tile_node_types()
    return [(node, graph) for node, graph in sdfg.all_nodes_recursive() if isinstance(node, tile_types)]


def _collect_non_tile_library_nodes(sdfg: SDFG) -> List[Tuple[nodes.LibraryNode, SDFGState]]:
    """Collect every non-tileops library node in ``sdfg``, recursively.

    These are library nodes that the tileops-anchored lowering passes leave
    untouched -- most importantly BLAS ``MatMul`` (and its specialized
    ``Gemm`` / ``Gemv`` / ``Dot`` / ``BatchedMatMul`` forms) produced by
    ``@`` / ``np.matmul`` in the source program.

    :param sdfg: SDFG to search (NestedSDFGs are included).
    :returns: List of ``(lib_node, state)`` pairs, where ``state`` is the
        SDFGState that owns the node.
    """
    tile_types = _tile_node_types()
    return [
        (node, graph)
        for node, graph in sdfg.all_nodes_recursive()
        if isinstance(node, nodes.LibraryNode) and not isinstance(node, tile_types)
    ]


def _collect_tileops_adjacent_edges(sdfg: SDFG) -> Tuple[Set[int], Set[int]]:
    """Find every edge whose memlet tree reaches a tileops library node.

    Walks the memlet tree of each tileops-incident edge and follows
    non-transient (connector) arrays across NestedSDFG boundaries up to the
    top-level SDFG, covering the always-on NestedSDFG body descent where the
    tileops nodes live inside a NestedSDFG body.

    :param sdfg: SDFG to search (NestedSDFGs included).
    :returns: ``(adjacent, boundary)`` sets of edge ``id()``s: ``adjacent``
        contains every edge in a tileops-reaching memlet tree; ``boundary``
        is the subset that are NestedSDFG boundary edges carrying a
        tileops-fed connector (their full-``W`` window subsets must never be
        narrowed below ``W`` — the inner connector array is ``W`` wide).
    """
    adjacent: Set[int] = set()
    boundary: Set[int] = set()
    worklist: List[Tuple[Any, SDFGState]] = []
    for node, state in _collect_tile_nodes(sdfg):
        worklist.extend((e, state) for e in state.in_edges(node))
        worklist.extend((e, state) for e in state.out_edges(node))
    while worklist:
        edge, state = worklist.pop()
        if id(edge) in adjacent:
            continue
        owning = state.sdfg
        # Mark the whole memlet tree; collect connector (non-transient)
        # arrays it touches for the NestedSDFG boundary crossing below.
        reached_connectors: Set[str] = set()
        for tree_edge in state.memlet_tree(edge):
            adjacent.add(id(tree_edge))
            if tree_edge.data is not None and tree_edge.data.data is not None:
                desc = owning.arrays.get(tree_edge.data.data)
                if desc is not None and not desc.transient:
                    reached_connectors.add(tree_edge.data.data)
        nsdfg_node = owning.parent_nsdfg_node
        parent_state = owning.parent
        if nsdfg_node is None or parent_state is None:
            continue
        for pe in parent_state.in_edges(nsdfg_node):
            if pe.dst_conn in reached_connectors and id(pe) not in adjacent:
                boundary.add(id(pe))
                worklist.append((pe, parent_state))
        for pe in parent_state.out_edges(nsdfg_node):
            if pe.src_conn in reached_connectors and id(pe) not in adjacent:
                boundary.add(id(pe))
                worklist.append((pe, parent_state))
    return adjacent, boundary


def _clamp_subset_to_shape(subset: Any, desc: data.Data) -> Tuple[Any, int]:
    """Narrow the provably-OOB dimensions of ``subset`` to ``desc``'s domain.

    :param subset: A :class:`~dace.subsets.Range` to clamp.
    :param desc: The data descriptor providing shape and offset.
    :returns: ``(new_subset, num_clamped_dims)``; ``new_subset`` is the input
        object when nothing changed.
    """
    from dace import subsets as sbs

    new_ranges = list(subset.ranges)
    num_clamped = 0
    for dim, (maxel, size, off) in enumerate(zip(subset.max_element(), desc.shape, desc.offset)):
        # Mirror validate_state's provable-OOB check.
        if ((maxel + off) >= size) == True:  # noqa: E712 -- sympy provability check
            begin, _, step = new_ranges[dim]
            new_ranges[dim] = (begin, size - 1 - off, step)
            num_clamped += 1
    if num_clamped:
        return sbs.Range(new_ranges), num_clamped
    return subset, 0


def clamp_propagated_oob_memlets(sdfg: SDFG) -> int:
    """Clamp provably out-of-bounds *tileops* memlet subsets to the array domain.

    The tile-op vectorizer emits full-width tile memlets (``x[i, j:j+W]``)
    whose accesses are mask-guarded at runtime. When a tiled dimension starts
    near the array end (e.g. a single-column assignment ``x[:, N-1] = v``),
    memlet propagation unions these to a *provably* out-of-bounds outer subset
    (``x[0:N, 0:N+W-1]``), which SDFG validation rejects (e.g. inside
    ``apply_gpu_transformations()``'s ``simplify()``). Since the mask
    guarantees no element beyond the array end is touched, the declared
    footprint is soundly narrowed to the array domain.

    The clamp is gated: only memlets already marked ``allow_oob`` (by
    :func:`mark_tile_op_memlets_allow_oob`) or whose memlet tree reaches a
    tileops library node — including across NestedSDFG boundaries — are
    narrowed. Any *other* provably-OOB memlet is a genuine bug with no mask
    guarding it: it is left untouched (with a diagnostic ``UserWarning``
    naming the memlet) so downstream validation rejects it loudly instead of
    silently compiling a truncated footprint.

    Both ``subset`` and ``other_subset`` are clamped, each against its own
    endpoint's descriptor (resolved from the memlet path the way
    ``validate_state`` does). After narrowing, ``volume`` is recomputed from
    the new ``subset``.

    Two kinds of edges are never narrowed:

    * Edges incident to library nodes: tile-op expansions require their own
      memlet subsets to match ``widths`` exactly (full-tile write contract).
    * NestedSDFG boundary edges whose connector feeds a tileops node
      (the always-on NestedSDFG body descent): the inner connector array is a
      full ``W``-wide window, so the boundary subset must stay ``W`` wide.
      When provably OOB they are marked ``allow_oob`` instead of clamped.

    :param sdfg: SDFG to fix up in place (NestedSDFGs included).
    :returns: Number of clamped memlet dimensions.
    """
    from dace import subsets as sbs

    adjacent, boundary = _collect_tileops_adjacent_edges(sdfg)
    clamped = 0
    for nsdfg in sdfg.all_sdfgs_recursive():
        for state in nsdfg.states():
            for e in state.edges():
                if isinstance(e.src, nodes.LibraryNode) or isinstance(e.dst, nodes.LibraryNode):
                    continue
                memlet = e.data
                if memlet is None or memlet.data is None or memlet.data not in nsdfg.arrays:
                    continue
                subset = memlet.subset
                if not isinstance(subset, sbs.Range):
                    continue
                desc = nsdfg.arrays[memlet.data]
                if subset.dims() != len(desc.shape):
                    continue

                # Resolve the other_subset endpoint's descriptor the way
                # validate_state does: from the memlet path endpoints.
                other_desc = None
                if memlet.other_subset is not None and isinstance(memlet.other_subset, sbs.Range):
                    path = state.memlet_path(e)
                    src_node, dst_node = path[0].src, path[-1].dst
                    other_node = (
                        dst_node
                        if isinstance(dst_node, nodes.AccessNode) and memlet.data != dst_node.data
                        else src_node
                    )
                    if isinstance(other_node, nodes.AccessNode) and other_node.data in nsdfg.arrays:
                        cand = nsdfg.arrays[other_node.data]
                        if memlet.other_subset.dims() == len(cand.shape):
                            other_desc = cand

                new_subset, n_sub = _clamp_subset_to_shape(subset, desc)
                new_other, n_other = (
                    _clamp_subset_to_shape(memlet.other_subset, other_desc)
                    if other_desc is not None
                    else (memlet.other_subset, 0)
                )
                if n_sub == 0 and n_other == 0:
                    continue

                if id(e) in boundary:
                    # NestedSDFG boundary edge feeding a tileops node: the
                    # full-W window must stay W wide (the inner connector
                    # array is W wide); defer the OOB to the runtime mask.
                    memlet.allow_oob = True
                    continue
                if not (memlet.allow_oob or id(e) in adjacent):
                    # Genuine OOB with no tileops mask anywhere: leave it
                    # for validation to reject loudly.
                    warnings.warn(
                        f"clamp_propagated_oob_memlets: memlet '{memlet}' (edge {e.src} -> {e.dst}, "
                        f"state '{state.label}', SDFG '{nsdfg.label}') is provably out-of-bounds but is "
                        "neither tileops-adjacent nor marked allow_oob; not clamping -- validation "
                        "will reject it"
                    )
                    continue

                if n_sub:
                    memlet.subset = new_subset
                    memlet.volume = new_subset.num_elements()
                if n_other:
                    memlet.other_subset = new_other
                clamped += n_sub + n_other
    return clamped


def _enclosing_map_chain(
    node: nodes.Node, state: SDFGState, scope_cache: Optional[_ScopeCache] = None
) -> List[Tuple[nodes.MapEntry, SDFGState]]:
    """Walk the chain of enclosing map scopes of ``node``, across NestedSDFGs.

    The in-state scope chain comes from ``state.scope_dict()`` (cached per
    state in ``scope_cache``); when the owning SDFG is nested, the walk
    continues from its ``parent_nsdfg_node`` in the parent state, transitively
    up to the top-level SDFG.

    :param node: The dataflow node whose enclosing maps are collected.
    :param state: The SDFGState that owns ``node``.
    :param scope_cache: Optional mutable per-state cache of scope dictionaries
        (avoids recomputing ``scope_dict()`` for repeated walks).
    :returns: List of ``(map_entry, state)`` pairs, innermost-first /
        outermost-last; empty if ``node`` is not inside any map at any level.
    """
    if scope_cache is None:
        scope_cache = {}
    chain: List[Tuple[nodes.MapEntry, SDFGState]] = []
    current_node: nodes.Node = node
    current_state: SDFGState = state
    while True:
        scope_dict = scope_cache.get(current_state)
        if scope_dict is None:
            scope_dict = current_state.scope_dict()
            scope_cache[current_state] = scope_dict
        scope = scope_dict.get(current_node)
        while scope is not None:
            chain.append((scope, current_state))
            scope = scope_dict.get(scope)
        owning_sdfg = current_state.sdfg
        nsdfg_node = owning_sdfg.parent_nsdfg_node
        parent_state = owning_sdfg.parent
        if nsdfg_node is None or parent_state is None:
            break
        current_node = nsdfg_node
        current_state = parent_state
    return chain


def _iter_cutile_scopes(sdfg: SDFG) -> Iterator[Tuple[nodes.MapEntry, SDFGState]]:
    """Iterate over every CuTile-scheduled MapEntry in ``sdfg``, recursively.

    :param sdfg: SDFG to search (NestedSDFGs are included).
    :returns: Iterator of ``(map_entry, state)`` pairs.
    """
    for node, graph in sdfg.all_nodes_recursive():
        if isinstance(node, nodes.MapEntry) and node.map.schedule == dtypes.ScheduleType.CuTile:
            yield node, graph


def _is_power_of_two(value: int) -> bool:
    """Whether ``value`` is a positive power of two.

    :param value: The integer to test.
    :returns: ``True`` iff ``value`` is in ``{1, 2, 4, 8, ...}``.
    """
    return value > 0 and (value & (value - 1)) == 0


def _warn_or_raise(message: str, strict: bool) -> None:
    """Emit ``message`` as a ``UserWarning``, or raise when ``strict``.

    :param message: The diagnostic message.
    :param strict: When ``True``, raise ``ValueError(message)`` instead of
        warning.
    :raises ValueError: When ``strict`` is ``True``.
    """
    if strict:
        raise ValueError(message)
    warnings.warn(message)


def mark_tile_op_memlets_allow_oob(sdfg: SDFG) -> int:
    """Set ``allow_oob=True`` on every memlet adjacent to a tileops node.

    Masked tile ops address a full ``W``-wide window whose tail lanes are
    inactive (the ``full_mask`` remainder strategy); at a non-divisible or
    single-point boundary the window's SUBSET exceeds the array bounds even
    though the masked runtime accesses (``ct.load`` / ``ct.gather`` /
    ``ct.scatter``) never touch the out-of-bounds lanes.  SDFG validation
    would reject such memlets ("Memlet subset out-of-bounds"), notably
    inside ``apply_gpu_transformations()`` whose simplify step re-propagates
    them outward.  Marking the LEAF memlets suffices for future
    propagations (``propagate_memlet`` shallow-copies the leaf, inheriting
    ``allow_oob``); the current memlet trees are marked too.

    CAVEAT (why :func:`clamp_propagated_oob_memlets` remains load-bearing):
    ``propagate_subset`` builds the propagated memlet as
    ``copy.copy(memlets[0])`` of the AGGREGATED memlet list, so with
    ``union_inner_edges=True`` a neighboring *unmarked* memlet on the same
    data can be first in that list and the ``allow_oob`` flag is silently
    dropped from the propagated result.  The clamp step (3b) re-narrows
    those re-propagated OOB subsets after ``apply_gpu_transformations()``.

    :param sdfg: The SDFG whose tileops-adjacent memlets are marked.
    :returns: The number of memlets marked.
    """
    marked = 0
    for node, state in _collect_tile_nodes(sdfg):
        for edge in list(state.in_edges(node)) + list(state.out_edges(node)):
            if edge.data is None or edge.data.data is None:
                continue
            for tree_edge in state.memlet_tree(edge):
                if not tree_edge.data.allow_oob:
                    tree_edge.data.allow_oob = True
                    marked += 1
    return marked


def _demote_residual_gpu_device_maps(sdfg: SDFG, strict: bool, pass_name: str) -> int:
    """Demote every remaining ``GPU_Device`` map to ``Sequential``.

    The Python/cuTile backend has no ``GPU_Device`` scope dispatcher, so any
    such map surviving to code generation raises ``KeyError: GPU_Device``.
    Demoted maps become host ("driver") Python loops over the ``GPU_Global``
    (cupy) operands.  Silent only for provably tiny maps (total volume at
    most :data:`_TINY_DEMOTION_VOLUME` — the intended scalar-control case);
    larger or symbolic volumes are diagnosed via :func:`_warn_or_raise`
    first, since the host loop pays one device round-trip per element.

    :param sdfg: The SDFG to re-stamp in place (NestedSDFGs included).
    :param strict: Whether the large-volume diagnostic raises.
    :param pass_name: Name of the calling pass (message prefix).
    :returns: The number of maps demoted.
    :raises ValueError: When ``strict`` and a residual map is not provably
        tiny.
    """
    demoted = 0
    for map_node, graph in sdfg.all_nodes_recursive():
        if not (isinstance(map_node, nodes.MapEntry) and map_node.map.schedule == dtypes.ScheduleType.GPU_Device):
            continue
        volume = map_node.map.range.num_elements()
        try:
            provably_tiny = int(volume) <= _TINY_DEMOTION_VOLUME
        except (TypeError, ValueError):
            provably_tiny = False  # Symbolic volume: not provably tiny.
        if not provably_tiny:
            _warn_or_raise(
                f"{pass_name}: demoting non-tileops GPU_Device map "
                f"'{map_node.map.label}' (state '{graph.label}', volume {volume}) to "
                "Sequential: it compiles to a per-element host loop over GPU data "
                "(one device round-trip per element), which is pathological for "
                "anything but tiny scalar-control maps",
                strict,
            )
        map_node.map.schedule = dtypes.ScheduleType.Sequential
        _promote_demoted_scope_transients(graph, map_node)
        demoted += 1
    return demoted


def _promote_demoted_scope_transients(state: SDFGState, map_entry: nodes.MapEntry) -> int:
    """Move ``Register`` transient Arrays inside a just-demoted map scope to ``GPU_Global``.

    ``apply_gpu_transformations(register_transients=True)`` stamps transients inside a
    GPU_Device map ``Register`` on the assumption they stay kernel-internal. After the map
    is demoted to a host driver loop, such a transient becomes a host (numpy) array
    bridging ``GPU_Global`` (cupy) operands — e.g. doitgen's matmul result buffer — and
    the generated ``cupy_slice[...] = numpy_array`` copy raises at runtime. Re-stamp them
    ``GPU_Global`` so the whole demoted body stays device-resident. Views and Scalars are
    left alone (a View follows its viewed array; Scalars feed host control flow).
    Descends into NestedSDFGs within the scope: their ``Register`` Array transients bridge
    the same ``GPU_Global`` operands and fail the same way on whole-array copies.

    The re-stamp is descriptor-level (``sdfg.arrays``), not scope-level: a transient whose
    descriptor is also accessed OUTSIDE the demoted scope changes storage at every use site,
    not just inside this scope.

    :param state: State containing the demoted map.
    :param map_entry: Entry node of the map that was just demoted to Sequential.
    :returns: Number of descriptors re-stamped.
    """
    promoted = 0
    scope = state.scope_subgraph(map_entry, include_entry=False, include_exit=False)
    # (state, nodes) pairs: the scope itself plus every NestedSDFG under it, recursively.
    frontier = [(state, scope.nodes())]
    while frontier:
        st, nds = frontier.pop()
        for node in nds:
            if isinstance(node, nodes.NestedSDFG):
                frontier.extend((ns, ns.nodes()) for ns in node.sdfg.states())
                continue
            if not isinstance(node, nodes.AccessNode):
                continue
            desc = st.sdfg.arrays.get(node.data)
            if (
                desc is None
                or not desc.transient
                or not isinstance(desc, data.Array)
                or isinstance(desc, data.View)
                or desc.storage != dtypes.StorageType.Register
            ):
                continue
            desc.storage = dtypes.StorageType.GPU_Global
            promoted += 1
    return promoted


def _diagnose_no_anchors(pass_name: str, sdfg: SDFG, strict: bool) -> None:
    """Diagnose an SDFG that carries zero tileops anchors.

    Two distinct cases:

    * **BLAS-only configuration** (supported): the SDFG has no tileops nodes
      but *does* contain other library nodes (e.g. BLAS ``Dot`` / ``MatMul``)
      that :class:`CuTileSetLibraryImplementations` lowers downstream.  This
      emits an informational ``UserWarning`` and never raises, even under
      ``strict``.
    * **Genuinely empty** (probable user error): no tileops nodes and no
      other library nodes either — the SDFG was most likely never vectorized.
      This warns, or raises ``ValueError`` when ``strict``.

    :param pass_name: Name of the calling pass (message prefix).
    :param sdfg: The anchor-less SDFG being diagnosed.
    :param strict: Whether the genuinely-empty case raises instead of warning.
    :raises ValueError: When ``strict`` and the SDFG contains no library
        nodes at all.
    """
    non_tile = _collect_non_tile_library_nodes(sdfg)
    if non_tile:
        warnings.warn(
            f"{pass_name}: no tileops library nodes found, but {len(non_tile)} non-tileops "
            "library node(s) are present (BLAS-only configuration); proceeding -- "
            "CuTileSetLibraryImplementations lowers them"
        )
    else:
        _warn_or_raise(
            f"{pass_name}: no tileops library nodes found and no other library nodes present; "
            "the SDFG appears not to have been vectorized -- run VectorizeCuTile (or "
            "VectorizeMultiDim with a target_isa=ISA.CUTILE, expand_tile_nodes=False config) "
            "first",
            strict,
        )


@properties.make_properties
class _CuTileLoweringPass(ppl.Pass):
    """Shared base of the cuTile lowering passes.

    Provides the ``strict`` knob (precondition violations warn by default and
    raise when ``strict=True``) and the common Pass plumbing.
    """

    CATEGORY: str = "Vectorization"

    strict = properties.Property(
        dtype=bool,
        default=False,
        desc="When True, precondition violations raise ValueError instead of emitting a UserWarning.",
    )

    def __init__(self, strict: bool = False):
        """Initialize the pass.

        :param strict: When ``True``, precondition violations raise
            ``ValueError`` instead of emitting a ``UserWarning``.
        """
        super().__init__()
        self.strict = strict

    def modifies(self) -> ppl.Modifies:
        return ppl.Modifies.Everything

    def should_reapply(self, modified: ppl.Modifies) -> bool:
        return bool(modified & ppl.Modifies.Everything)

    def depends_on(self) -> Set[Type[ppl.Pass]]:
        return set()


@properties.make_properties
@transformation.explicit_cf_compatible
class CuTileValidateTiles(_CuTileLoweringPass):
    """Precondition audit for the cuTile lowering sequence.

    Checks that (a) at least one tileops library node exists (the anchors all
    later passes stamp from), and (b) every anchor's ``widths`` are powers of
    two — a hard ``cuda.tile`` runtime requirement that raises ``ValueError``
    unconditionally (not gated by ``strict``).

    Zero anchors with other library nodes present (the BLAS-only
    configuration, e.g. ``cholesky`` / ``trisolv`` whose reductions became
    BLAS ``Dot`` nodes) is a supported case: informational warning only,
    even under ``strict``.
    """

    def apply_pass(self, sdfg: SDFG, pipeline_results: Dict[str, Any]) -> Optional[int]:
        """Validate the tile-op anchors of ``sdfg``.

        :param sdfg: The SDFG to validate (not modified).
        :param pipeline_results: Unused pipeline results.
        :returns: The number of tileops anchors found, or ``None`` if there
            are none (after diagnosing per :func:`_diagnose_no_anchors`).
        :raises ValueError: If any anchor has a non-power-of-2 tile width
            (always), or — when ``strict`` — if no anchors exist and the
            SDFG has no other library nodes either.
        """
        anchors = _collect_tile_nodes(sdfg)
        if not anchors:
            _diagnose_no_anchors("CuTileValidateTiles", sdfg, self.strict)
            return None
        for node, _ in anchors:
            for width in node.widths:
                if not _is_power_of_two(width):
                    raise ValueError(
                        f"{type(node).__name__} '{node.label}': tile width {width} in "
                        f"widths={list(node.widths)} is not a power of 2 "
                        "(cuda.tile runtime requirement)"
                    )
        return len(anchors)


@properties.make_properties
@transformation.explicit_cf_compatible
class CuTileSetTileStorage(_CuTileLoweringPass):
    """Stamp tile/mask transients with ``StorageType.CuTile_Tile``.

    Three rules, applied in order:

    a. **Anchored:** every AccessNode adjacent to a tileops anchor whose
       descriptor (in its owning SDFG) is a transient ``data.Array`` with
       ``StorageType.Register`` becomes ``CuTile_Tile``. Scalars are left
       untouched (e.g. TileReduce-adjacent accumulator Scalars stay Register).
    b. **NSDFG boundary propagation:** for NestedSDFGs inside a CuTile scope,
       a ``CuTile_Tile`` array on one side of a connector stamps the
       descriptor on the other side too (outer -> inner unconditionally on
       Arrays — the connector view of a tile is a tile; inner -> outer only on
       Register Array transients). Iterated to a fixpoint.
    c. **Recursive fallback:** any remaining Register Array transient whose
       AccessNode sits (transitively through NSDFG parents) inside a CuTile
       scope becomes ``CuTile_Tile``; rule (b) is then re-run once more.
    """

    def apply_pass(self, sdfg: SDFG, pipeline_results: Dict[str, Any]) -> Optional[int]:
        """Stamp tile-array storage inside cuTile kernels.

        :param sdfg: The SDFG to stamp in place.
        :param pipeline_results: Unused pipeline results.
        :returns: The number of data descriptors whose storage was changed to
            ``CuTile_Tile``, or ``None`` if none were (e.g. on a repeated
            run, or when the precondition fails).
        :raises ValueError: When ``strict`` and no CuTile-scheduled map
            exists.
        """
        if next(_iter_cutile_scopes(sdfg), None) is None:
            if not _collect_tile_nodes(sdfg) and _collect_non_tile_library_nodes(sdfg):
                # BLAS-only configuration: no cuTile kernels exist by design;
                # there is nothing to stamp. Informational, never raises.
                warnings.warn(
                    "CuTileSetTileStorage: no CuTile-scheduled map and no tileops anchors, "
                    "but non-tileops library nodes are present (BLAS-only configuration); "
                    "nothing to stamp"
                )
            else:
                _warn_or_raise(
                    "CuTileSetTileStorage: no CuTile-scheduled map found; run "
                    "sdfg.apply_gpu_transformations() + GPUDeviceToCuTile first",
                    self.strict,
                )
            return None

        scope_cache: _ScopeCache = {}
        stamped: Set[int] = set()  # id() of changed descriptors

        def _stamp(desc: data.Data) -> None:
            """Set ``desc.storage = CuTile_Tile`` and record the change."""
            if desc.storage != dtypes.StorageType.CuTile_Tile:
                desc.storage = dtypes.StorageType.CuTile_Tile
                stamped.add(id(desc))

        # Host driver loops can own array temporaries shared by several kernels.
        # Python host code can read GPU arrays, so keep those buffers on the device.
        for tile, state in _collect_tile_nodes(sdfg):
            owner = state.sdfg
            parent = owner.parent_nsdfg_node
            if parent is not None and any(
                entry.map.schedule == dtypes.ScheduleType.CuTile
                for entry, _ in _enclosing_map_chain(parent, owner.parent, scope_cache)
            ):
                continue
            for edge in (*state.in_edges(tile), *state.out_edges(tile)):
                desc = owner.arrays.get(edge.data.data)
                if (
                    isinstance(desc, data.Array)
                    and not isinstance(desc, data.View)
                    and desc.transient
                    and desc.storage == dtypes.StorageType.CPU_Heap
                ):
                    desc.storage = dtypes.StorageType.GPU_Global
                    stamped.add(id(desc))

        # Rule (a): AccessNodes adjacent to tileops anchors.
        for node, state in _collect_tile_nodes(sdfg):
            owning_arrays = state.sdfg.arrays
            neighbors = [e.src for e in state.in_edges(node)] + [e.dst for e in state.out_edges(node)]
            for neighbor in neighbors:
                if not isinstance(neighbor, nodes.AccessNode):
                    continue
                desc = owning_arrays.get(neighbor.data)
                if isinstance(desc, data.Array) and desc.transient and desc.storage == dtypes.StorageType.Register:
                    _stamp(desc)

        # Fusion can leave a widened kernel-local tile in GPU_Global storage.
        for tile, state in _collect_tile_nodes(sdfg):
            for edge in (*state.in_edges(tile), *state.out_edges(tile)):
                desc = state.sdfg.arrays.get(edge.data.data)
                if not (
                    isinstance(desc, data.Array)
                    and desc.transient
                    and desc.storage == dtypes.StorageType.GPU_Global
                    and tuple(desc.shape) == tuple(tile.widths)
                ):
                    continue
                accesses = [
                    (node, graph)
                    for node, graph in state.sdfg.all_nodes_recursive()
                    if isinstance(node, nodes.AccessNode) and node.data == edge.data.data and graph.sdfg is state.sdfg
                ]
                kernels = []
                for access, graph in accesses:
                    chain = _enclosing_map_chain(access, graph, scope_cache)
                    kernels.append(
                        next(
                            (entry for entry, _ in reversed(chain) if entry.map.schedule == dtypes.ScheduleType.CuTile),
                            None,
                        )
                    )
                if kernels and kernels[0] is not None and all(kernel is kernels[0] for kernel in kernels):
                    _stamp(desc)

        # Rule (b): NSDFG boundary propagation (fixpoint).
        def _propagate_nsdfg_boundaries() -> None:
            """Sync CuTile_Tile storage across NestedSDFG connectors until a
            fixpoint is reached."""
            changed = True
            while changed:
                changed = False
                before = len(stamped)
                for node, graph in sdfg.all_nodes_recursive():
                    if not isinstance(node, nodes.NestedSDFG):
                        continue
                    chain = _enclosing_map_chain(node, graph, scope_cache)
                    if not any(entry.map.schedule == dtypes.ScheduleType.CuTile for entry, _ in chain):
                        continue
                    outer_arrays = graph.sdfg.arrays
                    inner_arrays = node.sdfg.arrays
                    boundary = [(e.dst_conn, e) for e in graph.in_edges(node)] + [
                        (e.src_conn, e) for e in graph.out_edges(node)
                    ]
                    for conn, edge in boundary:
                        if conn is None or edge.data.data is None:
                            continue
                        outer_desc = outer_arrays.get(edge.data.data)
                        inner_desc = inner_arrays.get(conn)
                        if outer_desc is None or inner_desc is None:
                            continue
                        if (
                            outer_desc.storage == dtypes.StorageType.CuTile_Tile
                            and isinstance(inner_desc, data.Array)
                            and inner_desc.storage != dtypes.StorageType.CuTile_Tile
                        ):
                            _stamp(inner_desc)
                        elif (
                            inner_desc.storage == dtypes.StorageType.CuTile_Tile
                            and isinstance(outer_desc, data.Array)
                            and outer_desc.transient
                            and outer_desc.storage == dtypes.StorageType.Register
                        ):
                            _stamp(outer_desc)
                changed = len(stamped) > before

        _propagate_nsdfg_boundaries()

        # Rule (c): recursive fallback for Register Array transients inside
        # CuTile scopes (transitively through NSDFG parents).
        for node, graph in sdfg.all_nodes_recursive():
            if not isinstance(node, nodes.AccessNode):
                continue
            desc = graph.sdfg.arrays.get(node.data)
            if not (isinstance(desc, data.Array) and desc.transient and desc.storage == dtypes.StorageType.Register):
                continue
            chain = _enclosing_map_chain(node, graph, scope_cache)
            if any(entry.map.schedule == dtypes.ScheduleType.CuTile for entry, _ in chain):
                _stamp(desc)

        # Rule (c) may have created new boundary mismatches; settle them.
        _propagate_nsdfg_boundaries()

        return len(stamped) if stamped else None


@properties.make_properties
@transformation.explicit_cf_compatible
class GPUDeviceToCuTile(_CuTileLoweringPass):
    """Convert GPU_Device-scheduled maps to CuTile (tileops) or Sequential.

    This adapter pass runs AFTER ``sdfg.apply_gpu_transformations()``, which
    stamps top-level maps as ``ScheduleType.GPU_Device`` and inner maps as
    ``ScheduleType.Sequential``.  It performs two re-stamping steps:

    1. **Tileops-anchored -> CuTile.**  For each tileops library node the chain
       of enclosing maps (walked across NestedSDFG boundaries via
       :func:`_enclosing_map_chain`) is inspected; if the outermost map has
       ``ScheduleType.GPU_Device`` it is re-stamped ``ScheduleType.CuTile`` --
       these become the cuTile kernels.
    2. **Everything else -> Sequential.**  Any remaining ``GPU_Device`` map is
       a non-tileops map (e.g. the scalar / small-elementwise control steps of
       a sequential solver such as ``cholesky`` / ``trisolv`` / ``durbin``,
       which ``apply_gpu_transformations()`` blindly stamps ``GPU_Device``).
       The Python/cuTile backend has **no** ``GPU_Device`` scope dispatcher --
       only ``Sequential`` / CPU / ``CuTile`` -- so such a map would raise
       ``KeyError: ScheduleType.GPU_Device`` at code generation.  It is
       re-stamped ``ScheduleType.Sequential`` and emitted as a host ("driver")
       Python loop that operates directly on the ``GPU_Global`` (``cupy``)
       arrays, which are host-addressable in this backend.  Because that host
       loop pays one device round-trip per element, only maps whose total
       volume is provably at most :data:`_TINY_DEMOTION_VOLUME` (the intended
       scalar-control case) are demoted silently; larger or symbolic-volume
       maps are diagnosed via :func:`_warn_or_raise` first (warn by default,
       raise under ``strict``) and demoted only in non-strict mode.
    """

    def apply_pass(self, sdfg: SDFG, pipeline_results: Dict[str, Any]) -> Optional[int]:
        """Re-stamp GPU_Device maps to CuTile (tileops) or Sequential (rest).

        :param sdfg: The SDFG to transform in place.
        :param pipeline_results: Unused pipeline results.
        :returns: The number of maps re-stamped to ``CuTile``, or ``None``
            if there were no tileops anchors.
        :raises ValueError: When ``strict`` and either (a) no tileops anchors
            and no other library nodes exist, or (b) a residual ``GPU_Device``
            map with a large / not-provably-tiny volume must be demoted.
        """
        # Step 1: re-stamp tileops-anchored outermost maps to CuTile.  Absence
        # of anchors is not fatal here -- an SDFG whose only reductions became
        # BLAS library nodes (e.g. ``cholesky`` / ``trisolv`` with ``np.dot``)
        # has zero cuTile kernels but still carries GPU_Device host-control maps
        # that step 2 must demote -- so we diagnose (BLAS-only: informational;
        # genuinely empty: warn / strict-raise) and continue.
        anchors = _collect_tile_nodes(sdfg)
        if not anchors:
            _diagnose_no_anchors("GPUDeviceToCuTile", sdfg, self.strict)

        scope_cache: _ScopeCache = {}
        cutile_entries: Set[nodes.MapEntry] = set()
        for node, state in anchors:
            chain = _enclosing_map_chain(node, state, scope_cache)
            if not chain:
                _warn_or_raise(
                    f"GPUDeviceToCuTile: tileops node '{node.label}' has no "
                    "enclosing map at any level; it cannot be placed inside "
                    "a cuTile kernel",
                    self.strict,
                )
                continue
            outermost_entry = next(
                (
                    entry
                    for entry, _ in reversed(chain)
                    if entry.map.schedule in (dtypes.ScheduleType.GPU_Device, dtypes.ScheduleType.CuTile)
                ),
                None,
            )
            if outermost_entry is not None and outermost_entry.map.schedule == dtypes.ScheduleType.GPU_Device:
                outermost_entry.map.schedule = dtypes.ScheduleType.CuTile
                cutile_entries.add(outermost_entry)

        # Step 2: demote every remaining GPU_Device map (non-tileops host
        # control) to Sequential so the Python/cuTile backend can code-generate
        # it as a host driver loop over the GPU_Global (cupy) operands.  Runs
        # unconditionally -- these maps exist even when there are no anchors.
        _demote_residual_gpu_device_maps(sdfg, self.strict, "GPUDeviceToCuTile")

        return len(cutile_entries) if cutile_entries else None


@properties.make_properties
@transformation.explicit_cf_compatible
class CuTileSetImplementations(_CuTileLoweringPass):
    """Stamp every tileops node with the cuTile expansion, directly.

    Sets ``node.target_isa = "CUTILE"`` and ``node.implementation = "cutile"``
    on every tileops library node — no ``select_tile_implementation`` dispatch
    (cuTile selection is unconditional). A tile-op node type without a
    ``'cutile'`` implementation raises ``ValueError`` (loud failure instead of
    a silent ``'pure'`` fallback, which would emit CPP tasklets into a Python
    backend kernel). Must run before ``sdfg.expand_library_nodes()``.
    """

    def apply_pass(self, sdfg: SDFG, pipeline_results: Dict[str, Any]) -> Optional[int]:
        """Stamp the cuTile implementation on every tileops node.

        :param sdfg: The SDFG whose library nodes are stamped in place.
        :param pipeline_results: Unused pipeline results.
        :returns: The number of library nodes stamped, or ``None`` if there
            are none (e.g. when run after ``expand_library_nodes()``).
        :raises ValueError: If a tileops node lacks a ``'cutile'``
            implementation (always), or — when ``strict`` — if no tileops
            nodes exist.
        """
        anchors = _collect_tile_nodes(sdfg)
        if not anchors:
            _warn_or_raise(
                "CuTileSetImplementations: no tileops library nodes found (already expanded?); "
                "this pass must run before sdfg.expand_library_nodes()",
                self.strict,
            )
            return None
        for node, _ in anchors:
            if "cutile" not in node.implementations:
                raise ValueError(
                    f"{type(node).__name__} '{node.label}' has no 'cutile' implementation; "
                    "a tileops node type must ship a cuTile expansion before it can be "
                    "lowered for the cuTile backend"
                )
            from dace.libraries.tileops.dispatch import ISA

            node.target_isa = ISA.CUTILE
            node.implementation = "cutile"
        return len(anchors)


@properties.make_properties
@transformation.explicit_cf_compatible
class CuTileSetLibraryImplementations(_CuTileLoweringPass):
    """Select and expand implementations for non-tileops library nodes.

    The cuTile front door lowers elementwise / stencil maps into tileops
    library nodes, but any *other* library node in the program is left
    untouched by the tileops-anchored passes. The important case is BLAS
    ``MatMul`` (from ``@`` / ``np.matmul``) and its specialized ``Gemm`` /
    ``Gemv`` / ``Dot`` / ``BatchedMatMul`` forms: left alone they keep the
    ``GPU_Device`` schedule stamped by ``apply_gpu_transformations()`` and are
    never expanded, so the Python/cuTile backend raises
    ``KeyError: ScheduleType.GPU_Device`` at code generation.

    This pass gives every non-tileops library node a Python-backend-compatible
    implementation and drives its expansion, so no such node survives to
    codegen. Implementations are tried in :attr:`PREFERRED_IMPLEMENTATIONS`
    order -- ``'CuPy'`` first (it emits a ``cupy`` call, which the Python
    backend runs directly on the ``GPU_Global`` operands), then ``'pure'``.
    Nodes that expose neither (notably ``MatMul``, which only offers the
    ``'specialize'`` meta-expansion) are first expanded via ``'specialize'``;
    the concrete node it produces is picked up on the next iteration.

    Must run *after* ``GPUDeviceToCuTile`` (so the tileops kernels are already
    re-stamped to :class:`~dace.dtypes.ScheduleType.CuTile`) and *before* the
    Python-backend stamp. Tileops nodes are deliberately skipped -- they are
    handled by :class:`CuTileSetImplementations`.
    """

    #: Implementations to try, in priority order, for a non-tileops node.
    PREFERRED_IMPLEMENTATIONS: Tuple[str, ...] = ("Python", "CuPy", "pure")

    def _select_implementation(self, node: nodes.LibraryNode) -> Optional[str]:
        """Return the highest-priority available implementation, or ``None``.

        :param node: The library node to inspect.
        :returns: The first entry of :attr:`PREFERRED_IMPLEMENTATIONS` that the
            node exposes, or ``None`` if it exposes none of them.
        """
        for impl in self.PREFERRED_IMPLEMENTATIONS:
            if impl in node.implementations:
                return impl
        return None

    def apply_pass(self, sdfg: SDFG, pipeline_results: Dict[str, Any]) -> Optional[int]:
        """Select and expand implementations for every non-tileops library node.

        :param sdfg: The SDFG whose library nodes are expanded in place.
        :param pipeline_results: Unused pipeline results.
        :returns: The number of library nodes expanded, or ``None`` if there
            were none.
        :raises ValueError: When ``strict`` and a non-tileops library node
            exposes neither a preferred implementation nor ``'specialize'``.
        """
        expanded = 0
        # Fixed-point loop: expanding a meta-node (``specialize``) reveals a new
        # concrete node that must itself be selected on the next iteration.
        progressed = True
        while progressed:
            progressed = False
            for node, state in _collect_non_tile_library_nodes(sdfg):
                impl = self._select_implementation(node)
                if impl is None and "specialize" in node.implementations:
                    impl = "specialize"
                if impl is None:
                    _warn_or_raise(
                        f"CuTileSetLibraryImplementations: {type(node).__name__} '{node.label}' "
                        f"exposes none of {self.PREFERRED_IMPLEMENTATIONS} nor 'specialize'; the "
                        "Python/cuTile backend cannot code-generate it",
                        self.strict,
                    )
                    continue
                node.implementation = impl
                node.expand(state, impl)
                expanded += 1
                progressed = True

        # Expansion can introduce NEW GPU_Device-scheduled maps: the expansion
        # framework re-stamps a nested expansion's default-scheduled maps with
        # the library node's schedule (GPU_Device from
        # apply_gpu_transformations()), and GPUDeviceToCuTile has already run.
        # Re-run its demotion so no such map survives to codegen.
        if expanded:
            _demote_residual_gpu_device_maps(sdfg, self.strict, "CuTileSetLibraryImplementations")

        return expanded or None
