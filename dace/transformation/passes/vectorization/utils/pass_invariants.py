# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Invariant checker functions for K-dim vectorization passes.

Per user direction 2026-06-12: ``We should have pre condition and post
condition checks for all passes and by default and always and always run
them.`` + ``Does pre condition really need to pass? Can't be just a
function that checks the condition holds true in a graph / subgraph etc?``

This module exposes plain functions:

* Each checker takes an SDFG (and optionally other args) and returns
  ``None`` if the invariant holds, or a string describing the violation.
* :func:`assert_invariant` raises ``AssertionError`` on violation with a
  formatted message including the pass name + checker description +
  offending node / edge / state.

Each pass calls the checkers directly from its ``apply_pass``:

.. code-block:: python

    def apply_pass(self, sdfg, _):
        result = self._do_work(sdfg)
        assert_invariant(no_memlet_dim_mismatch(sdfg),
                         "WidenAccesses", "memlet dim consistent")
        return result

No mixin, no inheritance, no env gate. Always runs.
"""
from typing import Optional, Tuple

import dace
from dace.sdfg import SDFG, SDFGState
from dace.sdfg.nodes import AccessNode, MapEntry, MapExit, NestedSDFG


def assert_invariant(violation: Optional[str], pass_name: str, description: str) -> None:
    """Raise :class:`AssertionError` if ``violation`` is non-None.

    :param violation: The checker's return value (``None`` on success or
        the offending-node description).
    :param pass_name: The pass name to include in the error message.
    :param description: One-line description of the invariant.
    """
    if violation is None:
        return
    raise AssertionError(f"{pass_name}: invariant violated -- {description}: {violation}")


# ---------------------------------------------------------------------------
# Generic structural invariants (work at SDFG or per-state level).
# ---------------------------------------------------------------------------


def no_memlet_dim_mismatch(scope) -> Optional[str]:
    """For memlets connecting a tasklet / lib-node / NSDFG connector to
    an AccessNode (or two such connectors), ``subset`` and ``other_subset``
    must have matching rank. Accepts an SDFG OR a single :class:`SDFGState`.
    Returns ``None`` on success or a description.

    AN -> AN memlets are exempt: they describe pure copies between two
    descriptors that may have different ranks (e.g. a 4D slice copied
    into a 1D flat buffer) so the two subsets are intentionally
    different-rank when the descriptors are.

    ``MapEntry`` / ``MapExit`` pass-through edges are likewise exempt: they are
    scope plumbing, not the tasklet / lib-node / NSDFG-connector <-> AN edges
    this invariant targets, and they legitimately carry a different-rank
    ``other_subset`` when one side is a scalar staging element (e.g. a 2-D point
    ``a[jk, jc]`` copied through the map into the scalar ``c_slice`` after
    ``ConvertLengthOneArraysToScalars``).
    """
    states = _iter_states(scope)
    for sd, state in states:
        for edge in state.edges():
            mem = edge.data
            if mem is None or mem.subset is None or mem.other_subset is None:
                continue
            # Pure AN -> AN copies are allowed to carry different-rank
            # subsets when the two descriptors have different ranks.
            if isinstance(edge.src, AccessNode) and isinstance(edge.dst, AccessNode):
                continue
            # Scope pass-through edges (Map entry/exit) are out of scope -- see docstring.
            if isinstance(edge.src, (MapEntry, MapExit)) or isinstance(edge.dst, (MapEntry, MapExit)):
                continue
            if len(mem.subset.size()) != len(mem.other_subset.size()):
                return (f"{sd.name}.{state.label}: memlet ``{mem.data}`` subset dim={len(mem.subset.size())} "
                        f"!= other_subset dim={len(mem.other_subset.size())}")
    return None


def no_transient_scalar_stores(scope) -> Optional[str]:
    """No TILE (multi-element) memlet may be stored into a TRANSIENT Scalar.

    Per the K-dim design (user direction 2026-06-14): inside a body NSDFG a scalar
    *write* only targets a NON-transient program output (e.g. a reduction result,
    section 3.5); a TILE result written into a TRANSIENT scalar is a widening miss
    -- ``WidenAccesses`` should have widened that transient to a tile so the edge
    is ``tile -> tile``. This is the clean replacement for the old
    ``_maybe_elide_scalar_passthrough`` patch-fix (removed).

    Scalar load-staging -- a single element copied into a transient scalar for a
    broadcast, e.g. ``a_const`` from ``a[0]`` feeding a ``TileLoad(src_kind=
    "Scalar")`` -- is single-element and therefore allowed.

    Accepts an SDFG or a single :class:`SDFGState`. Returns ``None`` on success.
    """
    for sd, state in _iter_states(scope):
        for edge in state.edges():
            dst = edge.dst
            if not isinstance(dst, AccessNode):
                continue
            desc = sd.arrays.get(dst.data)
            if not (isinstance(desc, dace.data.Scalar) and desc.transient):
                continue
            mem = edge.data
            if mem is None or mem.subset is None:
                continue
            try:
                multi_element = any(bool(dace.symbolic.simplify(sz - 1) != 0) for sz in mem.subset.size())
            except Exception:  # noqa: BLE001 -- symbolic / non-Range subset: treat as scalar (skip)
                multi_element = False
            if multi_element:
                return (f"{sd.name}.{state.label}: tile (multi-element {tuple(mem.subset.size())}) stored into "
                        f"transient Scalar ``{dst.data}`` -- widen the transient to a tile "
                        f"(scalar stores are only allowed to a non-transient program output)")
    return None


def no_isolated_access_nodes(scope) -> Optional[str]:
    """No AccessNode may have zero in-edges AND zero out-edges. Accepts
    an SDFG or a single state.
    """
    for sd, state in _iter_states(scope):
        for node in state.nodes():
            if not isinstance(node, AccessNode):
                continue
            if state.in_degree(node) == 0 and state.out_degree(node) == 0:
                return f"{sd.name}.{state.label}: isolated AccessNode ``{node.data}``"
    return None


def no_duplicate_connector_edges(scope) -> Optional[str]:
    """Every NSDFG / Tasklet / lib-node connector has <=1 edge per direction.

    :class:`~dace.sdfg.nodes.MapEntry` and :class:`~dace.sdfg.nodes.MapExit`
    are skipped: their pass-through connectors are designed to fan-out
    (entry's ``OUT_X``) and fan-in (exit's ``IN_X``).
    """
    from dace.sdfg.nodes import MapEntry, MapExit
    for sd, state in _iter_states(scope):
        for node in state.nodes():
            if isinstance(node, (MapEntry, MapExit)):
                continue
            in_counts = {}
            for e in state.in_edges(node):
                if e.dst_conn is None:
                    continue
                in_counts.setdefault(e.dst_conn, 0)
                in_counts[e.dst_conn] += 1
            for conn, count in in_counts.items():
                if count > 1:
                    return (f"{sd.name}.{state.label}: {type(node).__name__} ``{node.label}``"
                            f"in-connector ``{conn}`` has {count} edges (max 1)")
            out_counts = {}
            for e in state.out_edges(node):
                if e.src_conn is None:
                    continue
                out_counts.setdefault(e.src_conn, 0)
                out_counts[e.src_conn] += 1
            for conn, count in out_counts.items():
                if count > 1:
                    return (f"{sd.name}.{state.label}: {type(node).__name__} ``{node.label}``"
                            f"out-connector ``{conn}`` has {count} edges (max 1)")
    return None


def mask_connectors_are_bool(scope) -> Optional[str]:
    """Every edge feeding a tile lib-node ``_mask`` connector must be sourced
    from a ``bool`` array. A mask selects per-lane, so a non-bool mask (e.g. a
    ``double`` 1.0/0.0 value mask) is invalid — comparison ops and lifted
    if-conditions produce ``bool``, and every mask consumer (TileBinop /
    TileUnop / TileITE ``_mask``) is defined over a boolean tile.

    Accepts an SDFG or a single :class:`SDFGState`. Returns ``None`` on
    success or a description identifying the first non-bool mask source.
    """
    import dace.dtypes as _dt
    for sd, state in _iter_states(scope):
        for edge in state.edges():
            if edge.dst_conn != "_mask":
                continue
            mem = edge.data
            if mem is None or mem.data is None:
                continue
            desc = sd.arrays.get(mem.data)
            if desc is None:
                continue
            if desc.dtype != _dt.bool_:
                return (f"{sd.name}.{state.label}: ``_mask`` connector on "
                        f"{type(edge.dst).__name__} ``{edge.dst.label}`` is fed by "
                        f"``{mem.data}`` of dtype {desc.dtype} (must be bool)")
    return None


def tile_mask_gen_dominates_consumers(scope) -> Optional[str]:
    """Every :class:`TileMaskGen` must sit in the start block of its own SDFG.

    The iteration mask is branch-independent ("which lanes are in bounds"), so its
    producer must DOMINATE every masked consumer -- otherwise a data-dependent
    ``if`` (-> TileITE) body reads ``_tile_iter_mask`` from a branch state the
    producer does not dominate (uninitialized lanes, flaky writes). The start block
    has no predecessors and dominates every reachable state, so producing the mask
    there is the simplest sufficient guarantee. ``GenerateTileIterationMask`` emits
    it in a dedicated ``_tile_mask_init`` start state; this is its post-condition.

    Accepts an SDFG or a single :class:`SDFGState`. Returns ``None`` on success or
    a description of the first mis-placed producer.
    """
    from dace.libraries.tileops import TileMaskGen
    for sd, state in _iter_states(scope):
        if not any(isinstance(n, TileMaskGen) for n in state.nodes()):
            continue
        if state is not sd.start_block:
            return (f"{sd.name}.{state.label}: TileMaskGen lives outside the SDFG start block "
                    f"``{sd.start_block.label}`` -- the iteration mask producer must dominate every "
                    f"masked consumer (emit it in the ``_tile_mask_init`` start state)")
    return None


def memlet_subset_matches_descriptor(scope) -> Optional[str]:
    """Every memlet's ``subset`` rank must match the rank of the descriptor it
    accesses (``len(sdfg.arrays[memlet.data].shape)``). A memlet that reads a
    ``(1,)`` scalar bridge with a 2-D ``[0:8, 0:8]`` tile subset (or vice
    versa) is invalid — the descriptor and the access disagree on rank, which
    ``sdfg.validate()`` later rejects. Surfacing it as a pass post-condition
    localizes which pass widened the memlet without widening the descriptor (or
    staged a too-narrow bridge under a widened consumer).

    Accepts an SDFG or a single :class:`SDFGState`. Returns ``None`` on success
    or a description of the first offender.
    """
    for sd, state in _iter_states(scope):
        for edge in state.edges():
            mem = edge.data
            if mem is None or mem.data is None or mem.subset is None:
                continue
            desc = sd.arrays.get(mem.data)
            if desc is None:
                continue
            if len(mem.subset.size()) != len(desc.shape):
                src = edge.src.label
                dst = edge.dst.label
                return (f"{sd.name}.{state.label}: memlet ``{mem.data}`` subset rank "
                        f"{len(mem.subset.size())} != descriptor rank {len(desc.shape)} "
                        f"(shape {tuple(desc.shape)}) on edge {src} -> {dst}")
    return None


def logical_binops_are_bool(scope) -> Optional[str]:
    """Every ``TileBinop`` with a logical op (``&&`` / ``||``) must have two
    ``bool`` inputs (``_a``, ``_b``) and a ``bool`` output (``_c``). A logical
    op over non-bool operands is invalid — the operands are predicates / masks
    and the result is a predicate.

    Accepts an SDFG or a single :class:`SDFGState`. Returns ``None`` on
    success or a description of the first offender.
    """
    import dace.dtypes as _dt
    from dace.libraries.tileops import TileBinop
    for sd, state in _iter_states(scope):
        for node in state.nodes():
            if not isinstance(node, TileBinop) or node.op not in ("&&", "||"):
                continue
            for conn in ("_a", "_b", "_c"):
                edges = ([e for e in state.in_edges(node) if e.dst_conn == conn] +
                         [e for e in state.out_edges(node) if e.src_conn == conn])
                for e in edges:
                    if e.data is None or e.data.data is None:
                        continue
                    desc = sd.arrays.get(e.data.data)
                    if desc is not None and desc.dtype != _dt.bool_:
                        return (f"{sd.name}.{state.label}: logical TileBinop ``{node.label}`` (op {node.op}) "
                                f"connector ``{conn}`` is ``{e.data.data}`` of dtype {desc.dtype} (must be bool)")
    return None


def _is_reduce_at_output_map(map_entry) -> bool:
    """True iff ``map_entry`` is tagged for the ``reduce_at_output`` tile-reduce lowering
    (its body reduction is relocated to a boundary ``TileReduce`` after widening, so the
    WCR invariants intentionally allow the transient in-body WCR)."""
    from dace.transformation.passes.vectorization.mark_reduce_at_output import REDUCE_AT_OUTPUT_MARKER
    # Substring, not ``endswith``: the reduce-at-output tag is applied early (after
    # LiftEinsum), then ``SplitMapForTileRemainder`` appends its own ``__tile_main`` /
    # ``__scalar_tail`` suffix, so the marker ends up in the MIDDLE of the label
    # (``comp_mean__reduce_out__tile_main``).
    return REDUCE_AT_OUTPUT_MARKER in map_entry.map.label


def _is_reduce_at_output_map_node(node) -> bool:
    """True iff ``node`` -- a :class:`MapEntry` OR :class:`MapExit` -- belongs to a
    ``reduce_at_output``-tagged map (both expose ``.map.label``)."""
    from dace.transformation.passes.vectorization.mark_reduce_at_output import REDUCE_AT_OUTPUT_MARKER
    return REDUCE_AT_OUTPUT_MARKER in node.map.label


def _nsdfg_in_reduce_at_output_scope(sd) -> bool:
    """True iff nested SDFG ``sd``'s enclosing map (via ``parent_nsdfg_node``) is a tagged
    ``reduce_at_output`` map."""
    node = sd.parent_nsdfg_node
    st = getattr(sd, "parent", None)
    if node is None or st is None:
        return False
    scope = st.scope_dict()
    entry = scope.get(node)
    while entry is not None:
        if isinstance(entry, MapEntry) and _is_reduce_at_output_map(entry):
            return True
        entry = scope.get(entry)
    return False


def no_wcr_in_map_body(scope) -> Optional[str]:
    """No edge inside a map scope may carry a write-conflict resolution.

    **Legacy vectorization precondition.** ``VectorizeCPU`` vectorizes the
    tasklets of a free map in place; a surviving WCR inside the map body is a
    loop-carried reduction the vectorizer does NOT lower (it would widen the
    body without resolving the conflict, racing the lanes). ``WCRToAugAssign``
    must first convert every such WCR into an explicit read-modify-write
    tasklet -- this checker is that pass's post-condition / the vectorizer's
    entry pre-condition.

    "The map body" is the set of nodes strictly between a ``MapEntry`` and its
    ``MapExit`` (:meth:`~dace.sdfg.state.SDFGState.all_nodes_between`); every
    edge incident to one of those nodes is a body edge. The reduction-out
    boundary edge ``MapExit -> AccessNode`` is incident only to the exit and an
    outer AccessNode -- neither is in the body -- so it is not flagged: that is
    the legitimate place a reduction's WCR lives once lifted out of the body.

    Accepts an SDFG or a single :class:`SDFGState`. Returns ``None`` on success
    or a description of the first offending edge.
    """
    for sd, state in _iter_states(scope):
        for node in state.nodes():
            if not isinstance(node, MapEntry):
                continue
            # A tagged ``reduce_at_output`` map keeps its reduction WCR in the body until a
            # boundary ``TileReduce`` is spliced (post-widening); do not flag it here.
            if _is_reduce_at_output_map(node):
                continue
            body = state.all_nodes_between(node, state.exit_node(node))
            if not body:
                continue
            for edge in state.all_edges(*body):
                if edge.data is None or edge.data.wcr is None:
                    continue
                return (f"{sd.name}.{state.label}: edge {edge.src} -> {edge.dst} carries WCR "
                        f"``{edge.data.wcr}`` inside a map body (convert it to an explicit "
                        f"read-modify-write via WCRToAugAssign before vectorizing)")
    return None


def no_wcr_inside_nested_sdfgs(scope) -> Optional[str]:
    """No edge INSIDE any nested SDFG may carry a write-conflict resolution.

    **Multi-dim vectorization precondition.** The tile emitters lower the body
    NSDFG of the tile map assuming every inner edge is a plain (conflict-free)
    write (design 3.5). An inner WCR that survives into tiling is silently
    dropped, degrading e.g. an in-place ``a[i] += b[i]`` to ``a[i] = b[i]``.
    ``WCRToAugAssign`` (incl. its AN->AN copy case) must eliminate every inner
    WCR first -- this checker is that pass's post-condition.

    The ALLOWED scalar-reduction-out form -- the NSDFG writes a scalar that
    exits via a WCR reduction on the ``NestedSDFG -> MapExit`` edge in the
    PARENT state -- is not flagged: that edge lives in the parent SDFG, not
    inside the nested SDFG, so it is skipped by the ``parent_nsdfg_node`` guard.

    Accepts an SDFG or a single :class:`SDFGState`. Returns ``None`` on success
    or a description of the first offending edge.
    """
    for sd, state in _iter_states(scope):
        if sd.parent_nsdfg_node is None:
            continue
        # Skip a body NSDFG whose enclosing map is a tagged ``reduce_at_output`` map: its
        # inner reduction WCR is intentionally kept until the boundary ``TileReduce`` splice.
        if _nsdfg_in_reduce_at_output_scope(sd):
            continue
        for edge in state.edges():
            if edge.data is None or edge.data.wcr is None:
                continue
            # A WCR on the reduction-out chain of a tagged ``reduce_at_output`` map --
            # ``AN ─[wcr]→ MapExit`` / ``MapExit ─[wcr]→ acc`` -- is the intended scalar-out
            # form kept until the boundary ``TileReduce`` splice. The tagged reduction map
            # can itself be nested inside another (untagged) map's body NSDFG (azimint's
            # ``j`` reduction inside the ``i`` map's ``loop_body``), so the ``sd``-level scope
            # skip above misses it; recognise it per-edge by the incident tagged map exit/entry.
            if any(isinstance(n, (MapEntry, MapExit)) and _is_reduce_at_output_map_node(n)
                   for n in (edge.src, edge.dst)):
                continue
            return (f"{sd.name}.{state.label}: edge {edge.src} -> {edge.dst} carries WCR "
                    f"``{edge.data.wcr}`` inside a nested SDFG (lift genuine reductions to the "
                    f"NSDFG -> MapExit boundary; convert in-place RMW via WCRToAugAssign before tiling)")
    return None


# ---------------------------------------------------------------------------
# K-dim pipeline invariants (require widths / K context).
# ---------------------------------------------------------------------------


def lane_dep_transients_widened(sdfg: SDFG, K: int, widths: Tuple[int, ...]) -> Optional[str]:
    """Every lane-dependent transient in a tile-tagged body NSDFG is at
    the tile shape ``widths`` OR is an exempt bridge name (gather idx
    tile / ITE materialised tile / cond broadcast tile / Scalar bridge).

    Per user example 2026-06-12: ``all non-scalar non-gather dims are
    widened``.
    """
    import dace.data as _dd
    for _state, nsdfg_node, _map_entry in _tile_tagged_bodies(sdfg, K):
        inner_sdfg = nsdfg_node.sdfg
        for name, desc in inner_sdfg.arrays.items():
            if not desc.transient:
                continue
            if name.startswith("_idx_") or name.startswith("_ite_sym_tile") or name.startswith("_cond_bcast"):
                continue
            if isinstance(desc, _dd.Scalar):
                continue
            if not isinstance(desc, _dd.Array):
                continue
            shape = tuple(desc.shape)
            if shape == tuple(widths):
                continue
            try:
                if all(bool(dace.symbolic.simplify(s - 1) == 0) for s in shape):
                    continue
            except Exception:  # noqa: BLE001
                pass
            return (f"{inner_sdfg.name}: lane-dep transient ``{name}`` has shape {shape} "
                    f"!= widths {tuple(widths)} (expected widened or Scalar bridge)")
    return None


def tile_main_map_step_is_widths(sdfg: SDFG, K: int, widths: Tuple[int, ...]) -> Optional[str]:
    """Every TILE_MAIN map has its last-K dim steps == ``widths``."""
    from dace.transformation.passes.vectorization.split_map_for_tile_remainder import TILE_MAIN_MARKER
    for sd in sdfg.all_sdfgs_recursive():
        for state in sd.states():
            for node in state.nodes():
                if not isinstance(node, MapEntry):
                    continue
                if not node.map.label.endswith(TILE_MAIN_MARKER):
                    continue
                if len(node.map.range) < K:
                    continue
                tail_steps = tuple(node.map.range[-K + d][2] for d in range(K))
                if tuple(str(s) for s in tail_steps) != tuple(str(s) for s in widths):
                    return (f"{sd.name}.{state.label}: TILE_MAIN map ``{node.map.label}`` last-K steps "
                            f"{tail_steps} != expected widths {tuple(widths)}")
    return None


# ---------------------------------------------------------------------------
# Helpers (private).
# ---------------------------------------------------------------------------


def _iter_states(scope):
    """Yield ``(sub_sdfg, state)`` for an SDFG (every state recursively)
    OR a single state passed directly.
    """
    if isinstance(scope, SDFGState):
        yield scope.sdfg, scope
        return
    if isinstance(scope, SDFG):
        for sd in scope.all_sdfgs_recursive():
            for state in sd.states():
                yield sd, state
        return
    raise TypeError(f"Invariant scope must be SDFG or SDFGState, got {type(scope).__name__}")


def _tile_tagged_bodies(sdfg: SDFG, K: int):
    """Yield ``(state, nsdfg_node, map_entry)`` for every tile-tagged body NSDFG."""
    from dace.transformation.passes.vectorization.split_map_for_tile_remainder import (SCALAR_TAIL_MARKER,
                                                                                       TILE_K1_TAIL_MARKER)
    from dace.transformation.passes.vectorization.utils.map_predicates import is_innermost_map
    for sd in sdfg.all_sdfgs_recursive():
        for state in sd.states():
            for node in state.nodes():
                if not isinstance(node, MapEntry):
                    continue
                try:
                    if not is_innermost_map(state, node):
                        continue
                except (StopIteration, ValueError):
                    continue
                if len(node.map.params) < K:
                    continue
                if node.map.label.endswith(SCALAR_TAIL_MARKER) or node.map.label.endswith(TILE_K1_TAIL_MARKER):
                    continue
                try:
                    scope = state.scope_subgraph(node, include_entry=False, include_exit=False).nodes()
                except (StopIteration, ValueError):
                    continue
                nsdfgs = [n for n in scope if isinstance(n, NestedSDFG)]
                if len(nsdfgs) != 1:
                    continue
                yield state, nsdfgs[0], node
