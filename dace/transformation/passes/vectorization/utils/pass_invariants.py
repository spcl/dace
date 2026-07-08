# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Invariant checkers for K-dim vectorization passes.

Per user direction 2026-06-12: pre/post-condition checks, always run, as plain functions (not a
pass-pass boolean). No mixin/inheritance/env gate.

* checker(SDFG or other args) → ``None`` if invariant holds, else violation string.
* :func:`assert_invariant` raises ``AssertionError`` on violation (pass name + description +
  offending node/edge/state).

Each pass calls checkers directly from ``apply_pass``:

.. code-block:: python

    def apply_pass(self, sdfg, _):
        result = self._do_work(sdfg)
        assert_invariant(no_memlet_dim_mismatch(sdfg),
                         "WidenAccesses", "memlet dim consistent")
        return result
"""
from typing import Optional, Tuple

import dace
from dace.sdfg import SDFG, SDFGState
from dace.sdfg.nodes import AccessNode, MapEntry, MapExit, NestedSDFG


def assert_invariant(violation: Optional[str], pass_name: str, description: str) -> None:
    """Raise :class:`AssertionError` if ``violation`` non-None.

    :param violation: checker return (``None`` on success, else offending-node description).
    :param pass_name: pass name for the error message.
    :param description: one-line invariant description.
    """
    if violation is None:
        return
    raise AssertionError(f"{pass_name}: invariant violated -- {description}: {violation}")


# ---------------------------------------------------------------------------
# Generic structural invariants (work at SDFG or per-state level).
# ---------------------------------------------------------------------------


def no_memlet_dim_mismatch(scope) -> Optional[str]:
    """``subset`` and ``other_subset`` ranks must match for memlets connecting a tasklet /
    lib-node / NSDFG connector to an AccessNode (or two such connectors).

    Exempt AN -> AN copies: pure copies between possibly different-rank descriptors (e.g. 4D
    slice → 1D flat buffer) → different-rank subsets intended.

    Exempt ``MapEntry`` / ``MapExit`` pass-through edges: scope plumbing, not the connector <-> AN
    edges targeted here; legitimately carry a different-rank ``other_subset`` when one side is a
    scalar staging element (e.g. 2-D point ``a[jk, jc]`` → scalar ``c_slice`` after
    ``ConvertLengthOneArraysToScalars``).
    """
    states = _iter_states(scope)
    for sd, state in states:
        for edge in state.edges():
            mem = edge.data
            if mem is None or mem.subset is None or mem.other_subset is None:
                continue
            # AN -> AN copies: different-rank subsets allowed (see docstring)
            if isinstance(edge.src, AccessNode) and isinstance(edge.dst, AccessNode):
                continue
            # Map entry/exit pass-through edges: out of scope (see docstring)
            if isinstance(edge.src, (MapEntry, MapExit)) or isinstance(edge.dst, (MapEntry, MapExit)):
                continue
            if len(mem.subset.size()) != len(mem.other_subset.size()):
                return (f"{sd.name}.{state.label}: memlet ``{mem.data}`` subset dim={len(mem.subset.size())} "
                        f"!= other_subset dim={len(mem.other_subset.size())}")
    return None


def no_transient_scalar_stores(scope) -> Optional[str]:
    """No TILE (multi-element) memlet may store into a TRANSIENT Scalar.

    K-dim design (user direction 2026-06-14): inside a body NSDFG a scalar *write* only targets a
    NON-transient program output (e.g. reduction result, section 3.5); TILE result into a TRANSIENT
    scalar = widening miss -- ``WidenAccesses`` should have widened that transient to a tile so the
    edge is ``tile -> tile``. Replaces old ``_maybe_elide_scalar_passthrough`` patch-fix.

    Allowed scalar load-staging: single element → transient scalar for a broadcast (e.g.
    ``a_const`` from ``a[0]`` feeding ``TileLoad(src_kind="Scalar")``).
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
    """No AccessNode may have zero in-edges AND zero out-edges. Accepts SDFG or a single state."""
    for sd, state in _iter_states(scope):
        for node in state.nodes():
            if not isinstance(node, AccessNode):
                continue
            if state.in_degree(node) == 0 and state.out_degree(node) == 0:
                return f"{sd.name}.{state.label}: isolated AccessNode ``{node.data}``"
    return None


def no_duplicate_connector_edges(scope) -> Optional[str]:
    """Every NSDFG / Tasklet / lib-node connector has <=1 edge per direction.

    Skips :class:`~dace.sdfg.nodes.MapEntry` / :class:`~dace.sdfg.nodes.MapExit`: their
    pass-through connectors fan-out (entry ``OUT_X``) and fan-in (exit ``IN_X``) by design.
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
    """Every edge feeding a tile lib-node ``_mask`` connector must source from a ``bool`` array.

    Mask selects per-lane → non-bool mask (e.g. ``double`` 1.0/0.0) invalid. Comparison ops and
    lifted if-conditions produce ``bool``; every mask consumer (TileBinop / TileUnop / TileITE
    ``_mask``) defined over a boolean tile.
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

    Iteration mask branch-independent ("which lanes in bounds") → producer must DOMINATE every
    masked consumer; else a data-dependent ``if`` (→ TileITE) body reads ``_tile_iter_mask`` from a
    branch state the producer doesn't dominate (uninitialized lanes, flaky writes). Start block has
    no predecessors + dominates every reachable state → simplest sufficient guarantee. Post-condition
    of ``GenerateTileIterationMask`` (emits it in a dedicated ``_tile_mask_init`` start state).
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
    """Every memlet's ``subset`` rank must match the accessed descriptor's rank
    (``len(sdfg.arrays[memlet.data].shape)``). E.g. a ``(1,)`` scalar bridge read with a 2-D
    ``[0:8, 0:8]`` tile subset (or vice versa) invalid -- ``sdfg.validate()`` later rejects it.
    Post-condition localizes which pass widened the memlet without widening the descriptor (or
    staged a too-narrow bridge under a widened consumer).
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
    """Every ``TileBinop`` with a logical op (``&&`` / ``||``) must have ``bool`` inputs
    (``_a``, ``_b``) and ``bool`` output (``_c``): operands = predicates / masks, result = predicate.
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


def no_wcr_in_map_body(scope) -> Optional[str]:
    """No edge inside a map scope may carry a write-conflict resolution.

    **Legacy vectorization precondition.** ``VectorizeCPU`` vectorizes a free map's tasklets in
    place; a surviving body WCR = loop-carried reduction it does NOT lower (widening the body
    without resolving the conflict races the lanes). ``WCRToAugAssign`` must first convert each such
    WCR to an explicit read-modify-write tasklet -- its post-condition / vectorizer entry pre-condition.

    Map body = nodes strictly between ``MapEntry`` and its ``MapExit``
    (:meth:`~dace.sdfg.state.SDFGState.all_nodes_between`); every incident edge is a body edge. The
    reduction-out boundary edge ``MapExit -> AccessNode`` touches only the exit + an outer
    AccessNode, not the body → not flagged: where a reduction's WCR legitimately lives once lifted out.
    """
    for sd, state in _iter_states(scope):
        for node in state.nodes():
            if not isinstance(node, MapEntry):
                continue
            body = state.all_nodes_between(node, state.exit_node(node))
            if not body:
                continue
            for edge in state.all_edges(*body):
                if edge.data is None or edge.data.wcr is None:
                    continue
                # Allowed: scalar/len-1 reduction-boundary WCR (``scalar/len1 -wcr-> MapExit -wcr->
                # AN``, e.g. ``acc = sum(A)``); resolved at boundary, no lane race. Per-element
                # scatter (``a[idx[i]] (op)= ...``, full-array sink) stays flagged. See
                # _is_reduction_boundary_wcr.
                if _is_reduction_boundary_wcr(sd, state, node, edge):
                    continue
                return (f"{sd.name}.{state.label}: edge {edge.src} -> {edge.dst} carries WCR "
                        f"``{edge.data.wcr}`` inside a map body (convert it to an explicit "
                        f"read-modify-write via WCRToAugAssign before vectorizing)")
    return None


def _is_reduction_boundary_wcr(sdfg, state, map_entry, edge) -> bool:
    """True iff ``edge`` is the allowed reduction-boundary WCR of ``map_entry``.

    Shape ``scalar / length-1 AN -wcr-> MapExit -wcr-> AccessNode``: WCR edge terminates at this
    map's exit and the reduction sink is a scalar / length-1 array. Resolved at the boundary by
    codegen (OpenMP ``reduction(op:var)`` / GPU block-reduce + atomic), never in the widened body
    → not a loop-carried in-body reduction the tiler would race.

    Rejects a per-element scatter (``a[idx[i]] (op)= ...``): its MapExit sink is a full array, not a
    scalar / length-1 accumulator.
    """
    map_exit = state.exit_node(map_entry)
    if edge.dst is not map_exit:
        return False
    conn = edge.dst_conn
    if not conn or not conn.startswith("IN_"):
        return False
    out_conn = "OUT_" + conn[len("IN_"):]
    outs = [e for e in state.out_edges(map_exit) if e.src_conn == out_conn]
    if len(outs) != 1 or not isinstance(outs[0].dst, AccessNode):
        return False
    desc = sdfg.arrays.get(outs[0].dst.data)
    if desc is None:
        return False
    # Scalar / length-1 array accumulator (genuine reduction target, not scatter into one element)
    if isinstance(desc, dace.data.Scalar):
        return True
    return isinstance(desc, dace.data.Array) and (desc.total_size == 1) == True


def no_wcr_inside_nested_sdfgs(scope) -> Optional[str]:
    """No edge INSIDE any nested SDFG may carry a write-conflict resolution.

    **Multi-dim vectorization precondition.** Tile emitters lower the body NSDFG assuming every
    inner edge is a plain conflict-free write (design 3.5). Inner WCR surviving into tiling is
    silently dropped, degrading e.g. in-place ``a[i] += b[i]`` → ``a[i] = b[i]``. ``WCRToAugAssign``
    (incl. its AN->AN copy case) must eliminate every inner WCR first -- its post-condition.

    ALLOWED scalar-reduction-out form (NSDFG writes a scalar exiting via a WCR reduction on the
    ``NestedSDFG -> MapExit`` edge in the PARENT state) not flagged: that edge lives in the parent
    SDFG, skipped by the ``parent_nsdfg_node`` guard.
    """
    for sd, state in _iter_states(scope):
        if sd.parent_nsdfg_node is None:
            continue
        for edge in state.edges():
            if edge.data is None or edge.data.wcr is None:
                continue
            return (f"{sd.name}.{state.label}: edge {edge.src} -> {edge.dst} carries WCR "
                    f"``{edge.data.wcr}`` inside a nested SDFG (lift genuine reductions to the "
                    f"NSDFG -> MapExit boundary; convert in-place RMW via WCRToAugAssign before tiling)")
    return None


# ---------------------------------------------------------------------------
# K-dim pipeline invariants (require widths / K context).
# ---------------------------------------------------------------------------


def lane_dep_transients_widened(sdfg: SDFG, K: int, widths: Tuple[int, ...]) -> Optional[str]:
    """Every lane-dependent transient in a tile-tagged body NSDFG is at tile shape ``widths`` OR an
    exempt bridge name (gather idx tile / ITE materialised tile / cond broadcast tile / Scalar
    bridge). Per user example 2026-06-12: all non-scalar non-gather dims widened.
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
    """Yield ``(sub_sdfg, state)`` for an SDFG (every state recursively) OR a single state directly."""
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
