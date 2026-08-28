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
from dace.dtypes import ReductionType
from dace.frontend.operations import detect_reduction_type
from dace.sdfg import SDFG, SDFGState
from dace.sdfg.nodes import AccessNode, MapEntry, MapExit, NestedSDFG
from dace.sdfg.state import ConditionalBlock, LoopRegion
from dace.transformation.dataflow.wcr_conversion import nested_connector_subset

#: Reduction ops a lifted array-slot boundary WCR may carry. The tile path folds the lanes with a
#: horizontal ``TileReduce`` and the boundary then combines one partial per tile, so the op must be
#: associative; a ``Custom`` (non-reassociable) WCR keeps the strict "no loose WCR" refusal.
_ASSOCIATIVE_REDUCTIONS = (ReductionType.Sum, ReductionType.Product, ReductionType.Min, ReductionType.Max,
                           ReductionType.Bitwise_And, ReductionType.Bitwise_Or, ReductionType.Bitwise_Xor,
                           ReductionType.Logical_And, ReductionType.Logical_Or)


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

    **Tile-vectorizer precondition.** A surviving body WCR is a loop-carried reduction the tile
    widener does NOT lower (widening the body without resolving the conflict races the lanes).
    ``WCRToAugAssign`` must first convert each such WCR to an explicit read-modify-write tasklet --
    its post-condition / the vectorizer's entry pre-condition.

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
                # Allowed: a LIFTED reduction boundary (``partial -wcr-> MapExit -> AN``, e.g.
                # ``acc = sum(A)`` or the per-row ``mean[j] = sum_i data[i, j]``); resolved at the
                # boundary, no lane race. A per-element scatter (``a[idx[i]] (op)= ...``) stays
                # flagged. See _is_lifted_reduction_wcr.
                if _is_lifted_reduction_wcr(sd, state, edge):
                    continue
                return (f"{sd.name}.{state.label}: edge {edge.src} -> {edge.dst} carries WCR "
                        f"``{edge.data.wcr}`` inside a map body (convert it to an explicit "
                        f"read-modify-write via WCRToAugAssign before vectorizing)")
    return None


def _reduction_chain_origin(state, edge):
    """The innermost ``body -> MapExit`` edge of the boundary chain ``edge`` sits on.

    A lifted reduction leaves its map scope as ``partial -[wcr]-> MapExit -> AccessNode``, and
    memlet propagation stamps the same WCR on every link (and on any further enclosing exit), so a
    checker may meet the chain at any of them. Walk back through the scope plumbing (``OUT_x`` on an
    exit ⟵ its single ``IN_x`` in-edge) to the ONE edge that carries the precise per-iteration write
    subset -- the only one a slot / lane analysis can be run on.

    :returns: The originating ``body -> MapExit`` edge, or ``None`` if ``edge`` is not on such a chain.
    """
    cur = edge
    while isinstance(cur.src, MapExit):
        conn = cur.src_conn
        if not conn or not conn.startswith("OUT_"):
            return None
        ins = [e for e in state.in_edges(cur.src) if e.dst_conn == "IN_" + conn[len("OUT_"):]]
        if len(ins) != 1:
            return None
        cur = ins[0]
    if not isinstance(cur.dst, MapExit) or not cur.dst_conn or not cur.dst_conn.startswith("IN_"):
        return None
    return cur


def _boundary_sink(state, origin):
    """The :class:`AccessNode` the lifted reduction at ``origin`` drains into, ONE map scope out.

    The reduction must resolve at the exit of its OWN map. A WCR that keeps escaping through a
    second, enclosing MapExit (correlation's ``corr[i, j] (+)= ...`` under ``comp_corr_row[i]``) is
    a reduction the tile path lowers WRONG today -- the partial leaves the tiled scope before the
    fold is resolved -- so it stays refused until that path exists.

    :returns: The sink AccessNode, or ``None`` if the chain forks or leaves through another scope.
    """
    outs = [e for e in state.out_edges(origin.dst) if e.src_conn == "OUT_" + origin.dst_conn[len("IN_"):]]
    if len(outs) != 1 or not isinstance(outs[0].dst, AccessNode):
        return None
    return outs[0].dst


def _iteration_symbols_in_scope(sdfg, state) -> set:
    """Symbol names bound by an ITERATION construct visible from ``state``: every map parameter in
    the state, every enclosing loop variable, plus the SDFG's own free symbols (sizes, and outer
    iterators arriving through a nested-SDFG symbol mapping).

    An index built only from these is affine in the loop nest, so it is a fixed accumulator slot per
    iteration point. Anything else is DATA-dependent -- a scatter ``a[idx[i]] (op)= ...`` promotes
    ``idx[i]`` to an interstate-assigned symbol, which is NOT in this set -- and must never be
    mistaken for a reduction slot (its target varies per lane, which the tile fold cannot express).
    """
    names = {str(s) for s in sdfg.free_symbols}
    for node in state.nodes():
        if isinstance(node, MapEntry):
            names |= set(node.map.params)
    region = state.parent_graph
    while region is not None and region is not sdfg:
        if isinstance(region, LoopRegion) and region.loop_variable:
            names.add(region.loop_variable)
        region = region.parent_graph
    return names


def _precise_write_subset(origin):
    """The per-iteration write subset of the ``body -> MapExit`` edge ``origin``.

    A body NestedSDFG that both reads and writes the accumulator array through ONE connector has its
    boundary memlet over-approximated to the bounding box of both (lu: ``A[i, j] (+)= -A[i, k] *
    A[k, j]`` widens to ``A[Min(i,k):Max(i,k)+1, Min(j,k):Max(j,k)+1]``), which hides the fixed
    single-element accumulator slot. Recover it from the inner writes.
    """
    if isinstance(origin.src, NestedSDFG) and origin.src_conn is not None:
        precise = nested_connector_subset(origin.src, origin.src_conn, writes=True, boundary_subset=origin.data.subset)
        if precise is not None:
            return precise
    return origin.data.subset


def _is_lifted_reduction_wcr(sdfg, state, edge) -> bool:
    """True iff ``edge`` belongs to an allowed LIFTED reduction boundary chain.

    Shape ``partial -wcr-> MapExit -> AccessNode``, accepted when the accumulator is either

    * a scalar / length-1 array (``acc = sum(A)``) -- codegen lowers it directly (OpenMP
      ``reduction(op:var)`` / GPU block-reduce + atomic); or
    * ONE element of a larger array whose index does not vary with the map's INNERMOST parameter
      -- the dim the tiler widens into lanes. ``mean[j] (+)= data[i, j]`` over a collapsed
      ``[j, i]`` map, trmm's ``tmp (+)= A[k, i] * B[k, j]``: within one tile the slot is fixed, so
      the body folds to a single ``TileReduce`` partial and the boundary resolves one accumulation
      per tile. Only a recognised associative op qualifies -- a ``Custom`` WCR is not reassociable
      across lanes.

    Rejects a per-lane scatter: a slot varying with the innermost param (``B[i] (+)= ...``) or
    addressed by a data-dependent symbol (``a[idx[i]] (op)= ...``) has no single accumulator per
    tile, so the widener would race the lanes. Also rejects an accumulator the map READS (lu) and a
    fold that escapes through a second enclosing scope (see :func:`_boundary_sink`).
    """
    origin = _reduction_chain_origin(state, edge)
    if origin is None:
        return False
    sink = _boundary_sink(state, origin)
    if sink is None:
        return False
    desc = sdfg.arrays.get(sink.data)
    if desc is None:
        return False
    # Scalar / length-1 array accumulator: the classic lifted scalar reduction.
    if isinstance(desc, dace.data.Scalar):
        return True
    if isinstance(desc, dace.data.Array) and (desc.total_size == 1) == True:  # noqa: E712 -- sympy
        return True
    subset = _precise_write_subset(origin)
    if subset is None or subset.num_elements() != 1:
        return False
    if detect_reduction_type(origin.data.wcr) not in _ASSOCIATIVE_REDUCTIONS:
        return False
    map_entry = state.entry_node(origin.dst)
    if map_entry is None or not map_entry.map.params:
        return False
    # The accumulator array must not also be READ inside the map: lu's ``A[i, j] (+)= -A[i, k] *
    # A[k, j]`` routes reads and the reduction write through ONE connector, so the body is a
    # recurrence over the same array rather than a fold of independent addends -- the tile
    # staging cannot separate the accumulator from its operands. Mirrors the same guard in
    # ``PrepareReductionForWidening``.
    if any(e.data is not None and e.data.data == sink.data for e in state.out_edges(map_entry)):
        return False
    syms = {str(s) for s in subset.free_symbols}
    if map_entry.map.params[-1] in syms:
        return False
    return syms <= _iteration_symbols_in_scope(sdfg, state)


def no_wcr_inside_nested_sdfgs(scope) -> Optional[str]:
    """No edge INSIDE any nested SDFG may carry a write-conflict resolution.

    **Multi-dim vectorization precondition.** Tile emitters lower the body NSDFG assuming every
    inner edge is a plain conflict-free write (design 3.5). Inner WCR surviving into tiling is
    silently dropped, degrading e.g. in-place ``a[i] += b[i]`` → ``a[i] = b[i]``. ``WCRToAugAssign``
    (incl. its AN->AN copy case) must eliminate every inner WCR first -- its post-condition.

    ALLOWED scalar-reduction-out form (NSDFG writes a scalar exiting via a WCR reduction on the
    ``NestedSDFG -> MapExit`` edge in the PARENT state) not flagged: that edge lives in the parent
    SDFG, skipped by the ``parent_nsdfg_node`` guard.

    A reduction whose OWN map lives inside the nested SDFG (trmm's ``tmp (+)= A[k, i] * B[k, j]``
    under a nested ``computecol`` scope) leaves through a MapExit of that inner SDFG, so the same
    lifted-boundary chain :func:`_is_lifted_reduction_wcr` accepts at top level is accepted here --
    nesting does not change how the tiler folds it, and forbidding it merely refuses kernels the
    top-level form vectorizes.
    """
    for sd, state in _iter_states(scope):
        if sd.parent_nsdfg_node is None:
            continue
        for edge in state.edges():
            if edge.data is None or edge.data.wcr is None:
                continue
            if _is_lifted_reduction_wcr(sd, state, edge):
                continue
            return (f"{sd.name}.{state.label}: edge {edge.src} -> {edge.dst} carries WCR "
                    f"``{edge.data.wcr}`` inside a nested SDFG (lift genuine reductions to the "
                    f"NSDFG -> MapExit boundary; convert in-place RMW via WCRToAugAssign before tiling)")
    return None


# ---------------------------------------------------------------------------
# K-dim pipeline invariants (require widths / K context).
# ---------------------------------------------------------------------------


def no_widened_scalar_tasklets(sdfg: SDFG, K: int, widths: Tuple[int, ...]) -> Optional[str]:
    """No plain Python :class:`~dace.sdfg.nodes.Tasklet` inside a tile-tagged body may still read or
    write a TILE.

    Every tile-shaped operand belongs to a tile lib node after ``ConvertTaskletsToTileOps``. A
    tasklet the converter could not classify keeps its scalar Python body while the widener has
    already swapped its connectors for ``(W,)`` buffers, so codegen emits the body verbatim against
    a pointer (``out = sqrt(inp / N)`` with ``inp`` a ``double*``): a compile error at best, a
    silently wrong per-lane value at worst. Neither is a result the vectorizer may hand back, so the
    orchestrator refuses the kernel and leaves it correct + scalar instead.
    """
    import dace.data as _dd
    for _state, nsdfg_node, _map_entry in _tile_tagged_bodies(sdfg, K):
        inner_sdfg = nsdfg_node.sdfg
        for state in inner_sdfg.states():
            for node in state.nodes():
                # Only PYTHON tasklets: a hand-written per-lane C++ body (the lane-id materialiser's
                # ``DACE_UNROLL for (__l0 ...)``) is already tile-aware by construction and is never
                # meant to become a lib node.
                if not isinstance(node, dace.nodes.Tasklet) or node.language is not dace.dtypes.Language.Python:
                    continue
                for edge in (*state.in_edges(node), *state.out_edges(node)):
                    if edge.data is None or edge.data.data is None:
                        continue
                    desc = inner_sdfg.arrays.get(edge.data.data)
                    if not isinstance(desc, _dd.Array) or tuple(desc.shape) != tuple(widths):
                        continue
                    return (f"{inner_sdfg.name}.{state.label}: tasklet ``{node.label}`` still holds "
                            f"the scalar body ``{node.code.as_string.strip()!r}`` while its operand "
                            f"``{edge.data.data}`` was widened to a {tuple(widths)} tile "
                            f"(ConvertTaskletsToTileOps could not classify it)")
    return None


def no_lane_collapsing_nested_sdfgs(sdfg: SDFG, K: int, widths: Tuple[int, ...]) -> Optional[str]:
    """No NestedSDFG inside a tile-tagged body may read/write a TILE through a single-element
    connector.

    A body NSDFG the branch front could not inline (npbench ``nbody``'s ``np.power(a, -1.5,
    out=a, where=I)``) keeps scalar inner descriptors while the tile body around it holds
    ``(W,)`` operands. Codegen then runs it ONCE at the tile base -- lane 0 decides for all W
    lanes and the result is broadcast -- with no compile error and no other invariant tripped.
    Refuse the kernel instead of handing back a silently wrong per-lane value.
    """
    import dace.data as _dd
    for _state, nsdfg_node, _map_entry in _tile_tagged_bodies(sdfg, K):
        inner_sdfg = nsdfg_node.sdfg
        for state in inner_sdfg.states():
            for node in state.nodes():
                if not isinstance(node, NestedSDFG):
                    continue
                edges = [(e, e.dst_conn) for e in state.in_edges(node) if e.dst_conn]
                edges += [(e, e.src_conn) for e in state.out_edges(node) if e.src_conn]
                for edge, conn in edges:
                    if edge.data is None or edge.data.data is None:
                        continue
                    outer = inner_sdfg.arrays.get(edge.data.data)
                    if not isinstance(outer, _dd.Array) or tuple(outer.shape) != tuple(widths):
                        continue
                    inner = node.sdfg.arrays.get(conn)
                    if inner is None:
                        continue
                    try:
                        collapsed = int(inner.total_size) == 1
                    except (TypeError, ValueError):
                        collapsed = False
                    if collapsed:
                        return (f"{inner_sdfg.name}.{state.label}: nested SDFG ``{node.label}`` reads/writes "
                                f"tile ``{edge.data.data}`` {tuple(widths)} through single-element connector "
                                f"``{conn}`` -- it would run once at the tile base (lane 0 for all lanes)")
    return None


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
            if isinstance(desc, _dd.View):
                continue  # alias of the viewed array: widened in place, never descriptor-swapped
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


def no_strided_map_param_in_surviving_condition(sdfg: SDFG, K: int) -> Optional[str]:
    """No STRIDED map still carries a conditional guarding one of its own params.

    Striding rebinds a map's params from per-iteration index to TILE BASE, so a guard over one
    (``if i + 1 < mid``, TSVC s276) is then evaluated ONCE per tile: lane 0 decides for all W
    lanes. ``is_tile_eligible`` refuses such maps up front, but a pre-check proves nothing about
    the post-state -- this is the post-condition, at the one place the rebinding happens.

    Keyed on the ACTUAL STEP, not on ``TILE_MAIN_MARKER``: a map strided by any other route
    (``FuseBranchedTailRemainder`` re-strides by label match alone) is covered too.

    :param sdfg: the SDFG after striding.
    :param K: number of tiled (innermost) dims.
    :returns: an error string, or ``None`` when the invariant holds.
    """
    from dace.transformation.passes.vectorization.utils.map_predicates import (map_body_has_tiled_param_dependent_branch
                                                                               )
    for sd in sdfg.all_sdfgs_recursive():
        for state in sd.states():
            for node in state.nodes():
                if not isinstance(node, MapEntry) or len(node.map.range) < K or len(node.map.params) < K:
                    continue
                if all(str(node.map.range[-K + d][2]) == '1' for d in range(K)):
                    continue  # not strided: nothing was rebound to a tile base
                if map_body_has_tiled_param_dependent_branch(state, node, tuple(node.map.params[-K:])):
                    return (f"{sd.name}.{state.label}: strided map ``{node.map.label}`` still holds a "
                            f"conditional guarding its own param -- that guard is evaluated once per "
                            f"tile, not per lane")
    return None


def no_conditional_interstate_assign_on_widened_data(sdfg: SDFG, widths: Tuple[int, ...]) -> Optional[str]:
    """No ``ConditionalBlock`` guarded by WIDENED data may assign a symbol on an interstate edge.

    The lane-varying analogue of :func:`no_strided_map_param_in_surviving_condition`. A symbol holds
    ONE value for the whole tile, so an assignment only some lanes reach cannot be represented.
    Widening turns the guard's operand into a ``(W,)`` buffer while the guard itself stays scalar
    control flow, and codegen then emits ``if (<bool[W]>)`` -- an array decaying to a never-null
    pointer, so every lane takes the branch. Branch lowering must first rewrite the assignment as
    dataflow (an ITE tasklet writing a scalar the widener widens per lane).

    :param sdfg: the SDFG after widening.
    :param widths: tile widths; a descriptor whose last dim is one of them is a per-lane buffer.
    :returns: an error string, or ``None`` when the invariant holds.
    """
    for block in sdfg.all_control_flow_blocks(recursive=True):
        if not isinstance(block, ConditionalBlock):
            continue
        for condition, region in block.branches:
            if condition is None:
                continue
            assigned = [
                e for r in region.all_control_flow_regions(recursive=True) for e in r.edges() if e.data.assignments
            ]
            if not assigned:
                continue
            for name in condition.get_free_symbols():
                desc = block.sdfg.arrays.get(name)
                if desc is None or not desc.shape or desc.shape[-1] not in widths:
                    continue
                keys = sorted(k for e in assigned for k in e.data.assignments)
                return (f"{block.sdfg.name}.{block.label}: conditional on widened ``{name}`` "
                        f"{desc.shape} assigns {keys} on an interstate edge -- one symbol cannot "
                        f"hold a per-lane value, so the guard runs for every lane")
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
