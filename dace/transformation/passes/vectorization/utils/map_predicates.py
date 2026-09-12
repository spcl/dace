# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Boolean predicates and defensive assertions on maps / SDFGs.

Policy (locked): ``assert_X`` siblings kept alongside their ``X`` counterparts; every
loud-failure helper stays available. Removing them shifts silent corruption into the pipeline.
"""
import ast
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

import dace
from dace import SDFGState, data as dt, subsets, symbolic
from dace.sdfg.state import ConditionalBlock, LoopRegion
from dace.transformation.passes.analysis import map_scope
from dace.transformation.passes.vectorization.utils.injectivity import scatter_write_is_injective
from dace.transformation.passes.vectorization.utils.tasklets import LANE_ID_MATERIALISER_PREFIX

# Same helper as dace.transformation.passes.analysis.map_scope.map_body_nodes; re-exported here
# (not redefined) so every existing `from ...map_predicates import map_body_nodes` keeps working.
# Canonicalization must not import from vectorization, so analysis/ is the shared home; this
# module imports FROM it, never the reverse.
map_body_nodes = map_scope.map_body_nodes


def has_maps(sdfg: dace.SDFG) -> bool:
    """True if the SDFG or any nested SDFG contains a MapEntry.

    :param sdfg: The SDFG to inspect.
    :returns: ``True`` if any MapEntry exists in the hierarchy.
    """
    for n, g in sdfg.all_nodes_recursive():
        if isinstance(n, dace.nodes.MapEntry):
            return True
    return False


def is_innermost_map(state: SDFGState, map_entry: dace.nodes.MapEntry) -> bool:
    """True if a map is innermost -- no nested maps, including inside nested SDFGs.

    :param state: The state containing the map entry.
    :param map_entry: The map entry node to test.
    :returns: ``True`` if the map has no inner maps.
    """
    nodes_between = map_body_nodes(state, map_entry)
    if any(isinstance(node, dace.nodes.MapEntry) for node in nodes_between):
        return False
    return not any(isinstance(node, dace.nodes.NestedSDFG) and has_maps(node.sdfg) for node in nodes_between)


def _sdfg_has_self_recurrent_assign(root_sdfg: dace.SDFG) -> bool:
    """True if some interstate assignment in ``root_sdfg`` (recursively) is self-referential
    -- ``k = f(k)`` (e.g. ``k = k + inc``, ``k = (k + i) + 1``).

    Self-recurrent symbol = loop-carried recurrence depending on iteration history → enclosing
    map cannot be tiled/vectorized. Refused REGARDLESS of use (index/arithmetic/...): never
    vectorize over an un-eliminated recurrence. Linearizable recurrences (plain affine IV) are
    rewritten away upstream (``InductionVariableSubstitution`` / ``NormalizeStridedMaps``), so
    whatever survives here is genuine.

    Reductions NOT caught: an accumulator is a WCR dataflow edge (``x ─[wcr:+]→ acc``), never an
    interstate symbol assignment.
    """
    for sd in root_sdfg.all_sdfgs_recursive():
        for e in sd.all_interstate_edges():
            for lhs, rhs in (e.data.assignments or {}).items():
                try:
                    rhs_syms = {str(s) for s in dace.symbolic.pystr_to_symbolic(rhs).free_symbols}
                except Exception:  # noqa: BLE001 -- unparseable rhs is not a recurrence we model
                    rhs_syms = set()
                if lhs in rhs_syms:
                    return True
    return False


def _inner_syms_for(node: dace.nodes.NestedSDFG, param_syms: set[str]) -> set[str]:
    """The nest's own names for ``param_syms``, read off its ``symbol_mapping``.

    A body NSDFG rebinds names across its boundary (``symbol_mapping = {_j: _loop_it_1}``), so an
    OUTER param name matched against an INNER condition is a name-identity coincidence. Every
    caller crossing the boundary must translate first.

    :param node: the nested-SDFG node whose boundary is being crossed.
    :param param_syms: outer names to translate.
    :returns: the inner names bound to an expression over any of ``param_syms``.
    """
    inner = set()
    for inner_key, outer_val in node.symbol_mapping.items():
        try:
            free = {str(s) for s in dace.symbolic.pystr_to_symbolic(str(outer_val)).free_symbols}
        except Exception:  # noqa: BLE001 -- unparseable mapping expr: cannot bind, skip
            continue
        if param_syms & free:
            inner.add(inner_key)
    return inner


def _loop_bound_uses_symbols(region: LoopRegion, param_syms: set[str]) -> bool:
    """True if ``region``'s init / condition / update references any name in ``param_syms``."""
    for code in (region.loop_condition, region.init_statement, region.update_statement):
        if code is None:
            continue
        try:
            used = {str(s) for s in dace.symbolic.symbols_in_code(code.as_string)}
        except Exception:  # noqa: BLE001 -- unparseable loop head is not a dependence we model
            continue
        if used & param_syms:
            return True
    return False


def _sdfg_loops_depend_on_symbols(sdfg: dace.SDFG, param_syms: set[str]) -> bool:
    """True if some ``LoopRegion`` in ``sdfg`` (descending nested SDFGs, remapping ``param_syms``
    through each ``symbol_mapping``) has a bound referencing a symbol in ``param_syms``."""
    for region in sdfg.all_control_flow_regions(recursive=False):
        if isinstance(region, LoopRegion) and _loop_bound_uses_symbols(region, param_syms):
            return True
    for state in sdfg.all_states():
        for node in state.nodes():
            if not isinstance(node, dace.nodes.NestedSDFG):
                continue
            inner_syms = _inner_syms_for(node, param_syms)
            if inner_syms and _sdfg_loops_depend_on_symbols(node.sdfg, inner_syms):
                return True
    return False


def map_body_has_param_dependent_loop(state: SDFGState, map_entry: dace.nodes.MapEntry) -> bool:
    """True if the map body has a nested loop whose bound depends on a tiled map param.

    A triangular ``for i in range(1, j+1)`` inside a ``j``-map (TSVC s232) has a per-lane trip
    count once ``j`` is tiled -- the tiled body runs ONE tile-start bound for all W lanes and
    applies the inner (recurrence) body uniformly, under-computing the higher rows. No single
    strided loop honours W divergent bounds, so the map is refused (kept scalar, bit-exact) rather
    than tiled incorrectly.
    """
    params = {str(p) for p in map_entry.map.params}
    for node in map_body_nodes(state, map_entry):
        if not isinstance(node, dace.nodes.NestedSDFG):
            continue
        inner_syms = _inner_syms_for(node, params)
        if inner_syms and _sdfg_loops_depend_on_symbols(node.sdfg, inner_syms):
            return True
    return False


def map_body_has_inner_loop(state: SDFGState, map_entry: dace.nodes.MapEntry) -> bool:
    """True if the innermost map's body still contains a sequential loop (a ``LoopRegion``).

    The tile emitter lane-widens a body by rewriting its *dataflow*: each scalar operation becomes
    a width-W tile op. A ``LoopRegion`` is not dataflow -- it is a sequential trip counter over a
    carried state, and the tile pipeline has no unroller to flatten one away. Tiling it would give
    W lanes ONE shared counter and ONE shared carry, i.e. run the recurrence for the tile BASE index
    and apply lane 0's answer to all W lanes. That is exactly the shape of polybench adi's
    tridiagonal sweeps, deriche's IIR filter passes, and lu's triangular update -- all genuine
    carried recurrences, none of them lane-parallel at any width.

    So the vectorizable map is a LEAF: no inner map (:func:`is_innermost_map`) and no loop.

    CONDITIONALS ARE DELIBERATELY NOT REFUSED HERE. A branch is lowerable by MASKING -- evaluate
    both sides and select per lane -- which is what the ``TileMaskGen`` / ``TileITE`` machinery
    exists for, so ``if a[i] > 0: c[i] = ...`` vectorizes fine (18 TSVC kernels: s271, s441, vif,
    ...). The branches that genuinely cannot be masked are already refused, precisely, by
    :func:`map_body_has_tiled_param_dependent_branch` (a guard over a param about to be strided,
    where striding rebinds the param to the tile base) -- a targeted gate, not a blanket one.

    :param state: The state containing the map entry.
    :param map_entry: The (assumed innermost) map entry to test.
    :returns: ``True`` if the body carries a loop at any nesting depth.
    """
    for node in map_body_nodes(state, map_entry):
        if not isinstance(node, dace.nodes.NestedSDFG):
            continue
        for region in node.sdfg.all_control_flow_regions(recursive=True):
            if isinstance(region, LoopRegion):
                return True
    return False


def _sdfg_conditions_depend_on_symbols(sdfg: dace.SDFG, param_syms: set[str]) -> bool:
    """True if some ``ConditionalBlock`` in ``sdfg`` (descending nested SDFGs, remapping
    ``param_syms`` through each ``symbol_mapping``) has a guard constraining a symbol in
    ``param_syms``. Mirror of :func:`_sdfg_loops_depend_on_symbols` for guards instead of bounds."""
    # Function-local: utils/ is the leaf layer (utils/__init__ eagerly imports this module), so a
    # module-level edge to a pass module would close the cycle the package already dodges.
    # branch_normalization.py:669 defers the same module for the same reason.
    from dace.transformation.passes.vectorization.same_write_set_if_else_to_ite_cfg import condition_guards_symbols
    for region in sdfg.all_control_flow_regions(recursive=False):
        if isinstance(region, ConditionalBlock) and condition_guards_symbols(region, param_syms):
            return True
    for state in sdfg.all_states():
        for node in state.nodes():
            if not isinstance(node, dace.nodes.NestedSDFG):
                continue
            if _inner_syms_for(node, param_syms) and _sdfg_conditions_depend_on_symbols(
                    node.sdfg, _inner_syms_for(node, param_syms)):
                return True
    return False


def map_body_has_tiled_param_dependent_branch(state: SDFGState, map_entry: dace.nodes.MapEntry,
                                              iter_vars: tuple[str, ...]) -> bool:
    """True if the map body has a conditional whose guard constrains one of ``iter_vars``.

    Sibling of :func:`map_body_has_param_dependent_loop`, refused for the same reason. Tiling
    rebinds the TILED params from per-iteration index to TILE BASE, so a guard over one
    (``if i + 1 < mid``, TSVC s276) is evaluated ONCE per tile and lane 0 decides for all W lanes
    -- silently mis-predicating every lane whose own value of the guard differs. No single scalar
    branch honours W divergent predicates, exactly as no single strided loop honours W divergent
    bounds.

    ``iter_vars`` is the caller's ``params[-K:]``, NOT every param: tiling rebinds nothing else. A
    guard over a non-tiled leading param of the same merged ``(j, i)`` map, over an enclosing map's
    param, or over an enclosing sequential loop variable (``for jk: for jl: if jk > 1``, the
    cloudsc level guard) is uniform across the tiled lanes and must stay tileable -- hence an
    explicit set rather than :func:`condition_guards_iteration_symbol`, whose breadth (every
    enclosing iteration symbol) is right for if-conversion and wrong here.

    :param state: state holding ``map_entry``.
    :param map_entry: the candidate map.
    :param iter_vars: the params that will be strided to the tile width.
    :returns: ``True`` if the map must be left scalar.
    """
    params = {str(p) for p in iter_vars}
    for node in map_body_nodes(state, map_entry):
        if not isinstance(node, dace.nodes.NestedSDFG):
            continue
        inner_syms = _inner_syms_for(node, params)
        if inner_syms and _sdfg_conditions_depend_on_symbols(node.sdfg, inner_syms):
            return True
    return False


NO_VECTORIZE_MARKER = "__no_vectorize"
"""Label suffix marking a map as program-level SCAFFOLDING the vectorizer must never touch.

A guard, a counter fill, a trap: emitted by a correctness pass, sized by the program rather than
by the data being widened, and frequently of a rank the requested tiling has no meaning at. The
scatter guard's other pieces are already invisible -- its check is a library node, its trap a C++
tasklet -- and this names the same exemption for the one piece that is an ordinary Python map.
"""


def is_tile_eligible(state: SDFGState, map_entry: dace.nodes.MapEntry, K: int | None = None) -> bool:
    """True if an (assumed innermost) ``map_entry`` can be safely tiled/vectorized.

    Refuses a body with a self-referential loop-carried recurrence (``k = f(k)``, e.g. TSVC
    s141's ``k = (k + i) + 1`` feeding ``flat_2d_array[k]``): per-iteration recurrence → map
    stays scalar, refused REGARDLESS of use. Also refuses a body whose nested loop bound depends
    on a tiled map param (triangular ``for i in range(1, j+1)``, TSVC s232): W lanes need divergent
    trip counts a single tiled loop cannot honour. Same for a body whose conditional guards one of
    the params about to be strided (``if i + 1 < mid``, TSVC s276): striding rebinds that param to
    the TILE BASE, so lane 0's answer is applied to all W lanes. Graceful-refuse gate: ineligible
    map left as correct scalar rather than tiled incorrectly.

    ``K`` scopes the guard check to ``params[-K:]``, the only params striding rebinds. Without it
    every param counts, which refuses a lane-uniform guard over a merged map's leading dim
    (``for jk: for jl: if jk > 1``, cloudsc) -- correct but needlessly scalar.

    (A provably-too-small tiled dim -- extent < the tile width -- is refused by the width-aware
    passes ``MarkTileDims`` / ``SplitMapForTileRemainder``, not here: this predicate is width
    agnostic, and the same map may be tiled at width 1 (a scalar tail) but not at width 8.)
    """
    for node in map_body_nodes(state, map_entry):
        if isinstance(node, dace.nodes.NestedSDFG) and _sdfg_has_self_recurrent_assign(node.sdfg):
            return False
    if map_body_has_param_dependent_loop(state, map_entry):
        return False
    params = list(map_entry.map.params)
    iter_vars = tuple(params[-K:]) if K else tuple(params)
    if map_body_has_tiled_param_dependent_branch(state, map_entry, iter_vars):
        return False
    return True


def is_foreign_language_tasklet(node: dace.nodes.Node) -> bool:
    """True for a non-Python tasklet the tile emitters did not mint themselves.

    The vectorizer's OWN per-lane index materialiser (``materialise_lane_id_index_tile``) is
    EXCLUDED, mirroring the tile-lib-node carve-out in :func:`map_body_has_library_node` and the
    one the exit invariant ``no_widened_scalar_tasklets`` already makes: ``InsertTileLoadStore``
    stages a gather index with it one pass before ``ConvertTaskletsToTileOps``, already at tile
    shape, so counting it makes a half-tiled map refuse its own remaining tile passes.
    """
    if not isinstance(node, dace.nodes.Tasklet) or node.language == dace.dtypes.Language.Python:
        return False
    return not node.label.startswith(LANE_ID_MATERIALISER_PREFIX)


def map_body_has_foreign_language_tasklet(state: SDFGState, map_entry: dace.nodes.MapEntry) -> bool:
    """True if the map's body holds a tasklet whose code is NOT Python (recursively).

    The tile emitters rewrite a body via its Python AST, so a non-Python tasklet cannot be
    widened; its free symbols are also invisible to ``get_free_symbols``, which can drop a map
    parameter it reads from raw code text out of a nested SDFG's symbol mapping. The pipeline's
    own lane-id index materialiser is not such a tasklet -- see
    :func:`is_foreign_language_tasklet`.
    """
    for node in map_body_nodes(state, map_entry):
        if is_foreign_language_tasklet(node):
            return True
        if isinstance(node, dace.nodes.NestedSDFG) and any(
                is_foreign_language_tasklet(n) for n, _ in node.sdfg.all_nodes_recursive()):
            return True
    return False


def map_body_has_library_node(state: SDFGState, map_entry: dace.nodes.MapEntry) -> bool:
    """True if the map's body contains an OPAQUE library node (recursively, incl. nested SDFGs).

    Opaque = an external un-expanded primitive the tile emitters cannot see through (a sort, a
    ``ScatterConflictCheck``, a ``Reduce``, a BLAS call): a map wrapping one is left scalar
    until the node is expanded (after vectorization). This keeps the scatter guard's
    ``ScatterConflictCheck`` opaque to tiling.

    The vectorizer's OWN tile-op library nodes (``TileLoad`` / ``TileStore`` / ``TileBinop`` /
    ...) are EXCLUDED: they are inserted into the body DURING tiling, so treating them as
    opaque would make a half-tiled map refuse its own remaining tile passes.
    """
    from dace.libraries.tileops.nodes import (TileBinop, TileITE, TileLoad, TileMaskGen, TileReduce, TileStore,
                                              TileUnop)
    tile_ops = (TileBinop, TileITE, TileLoad, TileMaskGen, TileReduce, TileStore, TileUnop)

    def _opaque(n: dace.nodes.Node) -> bool:
        return isinstance(n, dace.nodes.LibraryNode) and not isinstance(n, tile_ops)

    for node in map_body_nodes(state, map_entry):
        if _opaque(node):
            return True
        if isinstance(node, dace.nodes.NestedSDFG) and any(_opaque(n) for n, _ in node.sdfg.all_nodes_recursive()):
            return True
    return False


def _inner_lane_vars(outer_params: tuple[str, ...], nsdfg_node: dace.nodes.NestedSDFG) -> tuple[str, ...]:
    """The tile lane variables as NAMED inside a body ``NestedSDFG``: the outer map params plus
    any inner symbol whose ``symbol_mapping`` binds it to an expression over an outer map param.

    A body NSDFG may rebind the map param (``symbol_mapping = {_j: _loop_it_1}``); classifying an
    interior write with only the OUTER param names would then miss the lane dependency and admit a
    non-injective scatter. Including the inner alias keeps the refusal check sound across the
    boundary.
    """
    op = set(outer_params)
    lane = set(outer_params)
    for isym, mexpr in nsdfg_node.symbol_mapping.items():
        try:
            syms = {str(s) for s in symbolic.pystr_to_symbolic(str(mexpr)).free_symbols}
        except Exception:  # noqa: BLE001 -- unparseable mapping expr: cannot bind, skip
            continue
        if syms & op:
            lane.add(str(isym))
    return tuple(lane)


def lane_param_aliases(outer_params: tuple[str, ...], nsdfg_node: dace.nodes.NestedSDFG) -> dict[str, str]:
    """Inner names that RENAME an outer map param, each mapped to the param it denotes.

    Only a bare-symbol binding qualifies. ``symbol_mapping = {_it: i}`` renames ``i`` and carries a
    per-lane injectivity argument across the boundary unchanged; ``{_it: i // 2}`` does not, and
    reading the second as a lane variable would prove a colliding write injective. This is the
    NARROW counterpart of :func:`_inner_lane_vars`, which deliberately over-approximates because
    over-approximating can only refuse more.

    :param outer_params: the enclosing map's parameters.
    :param nsdfg_node: the body NestedSDFG whose ``symbol_mapping`` binds the inner names.
    :returns: inner name -> outer map param, renamings only.
    """
    aliases: dict[str, str] = {}
    for isym, mexpr in nsdfg_node.symbol_mapping.items():
        try:
            expr = symbolic.pystr_to_symbolic(str(mexpr))
        except Exception:  # noqa: BLE001 -- unparseable mapping expr: not a renaming we can trust
            continue
        if expr.is_Symbol and str(expr) in outer_params:
            aliases[str(isym)] = str(expr)
    return aliases


@dataclass(frozen=True, slots=True)
class PerLaneWrite:
    """One non-transient per-lane WRITE in a map body, as the tile-lowerability gate reads it.

    ``iter_vars`` is the classification superset (see :func:`_inner_lane_vars`); ``lane_aliases`` is
    the exact inner-name-to-map-param renaming (see :func:`lane_param_aliases`) the injectivity
    proof needs.
    """
    subset: subsets.Range
    sdfg: dace.SDFG
    state: SDFGState
    iter_vars: tuple[str, ...]
    desc: dt.Data
    lane_aliases: dict[str, str]


def map_body_per_lane_subsets(state: SDFGState, map_entry: dace.nodes.MapEntry) -> Iterator[PerLaneWrite]:
    """Yield a :class:`PerLaneWrite` for every NON-transient per-lane WRITE (store) in the map
    body -- the map-exit boundary edges (flat-body form) and the AN in-edges inside a body
    ``NestedSDFG`` (nested form). Only WRITES are yielded: a gather READ is always sound, so the
    tile-lowerability gate need only inspect stores. Transients are skipped: only real
    (caller-visible) arrays constrain what the tile emitter must lower.

    Body states are enumerated with ``all_states()`` so a write nested inside a control-flow region
    (loop / conditional) of the body NSDFG is not missed (top-level ``states()`` would skip it).

    :raises StopIteration: at the CALL, not on first iteration, when ``map_entry`` has no exit in its
        scope. Raised inside a generator it would surface as ``RuntimeError`` (PEP 479), past every
        caller that refuses such a map by catching ``StopIteration``.
    """
    mx = state.exit_node(map_entry)

    def writes() -> Iterator[PerLaneWrite]:
        sdfg = state.sdfg
        outer_params = tuple(map_entry.map.params)
        flat_aliases = {p: p for p in outer_params}
        for e in list(state.in_edges(mx)):  # writes OUT of the body
            if e.data is not None and e.data.data is not None and e.data.subset is not None:
                desc = sdfg.arrays.get(e.data.data)
                if desc is not None and not desc.transient:
                    yield PerLaneWrite(e.data.subset, sdfg, state, outer_params, desc, flat_aliases)
        for node in map_body_nodes(state, map_entry):
            if isinstance(node, dace.nodes.NestedSDFG):
                inner_iter = _inner_lane_vars(outer_params, node)
                inner_aliases = lane_param_aliases(outer_params, node)
                for ist in node.sdfg.all_states():
                    for an in ist.nodes():
                        if not isinstance(an, dace.nodes.AccessNode):
                            continue
                        d = node.sdfg.arrays.get(an.data)
                        if d is None or d.transient:
                            continue
                        for e in ist.in_edges(an):  # writes into the AN
                            if e.data is not None and e.data.subset is not None:
                                yield PerLaneWrite(e.data.subset, node.sdfg, ist, inner_iter, d, inner_aliases)

    return writes()


def map_body_is_tile_lowerable(state: SDFGState,
                               map_entry: dace.nodes.MapEntry,
                               K: int | None = None,
                               scan_cache: dict[int, Any] | None = None) -> bool:
    """True unless the body has a per-lane WRITE the tile emitter cannot soundly lower.

    A tile iter-var nested inside a non-affine function -- ``a[i mod K]`` (a residue-scan seed),
    ``a[i**2]`` -- is single-element but NOT contiguous across lanes. Such an access is a GATHER
    (:func:`classify_tile_access` routes it there; the emitter builds a per-lane index tile
    ``[f(i+0), .., f(i+W-1)]`` and gathers). A gather READ is always sound. A gather WRITE
    (scatter) is sound only when the per-lane index is injective across the tile -- the author
    asserts that for a data-dependent ``a[idx[i]] = ...`` scatter (an array-index gather), but a
    FUNCTION index ``a[i mod K] = ...`` is provably non-injective when the modulus/period is < W
    (lanes ``l`` and ``l + K`` collide) -> concurrent tile lanes race on the same slot. Refuse a
    map with such a write so it stays unit-stride and scalar (bit-exact, last-writer-wins order
    preserved). READS with a function index are left eligible (the gather is correct).

    This gate is a SAFETY gate, so it FAILS CLOSED: a store whose subset cannot be classified (or
    whose gather begin cannot be parsed) is refused rather than admitted -- we cannot prove it
    injective, so we keep the map scalar. Classification uses the scope's lane vars (see
    :func:`map_body_per_lane_subsets`), a superset of the true W lane vars: it can only refuse
    MORE, never fewer.

    One class of write is ADMITTED with a proof rather than refused. A dim that ``a[i, i]`` drives
    twice is re-marked GATHER by :func:`classify_tile_access` as an EMITTER-DISPATCH decision (a
    diagonal is not a per-dim contiguous window), not as a soundness verdict. At ``K == 1`` such a
    store is decided by :func:`scatter_write_is_injective`: one dim whose index has a nonzero
    affine coefficient in the lane var separates every pair of lanes, so the lanes write disjoint
    boxes of a plain ``Array``. ``K > 1`` keeps refusing -- the rank argument there is a different
    one and is not built.

    :param state: state holding ``map_entry``.
    :param map_entry: the candidate map.
    :param K: number of innermost dims the CALLER would tile. Only ``K == 1`` unlocks the
        injectivity proof above; ``None`` and ``K > 1`` keep the plain refusal.
    :param scan_cache: optional dict memoizing the whole-SDFG symbol-definition scan across calls
        (see :func:`is_vectorizable_map`).
    :returns: ``True`` if every per-lane write in the body can be soundly widened.
    """
    from dace.transformation.passes.vectorization.utils.tile_access import (PerDimKind, classify_tile_access,
                                                                            build_symbol_definition_map)
    # One symbol-definition map per body, not per subset. Building it scans every interstate edge and
    # scalar write in the body, and a large tiled state yields many subsets that all share the same
    # body -- rebuilding per subset made this predicate quadratic in the state size, enough to look
    # like a hang on a two-level tiled stencil. Scoped to this call, so no pass can mutate the body
    # out from under it.
    sym_defs_cache: dict[tuple[int, int], dict[str, Any]] = {}
    # ``scan_cache`` shares the raw body scan across CALLS, and only a caller that owns an unmutated
    # span may keep one. A call-scoped one is sound unconditionally -- nothing mutates inside a
    # predicate -- and without it a body NestedSDFG spanning S states is re-scanned once per state,
    # since ``sym_defs_cache`` keys on the state too.
    if scan_cache is None:
        scan_cache = {}
    lane_param = map_entry.map.params[-1] if K == 1 and map_entry.map.params else None
    for write in map_body_per_lane_subsets(state, map_entry):
        key = (id(write.sdfg), id(write.state))
        if key not in sym_defs_cache:
            sym_defs_cache[key] = build_symbol_definition_map(write.sdfg, write.state, scan_cache=scan_cache)
        try:
            rec = classify_tile_access(write.subset,
                                       iter_vars=write.iter_vars,
                                       inner_sdfg=write.sdfg,
                                       state=write.state,
                                       sym_defs=sym_defs_cache[key])
        except Exception:  # noqa: BLE001 -- a store we cannot classify, we cannot prove injective:
            return False  # fail closed, keep the map scalar (bit-exact) rather than risk a race.
        if lane_param is not None:
            # Exactly one SPELLING of the widened param may occur: two names for one lane fall
            # outside the single-variable argument the proof rests on.
            names = {n for n, outer in write.lane_aliases.items() if outer == lane_param}
            lane_names = names & write.subset.free_symbols
            if len(lane_names) == 1 and scatter_write_is_injective(write.subset, next(iter(lane_names)), write.desc):
                continue
        for d, kind in enumerate(rec.per_dim_kind):
            if kind is not PerDimKind.GATHER:
                continue
            # Distinguish an EXPRESSION gather (function of the iter-var, non-injective scatter)
            # from an ARRAY-index gather (``idx[i]`` -- author-asserted injective). Only the former
            # (a begin with a tile iter-var and no array subscript) is refused.
            try:
                beg = symbolic.pystr_to_symbolic(str(write.subset.ranges[d][0]))
                beg_syms = {str(s) for s in beg.free_symbols}
                has_subscript = len(beg.atoms(symbolic.Subscript)) > 0
            except Exception:  # noqa: BLE001 -- unparseable gather begin: cannot prove injective
                return False
            if not has_subscript and any(v in beg_syms for v in write.iter_vars):
                return False
    return True


def _tasklet_mixes_statements_with_conditional(tasklet: dace.nodes.Tasklet) -> bool:
    """A Python tasklet body that mixes straight-line statements with a control-flow STATEMENT.

    ``SplitTasklets`` lowers only a ONE-statement body to single-op SSA and declines everything
    else, and ``NormalizeMaskedWriteTasklets`` only rewrites a body that is a LONE bare ``if``.
    A body like correlation's ``out = sqrt(inp / N)`` followed by ``if out <= 0.1: out = 1.0``
    falls through both: it survives as a scalar Python statement whose connectors the widener then
    swaps for tile pointers, emitting ``double* / int``. Neither pass owns it, so the map itself
    must not be a tile candidate.
    """
    if tasklet.language is not dace.dtypes.Language.Python:
        return False
    try:
        body = ast.parse(tasklet.code.as_string).body
    except (SyntaxError, ValueError):
        return False  # not the shape modelled here; other gates decide
    return len(body) > 1 and any(isinstance(s, (ast.If, ast.For, ast.While)) for s in body)


def map_body_has_mixed_conditional_tasklet(state: SDFGState, map_entry: dace.nodes.MapEntry) -> bool:
    """True if the map body (including inside body NestedSDFGs) holds a tasklet neither
    ``SplitTasklets`` nor ``NormalizeMaskedWriteTasklets`` can lower -- see
    :func:`_tasklet_mixes_statements_with_conditional`."""
    for node in map_body_nodes(state, map_entry):
        if isinstance(node, dace.nodes.Tasklet):
            if _tasklet_mixes_statements_with_conditional(node):
                return True
        elif isinstance(node, dace.nodes.NestedSDFG):
            for inner, _ in node.sdfg.all_nodes_recursive():
                if isinstance(inner, dace.nodes.Tasklet) and _tasklet_mixes_statements_with_conditional(inner):
                    return True
    return False


def is_vectorizable_map(state: SDFGState,
                        map_entry: dace.nodes.MapEntry,
                        K: int | None = None,
                        scan_cache: dict[int, Any] | None = None) -> bool:
    """Innermost AND tile-eligible AND no library node inside AND a loop-free body AND body
    tile-lowerable: the shared tile-candidate gate.

    All tile passes select through this predicate so an un-vectorizable map (non-innermost,
    recurrence-indexed, wrapping an opaque library node or a non-Python tasklet, or carrying a
    per-lane access the tile emitter cannot soundly widen -- see :func:`map_body_is_tile_lowerable`) is refused
    CONSISTENTLY -- never tiled by one pass while another skips it (the desync that strides a map
    by W over a body that stays scalar).

    The cheap structural gates (``is_innermost_map`` / ``is_tile_eligible``) run FIRST so a
    non-innermost or ineligible map short-circuits before the expensive per-write body
    classification.

    :param state: state holding ``map_entry``.
    :param map_entry: the candidate map.
    :param K: number of innermost dims the CALLER would tile (``len(widths)``). Only the last K
        params get strided, so only a guard over those is per-lane; pass it and a guard over a
        non-tiled leading param stays tileable. ``None`` scopes to every param -- correct but
        conservative, for callers that do not tile.
    :param scan_cache: optional dict memoizing the whole-SDFG symbol-definition scan by SDFG identity
        across calls, so a per-map selection loop is not O(maps^2). The caller MUST keep it only for
        a span during which the SDFG is NOT mutated (see :func:`build_symbol_definition_map`). Default
        ``None`` disables the cache (identical behavior to before).
    :returns: ``True`` if every tile pass may treat this map as a candidate.
    """
    if map_entry.map.label.endswith(NO_VECTORIZE_MARKER):
        return False
    if not (is_innermost_map(state, map_entry) and is_tile_eligible(state, map_entry, K)):
        return False
    if map_body_has_library_node(state, map_entry):
        return False
    if map_body_has_foreign_language_tasklet(state, map_entry):
        return False
    if map_body_has_inner_loop(state, map_entry):
        return False
    if map_body_has_mixed_conditional_tasklet(state, map_entry):
        return False
    return map_body_is_tile_lowerable(state, map_entry, K, scan_cache=scan_cache)


def is_gpu_resident_map(state: SDFGState, map_entry: dace.nodes.MapEntry) -> bool:
    """True if a map executes on the GPU device.

    GPU-resident = itself ``GPU_Device``-scheduled (a kernel), OR a non-kernel (e.g.
    ``Sequential``) map nested inside a ``GPU_Device`` map. Nesting followed through enclosing
    map scopes in the state AND across NestedSDFG boundaries (``sdfg.parent_nsdfg_node``), so a
    sequential body map inside a nested SDFG still finds its parent GPU kernel.

    GPU tile vectorizer's applicability rule: half2 ``__device__`` intrinsics only compile
    inside a GPU kernel, so only GPU-resident innermost maps may be tiled; host-side maps skipped.

    :param state: The state containing the map entry.
    :param map_entry: The map entry node to test.
    :returns: ``True`` if the map runs inside a GPU kernel.
    """
    gpu_device = dace.dtypes.ScheduleType.GPU_Device
    if map_entry.map.schedule == gpu_device:
        return True
    # Walk scope tree up, crossing NestedSDFG boundaries; accept if any ancestor map is GPU_Device.
    node = map_entry
    cur_state = state
    while cur_state is not None:
        scope = cur_state.scope_dict()
        parent = scope.get(node)
        while parent is not None:
            if isinstance(parent, dace.nodes.MapEntry) and parent.map.schedule == gpu_device:
                return True
            parent = scope.get(parent)
        # Top scope of this state → ascend into enclosing NestedSDFG node (parent state) and
        # keep walking. ``parent_nsdfg_node`` is None at the top-level SDFG → terminates.
        owning_sdfg = cur_state.sdfg
        nsdfg_node = owning_sdfg.parent_nsdfg_node
        if nsdfg_node is None:
            return False
        node = nsdfg_node
        cur_state = owning_sdfg.parent
    return False


def map_consists_of_single_nsdfg_or_no_nsdfg(graph: dace.SDFGState, map_entry: dace.nodes.MapEntry) -> bool:
    """True if a map contains a single NestedSDFG or none at all.

    Reads the map's SCOPE (:func:`map_body_nodes`), so a body ending in a write-only scratch
    scalar is still inspected; ``all_nodes_between`` would answer "no NestedSDFG here" over a
    body holding one, and ``NestInnermostMapBodyIntoNSDFG`` re-nests on that answer.

    :param graph: The state containing the map.
    :param map_entry: The map entry to check.
    :returns: ``True`` if the map contains a single NestedSDFG or no NestedSDFG.
    """
    all_nodes = [
        k for k in map_body_nodes(graph, map_entry) if not isinstance(k, (dace.nodes.MapEntry, dace.nodes.MapExit))
    ]
    return (len(all_nodes) == 1 and isinstance(
        all_nodes[0], dace.nodes.NestedSDFG)) or not any(isinstance(n, dace.nodes.NestedSDFG) for n in all_nodes)


def get_single_nsdfg_inside_map(graph: dace.SDFGState, map_entry: dace.nodes.MapEntry) -> dace.nodes.NestedSDFG | None:
    """Return the sole NestedSDFG inside a map, or ``None`` if not exactly one.

    Reads the map's SCOPE (:func:`map_body_nodes`) -- see
    :func:`map_consists_of_single_nsdfg_or_no_nsdfg`.

    :param graph: The state containing the map.
    :param map_entry: The map entry to inspect.
    :returns: The single NestedSDFG node, or ``None``.
    """
    all_nodes = [
        k for k in map_body_nodes(graph, map_entry) if not isinstance(k, (dace.nodes.MapEntry, dace.nodes.MapExit))
    ]
    if len(all_nodes) == 1 and isinstance(all_nodes[0], dace.nodes.NestedSDFG):
        return all_nodes[0]
    return None


def _no_edge_attr_state(state: SDFGState, attr: str, recursive: bool) -> bool:
    """True iff no edge in ``state`` has the attribute set. ``recursive=True`` descends into NSDFGs."""
    for edge in state.edges():
        value = edge.data.wcr if attr == "wcr" else edge.data.other_subset
        if value is not None:
            return False
    if recursive:
        for node in state.nodes():
            if isinstance(node, dace.nodes.NestedSDFG):
                if not _no_edge_attr_sdfg(node.sdfg, attr, True):
                    return False
    return True


def _no_edge_attr_sdfg(sdfg: dace.SDFG, attr: str, recursive: bool) -> bool:
    """True iff no edge in any state of ``sdfg`` has the attribute set."""
    for state in sdfg.all_states():
        if not _no_edge_attr_state(state, attr, recursive):
            return False
    return True


def no_wcr(state: SDFGState, recursive: bool = True) -> bool:
    """True iff no edge in ``state`` has WCR set; recurses into NSDFGs by default."""
    return _no_edge_attr_state(state, "wcr", recursive)
