# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Collapse a manually-tiled loop nest back to a single unit-stride loop per axis."""

import copy
import dataclasses
import functools
from collections.abc import Iterator
from typing import Any

import sympy

import dace
from dace import SDFG, dtypes, properties, subsets, symbolic
from dace.sdfg import nodes
from dace.sdfg.graph import NodeNotFoundError
from dace.sdfg.narrowing import as_basic, as_expr, as_range
from dace.sdfg.state import BreakBlock, ControlFlowRegion, LoopRegion, SDFGState
from dace.transformation import pass_pipeline as ppl
from dace.transformation import transformation as xf
from dace.transformation.passes.analysis import loop_analysis
from dace.transformation.passes.canonicalize.fresh_names import lowest_free_suffix
from dace.transformation.passes.canonicalize.tracked_assumptions import record_assumption

#: Prefix for the synthesised unit-stride iterator that replaces the (i, ii) pair.
UNTILE_PREFIX = "_untile_k_"


def _try_extract_perfect_one_child(cfg: ControlFlowRegion) -> ControlFlowRegion | None:
    """Return the single non-empty child block of ``cfg`` if it has exactly one, otherwise ``None``."""
    candidate: ControlFlowRegion | None = None
    for b in cfg.nodes():
        if isinstance(b, SDFGState):
            if len(b.nodes()) > 0:
                return None
            continue
        if not isinstance(b, ControlFlowRegion):
            return None
        if candidate is not None:
            return None
        candidate = b
    return candidate


def _iter_candidate_inners(outer: LoopRegion) -> Iterator[LoopRegion]:
    """Walk down through perfect 1-child intermediate chains, yielding every descendant :class:`LoopRegion` as a
    potential tile-pair partner for ``outer``."""
    seen: dict[int, None] = {}
    current: ControlFlowRegion = outer
    while True:
        nxt = _try_extract_perfect_one_child(current)
        if nxt is None or id(nxt) in seen:
            return
        seen[id(nxt)] = None
        if isinstance(nxt, LoopRegion):
            yield nxt
        current = nxt


def _intermediate_chain_clean(outer: LoopRegion, inner: LoopRegion, outer_var: str) -> bool:
    """``True`` iff every LoopRegion strictly between ``outer`` and ``inner`` is free of references to
    ``outer.loop_variable`` in its iteration descriptors, and no edge from ``outer`` down to ``inner`` decides or
    assigns: such an edge runs per tile, and would run per point."""
    current = inner.parent_graph
    while True:
        if any(e.data.assignments or not e.data.is_unconditional() for e in current.edges()):
            return False
        if current is outer:
            return True
        if isinstance(current, LoopRegion):
            for code in (current.init_statement, current.loop_condition, current.update_statement):
                if code is None:
                    continue
                try:
                    names = free_symbol_names(code.as_string)
                except Exception:
                    names = frozenset()
                if outer_var in names:
                    return False
        current = current.parent_graph


def tiles_visit_points_in_order(outer: LoopRegion, inner: LoopRegion) -> bool:
    """Whether the nest from ``outer`` down to ``inner`` computes the same tile by tile as row by row."""
    tiles, points, region = {outer.loop_variable}, set(), inner.parent_graph
    while region is not outer:
        (points if _tile_size(loop_analysis.get_loop_stride(region)) is None else tiles).add(region.loop_variable)
        region = region.parent_graph
    points |= {r.loop_variable for r in [inner, *inner.all_control_flow_regions()] if isinstance(r, LoopRegion)}
    if any(e.data.assignments for e in inner.all_interstate_edges()):
        return False
    reads: dict[str, list[list]] = {}
    writes: dict[str, list[list]] = {}
    for st in inner.states():
        if not all(isinstance(n, (nodes.AccessNode, nodes.Tasklet)) for n in st.nodes()):
            return False
        for e in st.edges():
            if e.data.is_empty():
                continue
            sides = ((e.src, e.data.get_src_subset(e, st), reads), (e.dst, e.data.get_dst_subset(e, st), writes))
            for node, subset, accesses in sides:
                if isinstance(node, nodes.AccessNode):
                    if subset is None or subset.num_elements() != 1:
                        return False
                    exprs = [symbolic.pystr_to_symbolic(str(rng[0])) for rng in as_range(subset).ranges]
                    exprs = [x.subs({s: 0 for s in x.free_symbols if str(s) in tiles}) for x in exprs]
                    accesses.setdefault(node.data, []).append(exprs)
    if loop_analysis.control_reads(outer) & writes.keys():
        return False
    sdfg = outer.sdfg
    for data, written in writes.items():
        # A body's scratch is private to the point, in either order.
        if sdfg.arrays[data].transient and not loop_analysis.read_outside(sdfg, outer, set(), {data}):
            continue
        for element in written:
            axes = [[s for s in x.free_symbols if str(s) in points] for x in element]
            if any(len(axis) > 1 for axis in axes) or len(points) - sum(map(len, axes)) > 1:
                return False
            if any(x.coeff(s) != 1 for x, axis in zip(element, axes) for s in axis):
                return False
            for other in written + reads.get(data, []):
                distance = [symbolic.simplify(a - b) for a, b in zip(element, other)]
                if not all(d.is_number for d in distance) or (max(distance) > 0 and min(distance) < 0):
                    return False
    return True


def _is_constant_positive_int(expr: symbolic.SymbolicType) -> int | None:
    """If ``expr`` simplifies to a positive integer literal, return that value."""
    try:
        s = symbolic.simplify(expr)
    except Exception:
        return None
    if not as_basic(s).is_number or not as_basic(s).is_Integer:
        return None
    v = int(as_expr(s))
    return v if v > 0 else None


def _is_zero(expr: symbolic.SymbolicType) -> bool:
    try:
        s = symbolic.simplify(expr)
    except Exception:
        return False
    return as_basic(s).is_number and s == 0


def _tile_size(expr: symbolic.SymbolicType) -> tuple[symbolic.SymbolicType, int | None] | None:
    """Classify an outer-loop stride as a tile size: ``(K_expr, K_const)`` or ``None``."""
    try:
        s = symbolic.simplify(expr)
    except Exception:
        return None
    if as_basic(s).is_number:
        if not as_basic(s).is_Integer:
            return None
        v = int(as_expr(s))
        if v <= 1:
            return None
        return (s, v)
    # Symbolic: accept a bare symbol only (assumed non-negative by DaCe convention).
    if isinstance(s, sympy.Symbol):
        return (s, None)
    return None


def tiles_a_parent_window(enclosing: frozenset[str], start: symbolic.SymbolicType, span: symbolic.SymbolicType) -> bool:
    """``True`` iff a tile scope walks a fixed-width window opened by an enclosing scope with iterators ``enclosing``."""
    if not enclosing or not any(str(s) in enclosing for s in as_basic(start).free_symbols):
        return False
    return not any(str(s) in enclosing for s in as_basic(span).free_symbols)


def enclosing_loop_variables(block: ControlFlowRegion | SDFGState) -> frozenset[str]:
    """The iterators of the loops around ``block`` in its SDFG."""
    names = []
    graph = block.parent_graph
    while graph is not None:
        if isinstance(graph, LoopRegion) and graph.loop_variable:
            names.append(graph.loop_variable)
        graph = graph.parent_graph
    return frozenset(names)


def union_stop(
    sdfg: SDFG,
    start: symbolic.SymbolicType,
    end: symbolic.SymbolicType,
    K_expr: symbolic.SymbolicType,
    clamped: bool,
    enclosing: frozenset[str],
) -> symbolic.SymbolicType:
    """Exclusive end of the union of the tiles ``[o, o + K)`` over the origins ``o`` in ``[start, end]`` step ``K``."""
    stop_excl = symbolic.simplify(end + 1)
    if clamped:
        return stop_excl
    span = symbolic.simplify(stop_excl - start)
    tiles_end = symbolic.simplify(symbolic.int_ceil(span, K_expr) * K_expr)
    if (
        _diff_is_zero(tiles_end, span)
        or as_basic(tiles_end).is_number
        or not tiles_a_parent_window(enclosing, start, span)
    ):
        return symbolic.simplify(start + tiles_end)
    record_assumption(sdfg, sympy.Eq(sympy.Mod(span, K_expr), 0))
    return stop_excl


def count_loops(sdfg: SDFG) -> int:
    """Number of iterating LoopRegions anywhere in ``sdfg`` (nested SDFGs included)."""
    return sum(
        1
        for sd in sdfg.all_sdfgs_recursive()
        for r in sd.all_control_flow_regions()
        if isinstance(r, LoopRegion) and r.loop_variable
    )


def plus_one(end: symbolic.SymbolicType) -> symbolic.SymbolicType:
    """``end + 1``, pushed into a ``Min`` (``Min(N - 1, i + K - 1) + 1`` is ``Min(N, i + K)``): sympy keeps the sum
    of a ``Min`` and a constant unevaluated, which no ``Min`` test then recognizes."""
    if isinstance(end, sympy.Min):
        return sympy.Min(*(symbolic.simplify(arg + 1) for arg in end.args))
    return symbolic.simplify(end + 1)


def clamp_is_the_parent_limit(
    end_plus_one: symbolic.SymbolicType,
    outer_sym: symbolic.SymbolicType,
    K_expr: symbolic.SymbolicType,
    outer_limit: symbolic.SymbolicType,
) -> bool:
    """``True`` iff ``end_plus_one`` is ``Min(parent limit, i + K)`` -- the remainder clamp."""
    if not isinstance(end_plus_one, sympy.Min):
        return False
    tile_reach = symbolic.simplify(outer_sym + K_expr)
    rest = []
    saw_tile_reach = False
    for arg in end_plus_one.args:
        if not saw_tile_reach and _diff_is_zero(arg, tile_reach):
            saw_tile_reach = True
            continue
        rest.append(arg)
    if not saw_tile_reach or not rest:
        return False
    # The remaining terms are compared TOGETHER, not one by one.
    return _diff_is_zero(sympy.Min(*rest), outer_limit)


def _match_inner_case(
    inner: LoopRegion,
    outer_var: str,
    K_expr: symbolic.SymbolicType,
    K_const: int | None,
    outer_limit: symbolic.SymbolicType | None = None,
) -> tuple[str, symbolic.SymbolicType, bool, bool] | None:
    """:func:`classify_inner_range` of the loop ``inner``."""
    stride = loop_analysis.get_loop_stride(inner)
    start = loop_analysis.get_init_assignment(inner)
    end = loop_analysis.get_loop_end(inner)
    if stride is None or start is None or end is None:
        return None
    return classify_inner_range(start, end, stride, outer_var, K_expr, K_const, outer_limit)


def classify_inner_range(
    start: symbolic.SymbolicType,
    end: symbolic.SymbolicType,
    stride: symbolic.SymbolicType,
    outer_var: str,
    K_expr: symbolic.SymbolicType,
    K_const: int | None,
    outer_limit: symbolic.SymbolicType | None = None,
) -> tuple[str, symbolic.SymbolicType, bool, bool] | None:
    """Classify the inner range ``[start, end]`` (inclusive end) step ``stride``: ``(case, inner_stride,
    needs_div_assumption, clamped)`` or ``None``."""
    S_concrete = _is_constant_positive_int(stride)
    needs_div_assumption = False
    if S_concrete is not None:
        inner_stride: symbolic.SymbolicType = S_concrete
        if S_concrete == 1:
            pass  # unit inner stride: a genuine single-level untile, always sound
        elif K_const is not None:
            # Concrete tile + concrete cascade stride: a whole tile only when it
            # divides exactly (no partial-tile remainder).
            if K_const % S_concrete != 0:
                return None
        else:
            # Concrete stride into a symbolic tile: divisibility unprovable -> admit
            # under a recorded ``K % S == 0`` assumption.
            needs_div_assumption = True
    else:
        # Symbolic inner stride: accept only a bare positive symbol (a block-size parameter -- the inner tile ``T2`` of
        # a ``T1``/``T2`` double tile), never a compound expression.
        s_stride = symbolic.simplify(stride)
        if not isinstance(s_stride, sympy.Symbol):
            return None
        inner_stride = s_stride
        needs_div_assumption = True
    outer_sym = symbolic.pystr_to_symbolic(outer_var)
    s_sym = symbolic.simplify(start)
    e_sym = symbolic.simplify(end)
    # ``get_loop_end`` returns ``exclusive_upper_bound - 1`` regardless of step (i.e. for ``range(a, b, S)`` it returns
    # ``b - 1``, NOT the actual last admitted value ``a + S * floor((b - a - 1) / S)``).
    if _is_zero(s_sym):
        if _diff_is_zero(e_sym, K_expr - 1):
            return ("A", inner_stride, needs_div_assumption, False)
        return None
    # Case B: start == i, end == i + K - 1
    if _diff_is_zero(s_sym, outer_sym):
        if _diff_is_zero(e_sym, outer_sym + K_expr - 1):
            return ("B", inner_stride, needs_div_assumption, False)
        if outer_limit is not None and clamp_is_the_parent_limit(plus_one(e_sym), outer_sym, K_expr, outer_limit):
            return ("B", inner_stride, needs_div_assumption, True)
    return None


def _diff_is_zero(a: symbolic.SymbolicType, b: symbolic.SymbolicType) -> bool:
    """``simplify(a - b) == 0`` if both sides reduce to the same value, else ``False``."""
    try:
        # Sides reach here from different sources
        if isinstance(a, sympy.Basic) and isinstance(b, sympy.Basic):
            a, b = symbolic.equalize_symbols_across(a, b)
        diff = symbolic.simplify(a - b)
    except Exception:
        return False
    if as_basic(diff).is_number:
        try:
            return int(as_expr(diff)) == 0
        except (TypeError, ValueError):
            return False
    return False


@functools.lru_cache(maxsize=4096, typed=True)
def free_symbol_names(text: str) -> frozenset[str]:
    """Names of the free symbols of the expression ``text``."""
    return frozenset(str(s) for s in symbolic.pystr_to_symbolic(text).free_symbols)


def memlet_bound_texts(memlet: dace.Memlet) -> list[str]:
    """Every axis bound (lo/hi/stride) of ``memlet``'s subset, then of its other subset, as text."""
    return [
        str(bound)
        for subset in (memlet.subset, memlet.other_subset)
        if subset is not None
        for rng in as_range(subset).ranges
        for bound in rng
    ]


def body_bound_texts(inner: LoopRegion) -> list[str]:
    """Every memlet bound in ``inner``'s body, as text: where the audit looks for references to ``i`` / ``ii``."""
    return [
        text
        for st in inner.states()
        for e in st.edges()
        if e.data is not None and not e.data.is_empty()
        for text in memlet_bound_texts(e.data)
    ]


def depends_only_on_sum(ex: sympy.Basic, i_sym: sympy.Symbol, ii_sym: sympy.Symbol) -> bool:
    """``True`` iff ``ex`` reads ``i`` and ``ii`` only through the sum ``i + ii``."""
    try:
        # diff goes through symbol IDENTITY: a loop var minted from its name differentiates to 0 against an expression
        # carrying another instance, which reads as "equal partials" and lets the rewrite through.
        ex, i_sym, ii_sym = symbolic.equalize_symbols_across(ex, i_sym, ii_sym)
        return symbolic.simplify(sympy.diff(ex, i_sym) - sympy.diff(ex, ii_sym)) == 0
    except (TypeError, ValueError, AttributeError, NotImplementedError):
        return False


def _audit_combined_access(inner: LoopRegion, outer_var: str, inner_var: str, case: str) -> bool:
    """The structural safety check the docstring describes."""
    # Interstate edges and conditions take no part in the sum: only memlets and Map ranges do.
    control = {name for e in inner.all_interstate_edges() for name in e.data.free_symbols}
    for region in inner.all_control_flow_regions():
        if region is not inner:
            control |= region.used_symbols(all_symbols=True, with_contents=False)
    if outer_var in control or (case == "A" and inner_var in control):
        return False
    texts = body_bound_texts(inner)
    for st in inner.states():
        for node in st.nodes():
            if isinstance(node, nodes.NestedSDFG):
                texts.extend(str(value) for value in node.symbol_mapping.values())
            elif isinstance(node, nodes.EntryNode):
                texts.extend(str(bound) for rng in node.map.range.ranges for bound in rng)
            elif isinstance(node, nodes.CodeNode) and not code_node_allows(node, outer_var, inner_var, case):
                return False
    if case == "B":
        return all(outer_var not in free_symbol_names(text) for text in texts)
    return all(reads_only_through_sum(text, outer_var, inner_var) for text in texts)


@functools.lru_cache(maxsize=4096, typed=True)
def reads_only_through_sum(text: str, outer_var: str, inner_var: str) -> bool:
    """:func:`depends_only_on_sum` of the bound ``text``, for the loop variables named ``outer_var`` and ``inner_var``."""
    return depends_only_on_sum(
        symbolic.pystr_to_symbolic(text), symbolic.pystr_to_symbolic(outer_var), symbolic.pystr_to_symbolic(inner_var)
    )


@dataclasses.dataclass(frozen=True)
class MapTilePair:
    """Dimension ``dim`` of ``outer`` strides by the tile ``K_expr``; dimension ``inner_dim`` of ``inner`` covers one
    tile (``case``, ``stride``, ``needs_div`` and ``clamped`` as :func:`classify_inner_range` returns them)."""

    outer: nodes.MapEntry
    dim: int
    inner: nodes.MapEntry
    inner_dim: int
    case: str
    stride: symbolic.SymbolicType
    needs_div: bool
    clamped: bool
    K_expr: symbolic.SymbolicType


def code_node_allows(node: nodes.CodeNode, outer_var: str, inner_var: str, case: str) -> bool:
    """Code nodes may name neither iterator in case A, and not ``outer_var`` in case B."""
    named = {str(s) for s in node.free_symbols}
    return outer_var not in named and (case != "A" or inner_var not in named)


def map_body_reads_only(state: SDFGState, inner: nodes.MapEntry, outer_var: str, inner_var: str, case: str) -> bool:
    """Whether the scope of ``inner`` reads ``outer_var`` only as case ``case`` allows: never (B), or only through
    ``outer_var + inner_var`` (A)."""
    scope = state.scope_subgraph(inner)
    texts = [text for e in scope.edges() if not e.data.is_empty() for text in memlet_bound_texts(e.data)]
    for node in scope.nodes():
        if node is inner:
            continue
        if isinstance(node, nodes.EntryNode):
            texts.extend(str(bound) for rng in node.map.range.ranges for bound in rng)
        elif isinstance(node, nodes.NestedSDFG):
            texts.extend(str(value) for value in node.symbol_mapping.values())
        elif isinstance(node, nodes.CodeNode) and not code_node_allows(node, outer_var, inner_var, case):
            return False
    if case == "B":
        return all(outer_var not in free_symbol_names(text) for text in texts)
    return all(reads_only_through_sum(text, outer_var, inner_var) for text in texts)


def exact_bounds(rng: tuple) -> tuple:
    """A range's ``(begin, end, step)`` as exact expressions: a :class:`~dace.symbolic.SymExpr` bound (``MapTiling``
    writes one for a partial last tile, ``min(N, i + K) - 1`` approximated by ``i + K - 1``) by its main expression."""
    return tuple(bound.expr if isinstance(bound, symbolic.SymExpr) else bound for bound in rng)


def match_map_pair(
    state: SDFGState, outer: nodes.MapEntry, inner: nodes.MapEntry, between: list[nodes.MapEntry]
) -> MapTilePair | None:
    """The first tile pair between a dimension of ``outer`` and one of ``inner``, ``between`` the Maps that nest
    between them, or ``None``."""
    for dim, (outer_var, rng) in enumerate(zip(outer.map.params, outer.map.range.ranges)):
        begin, end, step = exact_bounds(rng)
        tile = _tile_size(step)
        if tile is None:
            continue
        K_expr, K_const = tile
        if any(
            outer_var in free_symbol_names(str(bound))
            for m in between
            for rng in m.map.range
            for bound in exact_bounds(rng)
        ):
            continue
        for inner_dim, (inner_var, inner_rng) in enumerate(zip(inner.map.params, inner.map.range.ranges)):
            low, high, stride = exact_bounds(inner_rng)
            match = classify_inner_range(
                low, high, stride, outer_var, K_expr, K_const, outer_limit=symbolic.simplify(end + 1)
            )
            if match is None:
                continue
            others = (
                str(bound)
                for j, rng in enumerate(inner.map.range.ranges)
                if j != inner_dim
                for bound in exact_bounds(rng)
            )
            if any(free_symbol_names(text) & {outer_var, inner_var} for text in others):
                continue
            if not map_body_reads_only(state, inner, outer_var, inner_var, match[0]):
                continue
            return MapTilePair(outer, dim, inner, inner_dim, *match, K_expr)
    return None


def map_tile_pairs(state: SDFGState) -> list[MapTilePair]:
    """The Map tile pairs of ``state`` that can collapse together: innermost first, no Map in two pairs."""
    sdict = state.scope_dict()
    children = state.scope_children()
    depth: dict[nodes.Node, int] = {}

    def depth_of(node: nodes.Node) -> int:
        if node not in depth:
            depth[node] = 0 if sdict[node] is None else 1 + depth_of(sdict[node])
        return depth[node]

    def eligible(entry: nodes.Node) -> bool:
        return (
            isinstance(entry, nodes.MapEntry)
            and entry.map.schedule not in dtypes.GPU_SCHEDULES
            and all(e.dst_conn is None or e.dst_conn.startswith("IN_") for e in state.in_edges(entry))
        )

    entries = [n for n in state.nodes() if eligible(n)]
    used: set[nodes.MapEntry] = set()
    pairs = []
    for inner in sorted(entries, key=depth_of, reverse=True):
        if inner in used:
            continue
        between: list[nodes.MapEntry] = []
        child, outer = inner, sdict[inner]
        while eligible(outer) and set(children[outer]) <= {state.exit_node(outer), child, state.exit_node(child)}:
            if outer not in used and not used.intersection(between):
                pair = match_map_pair(state, outer, inner, between)
                if pair is not None:
                    pairs.append(pair)
                    used.update([outer, inner, *between])
                    break
            between.append(outer)
            child, outer = outer, sdict[outer]
    return pairs


def untile_map_pair(sdfg: SDFG, state: SDFGState, pair: MapTilePair) -> None:
    """Widen the inner dimension of ``pair`` over the union of the tiles and pin the outer one to its first origin."""
    outer, inner = pair.outer, pair.inner
    outer_var = outer.map.params[pair.dim]
    inner_var = inner.map.params[pair.inner_dim]
    begin, end, _ = outer.map.range.ranges[pair.dim]
    if pair.needs_div:
        record_assumption(sdfg, sympy.Eq(sympy.Mod(pair.K_expr, pair.stride), 0))
    scope = state.scope_dict()[outer]
    names = []
    while scope is not None:
        names.extend(scope.map.params)
        scope = state.scope_dict()[scope]
    enclosing = enclosing_loop_variables(state) | frozenset(names)
    stop = union_stop(sdfg, symbolic.simplify(begin), end, pair.K_expr, pair.clamped, enclosing)
    ranges = list(inner.map.range.ranges)
    ranges[pair.inner_dim] = (begin, symbolic.simplify(stop - 1), pair.stride)
    inner.map.range = subsets.Range(ranges)
    if pair.case == "A":
        # ``outer + inner`` becomes the widened ``inner``; the audit makes ``outer`` cancel.
        state.scope_subgraph(inner).replace_dict({inner_var: f"(({inner_var}) - ({outer_var}))"})
    outer_ranges = list(outer.map.range.ranges)
    outer_ranges[pair.dim] = (begin, begin, 1)
    outer.map.range = subsets.Range(outer_ranges)


def eliminate_pinned_maps(sdfg: SDFG, state: SDFGState, entries: set[nodes.MapEntry]) -> None:
    """Drop the pinned dimensions of ``entries``, and each Map left with none."""
    from dace.transformation.dataflow.trivial_map_elimination import TrivialMapElimination

    xform = TrivialMapElimination()
    cfg_id = state.parent_graph.cfg_id
    for entry in sorted(entries, key=state.node_id):
        xform.setup_match(sdfg, cfg_id, state.block_id, {TrivialMapElimination.map_entry: state.node_id(entry)}, 0)
        if xform.can_be_applied(state, 0, sdfg):
            xform.apply(state, sdfg)


@properties.make_properties
@xf.explicit_cf_compatible
class UntileLoops(ppl.Pass):
    """Collapse a manually-tiled multi-level / multi-dim perfect nest of loops to a single loop, and one of Maps to a
    single Map."""

    CATEGORY: str = "Canonicalization"

    def modifies(self) -> ppl.Modifies:
        # Nodes: a Map whose every tile dimension collapsed into its partner is removed.
        return ppl.Modifies.CFG | ppl.Modifies.Symbols | ppl.Modifies.Memlets | ppl.Modifies.Nodes

    def should_reapply(self, modified: ppl.Modifies) -> bool:
        return False

    def untile_sweep(self, sdfg: SDFG) -> int:
        """One pass over every loop of ``sdfg``, collapsing each loop with its tile partner; returns the count."""
        rewritten = 0
        for sd in sdfg.all_sdfgs_recursive():
            # INNERMOST FIRST.
            for cfg in reversed(list(sd.all_control_flow_regions())):
                if isinstance(cfg, LoopRegion) and cfg.loop_variable and self.untile_loop(cfg, sd):
                    rewritten += 1
        return rewritten

    def untile_fixpoint(self, sdfg: SDFG) -> int:
        """Collapse tile pairs until none is left, and return how many were collapsed."""
        total = self.untile_sweep(sdfg)
        # Every collapse removes a loop, so 1 + the loops left bounds the sweeps still to come.
        sweeps_left = 1 + count_loops(sdfg) if total else 0
        rewritten = total
        while rewritten and sweeps_left:
            rewritten = self.untile_sweep(sdfg)
            total += rewritten
            sweeps_left -= 1
        return total

    def untile_map_fixpoint(self, sdfg: SDFG) -> int:
        """Collapse Map tile pairs, state by state, until none is left; returns how many were collapsed."""
        total = 0
        for sd in sdfg.all_sdfgs_recursive():
            for state in sd.states():
                # Every collapse removes a tile dimension, so their count bounds the sweeps.
                for _ in range(1 + sum(len(n.map.params) for n in state.nodes() if isinstance(n, nodes.MapEntry))):
                    pairs = map_tile_pairs(state)
                    if not pairs:
                        break
                    for pair in pairs:
                        untile_map_pair(sd, state, pair)
                    total += len(pairs)
                    eliminate_pinned_maps(sd, state, {pair.outer for pair in pairs})
        return total

    def apply_pass(self, sdfg: SDFG, _: dict[str, Any]) -> int | None:
        """Collapse every Map and loop tile pair, each as a fixpoint, then propagate memlets once."""
        total = self.untile_map_fixpoint(sdfg) + self.untile_fixpoint(sdfg)
        if not total:
            return None
        # Propagate once, at the end of the pass -- the in-place rewrites do not self-propagate.
        from dace.sdfg.propagation import propagate_memlets_sdfg

        propagate_memlets_sdfg(sdfg)
        return total

    def audit_candidate(
        self,
        candidate: LoopRegion,
        outer_var: str,
        case: str,
        clamped: bool,
        K_expr: symbolic.SymbolicType,
        K_const: int | None,
        sdfg: SDFG,
        outer_start: symbolic.SymbolicType,
    ) -> dict[str, Any] | None:
        """What :meth:`prepare_untile` needs to collapse ``candidate`` with its outer tile loop, or ``None`` to refuse."""
        return {} if _audit_combined_access(candidate, outer_var, candidate.loop_variable, case) else None

    def prepare_untile(
        self, sdfg: SDFG, outer_start: symbolic.SymbolicType, K_expr: symbolic.SymbolicType, audit: dict[str, Any]
    ) -> bool:
        """Rewrite ``sdfg`` before the substitution with what :meth:`audit_candidate` returned; ``False``, with the SDFG
        untouched, refuses the candidate."""
        return True

    def untile_loop(self, outer: LoopRegion, sdfg: SDFG) -> bool:
        """Collapse ``outer``, a loop over tiles, with the loop over one tile's points inside it; ``False``, with
        ``sdfg`` untouched, when ``outer`` is no such pair."""
        # The outer must be ``for i in range(0, N, K)`` with a positive tile ``K``
        outer_stride = loop_analysis.get_loop_stride(outer)
        if outer_stride is None:
            return False
        tile = _tile_size(outer_stride)
        if tile is None:
            return False  # stride <= 1, or a provably non-positive symbol
        # Start / end parsed only past the tile-stride gate -- two sympy round trips a unit-stride loop never needs.
        outer_start = loop_analysis.get_init_assignment(outer)
        outer_end = loop_analysis.get_loop_end(outer)
        if outer_start is None or outer_end is None:
            return False
        K_expr, K_const = tile
        # ``outer_start`` need not be 0: a tiled stencil walks tile origins over the interior ``[S, N)`` (e.g. ``for ii
        # in range(1, N-1-K, K)``).
        outer_start_sym = symbolic.simplify(outer_start)

        # Walk down through perfect 1-child intermediate chains and pick the first descendant LoopRegion whose shape +
        # audit match the tile-pair contract with ``outer``.
        case: str | None = None
        inner_stride: symbolic.SymbolicType = None
        needs_div_assumption = False
        clamped = False
        inner: LoopRegion | None = None
        audited: dict[str, Any] = {}
        for candidate in _iter_candidate_inners(outer):
            if not candidate.loop_variable:
                continue
            match = _match_inner_case(
                candidate, outer.loop_variable, K_expr, K_const, outer_limit=symbolic.simplify(outer_end + 1)
            )
            if match is None:
                continue
            # A ``break`` leaves one tile; in the collapsed loop it would leave them all
            if any(isinstance(jump, BreakBlock) for jump in loop_analysis.loop_jumps(candidate)):
                continue
            cand_case, cand_stride, cand_needs_div, cand_clamped = match
            audit = self.audit_candidate(
                candidate, outer.loop_variable, cand_case, cand_clamped, K_expr, K_const, sdfg, outer_start_sym
            )
            if audit is None:
                continue
            # Multi-dim sanity: any intermediate LoopRegion between outer and candidate must not reference outer's
            # iteration variable in its own bounds.
            if candidate is not outer and not _intermediate_chain_clean(outer, candidate, outer.loop_variable):
                continue
            if candidate.parent_graph is not outer and not tiles_visit_points_in_order(outer, candidate):
                continue
            inner = candidate
            case = cand_case
            inner_stride = cand_stride
            needs_div_assumption = cand_needs_div
            clamped = cand_clamped
            audited = audit
            break
        if inner is None or not self.prepare_untile(sdfg, outer_start_sym, K_expr, audited):
            return False

        # A cascade rung whose stride divides the tile only under an unprovable relation (symbolic tile and/or symbolic
        # stride) is admitted here; record ``K % S == 0`` so the terminal AssumeSymbolConstraints pass emits a runtime
        # trap.
        if needs_div_assumption:
            record_assumption(sdfg, sympy.Eq(sympy.Mod(K_expr, inner_stride), 0))

        # New iterator with step ``inner_stride``; > 1 is a cascade rung collapsed on a later sweep.
        k_var = f"{UNTILE_PREFIX}{lowest_free_suffix(sdfg, (UNTILE_PREFIX,))}"
        N_excl = union_stop(sdfg, outer_start_sym, outer_end, K_expr, clamped, enclosing_loop_variables(outer))

        # Body substitution: ``i + ii`` -> ``k`` (case A) or ``ii`` -> ``k`` (case B).
        i_sym = outer.loop_variable
        ii_sym = inner.loop_variable
        if case == "A":
            # ``i + ii`` -> ``k``.
            inner.replace_dict({ii_sym: f"({k_var}) - ({i_sym})"})
            inner.replace_dict({i_sym: "0"})
        else:
            # Case B: ``ii`` -> ``k``; ``i`` doesn't appear in any memlet.
            inner.replace_dict({ii_sym: k_var})

        # Splice the inner's body blocks into the inner's parent CFR (which may be ``outer`` directly for single-level
        # untile, or a nested intermediate LoopRegion for multi-dim untile).
        parent_of_inner: ControlFlowRegion = inner.parent_graph
        inner_was_start = parent_of_inner.start_block is inner
        # ``inner`` need NOT be the sole/start block of its parent: the parent LoopRegion may hold connective states
        # with interstate edges FEEDING ``inner`` and FLOWING OUT of it.
        pred_edges = list(parent_of_inner.in_edges(inner))
        succ_edges = list(parent_of_inner.out_edges(inner))
        try:
            inner_start_block = inner.start_block if inner.number_of_nodes() > 0 else None
        except (NodeNotFoundError, ValueError):
            inner_start_block = None
        inner_sinks = inner.sink_nodes()
        child_blocks = list(inner.nodes())
        inner_edges = list(inner.edges())
        # Detach the inner wrapper (drops its incident parent edges too), then splice its blocks up into the parent.
        parent_of_inner.remove_node(inner)
        for child in child_blocks:
            inner.remove_node(child)
            child_is_start = inner_was_start and (child is inner_start_block)
            parent_of_inner.add_node(child, is_start_block=child_is_start, ensure_unique_name=True)
        # Re-attach the inner's own body interstate edges.
        for ie in inner_edges:
            parent_of_inner.add_edge(ie.src, ie.dst, ie.data)
        # Reconnect the parent chain through the spliced body: predecessors of the old ``inner`` now flow into its entry
        # block; its exit block(s) flow to the old successors.
        if inner_start_block is not None:
            for pe in pred_edges:
                parent_of_inner.add_edge(pe.src, inner_start_block, copy.deepcopy(pe.data))
        for se in succ_edges:
            for sink in inner_sinks:
                parent_of_inner.add_edge(sink, se.dst, copy.deepcopy(se.data))

        # Rewrite the outer's iteration descriptors to drive ``k`` over ``[outer_start, N)`` in steps of
        # ``inner_stride``.
        outer.loop_variable = k_var
        outer.init_statement = dace.properties.CodeBlock(f"{k_var} = {symbolic.symstr(outer_start_sym)}")
        outer.loop_condition = dace.properties.CodeBlock(f"{k_var} < ({symbolic.symstr(N_excl)})")
        outer.update_statement = dace.properties.CodeBlock(f"{k_var} = {k_var} + {symbolic.symstr(inner_stride)}")
        return True


__all__ = ["UntileLoops"]
