# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Expose wavefront parallelism by skewing an outer loop with a chain of statements.

The outer loop ``u`` holds a chain of statements (inner loops, one-parameter maps, row copies), each
with its own iterator ``v`` (:mod:`.statements`). Every statement instance gets the schedule
``t = a*u + b*(v + s)``, ``s`` a per-statement alignment; ``t`` is sequential and each statement is a
parallel map at fixed ``t``, the statements in program order. The schedule is legal when every exact
dependence (:mod:`.dependences`) lands on a later ``t``, or on the same ``t`` with its source
statement sequenced first.

Where every dependence is non-negative along ``u`` and ``w = v + s + k*u`` the nest is tiled
instead: a sequential tile diagonal over a parallel tile-column map over the tile's rows, each row
running the statements clipped to the tile. On GPU a tile of loop statements also runs its own
``(1, 1)`` diagonal across a thread block.

A nest whose inner axes are already parallel at fixed ``u``, or whose outer axis is, is left alone, and so is a
time-step loop: one whose iterator indexes nothing.

References: Lamport, CACM '74; Wolf & Lam, TPDS '91; Bondhugula et al., PLDI '08.
"""

import zlib
from dataclasses import dataclass
from typing import Any

import islpy as isl
import sympy

import dace
from dace import SDFG, properties, symbolic
from dace.memlet import Memlet
from dace.sdfg import nodes
from dace.sdfg import tasklet_utils as tutil
from dace.sdfg.narrowing import as_expr, as_typeclass, ndrange_exprs, simplified
from dace.sdfg.propagation import propagate_memlets_state
from dace.sdfg.replace import replace_dict as replace_in_subgraph
from dace.sdfg.state import ConditionalBlock, ControlFlowRegion, LoopRegion, SDFGState, SymbolResolver
from dace.transformation import pass_pipeline as ppl
from dace.transformation import transformation as xf
from dace.transformation.interstate.loop_to_map import control_flow_reads
from dace.transformation.passes.analysis import canon_stats
from dace.transformation.passes.canonicalize.annotate_loop_kinds import (
    WAVEFRONT_DIAGONAL,
    WAVEFRONT_FRONT,
    WAVEFRONT_TILE_COLUMN,
    WAVEFRONT_TILE_DIAGONAL,
    WAVEFRONT_TILE_INTERIOR,
    skew_label,
    tile_label,
)
from dace.transformation.passes.canonicalize.fresh_names import lowest_free_suffix
from dace.transformation.passes.canonicalize.privatize_reduction_accumulator import privatize_reduction_accumulator
from dace.transformation.passes.parallelization.wavefront import polyhedron as poly
from dace.transformation.passes.parallelization.wavefront.dependences import Dependences, provably, subset_params
from dace.transformation.passes.parallelization.wavefront.statements import (
    Body,
    Statement,
    copy_axis,
    extract_body,
    free_names,
    is_time_step_loop,
    occurrence_index,
)

SKEW_T_PREFIX = "_skew_t_"
SKEW_P_PREFIX = "_skew_p_"

#: Tile extent per axis. CPU: 64x64 doubles measured 2.04-2.16x over the sequential nest at N=768 on 4
#: threads (the element diagonal: 0.17x). GPU: one launch per tile diagonal; 64 fills a gfx942 wavefront.
DEFAULT_TILE_SIZE = 64
DEFAULT_GPU_TILE_SIZE = 64

#: Largest inner-axis pre-skew ``w = v + k*u`` tried to make the nest tileable.
MAX_PRESKEW = 2

#: Diagonal skews ``tau = (a, b)`` in preference order; each is unimodular (``|a| == 1`` or ``|b| == 1``).
SKEW_CANDIDATES: tuple[tuple[int, int], ...] = ((1, 1), (1, -1), (2, 1), (2, -1), (1, 2), (1, -2))

#: Dimension names of the schedule polyhedra, and the probe names ``poly.skew_bounds`` returns bounds in.
DIM_U, DIM_W = "_wf_u", "_wf_w"
PROBE_T, PROBE_P = f"{SKEW_T_PREFIX}probe", f"{SKEW_P_PREFIX}probe"
TILE_I, TILE_J, TILE_NI, TILE_NJ = "_wf_ti", "_wf_tj", "_wf_ni", "_wf_nj"


def sym(name: str) -> sympy.Expr:
    return as_expr(name)


def declare_iterators(*names: str) -> None:
    """Mint the pass's new iterators at int64; loops bind them, so the SDFG does not declare them."""
    for name in names:
        symbolic.declare_symbol_dtype(name, as_typeclass(dace.int64))


def params_of(cons: list[sympy.Expr], dims: list[str]) -> list[str]:
    return sorted({n for c in cons for n in free_names(c)} - set(dims))


def statement_domain(body: Body, st: Statement, shift: int, k: int) -> list[sympy.Expr] | None:
    """``st``'s domain over ``(DIM_U, DIM_W)``, ``w = v + shift + k*u``, as exact ``>= 0`` terms."""
    u, w = sym(DIM_U), sym(DIM_W)
    v = w - shift - k * u
    subs = {body.u: u, st.var: v}
    out: list[sympy.Expr] = []
    for var, bound, upper in ((u, body.u_lo, False), (u, body.u_hi, True), (v, st.lo, False), (v, st.hi, True)):
        terms = symbolic.convex_bound_terms(poly.subs_by_name(bound, subs), upper)
        if terms is None:
            return None
        out += [simplified(t - var) if upper else simplified(var - t) for t in terms]
    return out


def bound_str(terms: list[sympy.Expr], subs: dict[str, sympy.Expr], fn: str) -> str:
    rendered = [symbolic.symstr(poly.subs_by_name(t, subs)) for t in terms]
    return rendered[0] if len(rendered) == 1 else f"{fn}({', '.join(rendered)})"


def combine(parts: list[str], fn: str) -> str:
    unique = list(dict.fromkeys(parts))
    return unique[0] if len(unique) == 1 else f"{fn}({', '.join(unique)})"


def skew_bounds(cons: list[sympy.Expr], tau: tuple[int, int], t_range=None) -> poly.SkewBounds | None:
    try:
        return poly.skew_bounds((DIM_U, DIM_W), params_of(cons, [DIM_U, DIM_W]), cons, tau, PROBE_T, PROBE_P, t_range)
    except ValueError:
        return None


@canon_stats.timed("isl")
def domain_bbox(cons: list[sympy.Expr]) -> list[sympy.Expr] | None:
    """``[u_lo, u_hi, w_lo, w_hi]`` of the integer set ``cons``, or ``None`` when a side is not one bound."""
    try:
        s, nmap = poly.make_set((DIM_U, DIM_W), params_of(cons, [DIM_U, DIM_W]), cons)
    except ValueError:
        return None
    inv = {safe: orig for orig, safe in nmap.items()}
    box: list[sympy.Expr] = []
    for keep in (0, 1):
        proj = s.project_out(isl.dim_type.set, 1 - keep, 1).coalesce()
        for pw in (proj.dim_min(0), proj.dim_max(0)):
            bound = poly.pwaff_bound(pw, inv)
            if bound is None:
                return None
            box.append(bound)
    return box


@dataclass
class TilePlan:
    """The tile-index bounds and grid of a ``bi x bj`` tiling of ``(u, w = v + shift + k*u)``."""

    bounds: poly.SkewBounds
    u_lo: sympy.Expr
    u_hi: sympy.Expr
    w_lo: sympy.Expr
    n_i: sympy.Expr
    n_j: sympy.Expr
    bi: int
    bj: int
    k: int


@dataclass
class Plan:
    body: Body
    tau: tuple[int, int]
    shifts: dict[int, int]
    guard_params: list[str]
    tiles: TilePlan | None
    #: Per-statement ``p`` bounds of the element diagonal, when untiled.
    fronts: dict[int, poly.SkewBounds]


def privatize_body_reductions(loop: LoopRegion) -> int:
    """Fold every top-level WCR map of a multi-block ``loop`` body into a transient scalar, so the reduction
    stays inside the NestedSDFG the lift nests the body into."""
    if len(loop.nodes()) == 1 and isinstance(loop.start_block, SDFGState):
        return 0
    count = 0
    for state in list(loop.states()):
        for map_exit in [n for n in state.nodes() if isinstance(n, nodes.MapExit)]:
            for edge in list(state.in_edges(map_exit)):
                count += privatize_reduction_accumulator(state, map_exit, edge)
    return count


def names_read_outside(outer: LoopRegion, sdfg: SDFG) -> set[str]:
    """The symbol names anything of ``sdfg`` outside ``outer`` reads."""
    inside_blocks = set(outer.all_control_flow_blocks(recursive=True))
    inside_edges = {id(e) for e in outer.all_interstate_edges(recursive=True)}
    names: set[str] = set()
    for state in sdfg.states():
        if state not in inside_blocks:
            names |= {str(s) for s in state.free_symbols}
    for edge in sdfg.all_interstate_edges(recursive=True):
        if id(edge) not in inside_edges:
            names |= {str(s) for s in edge.data.free_symbols}
    for region in sdfg.all_control_flow_regions(recursive=True):
        if region not in inside_blocks and region is not outer:
            names |= {str(s) for code in region.get_meta_codeblocks() for s in code.get_free_symbols()}
    return names


def copy_to_map(state: SDFGState, st: Statement) -> nodes.MapEntry:
    """Rewrite a copy statement as the one-parameter map its model describes."""
    e = st.edge
    assert e is not None
    src_ax, dst_ax, width = copy_axis(state, e)
    src_sub, dst_sub = e.data.get_src_subset(e, state), e.data.get_dst_subset(e, state)

    def indexed(sub, ax) -> str:
        parts = []
        for a, (lo, hi, _) in enumerate(ndrange_exprs(sub)):
            parts.append(symbolic.symstr(lo + sym(st.var)) if a == ax else symbolic.symstr(lo))
        return ", ".join(parts)

    me, mx = state.add_map(f"{e.dst.data}_copy", {st.var: f"0:{symbolic.symstr(width)}"})
    t = state.add_tasklet(f"{e.dst.data}_copy", {"__inp"}, {"__out"}, "__out = __inp")
    state.add_memlet_path(e.src, me, t, dst_conn="__inp", memlet=Memlet(f"{e.src.data}[{indexed(src_sub, src_ax)}]"))
    out = Memlet(f"{e.dst.data}[{indexed(dst_sub, dst_ax)}]", wcr=e.data.wcr)
    state.add_memlet_path(t, mx, e.dst, src_conn="__out", memlet=out)
    state.remove_edge(e)
    st.kind, st.node, st.edge = "map", me, None
    return me


@properties.make_properties
@xf.explicit_cf_compatible
class WavefrontSkew(ppl.Pass):
    """Skew genuine wavefront nests so every statement of the outer body runs as a parallel map per diagonal."""

    CATEGORY: str = "Canonicalization"

    target = properties.Property(
        dtype=str, default="cpu", choices=["cpu", "gpu"], desc="Target the tiling and tile interior are lowered for."
    )
    tile_i = properties.Property(dtype=int, default=DEFAULT_TILE_SIZE, desc="CPU tile extent along u.")
    tile_j = properties.Property(dtype=int, default=DEFAULT_TILE_SIZE, desc="CPU tile extent along w.")
    gpu_tile_i = properties.Property(dtype=int, default=DEFAULT_GPU_TILE_SIZE, desc="GPU tile extent along u.")
    gpu_tile_j = properties.Property(dtype=int, default=DEFAULT_GPU_TILE_SIZE, desc="GPU tile extent along w.")

    def __init__(self, target: str = "cpu") -> None:
        super().__init__()
        self.target = target

    def modifies(self) -> ppl.Modifies:
        return ppl.Modifies.CFG | ppl.Modifies.Symbols | ppl.Modifies.Nodes | ppl.Modifies.Edges

    def should_reapply(self, modified: ppl.Modifies) -> bool:
        return False

    def apply_pass(self, sdfg: SDFG, pipeline_results: dict[str, Any]) -> int | None:
        """Skew every eligible nest; the count, or ``None``."""
        with symbolic.serialization_symbol_dtypes({}, inherit=True):
            return self.skew_all(sdfg)

    def skew_all(self, sdfg: SDFG) -> int | None:
        skewed = 0
        for sd in sdfg.all_sdfgs_recursive():
            occurrences = None
            # Innermost first: an inner front is the wider one, and an outer nest holding a skewed diagonal stays.
            for region in reversed(list(sd.all_control_flow_regions())):
                if not (isinstance(region, LoopRegion) and region.loop_variable) or emitted_here(region):
                    continue
                parent = region.parent_graph
                if parent is None or region not in parent.nodes():
                    continue
                if occurrences is None:
                    occurrences = occurrence_index(sd)
                if self.try_skew(region, sd, occurrences):
                    skewed += 1
                    occurrences = None
        return skewed or None

    def try_skew(self, outer: LoopRegion, sdfg: SDFG, occurrences: dict | None = None) -> bool:
        """Skew ``outer`` if it is a genuine wavefront; ``True`` iff rewritten."""
        if emitted_here(outer):
            return False
        plan = self.plan(outer, sdfg, occurrence_index(sdfg) if occurrences is None else occurrences)
        if plan is None:
            return False
        self.commit(plan, sdfg)
        return True

    # ------------------------------------------------------------------------------------------------
    # Analysis

    def plan(self, outer: LoopRegion, sdfg: SDFG, occurrences: dict) -> Plan | None:
        body = extract_body(outer, sdfg, occurrences)
        if body is None:
            return None
        if is_time_step_loop(body):
            return None  # a loop over kernel invocations stays sequential
        for st in body.statements:
            if isinstance(st.block, LoopRegion) and any(
                isinstance(r, LoopRegion) and r.loop_variable and emitted_here(r)
                for r in [st.block, *st.block.all_control_flow_regions(recursive=True)]
            ):
                return None
        unsafe = body.assigned
        if (free_names(body.u_lo) | free_names(body.u_hi)) & (unsafe | {body.u}):
            return None
        if any((free_names(st.lo) | free_names(st.hi)) & (unsafe | {st.var}) for st in body.statements):
            return None
        facts = SymbolResolver().facts_at(outer)
        extents = [(body.u_lo, body.u_hi)] + [(st.lo, st.hi) for st in body.statements if st.kind == "loop"]
        if any(symbolic.is_small(hi - lo, facts) for lo, hi in extents):
            return None  # a front is at most as wide as the shorter axis
        written = {acc.data for st in body.statements for acc in st.accesses if acc.write}
        if written & control_flow_reads(outer):
            return None

        sdfg_facts = sdfg.facts()
        deps = Dependences(body, sdfg_facts)
        if not deps.complete or not all(deps.snapshot_safe(snap) for snap in body.snapshots):
            return None
        if deps.schedule_legal((1, 0), {}) or deps.outer_parallel():
            return None  # an axis is already parallel: LoopToMap reaches it

        found = self.pick_schedule(deps)
        guard_params: list[str] = []
        if found is None:
            eligible = [
                p for p in subset_params(body) if not provably(symbolic.Predicate.NONNEGATIVE, sym(p), sdfg_facts)
            ]
            if eligible:
                deps = Dependences(body, sdfg_facts, [sym(p) for p in eligible])
                found = self.pick_schedule(deps)
                if found is not None:
                    guard_params = list(eligible)
                    for p in eligible:
                        trial = [q for q in guard_params if q != p]
                        trial_deps = Dependences(body, sdfg_facts, [sym(q) for q in trial])
                        if self.pick_schedule(trial_deps) is not None:
                            guard_params, deps = trial, trial_deps
                    found = self.pick_schedule(deps)
        if found is None:
            return None
        tau, shifts = found

        tiles = self.plan_tiles(body, deps, shifts)
        fronts: dict[int, poly.SkewBounds] = {}
        if tiles is None:
            for st in body.statements:
                cons = statement_domain(body, st, shifts.get(st.order, 0), 0)
                bounds = None if cons is None else skew_bounds(cons, tau)
                if bounds is None:
                    return None
                fronts[st.order] = bounds
        read_after = names_read_outside(outer, sdfg)
        if ({body.u} | {st.var for st in body.statements} | body.assigned) & read_after:
            return None
        return Plan(body, tau, shifts, guard_params, tiles, fronts)

    def pick_schedule(self, deps: Dependences) -> tuple[tuple[int, int], dict[int, int]] | None:
        """The first legal ``tau``, under the alignment shifts and then without them."""
        aligned = deps.alignment()
        tried = [aligned] if not any(aligned.values()) else [aligned, {}]
        for shifts in tried:
            for tau in SKEW_CANDIDATES:
                if deps.schedule_legal(tau, shifts):
                    return tau, shifts
        return None

    def tile_extents(self) -> tuple[int, int]:
        if self.target == "gpu":
            return int(self.gpu_tile_i), int(self.gpu_tile_j)
        return int(self.tile_i), int(self.tile_j)

    def plan_tiles(self, body: Body, deps: Dependences, shifts: dict[int, int]) -> TilePlan | None:
        """A rectangular tiling of ``(u, v + shift + k*u)`` for the smallest pre-skew ``k`` that makes every
        dependence non-negative, or ``None``."""
        bi, bj = self.tile_extents()
        if bi < 1 or bj < 1:
            return None
        for k in range(MAX_PRESKEW + 1):
            if not deps.permutable(shifts, k):
                continue
            boxes = []
            for st in body.statements:
                cons = statement_domain(body, st, shifts.get(st.order, 0), k)
                box = None if cons is None else domain_bbox(cons)
                if box is None:
                    return None
                boxes.append(box)
            u_lo, u_hi = combine_bounds([b[0] for b in boxes], "min"), combine_bounds([b[1] for b in boxes], "max")
            w_lo, w_hi = combine_bounds([b[2] for b in boxes], "min"), combine_bounds([b[3] for b in boxes], "max")
            ti, tj = sym(TILE_I), sym(TILE_J)
            tile_domain = [ti, sym(TILE_NI) - 1 - ti, tj, sym(TILE_NJ) - 1 - tj]
            try:
                bounds = poly.skew_bounds((TILE_I, TILE_J), [TILE_NI, TILE_NJ], tile_domain, (1, 1), PROBE_T, PROBE_P)
            except ValueError:
                return None
            if bounds is None:
                return None
            n_i = symbolic.int_ceil(simplified(u_hi - u_lo + 1), bi)
            n_j = symbolic.int_ceil(simplified(w_hi - w_lo + 1), bj)
            return TilePlan(bounds, u_lo, u_hi, w_lo, n_i, n_j, bi, bj, k)
        return None

    # ------------------------------------------------------------------------------------------------
    # Rewrite

    def commit(self, plan: Plan, sdfg: SDFG) -> None:
        body = plan.body
        outer = body.outer
        commit_snapshots(body)
        for st in body.statements:
            if st.kind == "copy":
                assert isinstance(st.block, SDFGState)
                copy_to_map(st.block, st)
        if plan.guard_params:
            self.emit_positive_guard(outer, plan.guard_params)
        if plan.tiles is not None:
            self.rewrite_tiled(plan, sdfg)
        else:
            self.rewrite_diagonal(plan, sdfg)

    def fresh_names(self, sdfg: SDFG, count: int) -> tuple[str, list[str]]:
        nid = lowest_free_suffix(sdfg, (SKEW_T_PREFIX, SKEW_P_PREFIX))
        t_var = f"{SKEW_T_PREFIX}{nid}"
        p_vars = [f"{SKEW_P_PREFIX}{nid}" if i == 0 else f"{SKEW_P_PREFIX}{nid}_{i}" for i in range(count)]
        declare_iterators(t_var, *p_vars)
        return t_var, p_vars

    def rewrite_diagonal(self, plan: Plan, sdfg: SDFG) -> None:
        """``outer`` becomes the diagonal ``t``; each statement becomes a parallel map over its front."""
        body, (a, b) = plan.body, plan.tau
        outer, u = body.outer, body.u
        t_var, p_vars = self.fresh_names(sdfg, len(body.statements))
        t_lo = combine(
            [bound_str(plan.fronts[st.order].t_lo_terms, {PROBE_T: sym(t_var)}, "max") for st in body.statements], "min"
        )
        t_hi = combine(
            [bound_str(plan.fronts[st.order].t_hi_terms, {PROBE_T: sym(t_var)}, "min") for st in body.statements], "max"
        )
        set_header(outer, t_var, t_lo, t_hi)
        outer.pinned_sequential = True
        skew = skew_label(a, b, u, "v")
        outer.specialization_hint = WAVEFRONT_DIAGONAL.format(skew=skew)
        t = sym(t_var)
        for st, p_var in zip(body.statements, p_vars):
            fr = plan.fronts[st.order]
            subs = {PROBE_T: t, PROBE_P: sym(p_var)}
            p_lo, p_hi = bound_str(fr.p_lo_terms, subs, "max"), bound_str(fr.p_hi_terms, subs, "min")
            p, shift = sym(p_var), plan.shifts.get(st.order, 0)
            if abs(a) == 1:
                u_expr, v_expr = a * (t - b * p), p - shift
            else:
                u_expr, v_expr = p, b * (t - a * p) - shift
            repl = {u: symbolic.symstr(simplified(u_expr)), st.var: symbolic.symstr(simplified(v_expr))}
            hint = WAVEFRONT_FRONT.format(skew=skew)
            if st.kind == "loop":
                loop = st.block
                assert isinstance(loop, LoopRegion)
                set_header(loop, p_var, p_lo, p_hi)
                loop.replace_dict(repl)
                self.lift(outer, loop, sdfg, hint)
            else:
                rewrite_map(st, p_var, p_lo, p_hi, repl, hint, sdfg)

    def rewrite_tiled(self, plan: Plan, sdfg: SDFG) -> None:
        """``outer`` becomes the tile diagonal over a parallel tile-column map over the tile rows; every statement
        runs clipped to its tile, and on GPU along the tile's own diagonal."""
        body, tp = plan.body, plan.tiles
        assert tp is not None
        outer, u = body.outer, body.u
        t_var, (p_var,) = self.fresh_names(sdfg, 1)
        subs = {PROBE_T: sym(t_var), PROBE_P: sym(p_var), TILE_NI: tp.n_i, TILE_NJ: tp.n_j}
        tile = tile_label(tp.bi, tp.bj)
        skew = skew_label(1, 1, u, "w")

        p_loop = LoopRegion(
            f"{outer.label}_tile_diag",
            f"{p_var} <= ({bound_str(tp.bounds.p_hi_terms, subs, 'min')})",
            p_var,
            f"{p_var} = ({bound_str(tp.bounds.p_lo_terms, subs, 'max')})",
            f"{p_var} = {p_var} + 1",
        )
        i_tile, j_tile = sym(t_var) - sym(p_var), sym(p_var)
        i_lo = simplified(tp.u_lo + tp.bi * i_tile)
        j_lo = simplified(tp.w_lo + tp.bj * j_tile)
        i_loop = LoopRegion(
            f"{outer.label}_tile_row",
            f"{u} <= (min({symbolic.symstr(tp.u_hi)}, {symbolic.symstr(i_lo + tp.bi - 1)}))",
            u,
            f"{u} = ({symbolic.symstr(i_lo)})",
            f"{u} = {u} + 1",
        )
        i_loop.pinned_sequential = True
        i_loop.specialization_hint = WAVEFRONT_TILE_INTERIOR.format(tile=tile)
        move_body(outer, i_loop)
        p_loop.add_node(i_loop, is_start_block=True)
        outer.add_node(p_loop, is_start_block=True, ensure_unique_name=True)

        set_header(
            outer, t_var, bound_str(tp.bounds.t_lo_terms, subs, "max"), bound_str(tp.bounds.t_hi_terms, subs, "min")
        )
        outer.pinned_sequential = True
        outer.specialization_hint = WAVEFRONT_TILE_DIAGONAL.format(skew=skew, tile=tile)

        for st in body.statements:
            # ``w = v + shift + k*u`` in ``[j_lo, j_lo + bj - 1]``.
            first = simplified(j_lo - plan.shifts.get(st.order, 0) - tp.k * sym(u))
            lo = f"max({symbolic.symstr(st.lo)}, {symbolic.symstr(first)})"
            hi = f"min({symbolic.symstr(st.hi)}, {symbolic.symstr(first + tp.bj - 1)})"
            if st.kind == "loop":
                loop = st.block
                assert isinstance(loop, LoopRegion)
                set_header(loop, st.var, lo, hi)
                loop.pinned_sequential = True
                loop.specialization_hint = WAVEFRONT_TILE_INTERIOR.format(tile=tile)
            else:
                assert st.node is not None and isinstance(st.block, SDFGState)
                st.node.map.range = dace.subsets.Range([(as_expr(lo), as_expr(hi), 1)])
                st.node.map.schedule = dace.ScheduleType.Sequential
                propagate_memlets_state(sdfg, st.block)

        if self.target == "gpu":
            self.skew_tile_interior(plan, sdfg, i_loop, i_lo, j_lo)
        self.lift(outer, p_loop, sdfg, WAVEFRONT_TILE_COLUMN.format(tile=tile))

    def skew_tile_interior(
        self, plan: Plan, sdfg: SDFG, i_loop: LoopRegion, i_lo: sympy.Expr, j_lo: sympy.Expr
    ) -> None:
        """Run the tile's rows along its ``(1, 1)`` diagonal ``d = u + w``, each loop statement a guarded
        constant-width map across the thread block; left sequential when a front is not renderable."""
        body, tp = plan.body, plan.tiles
        assert tp is not None
        u = body.u
        u_s, w_s = sym(DIM_U), sym(DIM_W)
        d_lo = simplified(i_lo + j_lo)
        d_hi = simplified(i_lo + j_lo + tp.bi - 1 + tp.bj - 1)
        fronts = []
        for st in body.statements:
            cons = statement_domain(body, st, plan.shifts.get(st.order, 0), tp.k)
            if cons is None:
                return
            cons = cons + [
                u_s - i_lo,
                simplified(i_lo + tp.bi - 1 - u_s),
                w_s - j_lo,
                simplified(j_lo + tp.bj - 1 - w_s),
            ]
            bounds = skew_bounds(cons, (1, 1), (d_lo, d_hi))
            if bounds is None:
                return
            fronts.append(bounds)
        nid = lowest_free_suffix(sdfg, (SKEW_T_PREFIX, SKEW_P_PREFIX))
        d_var = f"{SKEW_T_PREFIX}{nid}"
        k_vars = [f"{SKEW_P_PREFIX}{nid}" if i == 0 else f"{SKEW_P_PREFIX}{nid}_{i}" for i in range(len(fronts))]
        declare_iterators(d_var, *k_vars)
        set_header(i_loop, d_var, symbolic.symstr(d_lo), symbolic.symstr(d_hi))
        i_loop.pinned_sequential = True
        i_loop.specialization_hint = WAVEFRONT_DIAGONAL.format(skew=skew_label(1, 1, u, "w"))
        d = sym(d_var)
        hint = WAVEFRONT_FRONT.format(skew=skew_label(1, 1, u, "w"))
        for st, bounds, k_var in zip(body.statements, fronts, k_vars):
            subs = {PROBE_T: d, PROBE_P: sym(k_var)}
            p_lo, p_hi = bound_str(bounds.p_lo_terms, subs, "max"), bound_str(bounds.p_hi_terms, subs, "min")
            coord = simplified(j_lo + sym(k_var))
            u_expr = simplified(d - coord)
            v_expr = simplified(coord - plan.shifts.get(st.order, 0) - tp.k * u_expr)
            repl = {u: symbolic.symstr(u_expr), st.var: symbolic.symstr(v_expr)}
            here = symbolic.symstr(coord)
            condition = f"({here}) >= ({p_lo}) and ({here}) <= ({p_hi})"
            if st.kind == "loop":
                loop = st.block
                assert isinstance(loop, LoopRegion)
                set_header(loop, k_var, "0", str(tp.bj - 1))
                loop.replace_dict(repl)
                guard_body(loop, sdfg, condition)
                self.lift(i_loop, loop, sdfg, hint)
            else:
                rewrite_map(st, k_var, "0", str(tp.bj - 1), repl, hint, sdfg)
                st.node.map.schedule = dace.ScheduleType.Default
                guard_map_body(st, sdfg, condition)
        for node, _ in i_loop.all_nodes_recursive():
            if isinstance(node, nodes.MapEntry):
                node.map.is_warp_tile = True

    def lift(self, scope: ControlFlowRegion, loop: LoopRegion, sdfg: SDFG, hint: str) -> None:
        """Lift ``loop`` to a map on the legality proof; ``hint`` names the fresh map."""
        from dace.transformation.passes.parallelization.parallelize_loops import ParallelizeLoops

        itervar = loop.loop_variable
        privatize_body_reductions(loop)
        ParallelizeLoops().parallelize_loop(sdfg, loop, proven=True)
        for node, _ in scope.all_nodes_recursive():
            if isinstance(node, nodes.MapEntry) and itervar in node.map.params and not node.specialization_hint:
                node.specialization_hint = hint

    def emit_positive_guard(self, outer: LoopRegion, params: list[str]) -> None:
        """Abort before ``outer`` when an offset the schedule assumed non-negative is negative."""
        parts = " or ".join(f"(({p}) < 0)" for p in params)
        tag = zlib.crc32(parts.encode()) & 0xFFFFFFF
        pre = outer.parent_graph.add_state_before(outer, label=f"_skew_guard_{tag:x}")
        tutil.add_abort_guard(pre, f"_skew_guard_{tag:x}", parts)


def emitted_here(loop: LoopRegion) -> bool:
    """Whether ``loop`` is a diagonal or tile row this pass emitted."""
    if loop.loop_variable.startswith(SKEW_T_PREFIX):
        return True
    hint = loop.specialization_hint or ""
    return any(hint.startswith(t.split("{")[0]) for t in (WAVEFRONT_TILE_INTERIOR, WAVEFRONT_DIAGONAL))


def combine_bounds(bounds: list[sympy.Expr], fn: str) -> sympy.Expr:
    unique = list(dict.fromkeys(simplified(b) for b in bounds))
    if len(unique) == 1:
        return unique[0]
    return symbolic.pystr_to_symbolic(f"{fn}({', '.join(symbolic.symstr(b) for b in unique)})")


def set_header(loop: LoopRegion, var: str, lo: str, hi: str) -> None:
    loop.loop_variable = var
    loop.init_statement = properties.CodeBlock(f"{var} = ({lo})")
    loop.loop_condition = properties.CodeBlock(f"{var} <= ({hi})")
    loop.update_statement = properties.CodeBlock(f"{var} = {var} + 1")


def rewrite_map(st: Statement, p_var: str, p_lo: str, p_hi: str, repl: dict[str, str], hint: str, sdfg: SDFG) -> None:
    """Re-index a map statement over its front ``p``: its parameter renamed, its body substituted."""
    state, entry = st.block, st.node
    assert isinstance(state, SDFGState) and entry is not None
    entry.map.params = [p_var]
    entry.map.range = dace.subsets.Range([(as_expr(p_lo), as_expr(p_hi), 1)])
    scope = state.scope_subgraph(entry, include_entry=False, include_exit=False)
    replace_in_subgraph(scope, repl)
    exit_node = state.exit_node(entry)
    for e in [*state.out_edges(entry), *state.in_edges(exit_node)]:
        for k, v in repl.items():
            e.data.replace({k: v})
    entry.specialization_hint = hint
    propagate_memlets_state(sdfg, state)


def move_body(src: ControlFlowRegion, dst: ControlFlowRegion) -> None:
    """Move every block and edge of ``src`` into ``dst``."""
    start, blocks, edges = src.start_block, list(src.nodes()), list(src.edges())
    for blk in blocks:
        dst.add_node(blk, is_start_block=blk is start)
    for e in edges:
        dst.add_edge(e.src, e.dst, e.data)
    for blk in blocks:
        src.remove_node(blk)


def guard_body(loop: LoopRegion, sdfg: SDFG, condition: str) -> None:
    """Move ``loop``'s body under a single-branch ConditionalBlock testing ``condition``."""
    branch = ControlFlowRegion(f"{loop.label}_active", sdfg=sdfg)
    move_body(loop, branch)
    guard = ConditionalBlock(f"{loop.label}_lane", sdfg=sdfg)
    guard.add_branch(condition, branch)
    loop.add_node(guard, is_start_block=True)


def guard_map_body(st: Statement, sdfg: SDFG, condition: str) -> None:
    """Nest a map statement's body and run it only under ``condition``."""
    from dace.transformation.helpers import nest_state_subgraph

    state, entry = st.block, st.node
    assert isinstance(state, SDFGState) and entry is not None
    inner = state.scope_subgraph(entry, include_entry=False, include_exit=False)
    node = nest_state_subgraph(sdfg, state, inner, name=f"{entry.map.label}_lane")
    nsdfg = node.sdfg
    for name in sorted(free_names(condition)):
        if name not in nsdfg.symbols:
            nsdfg.add_symbol(name, sdfg.symbols.get(name, as_typeclass(dace.int64)))
        node.symbol_mapping.setdefault(name, sym(name))
    branch = ControlFlowRegion(f"{entry.map.label}_active", sdfg=nsdfg)
    move_body(nsdfg, branch)
    guard = ConditionalBlock(f"{entry.map.label}_lane", sdfg=nsdfg)
    guard.add_branch(condition, branch)
    nsdfg.add_node(guard, is_start_block=True)


def commit_snapshots(body: Body) -> None:
    """Redirect every snapshot read onto its live array and empty the copy states."""
    for snap in body.snapshots:
        for st in body.statements:
            states = list(st.block.states()) if isinstance(st.block, LoopRegion) else [st.block]
            for state in states:
                for node in [n for n in state.data_nodes() if n.data == snap.snap]:
                    reader = next(
                        (n for n in state.data_nodes() if n.data == snap.live and state.in_degree(n) == 0), None
                    ) or state.add_access(snap.live)
                    for e in list(state.out_edges(node)):
                        for tree_edge in state.memlet_tree(e).traverse_children(include_self=True):
                            if tree_edge.edge.data.data == snap.snap:
                                tree_edge.edge.data.data = snap.live
                        state.add_edge(reader, e.src_conn, e.dst, e.dst_conn, e.data)
                        state.remove_edge(e)
                    if state.degree(node) == 0:
                        state.remove_node(node)
        for n in list(snap.state.nodes()):
            snap.state.remove_node(n)


__all__ = ["WavefrontSkew"]
