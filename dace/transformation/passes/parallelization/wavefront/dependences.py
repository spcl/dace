# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Exact memory-based dependences between the statement instances of a wavefront candidate.

For every pair of accesses to one container, at least one of them a write, a :class:`Piece` is the
integer set of instance pairs ``(src, dst)`` that touch a common cell with ``src`` first in program
order. Program order is ``(u, statement, v)``. Every schedule question is then one emptiness query
on a piece with one more constraint, decided by ISL. A constraint ISL cannot represent is dropped,
which only enlarges a piece: every answer stays conservative.
"""

from collections.abc import Callable, Iterable
from dataclasses import dataclass

import islpy as isl
import sympy

from dace import symbolic
from dace.sdfg.analysis.polyhedral_isl import subs_by_name, to_isl
from dace.sdfg.narrowing import as_expr, ndrange_exprs, simplified
from dace.symbolic_facts import RelationKind, relation_names
from dace.transformation.passes.parallelization.wavefront import polyhedron as poly
from dace.transformation.passes.parallelization.wavefront.statements import (
    Access,
    Body,
    Snapshot,
    Statement,
    free_names,
    sequenced,
)

#: The relations a convex constraint states; ``!=`` and divisibility are not convex.
AFFINE_RELATIONS = (RelationKind.LT, RelationKind.LE, RelationKind.EQ)


def is_empty(text: str) -> bool:
    return isl.Set(text).is_empty()


@dataclass
class Instance:
    """One side of a piece: its dimension names, constraints and per-axis cell bounds, rendered for ISL."""

    dims: list[str]
    cons: list[str]
    cell_lo: list[str | None]
    cell_hi: list[str | None]


@dataclass
class Piece:
    src: Statement
    dst: Statement
    #: 0 output (both write), 1 flow (the source writes), 2 anti (the source reads).
    kind: int
    dims: list[str]
    cons: list[str]


class Renderer:
    """Renders expressions under one global naming: instance dimensions by role, every other symbol a parameter."""

    def __init__(self, body: Body, facts: symbolic.Facts, assume: Iterable[sympy.Expr]) -> None:
        self.body = body
        self.facts = facts
        self.params: dict[str, str] = {}
        self.unsafe = set(body.assigned)
        self.assume = list(assume)

    def param(self, name: str) -> str:
        if name not in self.params:
            self.params[name] = f"P{len(self.params)}"
        return self.params[name]

    def render(self, expr: sympy.Expr | None, names: dict[str, str]) -> str | None:
        """``expr`` over ISL names, or ``None`` when it is not affine or reads a per-iteration symbol."""
        if expr is None:
            return None
        e = simplified(as_expr(expr))
        mapping: dict[str, sympy.Expr] = {}
        for name in sorted(free_names(e)):
            if name in names:
                mapping[name] = as_expr(names[name])
            elif name in self.unsafe:
                return None
            else:
                mapping[name] = as_expr(self.param(name))
        try:
            return to_isl(subs_by_name(e, mapping))
        except ValueError:
            return None

    def geq(self, expr: sympy.Expr | None, names: dict[str, str]) -> str | None:
        r = self.render(expr, names)
        return None if r is None else f"{r} >= 0"

    def bound(self, var: str, bound: sympy.Expr | None, upper: bool, names: dict[str, str]) -> list[str]:
        """``var <= bound`` (``upper``) or ``var >= bound`` as conjuncts; a non-convex or unrenderable bound drops."""
        if bound is None:
            return []
        terms = symbolic.convex_bound_terms(as_expr(bound), upper)
        out = []
        for term in terms or []:
            r = self.render(term, names)
            if r is not None:
                out.append(f"{var} <= {r}" if upper else f"{var} >= {r}")
        return out

    def instance(self, st: Statement, acc: Access, tag: str) -> Instance:
        """The instance of ``st`` performing ``acc``, its dimensions suffixed by ``tag``."""
        u, v = f"u{tag}", f"v{tag}"
        names = {self.body.u: u, st.var: v}
        dims = [u, v]
        for i, (name, _, _) in enumerate(acc.ctx):
            names[name] = f"x{tag}{i}"
            dims.append(names[name])
        cons = self.bound(u, self.body.u_lo, False, names) + self.bound(u, self.body.u_hi, True, names)
        cons += self.bound(v, st.lo, False, names) + self.bound(v, st.hi, True, names)
        for i, (_, lo, hi) in enumerate(acc.ctx):
            cons += self.bound(f"x{tag}{i}", lo, False, names) + self.bound(f"x{tag}{i}", hi, True, names)
        cons += [c for c in (self.geq(g, names) for g in acc.guard) if c is not None]
        return Instance(dims, cons, [self.render(b, names) for b in acc.lo], [self.render(b, names) for b in acc.hi])

    def header(self) -> str:
        return ", ".join(self.params.values())

    def facts_constraints(self) -> list[str]:
        """Sign facts of every parameter, the affine relations among the parameters, plus the assumptions under trial."""
        out = []
        for name, isl_name in self.params.items():
            sym = as_expr(name)
            if provably(symbolic.Predicate.POSITIVE, sym, self.facts):
                out.append(f"{isl_name} >= 1")
            elif provably(symbolic.Predicate.NONNEGATIVE, sym, self.facts):
                out.append(f"{isl_name} >= 0")
        known = set(self.params)
        for relation in self.facts.about(known).relations:
            if relation.kind not in AFFINE_RELATIONS or not relation_names(relation) <= known:
                continue
            gap = relation.rhs - relation.lhs - (1 if relation.kind is RelationKind.LT else 0)
            r = self.render(gap, {})
            if r is not None:
                out.append(f"{r} {'=' if relation.kind is RelationKind.EQ else '>='} 0")
        for a in self.assume:
            r = self.geq(a, {})
            if r is not None:
                out.append(r)
        return out


def affine(coeffs: dict[str, int], const: int = 0) -> str:
    """``sum(c * name) + const`` in ISL syntax."""
    parts = []
    for name, c in coeffs.items():
        if c:
            parts.append(
                f"{'+' if c > 0 else '-'} {abs(c)}*{name}" if abs(c) != 1 else f"{'+' if c > 0 else '-'} {name}"
            )
    if const or not parts:
        parts.append(f"{'+' if const >= 0 else '-'} {abs(const)}")
    text = " ".join(parts)
    return text[2:] if text.startswith("+ ") else "-" + text[2:]


def provably(pred: symbolic.Predicate, expr: sympy.Expr, facts: symbolic.Facts) -> bool:
    return symbolic.ask(symbolic.predicate_relation(pred, expr), facts) is symbolic.Truth.TRUE


def cell_constraints(src: Instance, dst: Instance) -> tuple[list[str], list[str]] | None:
    """Shared cell dimensions and both sides' bounds on them; ``None`` when the ranks differ."""
    if len(src.cell_lo) != len(dst.cell_lo):
        return None
    dims, cons = [], []
    for a in range(len(src.cell_lo)):
        c = f"c{a}"
        dims.append(c)
        for side in (src, dst):
            if side.cell_lo[a] is not None:
                cons.append(f"{c} >= {side.cell_lo[a]}")
            if side.cell_hi[a] is not None:
                cons.append(f"{c} <= {side.cell_hi[a]}")
    return dims, cons


def order_pieces(src: Statement, dst: Statement) -> list[list[str]]:
    """The program-order constraints ``src`` before ``dst``, one list per convex piece."""
    if src.order < dst.order:
        return [["ud >= us"]]
    if src.order > dst.order:
        return [["ud >= us + 1"]]
    return [["ud >= us + 1"], ["ud = us", "vd >= vs + 1"]]


class Dependences:
    """Every non-empty dependence piece of ``body``."""

    def __init__(self, body: Body, facts: symbolic.Facts, assume: Iterable[sympy.Expr] = ()) -> None:
        self.body = body
        self.renderer = Renderer(body, facts, assume)
        self.pieces: list[Piece] = []
        self.complete = True
        by_data: dict[str, list[tuple[Statement, Access]]] = {}
        for st in body.statements:
            for acc in st.accesses:
                by_data.setdefault(acc.data, []).append((st, acc))
        raw: list[tuple[Statement, Statement, bool, list[str], list[str]]] = []
        for name in sorted(by_data):
            group = by_data[name]
            if not any(acc.write for _, acc in group):
                continue
            inst = [
                (st, acc, self.renderer.instance(st, acc, "s"), self.renderer.instance(st, acc, "d"))
                for st, acc in group
            ]
            for i, (st_a, acc_a, a_s, a_d) in enumerate(inst):
                for j in range(i, len(inst)):
                    st_b, acc_b, b_s, b_d = inst[j]
                    if not (acc_a.write or acc_b.write):
                        continue
                    pairs = [(st_a, acc_a, a_s, st_b, acc_b, b_d)]
                    if i != j:
                        pairs.append((st_b, acc_b, b_s, st_a, acc_a, a_d))
                    for src, src_acc, src_inst, dst, dst_acc, dst_inst in pairs:
                        kind = 0 if src_acc.write and dst_acc.write else (1 if src_acc.write else 2)
                        cells = cell_constraints(src_inst, dst_inst)
                        if cells is None:
                            self.complete = False
                            continue
                        dims = src_inst.dims + dst_inst.dims + cells[0]
                        cons = src_inst.cons + dst_inst.cons + cells[1]
                        raw.extend((src, dst, kind, dims, cons + order) for order in order_pieces(src, dst))
        self.facts_cons = self.renderer.facts_constraints()
        for src, dst, kind, dims, cons in raw:
            piece = Piece(src, dst, kind, dims, cons + self.facts_cons)
            if not is_empty(self.text(piece)):
                self.pieces.append(piece)

    def text(self, piece: Piece, extra: Iterable[str] = (), dims: Iterable[str] = ()) -> str:
        body = " and ".join([*piece.cons, *extra]) or "true"
        return f"[{self.renderer.header()}] -> {{ [{', '.join([*piece.dims, *dims])}] : {body} }}"

    def any_nonempty(self, extra_for: Callable[[Piece], list[str] | None]) -> bool:
        """Whether some piece meets the constraints ``extra_for(piece)`` returns (``None``: skip the piece)."""
        for piece in self.pieces:
            extra = extra_for(piece)
            if extra is not None and not is_empty(self.text(piece, extra)):
                return True
        return False

    def snapshot_safe(self, snap: Snapshot) -> bool:
        """Whether reading ``snap.live`` in place of the snapshot keeps every value: no write after the copy
        reaches a read's cell earlier in the same ``u``, and every read stays inside the copied window."""
        r = self.renderer
        reads = [
            (st, acc) for st in self.body.statements for acc in st.accesses if acc.snapshot and acc.data == snap.live
        ]
        writes = [
            (st, acc)
            for st in self.body.statements
            for acc in st.accesses
            if acc.write and acc.data == snap.live and st.order > snap.order
        ]
        for wst, wacc in writes:
            w_inst = r.instance(wst, wacc, "s")
            for rst, racc in reads:
                r_inst = r.instance(rst, racc, "d")
                cells = cell_constraints(w_inst, r_inst)
                if cells is None:
                    return False
                for order in order_pieces(wst, rst):
                    cons = w_inst.cons + r_inst.cons + cells[1] + order + ["ud = us"] + self.facts_cons
                    if not is_empty(self.text(Piece(wst, rst, 1, w_inst.dims + r_inst.dims + cells[0], cons))):
                        return False
        for rst, racc in reads:
            inst = r.instance(rst, racc, "d")
            for a, (lo, hi, _) in enumerate(ndrange_exprs(snap.window)):
                wlo, whi = r.render(lo, {self.body.u: "ud"}), r.render(hi, {self.body.u: "ud"})
                if wlo is None or whi is None or a >= len(inst.cell_lo):
                    return False
                if inst.cell_lo[a] is None or inst.cell_hi[a] is None:
                    return False
                for outside in (f"{inst.cell_lo[a]} <= {wlo} - 1", f"{inst.cell_hi[a]} >= {whi} + 1"):
                    if not is_empty(self.text(Piece(rst, rst, 2, inst.dims, inst.cons + [outside] + self.facts_cons))):
                        return False
        return True

    def outer_parallel(self) -> bool:
        """No dependence crosses ``u``: the outer loop is already a parallel map."""
        return not self.any_nonempty(lambda p: ["ud >= us + 1"])

    def schedule_legal(self, tau: tuple[int, int], shifts: dict[int, int]) -> bool:
        """``t = a*u + b*(v + shift)`` orders every dependence: a later step, or the same step with ``src``
        sequenced before ``dst``."""
        a, b = tau

        def violation(p: Piece) -> list[str]:
            gap = 1 if sequenced(p.src, p.dst) else 0
            const = b * (shifts.get(p.src.order, 0) - shifts.get(p.dst.order, 0)) - gap
            return [f"{affine({'us': a, 'vs': b, 'ud': -a, 'vd': -b}, const)} >= 0"]

        return not self.any_nonempty(violation)

    def permutable(self, shifts: dict[int, int], k: int) -> bool:
        """Every dependence is non-negative along ``u`` and ``w = v + shift + k*u``: rectangular ``(u, w)`` tiles
        run in ``I + J`` order, each in program order, and within a tile the ``(1, 1)`` diagonal is legal."""

        def violation(p: Piece) -> list[str]:
            const = shifts.get(p.src.order, 0) - shifts.get(p.dst.order, 0) - 1
            return [f"{affine({'vs': 1, 'us': k, 'vd': -1, 'ud': -k}, const)} >= 0"]

        return not self.any_nonempty(violation)

    def same_step_offset(self, piece: Piece) -> int | None:
        """``vd - vs`` when it is one integer over all same-``u`` instance pairs of ``piece``."""
        s = isl.Set(self.text(piece, ["ud = us", "delta = vd - vs"], ["delta"]))
        if s.is_empty():
            return None
        n = s.dim(isl.dim_type.set)
        delta = s.project_out(isl.dim_type.set, 0, n - 1)
        lo = poly.pwaff_bound(delta.dim_min(0), {})
        hi = poly.pwaff_bound(delta.dim_max(0), {})
        if lo is None or hi is None or not lo.is_Integer or lo != hi:
            return None
        return int(lo)

    def alignment(self) -> dict[int, int]:
        """Per-statement shifts that put same-``u`` dependences between statements at zero ``v`` distance: output
        dependences first, then flow, then anti, each kept only if it agrees with the shifts already fixed."""
        parent: dict[int, tuple[int, int]] = {st.order: (st.order, 0) for st in self.body.statements}

        def find(x: int) -> tuple[int, int]:
            root, off = parent[x]
            if root == x:
                return x, 0
            r, o = find(root)
            parent[x] = (r, off + o)
            return r, off + o

        candidates = [p for p in self.pieces if p.src.order < p.dst.order]
        candidates.sort(key=lambda p: (p.kind, p.src.order, p.dst.order))
        for p in candidates:
            ra, oa = find(p.src.order)
            rb, ob = find(p.dst.order)
            if ra == rb:
                continue
            delta = self.same_step_offset(p)
            if delta is None:
                continue
            # shift[dst] - shift[src] = -delta, with shift[x] = off(x) + shift[root(x)].
            parent[rb] = (ra, oa - delta - ob)
        return {st.order: find(st.order)[1] for st in self.body.statements}


def subset_params(body: Body) -> list[str]:
    """Parameters some access subset reads: the offsets a runtime sign guard may pin."""
    names: set[str] = set()
    for st in body.statements:
        local = {st.var, body.u}
        for acc in st.accesses:
            local_acc = local | {name for name, _, _ in acc.ctx}
            for b in [*acc.lo, *acc.hi]:
                if b is not None:
                    names |= free_names(b) - local_acc
    return sorted(names - body.assigned)


__all__ = ["Dependences", "Piece", "subset_params"]
