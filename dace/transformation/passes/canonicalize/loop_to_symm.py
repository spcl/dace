# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Lift a hand-written symmetric matrix-multiply nest to a ``Symm`` BLAS node."""

import dataclasses
from typing import Any, NamedTuple

import sympy
from sympy.core.function import AppliedUndef

from dace import SDFG
from dace.sdfg.state import ControlFlowRegion, LoopRegion
from dace.transformation import pass_pipeline as ppl
from dace.transformation.passes.canonicalize.loop_iteration_value import (
    AxisKind,
    Domain,
    Refused,
    Scale,
    Summary,
    canonical,
    leaf,
    outer_loop_candidates,
    own_indices,
    plain,
    read_array,
    replace_loop_with_state,
    same,
    scale_of,
    shape_is,
    splice_blas_update,
    summarize,
    transients_dead_outside,
)
from dace.transformation.transformation import explicit_cf_compatible


class SymmMatch(NamedTuple):
    """Operands, coefficients and orientation of a recognized ``symm`` nest."""

    c: str
    a: str
    b: str
    alpha: Scale
    beta: Scale | None  #: ``None``: the prior ``C`` is dropped (``beta = 0``)
    side: str
    uplo: str
    folded: LoopRegion | None = None  #: the second loop of a split nest, spliced out with the first


class Layout(NamedTuple):
    """Where the iteration's two domains sit: the axis of ``C`` the iteration owns, its index ``r``, and the side
    ``O`` of ``r`` the scatter covers."""

    axis: int  #: ``0`` for side L (rows), ``1`` for side R (columns)
    r: sympy.Expr
    rising: bool
    below: bool  #: ``O = 0 .. r-1``, else ``r+1 .. extent-1``
    own: Domain
    others: Domain


def layout(summary: Summary, axis: int, cross: sympy.Expr) -> Layout | None:
    """The two domains of a nest that owns index ``r`` on ``axis`` of ``C``, each spanning the other axis
    ``0 .. cross - 1``."""
    if len(summary.domains) != 2 or any(len(d.axes) != 2 for d in summary.domains):
        return None
    n = summary.extent
    for r, rising in own_indices(summary):
        own = others = None
        below = False
        for domain in summary.domains:
            mine, span = domain.axes[axis], domain.axes[1 - axis]
            if span.kind is not AxisKind.RANGE or not (same(span.lo, 0) and same(span.hi, cross - 1)):
                return None
            if mine.kind is AxisKind.FIXED and same(mine.lo, r):
                own = domain
            elif mine.kind is AxisKind.RANGE and same(mine.lo, 0) and same(mine.hi, r - 1):
                others, below = domain, True
            elif mine.kind is AxisKind.RANGE and same(mine.lo, r + 1) and same(mine.hi, n - 1):
                others, below = domain, False
        if own is not None and others is not None:
            return Layout(axis, r, rising, below, own, others)
    return None


def at(axis: int, owned: sympy.Expr, cross: sympy.Expr) -> tuple[sympy.Expr, sympy.Expr]:
    """The ``C`` (and ``B``) index with ``owned`` on ``axis`` and ``cross`` on the other axis."""
    return (owned, cross) if axis == 0 else (cross, owned)


def match_side(sdfg: SDFG, summary: Summary, place: Layout) -> SymmMatch | None:
    c, axis, r = summary.output, place.axis, place.r
    # The scatter: C[p, j] + alpha * B[r, j] * a(r, p) over the rows (columns) p beside r.
    p, j = place.others.index[axis], place.others.index[1 - axis]
    scattered = canonical(place.others.value)
    prior = leaf(c, at(axis, p, j))
    if scattered.coeff(prior) != 1:
        return None
    update = sympy.expand(scattered - prior)
    reads = [f for f in sympy.Mul.make_args(update) if isinstance(f, AppliedUndef) and len(f.args) == 2]
    alpha = scale_of(sdfg, sympy.Mul(*(f for f in sympy.Mul.make_args(update) if f not in reads)))
    b_reads = [f for f in reads if f.args == at(axis, r, j)]
    a_reads = [f for f in reads if f not in b_reads]
    if alpha is None or len(b_reads) != 1 or len(a_reads) != 1:
        return None
    a, b = read_array(a_reads[0]), read_array(b_reads[0])
    if a_reads[0].args == (r, p):
        row_form = True
    elif a_reads[0].args == (p, r):
        row_form = False
    else:
        return None
    names = [c, a, b] if alpha.name is None else [c, a, b, alpha.name]
    if len(set(names)) != len(names):
        return None

    # Row (column) r: whatever scales the prior C is beta; the rest is symm's diagonal and inner-product terms.
    q = place.own.index[1 - axis]
    finished = canonical(place.own.value)
    own_prior = leaf(c, at(axis, r, q))
    beta_factor = finished.coeff(own_prior)
    beta = None if beta_factor == 0 else scale_of(sdfg, beta_factor)
    if beta_factor != 0 and beta is None:
        return None
    k = sympy.Dummy("k")
    lo, hi = (0, r - 1) if place.below else (r + 1, summary.extent - 1)
    entry = leaf(a, (r, k) if row_form else (k, r))
    alpha_value = alpha.value * (leaf(alpha.name, (0,)) if alpha.name is not None else 1)
    expected = (
        beta_factor * own_prior
        + alpha_value * leaf(b, at(axis, r, q)) * leaf(a, (r, r))
        + alpha_value * sympy.Sum(entry * leaf(b, at(axis, k, q)), (k, lo, hi))
    )
    if sympy.expand(finished - canonical(expected)) != 0:
        return None
    # The scatters into r land after r is finished, or the finish only adds.
    if place.below != place.rising and not (beta is not None and beta.is_one()):
        return None
    uplo = "L" if row_form == place.below else "U"
    return SymmMatch(c, a, b, alpha, beta, "L" if axis == 0 else "R", uplo)


def renamed(domain: Domain, rename: dict[sympy.Symbol, sympy.Symbol]) -> Domain:
    """``domain`` with the loop variables of ``rename`` replaced, in its axes, query index and value."""
    axes = tuple(dataclasses.replace(a, lo=a.lo.xreplace(rename), hi=a.hi.xreplace(rename)) for a in domain.axes)
    return Domain(axes, tuple(x.xreplace(rename) for x in domain.index), domain.value.xreplace(rename))


def split_pair(
    sdfg: SDFG, parent: ControlFlowRegion, loop: LoopRegion
) -> tuple[Summary, LoopRegion, Summary, Summary] | None:
    """The summary of ``loop`` and the next loop as one nest, that loop, and the two loops' own summaries, when each
    stores one domain of the same output over the same rows: distribution splits symm's scatter from its row
    finish."""
    edges = parent.out_edges(loop)
    if len(edges) != 1 or edges[0].data.assignments or not edges[0].data.is_unconditional():
        return None
    other = edges[0].dst
    if not isinstance(other, LoopRegion) or parent.in_degree(other) != 1:
        return None
    try:
        first, second = summarize(sdfg, loop), summarize(sdfg, other)
    except Refused:
        return None
    if (
        first.output != second.output
        or len(first.domains) != 1
        or len(second.domains) != 1
        or first.ascending != second.ascending
        or not same(first.extent, second.extent)
    ):
        return None
    (domain,) = second.domains
    # Each summary mints its own query symbols (``__q0``): the second loop's get names of their own
    rename = {x: sympy.Symbol(f"{x.name}_next") for x in domain.index if isinstance(x, sympy.Symbol)}
    merged = [first.domains[0], renamed(domain, {**rename, second.var: first.var})]
    written = {**first.written, **second.written}
    return Summary(first.output, first.var, first.extent, first.ascending, merged, written), other, first, second


def match_symm(sdfg: SDFG, loop: LoopRegion, parent: ControlFlowRegion | None = None) -> SymmMatch | None:
    """The symm ``loop`` computes; with ``parent``, also over ``loop`` and the next loop when distribution split the
    two (:attr:`SymmMatch.folded` names that loop)."""
    folded = None
    try:
        summary = summarize(sdfg, loop)
    except Refused:
        return None
    own, theirs = summary, summary
    if len(summary.domains) == 1 and parent is not None:
        pair = split_pair(sdfg, parent, loop)
        if pair is None:
            return None
        summary, folded, own, theirs = pair
    c = summary.output
    shape = sdfg.arrays[c].shape
    if len(shape) != 2:
        return None
    rows, cols = plain(shape[0]), plain(shape[1])
    for axis, cross in ((0, cols), (1, rows)):
        place = layout(summary, axis, cross)
        match = None if place is None else match_side(sdfg, summary, place)
        if match is None:
            continue
        n = summary.extent
        if not same(n, rows if axis == 0 else cols):
            return None
        if not (shape_is(sdfg, match.a, n, n) and shape_is(sdfg, match.b, rows, cols)):
            return None
        runtime = [s.name for s in (match.alpha, match.beta) if s is not None and s.name is not None]
        if any(name in summary.written for name in (match.a, match.b, *runtime)):
            return None
        if not transients_dead_outside(sdfg, loop, own):
            return None
        if folded is None:
            return match
        # Split in two: every scatter lands after its row's finish, or the finish only adds
        if place.own is not summary.domains[0] and not (match.beta is not None and match.beta.is_one()):
            return None
        if not transients_dead_outside(sdfg, folded, theirs):
            return None
        return match._replace(folded=folded)
    return None


@explicit_cf_compatible
class LoopToSymm(ppl.Pass):
    """Lift a hand-written symmetric matrix-multiply loop nest to a ``Symm`` node."""

    CATEGORY: str = "Canonicalization"

    def modifies(self) -> ppl.Modifies:
        return ppl.Modifies.CFG | ppl.Modifies.Nodes | ppl.Modifies.Memlets

    def should_reapply(self, modified: ppl.Modifies) -> bool:
        return bool(modified & ppl.Modifies.CFG)

    def apply_pass(self, sdfg: SDFG, _: dict[str, Any]) -> int | None:
        count = 0
        for parent, loop in outer_loop_candidates(sdfg):
            if loop not in parent.nodes():
                continue  # inside a nest an earlier lift spliced out
            match = match_symm(parent.sdfg, loop, parent)
            if match is not None:
                replace(parent, loop, match)
                count += 1
        return count or None


def replace(parent: ControlFlowRegion, loop: LoopRegion, match: SymmMatch) -> None:
    """Splice the recognized nest out and emit the ``Symm`` node in its place."""
    from dace.libraries.blas.nodes.symm import Symm  # local: the BLAS package imports transformations

    if match.folded is not None:
        replace_loop_with_state(parent, match.folded, match.folded.label + "_folded")
    node = Symm(loop.label + "_symm", side=match.side == "R", uplo=match.uplo == "U")
    splice_blas_update(parent, loop, node, {"_A": match.a, "_B": match.b}, match.c, match.alpha, match.beta)


__all__ = ["LoopToSymm"]
