# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Lift a hand-written symmetric rank-k or rank-2k update nest to a ``Syrk`` / ``Syr2k`` BLAS node."""

from typing import Any, NamedTuple

import sympy
from sympy.core.function import AppliedUndef

from dace import SDFG
from dace.sdfg.state import ControlFlowRegion, LoopRegion
from dace.transformation import pass_pipeline as ppl
from dace.transformation.passes.canonicalize.loop_iteration_value import (
    AxisKind,
    Refused,
    Scale,
    Summary,
    adjacent_scale,
    canonical,
    leaf,
    leaves,
    loop_range,
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
    times,
    transients_dead_outside,
    unit_stride,
)
from dace.transformation.transformation import explicit_cf_compatible


class RankKMatch(NamedTuple):
    """Operands, coefficients and orientation of a recognized rank-k / rank-2k nest."""

    c: str
    a: str
    b: str | None  #: ``None`` for syrk
    alpha: Scale
    beta: Scale | None  #: ``None``: the prior ``C`` is dropped (``beta = 0``)
    uplo: str
    trans: str


def triangle(summary: Summary) -> tuple[str, sympy.Expr, sympy.Expr] | None:
    """``(uplo, r, j)`` when the iteration stores exactly one row triangle of a square ``C``: ``C[r, 0:r+1]`` is
    ``'L'``, ``C[r, r:N]`` is ``'U'``; ``r`` is the row the iteration owns, ``j`` the query symbol of the column."""
    if len(summary.domains) != 1 or len(summary.domains[0].axes) != 2:
        return None
    (domain,) = summary.domains
    row, col = domain.axes
    if row.kind is not AxisKind.FIXED or col.kind is not AxisKind.RANGE:
        return None
    n = summary.extent
    for own in own_indices(summary):
        r = own.index
        if not same(row.lo, r):
            continue
        if same(col.lo, 0) and same(col.hi, r):
            return "L", r, domain.index[1]
        if same(col.lo, r) and same(col.hi, n - 1):
            return "U", r, domain.index[1]
    return None


class Pairing(NamedTuple):
    """One term ``alpha * Sum_k row[r, k] * col[j, k]`` (``[k, r]`` / ``[k, j]`` for ``trans='T'``)."""

    alpha: Scale
    row: str
    col: str
    extent: sympy.Expr
    trans: str


def pairing(sdfg: SDFG, term: sympy.Expr, r: sympy.Expr, j: sympy.Expr) -> Pairing | None:
    factors = sympy.Mul.make_args(term)
    sums = [f for f in factors if isinstance(f, sympy.Sum)]
    alpha = scale_of(sdfg, sympy.Mul(*(f for f in factors if not isinstance(f, sympy.Sum))))
    if len(sums) != 1 or alpha is None:
        return None
    (k, lo, hi), inner = sums[0].limits[0], sympy.Mul.make_args(sums[0].function)
    if not same(lo, 0) or hi.has(*r.free_symbols) or hi.has(j) or len(inner) != 2:
        return None
    if not all(isinstance(read, AppliedUndef) and len(read.args) == 2 for read in inner):
        return None
    for trans, (outer_axis, k_axis) in (("N", (0, 1)), ("T", (1, 0))):
        if not all(read.args[k_axis] == k for read in inner):
            continue
        rows = [read for read in inner if same(read.args[outer_axis], r)]
        cols = [read for read in inner if read.args[outer_axis] == j]
        if len(rows) == 1 and len(cols) == 1:
            return Pairing(alpha, read_array(rows[0]), read_array(cols[0]), hi + 1, trans)
    return None


def rank_k_operands(terms: list[Pairing]) -> list[str] | None:
    """The operand of ``A A^T`` or the two of ``A B^T + B A^T``, when every term shares one scale, transpose and
    inner extent; ``None`` otherwise."""
    first = terms[0]
    if any(t.alpha != first.alpha or t.trans != first.trans or not same(t.extent, first.extent) for t in terms):
        return None
    pairs = sorted((t.row, t.col) for t in terms)
    if len(pairs) == 1 and pairs[0][0] == pairs[0][1]:
        return [pairs[0][0]]
    if len(pairs) == 2 and pairs[0] == pairs[1][::-1] and pairs[0][0] != pairs[0][1]:
        return list(pairs[0])
    return None


def reduction_outside(sdfg: SDFG, loop: LoopRegion) -> tuple[Summary, sympy.Expr] | None:
    """``for k: <one triangle nest adding a term of k to C>`` (stride permutation moves the inner product loop out
    once distribution has split ``C *= beta`` away): the inner nest's summary and its value summed over ``k``."""
    body = loop.nodes()
    if len(body) != 1 or not isinstance(body[0], LoopRegion) or not unit_stride(loop):
        return None
    lo, hi = loop_range(loop, {})
    try:
        inner = summarize(sdfg, body[0])
    except Refused:
        return None
    shape = triangle(inner)
    k = sympy.Symbol(loop.loop_variable)
    if not same(lo, 0) or shape is None or inner.extent.has(k) or k in inner.written:
        return None
    _, r, j = shape
    prior = leaf(inner.output, (r, j))
    value = canonical(inner.domains[0].value)
    term = sympy.expand(value - prior)
    if value.coeff(prior) != 1 or any(read_array(read) == inner.output for read in leaves(term)):
        return None
    return inner, canonical(prior + sympy.Sum(term, (k, 0, hi)))


def match_rank_k(sdfg: SDFG, loop: LoopRegion) -> RankKMatch | None:
    try:
        summary = summarize(sdfg, loop)
        value = canonical(summary.domains[0].value) if summary.domains else None
    except Refused:
        found = reduction_outside(sdfg, loop)
        if found is None:
            return None
        summary, value = found
    shape = triangle(summary)
    c, n = summary.output, summary.extent
    if shape is None or not shape_is(sdfg, c, n, n):
        return None
    uplo, r, j = shape
    prior = leaf(c, (r, j))
    beta_factor = value.coeff(prior)
    beta = None if beta_factor == 0 else scale_of(sdfg, beta_factor)
    if beta_factor != 0 and beta is None:
        return None
    update = sympy.expand(value - beta_factor * prior)
    if any(read_array(read) == c for read in leaves(update)):
        return None
    terms = [pairing(sdfg, t, r, j) for t in sympy.Add.make_args(update)]
    if not terms or any(t is None for t in terms):
        return None
    first = terms[0]
    operands = rank_k_operands(terms)
    if operands is None:
        return None
    extent = (n, first.extent) if first.trans == "N" else (first.extent, n)
    if c in operands or not all(shape_is(sdfg, name, *extent) for name in operands):
        return None
    runtime = [s.name for s in (first.alpha, beta) if s is not None and s.name is not None]
    if any(name in summary.written for name in (*operands, *runtime)):
        return None
    if not transients_dead_outside(sdfg, loop, summary):
        return None
    return RankKMatch(c, operands[0], operands[1] if len(operands) == 2 else None, first.alpha, beta, uplo, first.trans)


@explicit_cf_compatible
class LoopToRankKUpdate(ppl.Pass):
    """Lift a hand-written symmetric rank-k / rank-2k update loop nest to a ``Syrk`` / ``Syr2k`` node."""

    CATEGORY: str = "Canonicalization"

    def modifies(self) -> ppl.Modifies:
        return ppl.Modifies.CFG | ppl.Modifies.Nodes | ppl.Modifies.Memlets

    def should_reapply(self, modified: ppl.Modifies) -> bool:
        return bool(modified & ppl.Modifies.CFG)

    def apply_pass(self, sdfg: SDFG, pipeline_results: dict[str, Any]) -> int | None:
        count = 0
        for parent, loop in outer_loop_candidates(sdfg):
            if loop not in parent.nodes():
                continue  # inside a nest an earlier lift spliced out
            match = match_rank_k(parent.sdfg, loop)
            if match is not None:
                replace(parent, loop, fold_scales(parent, loop, match))
                count += 1
        return count or None


def stored_triangle(summary: Summary) -> tuple[str, sympy.Expr] | None:
    """The triangle and extent ``summary`` stores, comparable across two loops."""
    shape = triangle(summary)
    return None if shape is None else (shape[0], summary.extent)


def fold_scales(parent: ControlFlowRegion, loop: LoopRegion, match: RankKMatch) -> RankKMatch:
    """``match`` with the scale loops of its triangle next to ``loop`` folded in: one before it multiplies ``beta``,
    one after it ``alpha`` and ``beta``. Each folded loop becomes an empty state."""
    sdfg = parent.sdfg
    key = (match.uplo, plain(sdfg.arrays[match.c].shape[0]))
    before = adjacent_scale(sdfg, parent, loop, match.c, stored_triangle, key, after=False)
    if before is not None and match.beta is not None:
        beta = times(match.beta, before[1])
        if beta is not None:
            replace_loop_with_state(parent, before[0], before[0].label + "_folded")
            match = match._replace(beta=beta)
    after = adjacent_scale(sdfg, parent, loop, match.c, stored_triangle, key, after=True)
    if after is not None:
        alpha = times(match.alpha, after[1])
        beta = None if match.beta is None else times(match.beta, after[1])
        if alpha is not None and (match.beta is None or beta is not None):
            replace_loop_with_state(parent, after[0], after[0].label + "_folded")
            match = match._replace(alpha=alpha, beta=beta)
    return match


def replace(parent: ControlFlowRegion, loop: LoopRegion, match: RankKMatch) -> None:
    # Deferred: the BLAS package imports transformations.
    from dace.libraries.blas.nodes.syr2k import Syr2k
    from dace.libraries.blas.nodes.syrk import Syrk

    node_class, suffix = (Syrk, "_syrk") if match.b is None else (Syr2k, "_syr2k")
    operands = {"_A": match.a} if match.b is None else {"_A": match.a, "_B": match.b}
    node = node_class(loop.label + suffix, uplo=match.uplo == "U", transA=match.trans == "T")
    splice_blas_update(parent, loop, node, operands, match.c, match.alpha, match.beta)


__all__ = ["LoopToRankKUpdate"]
