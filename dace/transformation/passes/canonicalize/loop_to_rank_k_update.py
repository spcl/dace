# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Lift a hand-written symmetric rank-k or rank-2k update nest to a ``Syrk`` / ``Syr2k`` BLAS node.

polybench ``syrk`` (``C := alpha*A*A^T + beta*C``) and ``syr2k`` (``C := alpha*A*B^T + alpha*B*A^T + beta*C``)
update one triangle of ``C``, one row per iteration of an outer loop over ``i``::

    for i in range(N):
        for j in range(i + 1):
            C[i, j] *= beta
        for k in range(M):
            for j in range(i + 1):
                C[i, j] += alpha * A[i, k] * A[j, k]                               # syrk
                C[i, j] += A[j, k] * alpha * B[i, k] + B[j, k] * alpha * A[i, k]   # syr2k

The slice spelling ``C[i, :i + 1] += alpha * A[i, k] * A[:i + 1, k]`` reaches ``lift_semantics`` as a chain of
elementwise loops over vector temporaries instead; both are recognized by what one outer iteration computes
(:func:`~dace.transformation.passes.canonicalize.loop_iteration_value.summarize`). The iteration must store only
one row triangle of a square ``C``: ``C[r, 0:r+1]`` (``uplo='L'``) or ``C[r, r:N]`` (``'U'``), where the row ``r``
is ``i`` or ``N - 1 - i`` (either loop direction), and there store exactly::

    beta * C[r, j] + alpha * Sum_k A[r, k] * A[j, k]                                 # Syrk
    beta * C[r, j] + alpha * Sum_k (A[r, k] * B[j, k] + B[r, k] * A[j, k])           # Syr2k

with the operands read ``[r, k]`` / ``[j, k]`` (``trans='N'``) or ``[k, r]`` / ``[k, j]`` (``'T'``) and written
nowhere in the nest. ``alpha`` and ``beta`` are numbers, runtime single-element containers, or their products; no
prior ``C`` term is ``beta = 0``. Iterations store disjoint rows and read ``C`` only at the element they store, so
the loop computes the BLAS update in either direction. Any deviation is a clean no-op.
"""

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
    canonical,
    leaf,
    leaves,
    outer_loop_candidates,
    own_indices,
    read_array,
    same,
    scale_of,
    shape_is,
    splice_blas_update,
    summarize,
    transients_dead_outside,
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


def match_rank_k(sdfg: SDFG, loop: LoopRegion) -> RankKMatch | None:
    try:
        summary = summarize(sdfg, loop)
    except Refused:
        return None
    shape = triangle(summary)
    c, n = summary.output, summary.extent
    if shape is None or not shape_is(sdfg, c, n, n):
        return None
    uplo, r, j = shape
    value = canonical(summary.domains[0].value)
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
    if any(t.alpha != first.alpha or t.trans != first.trans or not same(t.extent, first.extent) for t in terms):
        return None
    pairs = sorted((t.row, t.col) for t in terms)
    if len(pairs) == 1 and pairs[0][0] == pairs[0][1]:
        operands = [pairs[0][0]]
    elif len(pairs) == 2 and pairs[0] == pairs[1][::-1] and pairs[0][0] != pairs[0][1]:
        operands = list(pairs[0])
    else:
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
                replace(parent, loop, match)
                count += 1
        return count or None


def replace(parent: ControlFlowRegion, loop: LoopRegion, match: RankKMatch) -> None:
    # Deferred: the BLAS package imports transformations.
    from dace.libraries.blas.nodes.syr2k import Syr2k
    from dace.libraries.blas.nodes.syrk import Syrk

    node_class, suffix = (Syrk, "_syrk") if match.b is None else (Syr2k, "_syr2k")
    operands = {"_A": match.a} if match.b is None else {"_A": match.a, "_B": match.b}
    node = node_class(loop.label + suffix, uplo=match.uplo == "U", transA=match.trans == "T")
    splice_blas_update(parent, loop, node, operands, match.c, match.alpha, match.beta)


__all__ = ["LoopToRankKUpdate"]
