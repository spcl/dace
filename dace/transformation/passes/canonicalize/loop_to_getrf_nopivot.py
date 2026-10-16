# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Lift a hand-written LU elimination without pivoting to a ``GetrfNoPivot`` LAPACK node.

The right-looking (Doolittle) elimination of polybench/npbench ``lu`` and ``ludcmp``::

    for k in range(N):
        A[k + 1:, k] /= A[k, k]
        A[k + 1:, k + 1:] -= np.outer(A[k + 1:, k], A[k, k + 1:])

is recognized by what one outer iteration computes
(:func:`~dace.transformation.passes.canonicalize.loop_iteration_value.summarize`), not by its spelling: slices, a
scalar loop nest, a reciprocal kept in a temporary all reduce to the same values. Iteration ``k`` of an ``n x n``
array ``A`` has to store exactly two domains, in terms of the values at iteration start::

    A[i, k] = A[i, k] / A[k, k]                         i in k+1 .. n-1
    A[i, j] = A[i, j] - A[i, k] * A[k, j] / A[k, k]     i, j in k+1 .. n-1

with ``k`` visiting ``0 .. n-1`` (or ``0 .. n-2``: the last step is empty) in rising order. That is step ``k`` of
the elimination, so the nest composes to the factorization ``A = L @ U``, ``L`` unit lower triangular, packed in
place -- exactly ``GetrfNoPivot``. The nest may write no other non-transient container and leave no transient that
is read after it. Anything else -- a pivot search, a scaled row instead of a column (the transposed, Crout
factorization), a partial range -- is a clean no-op and the nest stays on the loop path.

On CPU the node is expanded right away: its ``pure`` expansion is the plain right-looking nest, which the
parallelism phase maps (a parallel trailing update per ``k``), so every spelling of the factorization reaches that
phase as the one nest it parallelizes best. On GPU the node stays, and the target finalization lowers it to the
vendor solver.
"""

from typing import Any

from dace import SDFG, properties
from dace import data as dt
from dace.sdfg import nodes
from dace.sdfg.state import ControlFlowRegion, LoopRegion
from dace.transformation import pass_pipeline as ppl
from dace.transformation.passes.canonicalize.loop_iteration_value import (
    AxisKind,
    Domain,
    Refused,
    Summary,
    canonical,
    full_memlet,
    leaf,
    outer_loop_candidates,
    own_indices,
    plain,
    replace_loop_with_state,
    same,
    summarize,
    transients_dead_outside,
)
from dace.transformation.transformation import explicit_cf_compatible


def spans(domain: Domain, axis: int, lo: Any, hi: Any) -> bool:
    """Whether ``domain`` covers ``lo .. hi`` along ``axis`` as a whole range."""
    a = domain.axes[axis]
    return a.kind is AxisKind.RANGE and same(a.lo, lo) and same(a.hi, hi)


def at(domain: Domain, axis: int, value: Any) -> bool:
    a = domain.axes[axis]
    return a.kind is AxisKind.FIXED and same(a.lo, value)


def is_elimination_step(summary: Summary, k: Any, n: Any) -> bool:
    """Whether the summarized iteration is elimination step ``k`` of an ``n x n`` matrix (the module docstring)."""
    a = summary.output
    column = [d for d in summary.domains if spans(d, 0, k + 1, n - 1) and at(d, 1, k)]
    trailing = [d for d in summary.domains if spans(d, 0, k + 1, n - 1) and spans(d, 1, k + 1, n - 1)]
    if len(summary.domains) != 2 or len(column) != 1 or len(trailing) != 1:
        return False
    pivot = leaf(a, (k, k))
    i = column[0].index[0]
    if not same(canonical(column[0].value), canonical(leaf(a, (i, k)) / pivot)):
        return False
    i, j = trailing[0].index
    update = leaf(a, (i, j)) - leaf(a, (i, k)) * leaf(a, (k, j)) / pivot
    return same(canonical(trailing[0].value), canonical(update))


def match_getrf_nopivot(sdfg: SDFG, loop: LoopRegion) -> str | None:
    """The array ``loop`` factors in place without pivoting, or ``None``."""
    try:
        summary = summarize(sdfg, loop)
    except Refused:
        return None
    desc = sdfg.arrays[summary.output]
    if len(desc.shape) != 2 or not isinstance(desc, dt.Array) or isinstance(desc, (dt.View, dt.Reference)):
        return None
    n = plain(desc.shape[0])
    if not same(plain(desc.shape[1]), n):
        return None
    if not (same(summary.extent, n) or same(summary.extent, n - 1)):
        return None
    try:
        if not any(rising and is_elimination_step(summary, k, n) for k, rising in own_indices(summary)):
            return None
    except Refused:
        return None
    if not transients_dead_outside(sdfg, loop, summary):
        return None
    return summary.output


@properties.make_properties
@explicit_cf_compatible
class LoopToGetrfNoPivot(ppl.Pass):
    """Lift a hand-written no-pivot LU elimination nest to a ``GetrfNoPivot`` node (expanded at once on CPU)."""

    CATEGORY: str = "Canonicalization"

    target = properties.Property(dtype=str, default="cpu", desc="'cpu' expands the lifted node to its pure nest")

    def __init__(self, target: str = "cpu") -> None:
        super().__init__()
        self.target = target

    def modifies(self) -> ppl.Modifies:
        return ppl.Modifies.CFG | ppl.Modifies.Nodes | ppl.Modifies.Memlets | ppl.Modifies.Descriptors

    def should_reapply(self, modified: ppl.Modifies) -> bool:
        return bool(modified & ppl.Modifies.CFG)

    def apply_pass(self, sdfg: SDFG, _: dict[str, Any]) -> int | None:
        count = 0
        for parent, loop in outer_loop_candidates(sdfg):
            if loop not in parent.nodes():
                continue  # inside a nest an earlier lift spliced out
            array = match_getrf_nopivot(parent.sdfg, loop)
            if array is not None:
                replace(parent, loop, array, expand=self.target == "cpu")
                count += 1
        return count or None


def replace(parent: ControlFlowRegion, loop: LoopRegion, array: str, expand: bool) -> None:
    """Splice the recognized nest out for a ``GetrfNoPivot`` on ``array``; with ``expand``, inline its pure nest."""
    from dace.libraries.lapack.nodes.getrf_nopivot import GetrfNoPivot  # local: the library imports transformations
    from dace.transformation.interstate.multistate_inline import InlineMultistateSDFG

    sdfg = parent.sdfg
    state = replace_loop_with_state(parent, loop, loop.label + "_getrf_nopivot")
    node = GetrfNoPivot(loop.label + "_getrf_nopivot")
    state.add_node(node)
    state.add_edge(state.add_read(array), None, node, "_x", full_memlet(sdfg, array))
    state.add_edge(node, "_x", state.add_write(array), None, full_memlet(sdfg, array))
    if not expand:
        return
    node.implementation = "pure"
    node.expand(sdfg, state)
    (nested,) = [n for n in state.nodes() if isinstance(n, nodes.NestedSDFG)]
    # Refused only for a shape the lift never produces (a view, an in-scope node); the nested nest stays correct.
    if InlineMultistateSDFG.can_be_applied_to(sdfg, nested_sdfg=nested):
        InlineMultistateSDFG.apply_to(sdfg, nested_sdfg=nested, verify=False, save=False)


__all__ = ["LoopToGetrfNoPivot", "match_getrf_nopivot"]
