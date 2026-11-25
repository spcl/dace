# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Parallelize loops that are data-parallel only under an algebraic constraint.

Some loops are safe to map iff a symbolic quantity satisfies a side condition
that cannot be discharged at compile time. Canonicalization historically *assumed*
such conditions (e.g. symbols are nonnegative), which is unsound when the
assumption is violated at runtime. This pass instead makes the assumption
explicit by *specializing*: it replaces the loop with a two-way conditional
``if (constraint) { parallel Map } else { original sequential loop }`` (see
:func:`~dace.transformation.passes.loop_specialization.specialize_loop_under_condition`).
A runtime value that satisfies the constraint takes the parallel path; one that
violates it still computes correctly on the sequential fallback -- ``if cond:
par else: seq``, never a silent miscompile and never an abort.

This is the same conditional-parallelization shape as
:class:`~dace.transformation.passes.parallelization.contracts.guarded_scatter.ScatterToGuardedMaps`,
but the predicate here is an algebraic condition over loop symbols.

Constraint types:

* **``coeff != 0``** -- a write whose index has a *symbolic* coefficient on the
  loop variable. Two sub-shapes qualify:

  * an in-place read-modify-write (``a[i * inc] = a[i * inc] + b[i]``; TSVC
    ``s171``), and
  * a plain injective store (``dst[i * SSYM] = src[i] * scale``; the TSVC-2.5
    ``ext_strided_store_ssym`` symbolic-stride scatter *store*).

  In both cases the write is injective -- and the loop therefore data-parallel --
  exactly when that coefficient is nonzero; permissive :class:`LoopToMap` lifts
  the true branch to a plain (WCR-free, vectorizable) Map, while the else branch
  keeps the sequential loop for ``inc == 0`` / ``SSYM == 0``. A loop-carried
  recurrence (``aa[c*i] = aa[c*i - 1]``, the array read at a subset *other* than
  the one written) is deliberately excluded: a ``coeff != 0`` condition does not
  make that data-parallel.
"""

from dataclasses import dataclass

from dace import SDFG, Memlet, dtypes, subsets, symbolic
from dace.optionals import required
from dace.ordered import OrderedSet
from dace.sdfg import nodes
from dace.sdfg.narrowing import as_expr
from dace.sdfg.state import ConditionalBlock, ControlFlowRegion, LoopRegion, SDFGState
from dace.transformation import pass_pipeline as ppl
from dace.transformation.passes.analysis import loop_analysis
from dace.transformation.passes.loop_specialization import (
    evaluate_before,
    invariant_host,
    specialize_loop_under_condition,
)
from dace.transformation.passes.symbol_propagation import consistent_bindings, resolve_bindings
from dace.transformation.transformation import explicit_cf_compatible


@dataclass(slots=True)
class ScopeFacts:
    """Whole-SDFG facts the stride matcher reads, built on first use and valid until ``sdfg`` mutates."""

    sdfg: SDFG
    invariant: set[str] | None = None
    bindings: dict[str, str | None] | None = None

    def invariant_names(self) -> set[str]:
        if self.invariant is None:
            self.invariant = set(self.sdfg.free_symbols) | set(self.sdfg.constants_prop)
        return self.invariant

    def binding_table(self) -> dict[str, str | None]:
        if self.bindings is None:
            self.bindings = consistent_bindings(self.sdfg)
        return self.bindings


@explicit_cf_compatible
class ParallelizeUnderConstraint(ppl.Pass):
    """Specialize into ``if cond: parallel-Map else: sequential-loop`` the loops
    that are data-parallel only under a symbolic constraint.

    With ``assume_constraint=True`` the runtime check is skipped and the loop is
    parallelized unconditionally -- the caller asserts the constraint (e.g.
    ``inc != 0``) always holds. Default ``False`` keeps the sound ``if cond: par
    else: seq`` guard."""

    CATEGORY: str = "Optimization Preparation"

    def __init__(self, assume_constraint: bool = False):
        super().__init__()
        self.assume_constraint = assume_constraint

    def modifies(self) -> ppl.Modifies:
        return ppl.Modifies.Everything

    def should_reapply(self, _modified: ppl.Modifies) -> bool:
        return False

    def depends_on(self):
        return set()

    def apply_pass(self, sdfg: SDFG, pipeline_results) -> int | None:
        """Specialize every constraint-parallel loop into ``if cond: par else: seq``. After :class:`ParallelizeLoops`
        in a pipeline, only the loops it refused on a non-unique write are candidates, and its verdict is reused.

        :param sdfg: SDFG to mutate in place.
        :returns: The number of loops specialized, or ``None`` if none.
        """
        from dace.transformation.interstate.loop_to_map import LoopToMap, Refusal
        from dace.transformation.passes.parallelization.parallelize_loops import REFUSALS, ParallelizeLoops

        refusals = pipeline_results.get(REFUSALS)

        specialized = 0
        # (host block, pointer, range) -> the symbol holding that pointer check's result, shared by every loop
        checks: dict[tuple[int, str, str], str] = {}
        for sd in sdfg.all_sdfgs_recursive():
            owner = sd
            while owner.parent_sdfg is not None:
                owner = owner.parent_sdfg
            # Probes and the matcher are read-only, so these stay valid until a specialization.
            facts = ScopeFacts(sd)
            for loop in list(sd.all_control_flow_regions(recursive=True)):
                if not (isinstance(loop, LoopRegion) and loop.loop_variable):
                    continue
                parent = loop.parent_graph
                if parent is None or loop not in parent.nodes():
                    continue
                refusal = None if refusals is None else refusals.get(loop)
                if (
                    self.version_carried_distance(loop, owner, refusal)
                    or self.version_segment_writes(loop, sd, owner, refusal, checks)
                    or self.version_invariant_branch(loop, owner, refusal)
                ):
                    facts = ScopeFacts(sd)
                    specialized += 1
                    continue
                if refusals is not None and (refusal is None or refusal.kind is not Refusal.NON_UNIQUE_WRITE):
                    continue
                # Cheap, read-only structural filter FIRST: only the symbolic-
                # stride injective-write shape is a candidate. Probing every loop
                # with LoopToMap (below) is both wasteful and unsafe on shapes it
                # cannot handle (e.g. 2-D recurrences), so gate on structure first.
                condition = self.symbolic_stride_condition(loop, sd, facts)
                if condition is None:
                    continue
                # Confirm the only blocker is the constrained condition: LoopToMap refuses it outright but accepts
                # with conflicts waived on the strided targets alone; any other write or read must prove itself.
                probe = LoopToMap()
                if refusal is None and ParallelizeLoops().accepts(loop, probe):
                    continue
                probe.contract_arrays = frozenset(self.strided_targets(loop, sd, facts))
                if not ParallelizeLoops().accepts(loop, probe):
                    continue

                # Specialize: the true branch is the loop lifted to a plain
                # (WCR-free, vectorizable) Map -- injective under ``condition`` --
                # and the else branch keeps the original sequential loop for when
                # the condition fails at runtime.
                def parallelize(par_loop: LoopRegion, par_region: ControlFlowRegion, own: SDFG) -> None:
                    ParallelizeLoops().parallelize_loop(own, par_loop, proven=True)

                specialize_loop_under_condition(loop, condition, parallelize, owner, assume=self.assume_constraint)
                facts = ScopeFacts(sd)
                specialized += 1
        return specialized or None

    def version_invariant_branch(self, loop: LoopRegion, owner: SDFG, refusal) -> bool:
        """Version ``loop`` on the condition of the one branch that stores the container it was refused on, when the
        loop does not change that condition: ``if (!cond) { the loop without the branch, parallel } else { loop }``
        (CloudSC's ``IF (LAERICESED .AND. JM == NCLDQI) ZVQX(NCLDQI) = ...`` in the column loop).

        :param refusal: Why :class:`ParallelizeLoops` kept ``loop`` a loop, or ``None`` when no record exists.
        :returns: Whether ``loop`` was versioned.
        """
        from dace.transformation.interstate.loop_to_map import LoopToMap, Refusal
        from dace.transformation.passes.parallelization.parallelize_loops import ParallelizeLoops

        if refusal is None:
            # Cheap structural filter first: a loop without a one-branch conditional is never a candidate
            if not any(
                isinstance(block, ConditionalBlock) and len(block.branches) == 1
                for block in loop.all_control_flow_blocks()
            ):
                return False
            probe = LoopToMap()
            if ParallelizeLoops().accepts(loop, probe):
                return False
            refusal = probe.last_refusal
        if refusal is None or refusal.kind is not Refusal.NON_UNIQUE_WRITE or not refusal.container:
            return False
        found = invariant_branch_storing(loop, refusal.container)
        if found is None:
            return False
        position, condition = found
        probe = LoopToMap()
        probe.contract_arrays = probe.contract_reads = frozenset({refusal.container})
        if not ParallelizeLoops().accepts(loop, probe):
            return False

        def parallelize(par_loop: LoopRegion, par_region: ControlFlowRegion, own: SDFG) -> None:
            # The clone runs only where the branch does not: it stores nothing there
            drop_block(list(par_loop.all_control_flow_blocks())[position])
            ParallelizeLoops().parallelize_loop(own, par_loop, proven=True)

        specialize_loop_under_condition(loop, f"not ({condition})", parallelize, owner)
        return True

    def version_carried_distance(self, loop: LoopRegion, owner: SDFG, refusal) -> bool:
        """Version ``loop`` on its :class:`CarriedDistance`, the test evaluated once before the outermost loop that
        leaves its operands unchanged. A fallback an earlier guard pinned is a candidate too: the guard it failed
        (``k <= 0`` for a snapshot) says nothing about this one (``k >= N``).

        :param refusal: Why :class:`ParallelizeLoops` kept ``loop`` a loop, or ``None`` when no record exists.
        :returns: Whether ``loop`` was versioned.
        """
        from dace.transformation.interstate.loop_to_map import LoopToMap, Refusal
        from dace.transformation.passes.parallelization.parallelize_loops import ParallelizeLoops

        if refusal is not None and refusal.kind not in (Refusal.CARRIED_DEPENDENCE, Refusal.PINNED):
            return False
        distance = carried_distance(loop)
        if distance is None:
            return False
        pinned = loop.pinned_sequential
        if refusal is None and not pinned and ParallelizeLoops().accepts(loop, LoopToMap()):
            return False
        # The write is one-to-one and the check rules out the read meeting another iteration's write
        probe = LoopToMap()
        probe.contract_arrays = probe.contract_reads = frozenset({distance.array})
        loop.pinned_sequential = False
        try:
            accepted = ParallelizeLoops().accepts(loop, probe)
        finally:
            loop.pinned_sequential = pinned
        if not accepted:
            return False

        def parallelize(par_loop: LoopRegion, par_region: ControlFlowRegion, own: SDFG) -> None:
            ParallelizeLoops().parallelize_loop(own, par_loop, proven=True)

        host = None if self.assume_constraint else invariant_host(loop, distance.condition)
        flag = None if host is None else evaluate_before(host, f"{distance.array}_independent", distance.condition)
        # The class line an earlier guard gave this fallback; both arms are labelled anew below
        loop.specialization_hint = None
        specialize_loop_under_condition(
            loop,
            flag or distance.condition,
            parallelize,
            owner,
            assume=self.assume_constraint,
            labels=distance.labels(flag, host),
        )
        return True

    def version_segment_writes(
        self, loop: LoopRegion, sdfg: SDFG, owner: SDFG, refusal, checks: dict[tuple[int, str, str], str]
    ) -> bool:
        """Version a loop whose iteration ``n`` writes only ``P[n] .. P[n + 1]`` of its targets (a CSR row's segment)
        on ``P`` being non-decreasing, which makes the segments disjoint. The check is one pass over ``P``, placed
        before the outermost loop that writes neither ``P`` nor the loop's bounds, and shared by every loop
        segmented by the same stretch of ``P``.

        :param checks: The checks emitted so far, keyed by host block, pointer and range.
        :returns: Whether ``loop`` was versioned.
        """
        from dace.transformation.interstate.loop_to_map import LoopToMap, Refusal
        from dace.transformation.passes.parallelization.parallelize_loops import ParallelizeLoops

        if refusal is not None and refusal.kind not in (Refusal.NON_UNIQUE_WRITE, Refusal.OVERLAPPING_WRITES):
            return False
        segments = segment_write(loop, sdfg)
        if segments is None:
            return False
        pointer, targets = segments
        first, last = loop_analysis.get_init_assignment(loop), loop_analysis.get_loop_end(loop)
        if refusal is None and ParallelizeLoops().accepts(loop, LoopToMap()):
            return False
        probe = LoopToMap()
        probe.contract_arrays = frozenset(targets)
        if not ParallelizeLoops().accepts(loop, probe):
            return False
        low, high = symbolic.symstr(first), symbolic.symstr(last + 1)
        host = pointer_check_host(loop, pointer, OrderedSet(symbolic.free_symbol_names([first, last])))
        key = (id(host), pointer, f"{low}:{high}")
        if key not in checks:
            checks[key] = nondecreasing_check(sdfg, host, pointer, low, high)
        count = checks[key]

        def parallelize(par_loop: LoopRegion, par_region: ControlFlowRegion, own: SDFG) -> None:
            ParallelizeLoops().parallelize_loop(own, par_loop, proven=True)

        where = "" if host is loop else f", checked once before the {host.loop_variable} loop"
        written = ", ".join(sorted(targets))
        labels = (
            f"parallel -- the runtime check found {pointer} non-decreasing ({count} == 0{where}), so the ranges "
            f"{pointer}[n] .. {pointer}[n + 1] of different iterations n are disjoint: each iteration writes its own "
            f"elements of {written}",
            f"sequential -- the runtime check found {pointer} decreasing ({count} > 0{where}): two iterations' ranges "
            f"of {written} may overlap and the later write must win, so this arm keeps the order",
        )
        specialize_loop_under_condition(
            loop, f"{count} == 0", parallelize, owner, assume=self.assume_constraint, labels=labels
        )
        return True

    def symbolic_stride_condition(self, loop: LoopRegion, sdfg: SDFG, facts: ScopeFacts | None = None) -> str | None:
        """The parallel-validity condition for a symbolic-stride **injective write**.

        Targets a non-transient array written at a subset with a *symbolic*
        (non-numeric) coefficient ``c`` on the loop variable, where that write is
        injective -- hence the loop data-parallel -- iff ``c != 0``. Two write
        shapes qualify:

        * the in-place read-modify-write ``a[c * i + d] = a[c * i + d] <op> ...``
          (TSVC ``s171``): the array is read and written at the *same* subset, and
        * the plain store ``dst[c * i + d] = f(i)`` (TSVC-2.5
          ``ext_strided_store_ssym``): the array is written but not read at these
          positions.

        Non-aliasing recurrences (``aa[c*i] = aa[c*i - 1] ...`` -- the array read
        at a subset *other* than the one written) are excluded: a ``c != 0``
        condition does not make a loop-carried recurrence data-parallel. Returns
        ``"(c) != 0"`` when exactly one such coefficient governs the body, else
        ``None``.

        :param loop: The candidate loop region.
        :param sdfg: The SDFG owning ``loop``'s arrays.
        :param facts: Cached whole-SDFG facts of an unmutated ``sdfg``; ``None`` builds fresh ones.
        :returns: The parallel-validity condition string, or ``None``.
        """
        found = self.stride_coefficients(loop, sdfg, facts)
        if found is None or len(set(found.values())) != 1:
            return None
        return f"({next(iter(found.values()))}) != 0"

    def strided_targets(self, loop: LoopRegion, sdfg: SDFG, facts: ScopeFacts | None = None) -> set[str]:
        """The arrays :meth:`symbolic_stride_condition` proves injective: the only ones whose conflicts it answers for."""
        return set(self.stride_coefficients(loop, sdfg, facts) or ())

    def stride_coefficients(
        self, loop: LoopRegion, sdfg: SDFG, facts: ScopeFacts | None = None
    ) -> dict[str, str] | None:
        """Array -> symbolic coefficient of every injective symbolic-stride write of ``loop``, keyed by SPELLING: two
        writes can carry the same coefficient as two symbol instances. ``None`` when a write fails closed."""
        from dace.transformation.interstate.loop_to_map import loop_varying_symbols  # avoid an import cycle

        if facts is None:
            facts = ScopeFacts(sdfg)
        loop_var = symbolic.pystr_to_symbolic(loop.loop_variable)
        # (array, subset-string) pairs read from / written to non-transient arrays.
        # Resolve descriptors against each state's OWN SDFG -- ``loop.states()``
        # may descend into nested SDFGs whose data names are absent from ``sdfg``.
        reads: set = set()
        writes: dict = {}
        for state in loop.states():
            arrays = state.sdfg.arrays
            for e in state.edges():
                if e.data is None or e.data.is_empty() or e.data.subset is None:
                    continue
                src_desc = arrays.get(e.src.data) if isinstance(e.src, nodes.AccessNode) else None
                if src_desc is not None and not src_desc.transient:
                    reads.add((e.src.data, str(e.data.subset)))
                dst_desc = arrays.get(e.dst.data) if isinstance(e.dst, nodes.AccessNode) else None
                if dst_desc is not None and not dst_desc.transient:
                    writes[(e.dst.data, str(e.data.subset))] = e.data.subset
        reads_by_arr: dict = {}
        for rarr, rkey in reads:
            reads_by_arr.setdefault(rarr, set()).add(rkey)
        # Symbols bound outside the loop -- a subset built only from these is genuinely
        # loop-invariant, and anything else is a relation the matcher failed to resolve.
        # Names, not symbol objects: ``SDFG.free_symbols`` is a set of strings. Built on first need.
        coeffs: dict[str, str] = {}
        varying = loop_varying_symbols(loop)
        for (arr, key), subset in writes.items():
            # Admit an injective write: either a same-subset read-modify-write
            # (``key`` is among the array's read subsets) or a plain store (the
            # array is not read at all here). Refuse when the array is read at a
            # subset OTHER than the one written -- that is a loop-carried
            # recurrence (``aa[c*i] = aa[c*i - 1]``) which a ``c != 0`` guard
            # cannot parallelize.
            if reads_by_arr.get(arr, set()) - {key}:
                continue
            for rb, re_, _ in subset.ndrange():
                if rb != re_:
                    continue
                # ``in free_symbols`` and ``coeff`` go through symbol IDENTITY, and the subset's
                # instances come from a different source than the reparsed loop variable -- one
                # name, several instances, and both operations then quietly answer "not there".
                expr, lvar = symbolic.equalize_symbols_across(symbolic.pystr_to_symbolic(rb), loop_var)
                if lvar not in expr.free_symbols:
                    # The loop variable may be hidden behind a promoted index symbol
                    # (``a[__sym_i_times_inc]``), which SymbolPropagation binds but deliberately
                    # does not substitute into the graph. Resolve the binding for the QUERY --
                    # but only for a subset that actually carries an unexplained symbol, since
                    # collecting the bindings walks every interstate edge.
                    if not {str(sym) for sym in expr.free_symbols} - facts.invariant_names():
                        continue
                    expr, lvar = symbolic.equalize_symbols_across(
                        resolve_bindings(expr, sdfg, bindings=facts.binding_table()), loop_var
                    )
                if lvar not in expr.free_symbols:
                    # Fail closed. A write whose index still carries a symbol this pass cannot
                    # relate to the loop variable is NOT known to be loop-invariant -- it may well
                    # be a data-dependent scatter -- and ``coeff() == 0`` must not be read as one.
                    if {str(sym) for sym in expr.free_symbols} - facts.invariant_names():
                        return None
                    continue
                # Injective for ``c != 0`` only when affine in the loop variable: ``c*i + i**2`` is not.
                affine = symbolic.affine_coefficients(as_expr(expr), as_expr(lvar))
                if affine is None:
                    return None
                coeff = affine[0]
                # Symbolic (not provably numeric) nonzero coefficient only.
                if coeff == 0 or not required(coeff).free_symbols:
                    continue
                # ``c != 0`` keeps ``c*i + d`` one-to-one only while ``d`` holds still: ``a[c*i + off[i]]`` and
                # ``a[c*i + j]`` over an inner ``j`` collide at ``c == 1``.
                if symbolic.free_symbol_names([affine[1]]).keys() & varying:
                    return None
                # The guard runs before the loop, so ``c`` must hold still in it; and a second subset of the array can
                # land on this one under ``c != 0``: ``a[c*i]`` and ``a[c*i + 1]`` collide at ``c == 1``.
                if symbolic.free_symbol_names([coeff]).keys() & varying or sum(a == arr for a, _ in writes) > 1:
                    return None
                coeffs.setdefault(arr, symbolic.symstr(coeff))
        return coeffs


@dataclass(frozen=True, slots=True)
class CarriedDistance:
    """The one carried dependence of a loop: iteration ``n`` reads the element of ``array`` that iteration
    ``source`` writes, ``distance`` apart in the index. The loop is parallel exactly when ``distance`` is 0 or
    reaches ``limit``, past every other iteration."""

    array: str
    distance: str
    limit: str
    source: str

    @property
    def condition(self) -> str:
        d, lim = self.distance, self.limit
        return f"(({d}) == 0) or (({d}) >= ({lim})) or (({d}) <= -({lim}))"

    def labels(self, flag: str | None, host: LoopRegion | None) -> tuple[str, str]:
        """The class lines of the parallel arm and of the fallback arm; ``flag`` holds the check, evaluated before
        the ``host`` loop."""
        d, lim = self.distance, self.limit if self.limit.isidentifier() else f"({self.limit})"
        check = "the runtime check"
        if flag is not None and host is not None:
            check = f"the runtime check {flag} (evaluated once, before the {host.loop_variable} loop)"
        return (
            f"parallel -- {check} found {d} == 0 or {d} >= {lim} or {d} <= -{lim}: no iteration reads an element "
            f"of {self.array} another one writes",
            f"sequential -- {check} found {d} != 0 and -{lim} < {d} < {lim}: iteration n reads the element of "
            f"{self.array} that iteration {self.source} writes; this arm keeps the order",
        )


def non_transient_accesses(loop: LoopRegion) -> tuple[dict, dict]:
    """``(writes, reads)``: per non-transient array, its subsets in ``loop`` keyed by spelling."""
    writes: dict[str, dict[str, subsets.Range]] = {}
    reads: dict[str, dict[str, subsets.Range]] = {}
    for state in loop.states():
        for e in state.edges():
            if e.data.is_empty():
                continue
            for node, table, subset in (
                (e.src, reads, e.data.get_src_subset(e, state)),
                (e.dst, writes, e.data.get_dst_subset(e, state)),
            ):
                if isinstance(node, nodes.AccessNode) and subset is not None:
                    if not state.sdfg.arrays[node.data].transient:
                        table.setdefault(node.data, {})[str(subset)] = subset
    return writes, reads


def read_in_control_flow(loop: LoopRegion, array: str) -> bool:
    """Whether an interstate edge or a loop or branch condition inside ``loop`` reads ``array``."""
    arrays = loop.sdfg.arrays
    if any(m.data == array for e in loop.all_interstate_edges() for m in e.data.get_read_memlets(arrays)):
        return True
    return any(
        m.data == array
        for block in loop.all_control_flow_blocks()
        if isinstance(block, (LoopRegion, ConditionalBlock))
        for m in block.get_meta_read_memlets()
    )


def carried_distance(loop: LoopRegion) -> CarriedDistance | None:
    """The :class:`CarriedDistance` of a loop writing one array at one point ``a[c*i + d1]`` and reading it at one
    other point ``a[c*i + d2]``, with ``c`` a nonzero number and ``d2 - d1`` symbolic and fixed before the loop
    (``s[i + k] += s[i]``); ``None`` for any other shape. The other dimensions of the two points must match."""
    from dace.transformation.interstate.loop_to_map import loop_varying_symbols  # avoid an import cycle

    first, last = loop_analysis.get_init_assignment(loop), loop_analysis.get_loop_end(loop)
    stride = loop_analysis.get_loop_stride(loop)
    if first is None or last is None or stride is None or not (stride.is_number and stride > 0):
        return None
    writes, reads = non_transient_accesses(loop)
    pairs = []
    for array, written in writes.items():
        other = [subset for key, subset in reads.get(array, {}).items() if key not in written]
        if len(written) == 1 and len(other) == 1:
            pairs.append((array, next(iter(written.values())), other[0]))
    if len(pairs) != 1:
        return None
    array, write, read = pairs[0]
    if write.dims() != read.dims() or read_in_control_flow(loop, array):
        return None
    differ = [dim for dim in range(write.dims()) if str(write[dim]) != str(read[dim])]
    if len(differ) != 1:
        return None
    (wb, we, wstep), (rb, re_, rstep) = write[differ[0]], read[differ[0]]
    if wb != we or rb != re_:
        return None
    wexpr, rexpr, var = symbolic.equalize_symbols_across(
        symbolic.pystr_to_symbolic(wb), symbolic.pystr_to_symbolic(rb), symbolic.pystr_to_symbolic(loop.loop_variable)
    )
    affine = symbolic.affine_coefficients(wexpr, var)
    if affine is None or not affine[0].is_number or affine[0] == 0:
        return None
    distance = symbolic.simplify(rexpr - wexpr)
    varying = loop_varying_symbols(loop) | {loop.loop_variable}
    names = symbolic.free_symbol_names([distance]).keys()
    # A numeric distance is the static test's to decide; a varying one is no single runtime test
    if not names or names & varying or symbolic.free_symbol_names([affine[1]]).keys() & varying:
        return None
    # Two iterations at most ``limit - 1`` apart in the index: a longer distance never reaches another iteration
    limit = symbolic.simplify(abs(affine[0]) * (last - first) + 1)
    shown = -distance if distance.could_extract_minus_sign() else distance
    step = symbolic.simplify(distance / affine[0])
    source = f"n - {symbolic.symstr(-step)}" if step.could_extract_minus_sign() else f"n + {symbolic.symstr(step)}"
    return CarriedDistance(array, symbolic.symstr(shown), symbolic.symstr(limit), source)


def segment_write(loop: LoopRegion, sdfg: SDFG) -> tuple[str, OrderedSet[str]] | None:
    """``(P, targets)`` when every write of ``loop`` to the ``targets`` (the arrays it writes at a subset that moves)
    lies in ``P[n] .. P[n + 1]`` for iteration ``n`` in one dimension, and the targets are read only where written;
    ``None`` for any other loop. The loop's start must be numeric or fixed and its stride 1."""
    from dace.transformation.passes.canonicalize.segment_confined_loops import SegmentLoop  # avoid an import cycle

    if loop_analysis.get_loop_stride(loop) != 1 or loop_analysis.get_loop_end(loop) is None:
        return None
    if loop_analysis.get_init_assignment(loop) is None:
        return None
    analysis = SegmentLoop(sdfg, loop)
    writes, reads = non_transient_accesses(loop)
    pointers: OrderedSet[str] = OrderedSet()
    targets: OrderedSet[str] = OrderedSet()
    for array, written in writes.items():
        bounds = [segment_bounds(subset, analysis) for subset in written.values()]
        if not any(bounds):
            continue
        if not all(bounds) or set(reads.get(array, {})) - set(written):
            return None
        pointers.update(bounds)
        targets.add(array)
    if len(pointers) != 1 or next(iter(pointers)) in writes:
        return None
    return next(iter(pointers)), targets


def segment_bounds(subset: subsets.Range, analysis) -> str | None:
    """The pointer ``P`` when a dimension of ``subset`` spans ``lo .. hi - 1`` with ``lo`` bound to ``P[n]`` and ``hi``
    to ``P[n + 1]`` for iteration ``n`` of ``analysis``'s loop."""
    for begin, end, step in subset.ndrange():
        low = analysis.pointer_binding(str(begin))
        top = analysis.pointer_binding(str(symbolic.simplify(end + 1)))
        if low is not None and top is not None and low[0] == top[0] and (low[1], top[1]) == (0, 1):
            return low[0]
    return None


def pointer_check_host(loop: LoopRegion, pointer: str, bounds: OrderedSet[str]) -> LoopRegion:
    """The outermost loop around ``loop`` (or ``loop`` itself) whose iterations write neither ``pointer`` nor a
    symbol of ``bounds``: a check of ``pointer`` before it holds for every run of ``loop`` inside."""
    from dace.transformation.interstate.loop_to_map import loop_varying_symbols  # avoid an import cycle

    host = loop
    region = loop.parent_graph
    while isinstance(region, LoopRegion) and region.loop_variable:
        if pointer in loop_analysis.written_names(region):
            break
        if any(name in bounds for name in (*loop_varying_symbols(region), region.loop_variable)):
            break
        host = region
        region = region.parent_graph
    return host


def nondecreasing_check(sdfg: SDFG, before: LoopRegion, pointer: str, low: str, high: str) -> str:
    """Count, in one parallel pass right before ``before``, the positions ``c`` in ``[low, high)`` where
    ``pointer[c + 1] < pointer[c]``; returns the new symbol holding the count."""
    count = sdfg.find_new_symbol(f"{pointer}_descents")
    sdfg.add_symbol(count, dtypes.int64)
    total, total_desc = sdfg.add_array(f"{count}_total", [1], dtypes.int64, transient=True, find_new_name=True)
    region = before.parent_graph
    scan = region.add_state_before(
        before, f"{count}_check", is_start_block=region.start_block is before, assignments={count: f"{total}[0]"}
    )
    zero = region.add_state_before(scan, f"{count}_zero", is_start_block=region.start_block is scan)
    init = zero.add_tasklet(f"{count}_zero", {}, {"out"}, "out = 0")
    zero.add_edge(init, "out", zero.add_write(total), None, Memlet(f"{total}[0]"))
    index = f"{count}_c"
    entry, exit_ = scan.add_map(f"{count}_scan", {index: f"{low}:{high}"})
    entry.specialization_hint = (
        f"runtime check: one pass over {pointer}[{low}:{high} + 1] (O(n)) counts the positions where {pointer} "
        f"decreases into {count}; the loops after it branch on {count}"
    )
    step = scan.add_tasklet(f"{count}_step", {"lo", "hi"}, {"out"}, "out = 1 if hi < lo else 0")
    source = scan.add_read(pointer)
    scan.add_memlet_path(source, entry, step, dst_conn="lo", memlet=Memlet(f"{pointer}[{index}]"))
    scan.add_memlet_path(source, entry, step, dst_conn="hi", memlet=Memlet(f"{pointer}[{index} + 1]"))
    scan.add_memlet_path(
        step, exit_, scan.add_write(total), src_conn="out", memlet=Memlet(f"{total}[0]", wcr="lambda a, b: a + b")
    )
    return count


__all__ = ["CarriedDistance", "ParallelizeUnderConstraint", "carried_distance", "segment_write"]


def invariant_branch_storing(loop: LoopRegion, container: str) -> tuple[int, str] | None:
    """The position in ``loop.all_control_flow_blocks()`` of the one-branch conditional of ``loop`` that holds every
    store to ``container`` in the loop, and its condition; ``None`` unless the loop leaves that condition unchanged:
    no symbol it reads varies in the loop and no container it reads is written there."""
    from dace.transformation.interstate.loop_to_map import loop_varying_symbols

    blocks = list(loop.all_control_flow_blocks())
    holders = [
        block
        for block in blocks
        if isinstance(block, SDFGState)
        and any(dn.data == container and block.in_degree(dn) > 0 for dn in block.data_nodes())
    ]
    if not holders:
        return None
    guards = set()
    for state in holders:
        region = state.parent_graph
        while region is not loop and not isinstance(region, ConditionalBlock):
            region = region.parent_graph
        if region is loop:
            return None
        guards.add(region)
    if len(guards) != 1:
        return None
    guard = guards.pop()
    if len(guard.branches) != 1 or guard.branches[0][0] is None:
        return None
    condition = guard.branches[0][0]
    names = {str(name) for name in condition.get_free_symbols()}
    written = {
        dn.data for block in blocks if isinstance(block, SDFGState) for dn in block.data_nodes() if block.in_degree(dn)
    }
    if names & (loop_varying_symbols(loop) | {loop.loop_variable} | written):
        return None
    return blocks.index(guard), condition.as_string


def drop_block(block) -> None:
    """Replace ``block`` with an empty state on the same edges."""
    graph = block.parent_graph
    state = graph.add_state(f"{block.label}_dropped", is_start_block=graph.start_block is block)
    for edge in graph.in_edges(block):
        graph.add_edge(edge.src, state, edge.data)
    for edge in graph.out_edges(block):
        graph.add_edge(state, edge.dst, edge.data)
    graph.remove_node(block)
