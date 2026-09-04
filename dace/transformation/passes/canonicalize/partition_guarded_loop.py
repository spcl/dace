# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Partition the iteration space of a loop whose carried dependence is GUARDED by a predicate.

The shape, straight out of the MPR-PPG figure::

    for i in range(1, N - 1):
        if i * i < N:
            A[i] = A[i - 1] + B[i]      # S1 -- loop-carried RAW on A
        else:
            A[i] = C[i] + B[i]          # S2 -- no carried dependence

``S1`` is the only carrier and it runs only where ``i * i < N`` holds. That predicate FALLS in
``i`` -- once false it never becomes true again -- so the iteration space is a prefix on which the
dependence may exist and a suffix on which it provably does not. Cutting there leaves a sequential
``LoopRegion`` followed by one ``LoopToMap`` lifts::

    _mpr_split_0 = 1
    for (; _mpr_split_0 <= N - 2 and _mpr_split_0 ** 2 < N; _mpr_split_0 += 1);  # find the cut
    for i = 1; i < _mpr_split_0; i += 1:        # sequential prefix, body unchanged
        if i * i < N: A[i] = A[i - 1] + B[i]
        else:         A[i] = C[i] + B[i]
    for i = _mpr_split_0; i < N - 1; i += 1:    # parallel suffix, else branch only
        A[i] = C[i] + B[i]

**Why the cut is scanned for and not solved for.** The exact cut for ``i * i < N`` is
``ceil(sqrt(N)) - 1``, which is neither affine nor expressible with ``int_floor`` / ``int_ceil``,
and a floating-point ``sqrt`` folded to an integer is off by one at the squares. Walking the
predicate is EXACT for any monotone predicate and costs one scalar test per prefix iteration, next
to a prefix iteration of the real body. That is what makes the non-affine case reachable at all:
:class:`~dace.transformation.passes.parallelization_prep.BestEffortLoopPeeling` already index-set-
splits the affine ``if i < k`` guard, where the cut IS a closed form (``_RANGE_GUARD_BOUNDARY``),
and stops exactly where a closed form stops existing.

**What must be proven, and what must not.** Regrouping ``[lo, hi]`` as ``[lo, t - 1] + [t, hi]``
runs the same iterations of the same body in the same order whatever ``t`` is, so the split itself
needs no dependence analysis. The one claim needing proof is that the suffix may DROP the guarded
branch, i.e. that the predicate is false on every iteration the suffix runs -- which is exactly
:func:`provably_falling` plus the scan's own exit test. Everything else this pass refuses it
refuses because the partition would not be worth making, not because the rewrite would be wrong;
most importantly the suffix is probed with ``LoopToMap.can_be_applied_to`` and the split is thrown
away unless the suffix really is DOALL, so a loop whose else branch carries a dependence of its own
is left exactly as it was.
"""
import copy
from typing import Any, Dict, List, Optional, Set, Tuple

import sympy

from dace import InterstateEdge, SDFG, properties, symbolic
from dace.properties import CodeBlock
from dace.sdfg.state import ConditionalBlock, ControlFlowRegion, LoopRegion
from dace.transformation import helpers as xfh
from dace.transformation import pass_pipeline as ppl
from dace.transformation import transformation as xf
from dace.transformation.passes.analysis import loop_analysis

#: Prefix of the symbol the scan loop leaves the partition point in.
SPLIT_PREFIX = '_mpr_split_'

#: Name of the fresh non-negative offset the monotonicity proof shifts the loop variable by, so
#: ``i >= start`` reaches sympy's assumption engine as ``i = start + offset, offset >= 0``.
SHIFT_NAME = '_mpr_shift'

#: Relations a monotone predicate may be spelled with, grouped by which way their left-minus-right
#: difference has to move for the predicate to fall. ``==`` / ``!=`` are absent because neither is
#: monotone in the loop variable, and a conjunction is absent because its pieces can fall in
#: opposite directions.
FALLS_WHEN_RISING = (sympy.StrictLessThan, sympy.LessThan)
FALLS_WHEN_FALLING = (sympy.StrictGreaterThan, sympy.GreaterThan)


def loops_of(sdfg: SDFG) -> List[LoopRegion]:
    """Every ``LoopRegion`` under ``sdfg``, in a deterministic order.

    The order is what lets a candidate be re-found by INDEX inside a ``deepcopy`` of the SDFG,
    which is how the suffix is probed without touching the graph the caller handed in.
    """
    return [b for b in sdfg.all_control_flow_regions(recursive=True) if isinstance(b, LoopRegion)]


def guarded_body(loop: LoopRegion) -> Optional[Tuple[ConditionalBlock, CodeBlock, ControlFlowRegion]]:
    """``(conditional, guard, else region)`` for a body that is exactly one two-way conditional.

    Anything else is refused: a body with sibling blocks beside the conditional would have to be
    replicated into both segments with its own dependences re-argued, and a one-armed ``if`` leaves
    the suffix with an empty body, which is a dead-code question rather than a partitioning one.
    """
    blocks = loop.nodes()
    if len(blocks) != 1 or not isinstance(blocks[0], ConditionalBlock):
        return None
    conditional = blocks[0]
    if len(conditional.branches) != 2:
        return None
    guard, _ = conditional.branches[0]
    otherwise, else_region = conditional.branches[1]
    if guard is None or otherwise is not None:
        return None
    return conditional, guard, else_region


def predicate_of(guard: CodeBlock, loop_var: str, array_names: Set[str]) -> Optional[sympy.Rel]:
    """``guard`` as a single comparison over the loop variable and loop-invariant symbols.

    ``None`` for anything whose truth is not a function of ``loop_var`` alone: a guard reading an
    array element is a data-dependent predicate no compile-time monotonicity argument covers, and a
    guard not mentioning the loop variable partitions nothing.

    The names come from the CODE BLOCK, not from the parsed relation: ``D[i] < 0`` parses to a
    sympy ``Subscript(D, i)`` whose only free SYMBOL is ``i``, so an array test on the relation
    would read a data-dependent guard as loop-invariant.
    """
    try:
        relation = symbolic.pystr_to_symbolic(guard.as_string)
    except (SyntaxError, TypeError, ValueError):
        return None
    if not isinstance(relation, FALLS_WHEN_RISING + FALLS_WHEN_FALLING):
        return None
    names = guard.get_free_symbols()
    if loop_var not in names or names & array_names:
        return None
    return relation


def loop_variable_in(relation: sympy.Rel, loop_var: str) -> Optional[sympy.Symbol]:
    """The symbol OBJECT named ``loop_var`` inside ``relation``.

    Never ``symbolic.symbol(loop_var)``: a freshly minted symbol carries the default dtype and its
    own assumptions, and two same-named symbols that disagree on either do not cancel under
    ``subs`` -- the substitutions below would silently leave the loop variable in place.
    """
    return next((s for s in relation.free_symbols if str(s) == loop_var), None)


def provably_falling(relation: sympy.Rel, ivar: sympy.Symbol, start, stride) -> bool:
    """Whether ``relation`` is true on a PREFIX of the iteration space and false ever after.

    Written as ``delta REL 0``, the predicate falls exactly when ``delta`` moves the way the
    relation's direction forbids: for ``<`` / ``<=`` it must never decrease, for ``>`` / ``>=`` it
    must never increase. One step is enough because the iterations are visited in order -- a
    per-step sign over the whole range gives the sign over any span of it.

    The step is evaluated at ``i = start + offset`` with ``offset >= 0`` rather than at a bare
    ``i``, which is what carries the loop's own lower bound into the proof (``2 * i + 1 >= 0`` is
    not a fact about an integer, ``2 * (1 + offset) + 1 >= 0`` is). Undecided is ``False``: a
    predicate whose direction sympy cannot settle is refused, never guessed.
    """
    delta = symbolic.simplify(relation.lhs - relation.rhs)
    step = symbolic.simplify(delta.subs(ivar, ivar + stride) - delta)
    offset = sympy.Symbol(SHIFT_NAME, integer=True, nonnegative=True)
    step = symbolic.simplify(step.subs(ivar, start + offset))
    if isinstance(relation, FALLS_WHEN_RISING):
        return step.is_nonnegative is True
    return step.is_nonpositive is True


def provably_true_at(relation: sympy.Rel, ivar: sympy.Symbol, value) -> bool:
    """Whether ``relation`` is DECIDED true at ``ivar = value``. Undecided is ``False``."""
    return symbolic.simplify(relation.subs(ivar, value)) is sympy.true


def specialize_to_else(loop: LoopRegion) -> None:
    """Replace the loop's guarded conditional by its unconditional branch.

    Sound only where the guard is false on every iteration the loop runs, which for the suffix is
    exactly what the scan's exit test and :func:`provably_falling` together establish.
    """
    conditional, _, else_region = guarded_body(loop)
    xfh.move_branch_cfg_up_discard_conditions(conditional, else_region)


def split_at_guard(loop: LoopRegion, relation: sympy.Rel, ivar: sympy.Symbol, cut: str, tag: str) -> LoopRegion:
    """Rewrite ``loop`` into scan + sequential prefix + specialized parallel suffix; return the suffix.

    ``loop`` itself becomes the prefix -- kept rather than cloned, so its blocks keep their
    identity and every guid with it -- and the suffix is the clone. Both segments keep a canonical
    ``for`` shape (``i < cut`` and ``i = cut``), which is what every downstream bound reader needs;
    the predicate is walked in a separate iterator-less ``while`` whose only effect is ``cut``.
    """
    parent = loop.parent_graph
    var = loop.loop_variable
    start = loop_analysis.get_init_assignment(loop)
    end = loop_analysis.get_loop_end(loop)
    stride = symbolic.symstr(loop_analysis.get_loop_stride(loop))
    at_cut = symbolic.symstr(relation.subs(ivar, symbolic.pystr_to_symbolic(cut)))

    parent.add_state_before(loop,
                            f'{tag}_seed',
                            is_start_block=parent.start_block is loop,
                            assignments={cut: symbolic.symstr(start)})

    scan = LoopRegion(f'{tag}_scan',
                      condition_expr=f'({cut} <= ({symbolic.symstr(end)})) and ({at_cut})',
                      update_expr=f'{cut} = {cut} + ({stride})',
                      sdfg=loop.sdfg)
    scan.add_state(f'{tag}_scan_body', is_start_block=True)
    parent.add_node(scan)
    for edge in list(parent.in_edges(loop)):
        parent.add_edge(edge.src, scan, edge.data)
        parent.remove_edge(edge)
    parent.add_edge(scan, loop, InterstateEdge())

    suffix = copy.deepcopy(loop)
    suffix.label = tag
    parent.add_node(suffix)
    for edge in list(parent.out_edges(loop)):
        parent.add_edge(suffix, edge.dst, edge.data)
        parent.remove_edge(edge)
    parent.add_edge(loop, suffix, InterstateEdge())

    loop.loop_condition = CodeBlock(f'{var} < {cut}')
    suffix.init_statement = CodeBlock(f'{var} = {cut}')
    specialize_to_else(suffix)
    parent.reset_cfg_list()
    return suffix


@properties.make_properties
@xf.explicit_cf_compatible
class PartitionGuardedLoop(ppl.Pass):
    """Split a loop whose carried dependence is guarded by a monotone predicate.

    See the module docstring for the shape and for why the partition point is scanned for.
    """

    CATEGORY: str = 'Optimization Preparation'

    def modifies(self) -> ppl.Modifies:
        return ppl.Modifies.CFG | ppl.Modifies.Symbols

    def should_reapply(self, _modified: ppl.Modifies) -> bool:
        return False

    def depends_on(self):
        return set()

    def fresh_cut_symbol(self, sdfg: SDFG, loop: LoopRegion) -> Optional[str]:
        """A new SDFG symbol to hold the partition point, typed like the loop variable."""
        dtype = loop.new_symbols(sdfg.symbols).get(loop.loop_variable) or sdfg.symbols.get(loop.loop_variable)
        if dtype is None:
            return None
        index = 0
        while f'{SPLIT_PREFIX}{index}' in sdfg.symbols:
            index += 1
        name = f'{SPLIT_PREFIX}{index}'
        sdfg.add_symbol(name, dtype)
        return name

    def candidate(self, loop: LoopRegion) -> Optional[Tuple[sympy.Rel, sympy.Symbol]]:
        """``(predicate, loop-variable symbol)`` for a partitionable loop, else ``None``."""
        sdfg = loop.sdfg
        if sdfg is None or not loop.loop_variable or loop.inverted:
            return None
        if loop.has_break or loop.has_continue or loop.has_return:
            return None  # an early exit makes "which iterations ran" a runtime question
        bounds = loop.loop_condition.get_free_symbols()
        if loop.init_statement is not None:
            bounds |= loop.init_statement.get_free_symbols()
        if any(name.startswith(SPLIT_PREFIX) for name in bounds):
            return None  # already a segment this pass cut: re-cutting it only grows the graph
        start = loop_analysis.get_init_assignment(loop)
        end = loop_analysis.get_loop_end(loop)
        stride = loop_analysis.get_loop_stride(loop)
        if start is None or end is None or stride is None or symbolic.simplify(stride - 1) != 0:
            return None  # unit stride only: it is what makes ``end`` an iteration the loop attains
        body = guarded_body(loop)
        if body is None:
            return None
        relation = predicate_of(body[1], loop.loop_variable, set(sdfg.arrays))
        if relation is None:
            return None
        moved = {name for e in loop.all_interstate_edges() for name in e.data.assignments}
        if (body[1].get_free_symbols() - {loop.loop_variable}) & moved:
            return None  # the predicate's own symbols move inside the loop: not a function of i
        ivar = loop_variable_in(relation, loop.loop_variable)
        if ivar is None or not provably_falling(relation, ivar, start, stride):
            return None
        if provably_true_at(relation, ivar, end):
            return None  # true at the last iteration and falling: the whole space is sequential
        return relation, ivar

    def apply_pass(self, sdfg: SDFG, _pipeline_results: Dict[str, Any]) -> Optional[int]:
        """Partition every loop that passes :meth:`candidate` and whose suffix is provably DOALL.

        The suffix is built on a ``deepcopy`` and handed to ``LoopToMap.can_be_applied_to``; the
        SDFG the caller passed in is touched only once that answers yes, so a refusal leaves it
        bit-identical. Reusing ``LoopToMap``'s own dependence analysis is the point -- a second,
        weaker one written here is how a partition that "parallelizes" a loop that still carries a
        dependence gets shipped.
        """
        from dace.transformation.interstate.loop_to_map import LoopToMap
        from dace.transformation.passes.scalar_fission import PrivatizeScalars
        count = 0
        for loop in loops_of(sdfg):
            match = self.candidate(loop)
            if match is None:
                continue
            relation, ivar = match
            index = next(i for i, other in enumerate(loops_of(sdfg)) if other is loop)
            probe = copy.deepcopy(sdfg)
            probe_loop = loops_of(probe)[index]
            probe_cut = self.fresh_cut_symbol(probe, probe_loop)
            if probe_cut is None:
                continue
            probe_suffix = split_at_guard(probe_loop, relation, ivar, probe_cut, f'{probe_loop.label}_mpr')
            # Privatize the throwaway copy first: a per-iteration scalar temporary is a FALSE
            # write/write dependence that hides the answer to the only question being asked here,
            # and it is the same prep the pipeline runs ahead of its own LoopToMap.
            PrivatizeScalars().apply_pass(probe, {})
            if not LoopToMap.can_be_applied_to(probe_suffix.sdfg, loop=probe_suffix):
                continue  # the suffix still carries a dependence: the partition buys nothing
            cut = self.fresh_cut_symbol(sdfg, loop)
            if cut is None:
                continue
            split_at_guard(loop, relation, ivar, cut, f'{loop.label}_mpr')
            count += 1
        return count or None

    def report(self, pass_retval: int) -> str:
        return f'Partitioned {pass_retval} guarded loop(s) into a sequential prefix and a parallel suffix.'
