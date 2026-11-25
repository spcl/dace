# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Specialize a loop into ``if (condition) { parallel } else { sequential }``.

A loop data-parallel only under a runtime predicate (symbolic stride nonzero, wrap
offset below the modulus) is replaced by a two-way conditional: true branch = parallel
form, else = original sequential loop. Values satisfying the predicate take the fast
path; violating ones still compute correctly on the fallback. Contrast the abort-only
trap guard for whole-pipeline preconditions (symbol nonnegativity) with no sequential
alternative.

The parallel form comes from a caller-supplied callback mutating the true-branch clone
in place (lift to Map, split into affine segments, ...): specialization owns the
control-flow surgery, the callback owns what "parallel" means.
"""

import ast
import copy
from collections.abc import Callable, Sequence

import sympy

from dace import dtypes, symbolic
from dace.cpf_lowering import LoopClass, Reason
from dace.memlet import Memlet
from dace.properties import CodeBlock
from dace.sdfg import SDFG, nodes
from dace.sdfg.state import ConditionalBlock, ControlFlowBlock, ControlFlowRegion, LoopRegion, SDFGState

#: The class line of a parallel arm: the guard is the evidence the iterations are independent. ``{frequency}`` says how
#: often the test runs (:func:`test_frequency`).
GUARDED = (
    LoopClass.PARALLEL.prefix
    + Reason.GUARDED.value
    + "{guard}) ({frequency}){meaning}: when it holds, no iteration writes an element another iteration reads or writes"
)
#: The class line of the sequential arm a guard falls back to: the original loop, for the inputs the guard rejects.
#: The negation is simplified (``inc == 0``, never ``not (not ...)``, review #10), the arm says why it keeps the
#: order, not that a dependence was proven (review #1), and ``{effect}`` says what the loop then does
#: (:func:`fallback_effect`), empty when that is not known.
GUARD_FALLBACK = (
    LoopClass.SEQUENTIAL.prefix
    + Reason.FALLBACK.value
    + "{guard}), taken when {negated}{meaning}{effect}: two iterations may then touch the same "
    "element, so "
    "this arm keeps the order"
)

#: The class line of a fallback arm whose loop then accumulates into one floating-point element: a reduction, whose
#: order the arm keeps only for its rounding. ``{effect}`` as in :data:`GUARD_FALLBACK`.
GUARD_FALLBACK_REDUCTION = (
    LoopClass.SEQUENTIAL.prefix
    + Reason.FALLBACK.value
    + "{guard}), taken when {negated}{meaning}{effect}: a reduction, kept in order here because the "
    "order "
    "of the {noun} fixes its floating-point rounding; parallel with reduction({op}:{element}) if a different rounding "
    "is acceptable"
)

RELATION_OPERATORS = {"==", "!=", "<", "<=", ">", ">="}


def readable(expr: sympy.Basic) -> str:
    """``expr`` as a reader writes a test: ``inc != 0``, ``N > 0 and M > 0``, without a parenthesis around an operand."""
    if isinstance(expr, sympy.core.relational.Relational) and expr.rel_op in RELATION_OPERATORS:
        return f"{expr.lhs} {expr.rel_op} {expr.rhs}"
    if isinstance(expr, (sympy.And, sympy.Or)):
        joiner = " and " if isinstance(expr, sympy.And) else " or "
        return joiner.join(
            f"({readable(a)})" if isinstance(a, (sympy.And, sympy.Or)) else readable(a) for a in expr.args
        )
    if isinstance(expr, sympy.Not):
        return f"not ({readable(expr.args[0])})"
    return str(expr)


def condition_text(condition: str) -> str:
    """``condition`` simplified for a reader (``not (a > 0)`` reads ``a <= 0``), or as given if it does not parse."""
    try:
        return readable(symbolic.pystr_to_symbolic(condition))
    except Exception:  # pylint: disable=broad-exception-caught
        # A comment must not fail the specialization that asked for it
        return condition


def label_arm(node, kind: str) -> None:
    """Put ``kind`` first in ``node``'s hint, keeping a note a pass left below it."""
    hint = (node.specialization_hint or "").strip()
    node.specialization_hint = f"{kind}\n{hint}" if hint else kind


def label_arms(
    par_region: ControlFlowRegion,
    seq_region: ControlFlowRegion,
    guard: str,
    labels: tuple[str, str] | None = None,
    meaning: str = "",
) -> None:
    """Name ``guard``, the condition the parallel arm runs under, on that arm's outermost maps and on the pinned
    loops of the sequential arm. ``labels``, when given, are the two arms' class lines in place of the generic
    :data:`GUARDED` and :data:`GUARD_FALLBACK`; ``meaning`` (``, where ...``) says what the guard's symbols are."""
    if labels is None:
        negated = condition_text(f"not ({guard})")
        loops = [r for r in seq_region.all_control_flow_regions() if isinstance(r, LoopRegion) and r.pinned_sequential]
        labels = (
            GUARDED.format(guard=condition_text(guard), frequency=test_frequency(par_region), meaning=meaning),
            fallback_line(guard, negated, loops[0] if loops else None, meaning),
        )
    for block in par_region.all_control_flow_blocks():
        if isinstance(block, SDFGState):
            for node in block.nodes():
                if isinstance(node, nodes.MapEntry) and block.entry_node(node) is None:
                    label_arm(node, labels[0])
    for region in seq_region.all_control_flow_regions():
        if isinstance(region, LoopRegion) and region.pinned_sequential:
            label_arm(region, labels[1])


def test_frequency(region: ControlFlowRegion) -> str:
    """How often the test of a conditional holding ``region`` runs: once per call, or once per iteration of the
    loop or map it sits in (through the nested SDFGs that hold it)."""
    block = region
    while block is not None:
        if isinstance(block, LoopRegion) and block.loop_variable:
            return f"one test per iteration of the {block.loop_variable} loop"
        if isinstance(block, SDFG):
            node = block.parent_nsdfg_node
            if node is None:
                break
            if block.parent.entry_node(node) is not None:
                return "one test per iteration of the enclosing map"
            block = block.parent
        else:
            block = block.parent_graph
    return "one test per call"


ACCUMULATIONS = {ast.Add: "a sum", ast.Sub: "a difference", ast.Mult: "a product"}
#: The reduction clause operator of each accumulation.
ACCUMULATION_OPERATORS = {"a sum": "+", "a difference": "-", "a product": "*"}


def fallback_line(guard: str, negated: str, loop: LoopRegion | None, meaning: str = "") -> str:
    """The class line of a guard's fallback arm: :data:`GUARD_FALLBACK_REDUCTION` when ``loop`` then accumulates into one
    floating-point element that no iteration indexes apart, else :data:`GUARD_FALLBACK`."""
    effect, kind, element = fallback_effect(loop, negated) if loop is not None else ("", "", None)
    if kind and element is not None:
        return GUARD_FALLBACK_REDUCTION.format(
            guard=condition_text(guard),
            negated=negated,
            meaning=meaning,
            effect=effect,
            noun=kind.split(" ", 1)[1],
            op=ACCUMULATION_OPERATORS[kind],
            element=element,
        )
    return GUARD_FALLBACK.format(guard=condition_text(guard), negated=negated, meaning=meaning, effect=effect)


def fallback_effect(loop: LoopRegion, negated: str) -> tuple[str, str, str | None]:
    """What ``loop`` does where ``negated`` holds, for the class line of a guard's fallback arm: `` (the loop then makes
    a sum into a[0])`` for ``a[i * inc] = a[i * inc] + b[i]`` under ``inc == 0``, or ``""`` when no single write says
    it; then the accumulation (``a sum``, ``""`` for none) and, when it goes to one floating-point element the loop's
    counter does not index, that element (``a[0]``), else ``None``."""
    try:
        facts = symbolic.pystr_to_symbolic(negated)
        equalities = {
            r.lhs: r.rhs for r in (facts.args if isinstance(facts, sympy.And) else (facts,)) if isinstance(r, sympy.Eq)
        }
    except Exception:  # pylint: disable=broad-exception-caught
        # A comment must not fail the specialization that asked for it
        return "", "", None
    for state in loop.states():
        for node in state.nodes():
            if not isinstance(node, nodes.Tasklet):
                continue
            written = [
                e for e in state.out_edges(node) if isinstance(e.dst, nodes.AccessNode) and not e.data.is_empty()
            ]
            if len(written) != 1 or state.sdfg.arrays[written[0].data.data].transient:
                continue
            target = written[0].data
            where = copy.deepcopy(target.subset)
            where.replace(equalities)
            # A symbol the body assigns from an index array reads as that read: ``b[idx[i]]``, not ``b[sym0]``
            where.replace(
                {
                    symbolic.symbol(name): symbolic.pystr_to_symbolic(value)
                    for name, value in body_assignments(loop).items()
                }
            )
            kind = accumulation(node, state, target)
            if not kind:
                return f" (the loop then writes {target.data}[{where}])", "", None
            point = where.num_elements() == 1 and loop.loop_variable not in {str(s) for s in where.free_symbols}
            floating = state.sdfg.arrays[target.data].dtype in (dtypes.float16, dtypes.float32, dtypes.float64)
            element = f"{target.data}[{where}]" if point and floating else None
            return f" (the loop then makes {kind} into {target.data}[{where}])", kind, element
    return "", "", None


def body_assignments(loop: LoopRegion) -> dict[str, str]:
    """The symbols the interstate edges inside ``loop`` assign, each to the one value it gets."""
    found: dict[str, list[str]] = {}
    for edge in loop.all_interstate_edges():
        for name, value in edge.data.assignments.items():
            found.setdefault(name, []).append(value)
    return {name: values[0] for name, values in found.items() if len(set(values)) == 1}


def accumulation(node: nodes.Tasklet, state: SDFGState, target: Memlet) -> str:
    """``a sum`` (or product, difference) if ``node`` adds to the element ``target`` names something else, else ``""``:
    one of its inputs reads that element and its one statement applies the operator to it."""
    reads = [e for e in state.in_edges(node) if e.data.data == target.data and str(e.data.subset) == str(target.subset)]
    body = node.code.code
    if not reads or len(body) != 1 or not isinstance(body[0], ast.Assign) or not isinstance(body[0].value, ast.BinOp):
        return ""
    return ACCUMULATIONS.get(type(body[0].value.op), "")


def _owner_sdfg(loop: LoopRegion) -> SDFG:
    """The top-level SDFG owning ``loop`` (walk up nested-SDFG parents)."""
    sd = loop.sdfg
    while sd.parent_sdfg is not None:
        sd = sd.parent_sdfg
    return sd


def version_block(
    block: ControlFlowBlock,
    label: str,
    first: tuple[str | CodeBlock, ControlFlowBlock, str],
    second: tuple[None, ControlFlowBlock, str],
) -> tuple[ConditionalBlock, ControlFlowRegion, ControlFlowRegion]:
    """Put a two-arm conditional in the place of ``block``, the one loop-versioning builder.

    :param block: The block to replace, removed from its region.
    :param label: The label of the conditional.
    :param first: ``(condition, version, label of its region)`` of the guarded arm.
    :param second: ``(None, version, label of its region)`` of the ``else`` arm.
    :returns: The conditional and the regions of its two arms, in that order.
    """
    parent = block.parent_graph
    in_edges = list(parent.in_edges(block))
    out_edges = list(parent.out_edges(block))
    conditional = ConditionalBlock(label=label)
    parent.add_node(conditional, is_start_block=parent.start_block is block, ensure_unique_name=True)
    for e in in_edges:
        parent.add_edge(e.src, conditional, copy.deepcopy(e.data))
    for e in out_edges:
        parent.add_edge(conditional, e.dst, copy.deepcopy(e.data))
    parent.remove_node(block)
    regions = []
    for condition, version, region_label in (first, second):
        region = ControlFlowRegion(label=region_label, sdfg=parent.sdfg)
        region.add_node(version, is_start_block=True, ensure_unique_name=True)
        conditional.add_branch(condition, region)
        regions.append(region)
    return conditional, *regions


def specialize_loop_under_condition(
    loop: LoopRegion,
    condition,
    parallelize: Callable[[LoopRegion, ControlFlowRegion, SDFG], None],
    owner_sdfg: SDFG | None = None,
    assume: bool = False,
    pin: bool = True,
    labels: tuple[str, str] | None = None,
) -> ConditionalBlock | None:
    """Replace ``loop`` with ``if (condition) { parallel } else { original loop }``.

    :param loop: Loop to specialize; removed from its parent graph, deep-copied into each branch.
    :param condition: Predicate (``CodeBlock`` or C/Python string) under which the parallel
        form is valid -- the true-branch guard.
    :param parallelize: Callback ``(par_loop, par_region, owner)`` mutating the true-branch clone
        into its parallel form (run ``LoopToMap``, split into affine segments, ...). Runs after
        the conditional is spliced in, so ``par_loop.parent_graph is par_region`` with SDFG links set.
    :param owner_sdfg: Top-level SDFG (derived from ``loop`` if omitted).
    :param assume: When ``True``, the caller *asserts* ``condition`` always holds at runtime.
        Skip the two-way conditional entirely and parallelize ``loop`` in place -- emit only the
        parallel form, with no sequential fallback. Unsound if the condition is ever violated
        (the caller owns that contract); returns ``None`` since no ``ConditionalBlock`` is built.
    :param pin: Pin the fallback sequential. A caller whose fallback the facts there (the failed ``condition``) keep
        it from specializing again, and which another pass may still parallelize, passes ``False``.
    :param labels: The class lines of the parallel and the fallback arm (see :func:`label_arms`).
    :returns: The inserted :class:`ConditionalBlock`, or ``None`` in ``assume`` mode.
    """
    if owner_sdfg is None:
        owner_sdfg = _owner_sdfg(loop)
    # assume mode: the condition is taken as always-true, so there is no fallback to build --
    # parallelize the original loop in its own parent graph and return (no ConditionalBlock).
    if assume:
        parallelize(loop, loop.parent_graph, owner_sdfg)
        return None
    if not isinstance(condition, CodeBlock):
        condition = CodeBlock(condition)

    # True branch: a clone the caller parallelizes. Else branch (condition ``None``):
    # a clone kept as sequential fallback. Deep-copy both (not re-host the original) to
    # keep re-parenting simple and leave no stale refs to the removed loop.
    par_loop = copy.deepcopy(loop)
    # The clone runs only under ``condition``, which is what ``parallelize`` proves the parallel form under: a pin
    # ``loop`` carries as an earlier specialization's fallback does not hold there (LoopToScan's ``K < 0`` fallback,
    # versioned again on ``0 <= K`` by BreakAntiDependence)
    par_loop.pinned_sequential = False
    seq_loop = copy.deepcopy(loop)
    # Pin the fallback sequential so no later parallelizer (LoopToMap / LoopToReduce)
    # lifts it back to a Map -- it exists for values violating ``condition`` and must
    # stay sequential.
    seq_loop.pinned_sequential = pin
    conditional, par_region, seq_region = version_block(
        loop,
        f"{loop.label}_specialize",
        (condition, par_loop, f"{loop.label}_par"),
        (None, seq_loop, f"{loop.label}_seq"),
    )

    # Both clones still name the loop's scratch transients: a transient accessed outside the loop is one
    # LoopToMap cannot privatize, so the fallback gets its own copies (only one branch ever runs)
    from dace.transformation.passes.parallelization.contracts.guarded_scatter import privatize_clone_local_transients

    privatize_clone_local_transients(owner_sdfg, par_loop, seq_loop)
    give_clone_its_own_iterators(owner_sdfg, conditional, seq_loop)
    parallelize(par_loop, par_region, owner_sdfg)
    label_arms(par_region, seq_region, condition.as_string.strip(), labels)
    return conditional


def invariant_host(block: ControlFlowBlock, condition: str) -> LoopRegion | None:
    """The outermost loop enclosing ``block`` (in its SDFG) that leaves every operand of ``condition`` unchanged,
    where a guard of ``block`` on ``condition`` can be evaluated once; ``None`` if there is none or ``condition``
    reads a container."""
    from dace.transformation.interstate.loop_to_map import loop_varying_symbols  # avoid an import cycle

    operands = CodeBlock(condition).get_free_symbols()
    if operands & block.sdfg.arrays.keys():
        return None
    host = None
    region = block.parent_graph
    while isinstance(region, LoopRegion) and region.loop_variable:
        if operands & (loop_varying_symbols(region) | {region.loop_variable}):
            break
        host = region
        region = region.parent_graph
    return host


def evaluate_before(host: ControlFlowBlock, name: str, condition: str) -> str:
    """Save ``condition`` into a new boolean symbol named after ``name`` on entry to ``host``; returns its name."""
    sdfg = host.sdfg
    flag = sdfg.find_new_symbol(name)
    sdfg.add_symbol(flag, dtypes.bool)
    region = host.parent_graph
    region.add_state_before(
        host, f"{flag}_check", is_start_block=region.start_block is host, assignments={flag: condition}
    )
    return flag


def give_clone_its_own_iterators(owner_sdfg: SDFG, conditional: ConditionalBlock, clone: LoopRegion) -> None:
    """Rename the iterators of ``clone`` (a fallback branch of ``conditional``) to fresh ``_loop_it_<N>`` names.

    The two branches of a specialization bind the same iterator, so the parallel branch's iterator looks read
    after its loop (in the fallback) and ``LoopToMap`` refuses it. An iterator something after ``conditional``
    still reads keeps its name, so that reader sees the value the branch that ran left.
    """
    from dace.transformation.passes.unique_loop_iterators import (
        LOOP_ITERATOR_PREFIX,
        first_free_id,
        rename_iterator,
        symbol_readers,
    )

    loops = [r for r in clone.all_control_flow_regions() if isinstance(r, LoopRegion) and r.loop_variable]
    inside = set(conditional.all_control_flow_blocks(recursive=True))
    readers = symbol_readers(owner_sdfg, {loop.loop_variable for loop in loops})
    next_id = first_free_id(owner_sdfg)
    for loop in reversed(loops):  # innermost first: a rename covers the loop's whole subtree
        old = loop.loop_variable
        if readers.get(old, set()) - inside:
            continue
        rename_iterator(loop, old, f"{LOOP_ITERATOR_PREFIX}{next_id}")
        next_id += 1


def specialize_loop_under_relations(
    loop: LoopRegion,
    relations: Sequence[symbolic.Relation],
    parallelize: Callable[[LoopRegion, ControlFlowRegion, SDFG], None],
    owner_sdfg: SDFG | None = None,
) -> ConditionalBlock | None:
    """Version ``loop`` on a sign or order fact the facts cannot prove: ``if relations: parallel else: original``.

    The true-branch clone declares ``relations`` as facts of its own scope before ``parallelize`` runs, so every
    proof about the clone or its body reads them, and none outside it (a pass hoisting code out of the branch asks
    there). This is a contract guard, the runtime test of exactly what the parallel form assumes.

    :param loop: Loop to version (see :func:`specialize_loop_under_condition`).
    :param relations: What the parallel form needs; their conjunction is the branch condition.
    :param parallelize: Callback mutating the true-branch clone into its parallel form.
    :param owner_sdfg: Top-level SDFG (derived from ``loop`` if omitted).
    :returns: The inserted :class:`ConditionalBlock`.
    """
    condition = " and ".join(
        f"(({symbolic.symstr(r.lhs)}) {'<' if r.kind is symbolic.RelationKind.LT else '<='} ({symbolic.symstr(r.rhs)}))"
        for r in relations
    )

    def declare_then_parallelize(par_loop: LoopRegion, par_region: ControlFlowRegion, owner: SDFG) -> None:
        repo = par_loop.sdfg.symbol_repo
        for relation in relations:
            # A fact can only name declared symbols; the branch condition still guards the clone
            if symbolic.relation_names(relation) <= repo.params.types.keys():
                repo.add_relation(relation, at=par_loop)
        parallelize(par_loop, par_region, owner)

    return specialize_loop_under_condition(loop, condition, declare_then_parallelize, owner_sdfg)


__all__ = [
    "GUARDED",
    "GUARD_FALLBACK",
    "GUARD_FALLBACK_REDUCTION",
    "evaluate_before",
    "invariant_host",
    "label_arms",
    "specialize_loop_under_condition",
    "specialize_loop_under_relations",
    "version_block",
]
