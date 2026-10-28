# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Remove a loop guard implied by the per-iteration guards inside the loop."""

import re
from typing import Any, NamedTuple

import sympy

from dace import SDFG, properties, subsets, symbolic
from dace.sdfg import nodes
from dace.sdfg.state import ConditionalBlock, ControlFlowBlock, ControlFlowRegion, LoopRegion, SDFGState
from dace.transformation import pass_pipeline as ppl
from dace.transformation import transformation
from dace.transformation.helpers import move_branch_cfg_up_discard_conditions
from dace.transformation.passes.analysis import loop_analysis

LOOP_INDEX = "__i"
SUBSCRIPT = re.compile(r"(\w+)\[([^\[\]]*)\]")


class Predicate(NamedTuple):
    """A per-iteration condition with the loop variable renamed to ``LOOP_INDEX``, and the elements it reads."""

    expression: str
    reads: frozenset[tuple[str, str]]


class Match(NamedTuple):
    guard: ConditionalBlock
    counter: str
    flag: str | None
    increments: list[ConditionalBlock]


@properties.make_properties
@transformation.explicit_cf_compatible
class RemoveRedundantLoopGuard(ppl.Pass):
    """Removes ``if c != 0`` around ``for i: if p(i): S(i)`` when ``c`` counts the iterations where ``p(i)`` holds.

    The outer guard is a hand-hoisted summary ``exists i: p(i)`` of the per-iteration guards, so it is implied by
    them: sinking it back is exact. The shape, as Fortran ``IF (c == 0) CYCLE`` reaches the SDFG after
    ``ContinueToCondition``::

        c = 0
        for i in R: ...; if p(i): c = c + k            (k > 0, the only writes of c)
        if c != 0: for i in R: ...; if p(i): S(i)      (every effect of S under p(i))

    Safe when:

    * ``c == 0`` iff no ``p(i)`` holds: ``c`` starts at 0, only grows (it would wrap only past 2**31 steps), and is
      read by nothing but the guard;
    * the guarded loop does nothing when no ``p(i)`` holds: outside the single-branch ``if p(i)`` it only computes
      transient scalars and symbols that nothing after it reads;
    * ``p(i)`` is the same expression of the same elements in both loops, the counting loop writes those elements
      only at the element it reads, and the guarded loop not at all;
    * both loops run over the same range, unchanged in between.

    ``p(i)`` may be the condition itself, a symbol assigned in the loop body, or a scalar computed by one tasklet in
    the loop body; the guard may test ``c`` directly or through a symbol. The loop then runs unguarded and with no
    reduction in between, so ``LoopToMap`` and ``MapCollapse`` can make the enclosing nest fully parallel (ICON's
    CFL ``clip_count``).
    """

    CATEGORY: str = "Simplification"

    def modifies(self) -> ppl.Modifies:
        return ppl.Modifies.CFG | ppl.Modifies.InterstateEdges

    def should_reapply(self, modified: ppl.Modifies) -> bool:
        return bool(modified & ppl.Modifies.CFG)

    def apply_pass(self, sdfg: SDFG, pipeline_results: dict[str, Any]) -> int | None:
        dropped = 0
        for sd in sdfg.all_sdfgs_recursive():
            # A drop copies the guarded loop up and deletes the increments, so the walk restarts after each one.
            while (match := next_match(sd)) is not None:
                drop(match)
                dropped += 1
        return dropped or None


def next_match(sdfg: SDFG) -> Match | None:
    guards = (b for b in sdfg.all_control_flow_blocks(recursive=True) if isinstance(b, ConditionalBlock))
    return next((m for m in map(match_guard, guards) if m is not None), None)


def is_empty(block: ControlFlowBlock) -> bool:
    return isinstance(block, SDFGState) and block.number_of_nodes() == 0


def single_condition(block: ConditionalBlock) -> str | None:
    """The condition of a ``ConditionalBlock`` with one branch and no ``else``."""
    if len(block.branches) != 1 or block.branches[0][0] is None:
        return None
    return block.branches[0][0].as_string


def single_symbol(code: str) -> str | None:
    expr = symbolic.pystr_to_symbolic(code)
    return str(expr) if isinstance(expr, sympy.Symbol) else None


def nonzero_test(expr: str) -> str | None:
    """``c`` if ``expr`` is ``c != 0``, ``c > 0`` or ``not (c == 0)``."""
    test = symbolic.pystr_to_symbolic(expr)
    if isinstance(test, sympy.Not) and isinstance(test.args[0], sympy.Eq):
        test = sympy.Ne(*test.args[0].args)
    if not isinstance(test, (sympy.Ne, sympy.StrictGreaterThan)) or test.rhs != 0:
        return None
    return str(test.lhs) if isinstance(test.lhs, sympy.Symbol) else None


def assignments_of(sdfg: SDFG, name: str) -> list:
    return [
        e for cfg in sdfg.all_control_flow_regions(recursive=True) for e in cfg.edges() if name in e.data.assignments
    ]


def inside(block: ControlFlowBlock, region: ControlFlowRegion) -> bool:
    graph = block.parent_graph
    while graph is not None and graph is not region:
        graph = graph.parent_graph
    return graph is region


def block_reads(block: ControlFlowBlock) -> set[str]:
    """Symbols a block reads itself, without the blocks nested in it."""
    if isinstance(block, SDFGState):
        return set(block.free_symbols)
    codes = [c for c, _ in block.branches if c] if isinstance(block, ConditionalBlock) else []
    if isinstance(block, LoopRegion):
        codes = [c for c in (block.init_statement, block.loop_condition, block.update_statement) if c]
    return {s for c in codes for s in c.get_free_symbols()}


def read_only_by(sdfg: SDFG, name: str, edges: set[int], blocks: set[int], region: ControlFlowRegion = None) -> bool:
    """Whether ``name`` is read only on ``edges``, by ``blocks`` or inside ``region``."""
    for cfg in sdfg.all_control_flow_regions(recursive=True):
        if region is not None and (cfg is region or inside(cfg, region)):
            continue
        if any(name in e.data.read_symbols() for e in cfg.edges() if id(e) not in edges):
            return False
        if any(
            name in block_reads(b) for b in cfg.nodes() if id(b) not in blocks and not isinstance(b, ControlFlowRegion)
        ):
            return False
        if id(cfg) not in blocks and not isinstance(cfg, SDFG) and name in block_reads(cfg):
            return False
    return True


def guard_test(guard: ConditionalBlock) -> tuple[str, Any] | None:
    """The guard's nonzero test, ``(expression, edge assigning its flag or None)``."""
    condition = single_condition(guard)
    flag = single_symbol(condition) if condition else None
    if flag is None or flag in guard.sdfg.arrays:
        return (condition, None) if condition else None
    edges = assignments_of(guard.sdfg, flag)
    if (
        len(edges) != 1
        or edges[0].data.assignments.keys() != {flag}
        or edges[0].src.parent_graph is not guard.parent_graph
    ):
        return None
    return edges[0].data.assignments[flag], edges[0]


def preceding_loop(guard: ConditionalBlock, flag_edge) -> LoopRegion | None:
    """The loop the guard follows through empty states and edges assigning nothing but the flag."""
    graph, block = guard.parent_graph, guard
    while True:
        edges = graph.in_edges(block)
        if len(edges) != 1 or (edges[0].data.assignments and edges[0] is not flag_edge):
            return None
        block = edges[0].src
        if isinstance(block, LoopRegion):
            return block
        if not is_empty(block):
            return None


def guarded_loop(guard: ConditionalBlock) -> LoopRegion | None:
    blocks = [b for b in guard.branches[0][1].nodes() if not is_empty(b)]
    return blocks[0] if len(blocks) == 1 and isinstance(blocks[0], LoopRegion) else None


def same_range(a: LoopRegion, b: LoopRegion) -> bool:
    bounds = [
        (f(a), f(b))
        for f in (loop_analysis.get_init_assignment, loop_analysis.get_loop_end, loop_analysis.get_loop_stride)
    ]
    return all(x is not None and y is not None and symbolic.simplify(x - y) == 0 for x, y in bounds)


def normalized(loop: LoopRegion, text: str) -> str:
    return re.sub(rf"\b{re.escape(loop.loop_variable)}\b", LOOP_INDEX, text)


def element(text: str) -> str:
    return str(subsets.Range.from_string(text))


def from_expression(loop: LoopRegion, text: str) -> Predicate:
    text = normalized(loop, text)
    reads = {(m.group(1), element(m.group(2))) for m in SUBSCRIPT.finditer(text) if m.group(1) in loop.sdfg.arrays}
    return Predicate(str(symbolic.pystr_to_symbolic(text)), frozenset(reads))


def from_tasklet(loop: LoopRegion, flag: str) -> Predicate | None:
    """The predicate a single tasklet in ``loop``'s body stores into the scalar ``flag``."""
    writers = [
        (s, e)
        for s in loop.nodes()
        if isinstance(s, SDFGState)
        for n in s.data_nodes()
        if n.data == flag
        for e in s.in_edges(n)
    ]
    if len(writers) != 1 or not isinstance(writers[0][1].src, nodes.Tasklet):
        return None
    state, out = writers[0]
    code, reads = out.src.code.as_string, set()
    for edge in state.in_edges(out.src):
        read = normalized(loop, f"{edge.data.data}[{edge.data.subset}]")
        reads.add((edge.data.data, element(normalized(loop, str(edge.data.subset)))))
        code = re.sub(rf"\b{edge.dst_conn}\b", read, code)
    code = re.sub(rf"\b{out.src_conn}\b\s*=", "", code)
    return Predicate(str(symbolic.pystr_to_symbolic(code)), frozenset(reads))


def predicate(loop: LoopRegion, cond: ConditionalBlock) -> Predicate | None:
    """The condition of ``cond``, a single-branch conditional directly in ``loop``'s body, as a ``Predicate``."""
    condition = single_condition(cond)
    if condition is None or cond.parent_graph is not loop:
        return None
    flag = single_symbol(condition)
    if flag is None:
        return from_expression(loop, condition)
    if flag in loop.sdfg.arrays:
        return from_tasklet(loop, flag)
    edges = [e for e in loop.edges() if flag in e.data.assignments]
    if len(edges) != 1 or len(assignments_of(loop.sdfg, flag)) != 1:
        return None
    return from_expression(loop, edges[0].data.assignments[flag])


def increments(loop: LoopRegion, counter: str) -> list[ConditionalBlock] | None:
    """The branches directly in ``loop`` that only add a positive constant to ``counter``."""
    found = []
    for edge in assignments_of(loop.sdfg, counter):
        branch = edge.src.parent_graph
        cond = branch.parent_graph
        if not isinstance(cond, ConditionalBlock) or cond.parent_graph is not loop:
            continue
        step = symbolic.pystr_to_symbolic(edge.data.assignments[counter]) - symbolic.pystr_to_symbolic(counter)
        if not (step.is_Integer and step > 0) or edge.data.assignments.keys() != {counter}:
            return None
        if any(not is_empty(b) for b in branch.nodes()) or len(branch.edges()) != 1:
            return None
        found.append(cond)
    return found


def counter_reset_before(loop: LoopRegion, counter: str) -> Any | None:
    """The edge setting ``counter = 0`` on the single-predecessor chain leading to ``loop``."""
    graph, block = loop.parent_graph, loop
    while len(graph.in_edges(block)) == 1:
        edge = graph.in_edges(block)[0]
        if counter in edge.data.assignments:
            return edge if symbolic.pystr_to_symbolic(edge.data.assignments[counter]) == 0 else None
        block = edge.src
    return None


def written_elements(loop: LoopRegion) -> list[tuple[str, str]]:
    return [
        (e.data.data, element(normalized(loop, str(e.data.subset))))
        for s in loop.all_control_flow_blocks(recursive=True)
        if isinstance(s, SDFGState)
        for n in s.data_nodes()
        for e in s.in_edges(n)
        if e.data.data is not None
    ]


def idle_without(loop: LoopRegion, key: Predicate) -> bool:
    """Whether every effect of ``loop``'s body sits under a single-branch conditional on ``key``."""
    sdfg = loop.sdfg
    for block in loop.nodes():
        if isinstance(block, ConditionalBlock):
            if predicate(loop, block) != key:
                return False
        elif not isinstance(block, SDFGState):
            return False
        elif any(
            not sdfg.arrays[n.data].transient or sdfg.arrays[n.data].total_size != 1
            for n in block.data_nodes()
            if block.in_degree(n) > 0
        ):
            return False
    private = {name for e in loop.edges() for name in e.data.assignments}
    return all(read_only_by(sdfg, name, set(), set(), region=loop) for name in private)


def predicate_is_stable(counting: LoopRegion, guarded: LoopRegion, key: Predicate) -> bool:
    """The counting loop writes the predicate's inputs only at the element it reads; the guarded loop not at all."""
    inputs = dict(key.reads)
    if len(inputs) != len(key.reads) or any(data in inputs for data, _ in written_elements(guarded)):
        return False
    return all(subset == inputs[data] for data, subset in written_elements(counting) if data in inputs)


def counting_predicate(counting: LoopRegion, counter: str) -> tuple[list[ConditionalBlock], Predicate] | None:
    """The increments of ``counter`` in ``counting`` and the single predicate they share."""
    found = increments(counting, counter)
    if not found or counter_reset_before(counting, counter) is None:
        return None
    if len(assignments_of(counting.sdfg, counter)) != len(found) + 1:
        return None
    keys = {predicate(counting, cond) for cond in found}
    return (found, keys.pop()) if len(keys) == 1 and None not in keys else None


def match_guard(guard: ConditionalBlock) -> Match | None:
    test = guard_test(guard)
    if test is None:
        return None
    expression, flag_edge = test
    counter = nonzero_test(expression)
    counting, guarded = preceding_loop(guard, flag_edge), guarded_loop(guard)
    if counter is None or counting is None or guarded is None or not same_range(counting, guarded):
        return None
    counted = counting_predicate(counting, counter)
    if (
        counted is None
        or not idle_without(guarded, counted[1])
        or not predicate_is_stable(counting, guarded, counted[1])
    ):
        return None
    found = counted[0]
    edges = {id(e) for cond in found for e in cond.branches[0][1].edges()}
    flag = None if flag_edge is None else next(iter(flag_edge.data.assignments))
    if flag_edge is not None:
        edges.add(id(flag_edge))
        if not read_only_by(guard.sdfg, flag, set(), {id(guard)}):
            return None
    if not read_only_by(guard.sdfg, counter, edges, {id(guard)}):
        return None
    return Match(guard, counter, flag, found)


def replace_with_empty_state(block: ControlFlowBlock) -> None:
    graph = block.parent_graph
    state = graph.add_state(f"{block.label}_dropped", is_start_block=graph.start_block is block)
    for edge in graph.in_edges(block):
        graph.add_edge(edge.src, state, edge.data)
    for edge in graph.out_edges(block):
        graph.add_edge(state, edge.dst, edge.data)
    graph.remove_node(block)


def drop(match: Match) -> None:
    names = {match.counter, match.flag} - {None}
    for cfg in match.guard.sdfg.all_control_flow_regions(recursive=True):
        for edge in cfg.edges():
            for name in names:
                edge.data.assignments.pop(name, None)
    for cond in match.increments:
        replace_with_empty_state(cond)
    move_branch_cfg_up_discard_conditions(match.guard, match.guard.branches[0][1])
