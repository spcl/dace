# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Lift a loop-carried counter ``c = c + s(i)`` to an exclusive scan of its increments.

``for i: start[i] = c; if cond(i): c = c + s(i)`` carries ``c`` from each iteration to the next, so it runs in order.
Its value at the start of iteration ``i`` is ``c0 + sum(inc[0:i])`` with ``inc[i] = s(i)`` where the bump runs and 0
where it does not: an exclusive scan. The lift runs three phases:

1. the loop with its body's dataflow dropped, storing each iteration's increment into ``inc``;
2. ``rank = exclusive_scan(inc)`` and ``total = sum(inc)``;
3. the loop as written, with ``c = c0 + rank[i]`` bound at the start of each iteration and ``c = c0 + total`` after it.

Phases 1 and 3 carry nothing from one iteration to the next. A counter that indexes a write (stream compaction,
``out[c] = x``) is :class:`~dace.transformation.passes.canonicalize.loop_to_stream_compaction.LoopToStreamCompaction`'s:
its scatter needs a proof that the cursor never lands twice on one element. A counter that only feeds a reduction (read
after the loop alone) is left to the reduction lifts.
"""

import ast
import copy
from typing import Any, NamedTuple

import sympy

from dace import SDFG, InterstateEdge, dtypes, properties, subsets, symbolic
from dace.memlet import Memlet
from dace.sdfg import nodes
from dace.sdfg.state import (
    BreakBlock,
    ConditionalBlock,
    ContinueBlock,
    ControlFlowRegion,
    LoopRegion,
    ReturnBlock,
    SDFGState,
    SymbolResolver,
)
from dace.transformation import pass_pipeline as ppl
from dace.transformation import transformation
from dace.transformation.passes.analysis.loop_analysis import get_init_assignment, get_loop_end, get_loop_stride
from dace.transformation.passes.canonicalize.lift_break_to_find_first import linear_chain, names_in, written_data
from dace.transformation.passes.canonicalize.loop_to_stream_compaction import LoopToStreamCompaction

#: Prefixes of the transients and symbols the lift introduces.
INCREMENT_PREFIX = "counter_increment_"
RANK_PREFIX = "counter_rank_"
TOTAL_PREFIX = "counter_total_"
BASE_PREFIX = "counter_base_"
STEP_PREFIX = "counter_step_"


class CounterMatch(NamedTuple):
    """A liftable counter loop: the counter, its step as Python text, and the edge that bumps it."""

    loop: LoopRegion
    counter: str
    step: str
    bump: InterstateEdge
    derived: frozenset[str]
    #: The bump sits on the one path through the body: every iteration stores its increment.
    every_iteration: bool


def step_of(counter: str, rhs: str) -> str | None:
    """``s`` of ``counter + s``, ``s + counter`` or ``counter - s`` (as ``-(s)``) when ``s`` does not read ``counter``."""
    tree = ast.parse(rhs, mode="eval").body
    if not isinstance(tree, ast.BinOp) or not isinstance(tree.op, (ast.Add, ast.Sub)):
        return None
    left_is = isinstance(tree.left, ast.Name) and tree.left.id == counter
    right_is = isinstance(tree.right, ast.Name) and tree.right.id == counter and isinstance(tree.op, ast.Add)
    other = tree.right if left_is else tree.left if right_is else None
    if other is None or counter in {n.id for n in ast.walk(other) if isinstance(n, ast.Name)}:
        return None
    text = ast.unparse(other)
    return f"-({text})" if isinstance(tree.op, ast.Sub) else text


def attached(region: ControlFlowRegion, sdfg: SDFG) -> bool:
    """Whether ``region`` still hangs off ``sdfg``: an earlier lift in the sweep may have detached it."""
    current = region
    while current is not sdfg:
        parent = current.parent_graph
        if parent is None or current not in parent.nodes():
            return False
        current = parent
    return True


@properties.make_properties
@transformation.explicit_cf_compatible
class LiftCounterScan(ppl.Pass):
    """Lift every loop whose one carried counter ``c = c + s(i)`` it can prove to an exclusive scan of the increments."""

    CATEGORY: str = "Canonicalization"

    def modifies(self) -> ppl.Modifies:
        return ppl.Modifies.CFG | ppl.Modifies.States | ppl.Modifies.Nodes | ppl.Modifies.Descriptors

    def should_reapply(self, modified: ppl.Modifies) -> bool:
        return False

    def apply_pass(self, sdfg: SDFG, pipeline_results: dict[str, Any]) -> int | None:
        """:returns: how many loops were lifted, or ``None`` if none were."""
        lifted = 0
        for sd in list(sdfg.all_sdfgs_recursive()):
            for loop in [r for r in sd.all_control_flow_regions() if isinstance(r, LoopRegion)]:
                if not attached(loop, sd):
                    continue
                match = self.match(sd, loop)
                if match is not None:
                    self.lift(sd, match)
                    lifted += 1
        if lifted:
            sdfg.reset_cfg_list()
        return lifted or None

    def match(self, sdfg: SDFG, loop: LoopRegion) -> CounterMatch | None:
        """The match for ``loop``, or ``None`` when it is not the pattern or the lift is not sound."""
        if not loop.loop_variable or loop.inverted or get_loop_stride(loop) != 1:
            return None
        if get_init_assignment(loop) is None or get_loop_end(loop) is None:
            return None
        if any(isinstance(b, (BreakBlock, ContinueBlock, ReturnBlock)) for b in loop.all_control_flow_blocks()):
            return None
        located = [(region, edge) for region in loop.all_control_flow_regions() for edge in region.edges()]
        edges = [edge for region, edge in located]
        recurrent = [
            (region, edge, name, rhs)
            for region, edge in located
            for name, rhs in edge.data.assignments.items()
            if name in names_in(rhs)
        ]
        if len(recurrent) != 1:
            return None  # no carried symbol, or several: no single scan
        region, bump, counter, rhs = recurrent[0]
        step = step_of(counter, rhs)
        if step is None or counter == loop.loop_variable or not bump.data.is_unconditional():
            return None
        if counter not in sdfg.symbols or sdfg.symbols[counter] not in dtypes.INTEGER_TYPES:
            return None
        if sum(1 for edge in edges for name in edge.data.assignments if name == counter) != 1:
            return None
        # The bump runs at most once per iteration: no loop between it and ``loop``
        enclosing = region
        while enclosing is not loop:
            if isinstance(enclosing, LoopRegion):
                return None
            enclosing = enclosing.parent_graph
        derived = self.derived_symbols(edges, counter, bump)
        moving = derived | {counter}
        if names_in(step) & moving or self.conditions_read(loop) & moving:
            return None  # the increments are not computable without the counter
        if not self.meta_reads_stay_clear(sdfg, loop):
            return None  # a test or step reads data the body writes: phase 1 drops those writes
        states = list(loop.states())
        if not any(s.free_symbols & moving for s in states):
            return None  # the counter is only read after the loop: a reduction, not a scan
        if self.indexes_a_write(states, moving):
            return None  # a cursor: the compaction lift owns it
        every_iteration = region is loop and linear_chain(loop) is not None
        assigned = {name for edge in edges for name in edge.data.assignments} | {loop.loop_variable}
        if every_iteration and "[" not in step and not names_in(step) & assigned:
            return None  # ``c = c + K`` every iteration: the induction-variable substitution closes it
        return CounterMatch(loop, counter, step, bump.data, frozenset(derived), every_iteration)

    def derived_symbols(self, edges: list, counter: str, bump: Any) -> set[str]:
        """The symbols the loop assigns from the counter, directly or through another such symbol."""
        derived: set[str] = set()
        changed = True
        while changed:
            changed = False
            for edge in edges:
                for name, rhs in edge.data.assignments.items():
                    if edge is bump and name == counter:
                        continue
                    if name not in derived and names_in(rhs) & (derived | {counter}):
                        derived.add(name)
                        changed = True
        return derived

    def conditions_read(self, loop: LoopRegion) -> set[str]:
        """Every name a branch condition, an interstate condition or a nested loop's header in ``loop`` reads."""
        read: set[str] = set()
        for region in loop.all_control_flow_regions():
            for edge in region.edges():
                read |= names_in(edge.data.condition.as_string)
            if isinstance(region, ConditionalBlock):
                for condition, branch in region.branches:
                    if condition is not None:
                        read |= names_in(condition.as_string)
            if isinstance(region, LoopRegion) and region is not loop:
                for code in (region.init_statement, region.loop_condition, region.update_statement):
                    if code is not None:
                        read |= {n.id for n in ast.walk(ast.parse(code.as_string)) if isinstance(n, ast.Name)}
        return read

    def meta_reads_stay_clear(self, sdfg: SDFG, loop: LoopRegion) -> bool:
        """Whether no interstate edge or condition in ``loop`` reads a container the loop's dataflow writes."""
        written = written_data(list(loop.nodes()))
        return not any(m.data in written for m in LoopToStreamCompaction().meta_reads(loop, sdfg))

    def indexes_a_write(self, states: list[SDFGState], moving: set[str]) -> bool:
        """Whether a write subset or a nested SDFG's symbol mapping in ``states`` reads one of ``moving``."""
        for state in states:
            for node in state.nodes():
                if isinstance(node, nodes.NestedSDFG) and any(
                    {str(f) for f in symbolic.pystr_to_symbolic(str(v)).free_symbols} & moving
                    for v in node.symbol_mapping.values()
                ):
                    return True
            for edge in state.edges():
                if edge.data.is_empty() or not isinstance(edge.dst, nodes.AccessNode):
                    continue
                written = edge.data.get_dst_subset(edge, state)
                if written is None or {str(f) for f in written.free_symbols} & moving:
                    return True
        return False

    def lift(self, sdfg: SDFG, match: CounterMatch) -> None:
        """Rewrite ``match.loop`` into the increments loop, the scan and the loop with the counter bound per iteration."""
        loop, counter = match.loop, match.counter
        parent = loop.parent_graph
        dtype = sdfg.symbols[counter]
        begin = get_init_assignment(loop)
        trip = symbolic.simplify(get_loop_end(loop) - begin + 1)
        facts = SymbolResolver().facts_at(parent)
        shape = [trip if symbolic.provably_nonnegative(trip, facts) else sympy.Max(0, trip)]
        increments = sdfg.add_array(INCREMENT_PREFIX + loop.label, shape, dtype, transient=True, find_new_name=True)[0]
        rank = sdfg.add_array(RANK_PREFIX + loop.label, shape, dtype, transient=True, find_new_name=True)[0]
        total = sdfg.add_scalar(TOTAL_PREFIX + loop.label, dtype, transient=True, find_new_name=True)[0]
        base = sdfg.find_new_symbol(BASE_PREFIX + loop.label)
        step = sdfg.find_new_symbol(STEP_PREFIX + loop.label)
        sdfg.add_symbol(base, dtype)
        sdfg.add_symbol(step, dtype)
        position = f"{loop.loop_variable} - ({symbolic.symstr(begin)})"

        entry = parent.add_state(f"{loop.label}_counter_entry", is_start_block=parent.start_block is loop)
        for edge in list(parent.in_edges(loop)):
            parent.add_edge(edge.src, entry, edge.data)
            parent.remove_edge(edge)
        exit_state = parent.add_state(f"{loop.label}_counter_exit")
        for edge in list(parent.out_edges(loop)):
            parent.add_edge(exit_state, edge.dst, edge.data)
            parent.remove_edge(edge)

        increments_loop = self.increments_loop(sdfg, match, increments, step, position)
        scan = parent.add_state(f"{loop.label}_counter_scan")
        LoopToStreamCompaction().emit_scan_and_count(scan, increments, rank, total, shape)
        parent.add_edge(entry, increments_loop, InterstateEdge(assignments={base: counter}))
        parent.add_edge(increments_loop, scan, InterstateEdge())
        parent.add_edge(scan, loop, InterstateEdge())
        parent.add_edge(loop, exit_state, InterstateEdge(assignments={counter: f"{base} + {total}"}))

        # Phase 3: each iteration starts from its own value of the counter
        start = loop.start_block
        bind = loop.add_state(f"{loop.label}_counter_bind", is_start_block=True)
        loop.add_edge(bind, start, InterstateEdge(assignments={counter: f"{base} + {rank}[{position}]"}))

    def increments_loop(self, sdfg: SDFG, match: CounterMatch, increments: str, step: str, position: str) -> LoopRegion:
        """Phase 1: a copy of the loop without its dataflow and the counter, storing each iteration's increment."""
        loop = match.loop
        copied = copy.deepcopy(loop)
        copied.label = f"{loop.label}_increments"
        loop.parent_graph.add_node(copied, ensure_unique_name=True)
        for state in list(copied.states()):
            state.remove_nodes_from(list(state.nodes()))
        for region in list(copied.all_control_flow_regions()):
            for edge in list(region.edges()):
                assignments = edge.data.assignments
                if match.counter not in assignments and not assignments.keys() & match.derived:
                    continue
                if match.counter not in assignments:
                    edge.data.assignments = {k: v for k, v in assignments.items() if k not in match.derived}
                    continue
                edge.data.assignments = {
                    (step if k == match.counter else k): (match.step if k == match.counter else v)
                    for k, v in assignments.items()
                    if k not in match.derived
                }
                store = region.add_state(f"{copied.label}_store")
                self.store(store, increments, step, position)
                region.add_edge(edge.src, store, edge.data)
                region.add_edge(store, edge.dst, InterstateEdge())
                region.remove_edge(edge)
        if match.every_iteration:
            return copied
        # An iteration that skips the bump adds nothing
        start = copied.start_block
        zero = copied.add_state(f"{copied.label}_zero", is_start_block=True)
        self.store(zero, increments, "0", position)
        copied.add_edge(zero, start, InterstateEdge())
        return copied

    def store(self, state: SDFGState, array: str, value: str, position: str) -> None:
        """``array[position] = value`` in ``state``."""
        tasklet = state.add_tasklet(f"{state.label}_value", {}, {"__out"}, f"__out = {value}")
        index = symbolic.pystr_to_symbolic(position)
        state.add_edge(
            tasklet,
            "__out",
            state.add_write(array),
            None,
            Memlet(data=array, subset=subsets.Range([(index, index, 1)])),
        )


__all__ = ["LiftCounterScan"]
