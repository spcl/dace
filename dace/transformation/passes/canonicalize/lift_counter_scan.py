# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Lift a loop-carried counter ``c = c + s(i)`` to an exclusive scan of its increments.

``for i: start[i] = c; if cond(i): c = c + s(i)`` carries ``c`` from each iteration to the next, so it runs in order.
Its value at the start of iteration ``i`` is ``c0 + sum(inc[0:i])`` with ``inc[i] = s(i)`` where the bump runs and 0
where it does not: an exclusive scan. The lift runs three phases:

1. the loop with its body's dataflow dropped (but what computes a test, a bound or the step), storing each iteration's
   increment into ``inc``; a bump inside inner loops (QE's ``ijh``) adds up its iteration's bumps;
2. ``rank = exclusive_scan(inc)`` and ``total = sum(inc)``;
3. the loop as written, with ``c = c0 + rank[i]`` bound at the start of each iteration and ``c = c0 + total`` after it.

Phases 1 and 3 carry nothing from one iteration to the next, but for the arrays the counter indexes: each iteration
must touch only its own segment ``[c, c + s)`` of them (QE's ``vkb(:, s + l)`` for ``l < nh``, cp2k's ``lx(n + l)``).
The lift proves that segment bound for every access through the counter, so the segments of two iterations are
disjoint whenever every ``s >= 0``; when the sign of ``s`` is not provable, phase 2 also takes ``min(inc)`` and phase 3
runs in parallel under ``min(inc) >= 0``, in order otherwise. A constant step is not the lift's: the induction-variable
substitution closes an unconditional one, and the compaction lift
(:class:`~dace.transformation.passes.canonicalize.loop_to_stream_compaction.LoopToStreamCompaction`) takes a guarded one.
A counter read only after the loop is left to the reduction lifts.

A counter bumped at several sites (TSVC s123: ``j = j + 1`` always, again under a guard) sums the steps of the bumps
each iteration runs into ``inc``; phase 3 keeps the body's own bumps. Every step must be ``>= 0``, and every access
through the counter lies in ``(c, c + inc]``: after bumps of steps ``P`` it reads ``c + d`` with ``1 - P <= d <= 0``.
"""

import ast
import copy
from typing import Any, NamedTuple

import sympy

from dace import SDFG, InterstateEdge, dtypes, properties, subsets, symbolic
from dace.libraries.standard.nodes.reduce import Reduce
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
from dace.transformation.interstate.loop_to_map import LoopLift
from dace.transformation.passes.analysis.loop_analysis import get_init_assignment, get_loop_end, get_loop_stride
from dace.transformation.passes.canonicalize.lift_break_to_find_first import linear_chain, names_in, written_data
from dace.transformation.passes.canonicalize.loop_to_stream_compaction import LoopToStreamCompaction
from dace.transformation.passes.loop_specialization import specialize_loop_under_condition
from dace.transformation.passes.parallelization.parallelize_loops import ParallelizeLoops

#: Prefixes of the transients and symbols the lift introduces.
INCREMENT_PREFIX = "counter_increment_"
RANK_PREFIX = "counter_rank_"
TOTAL_PREFIX = "counter_total_"
BASE_PREFIX = "counter_base_"
STEP_PREFIX = "counter_step_"
MIN_PREFIX = "counter_min_"
#: Rounds of lifting, one per counter nested in a lifted loop
MAX_ROUNDS = 8


class CounterMatch(NamedTuple):
    """A liftable counter loop: the counter, its step as Python text, and the edge that bumps it."""

    loop: LoopRegion
    counter: str
    step: str
    bump: InterstateEdge
    derived: frozenset[str]
    #: The bump sits on the one path through the body: every iteration stores its increment.
    every_iteration: bool
    #: The arrays the counter indexes, each iteration within its own segment ``[c, c + s)``.
    cursor_arrays: frozenset[str] = frozenset()
    #: Whether ``s >= 0`` is proven; else phase 3 runs in parallel under a check of the increments.
    nonnegative: bool = True
    #: The frontend's ``t = c + s; c = t``: the symbol the step is added into, on its own edge before the bump.
    temp: str | None = None
    #: The bump sits in a loop inside ``loop``: an iteration's increment is the sum of its bumps.
    nested: bool = False
    #: The transients the body computes for a test, a bound or a step: phase 1 keeps the dataflow that writes them.
    kept: frozenset[str] = frozenset()
    #: Several bump sites (TSVC s123): an iteration's increment is the sum of the bumps it runs, each its own step.
    multi: bool = False


def additive_terms(node: ast.expr, sign: int = 1) -> list[tuple[int, ast.expr]]:
    """The terms of a sum of ``+`` and ``-``, each with its sign."""
    if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add, ast.Sub)):
        right = sign if isinstance(node.op, ast.Add) else -sign
        return additive_terms(node.left, sign) + additive_terms(node.right, right)
    return [(sign, node)]


def step_of(counter: str, rhs: str) -> str | None:
    """``s`` of ``counter + s`` (in any order of the terms of a sum) when ``s`` does not read ``counter``."""
    terms = additive_terms(ast.parse(rhs, mode="eval").body)
    own = [sign for sign, term in terms if isinstance(term, ast.Name) and term.id == counter]
    rest = [(sign, term) for sign, term in terms if not (isinstance(term, ast.Name) and term.id == counter)]
    if own != [1] or not rest or any(counter in names_in(ast.unparse(term)) for sign, term in rest):
        return None
    return " + ".join(f"({ast.unparse(term)})" if sign > 0 else f"-({ast.unparse(term)})" for sign, term in rest)


def writers(state: SDFGState, containers: set[str] | frozenset[str]) -> set[nodes.Node]:
    """The access nodes in ``state`` that write ``containers``, with every node upstream of them."""
    keep = {n for n in state.data_nodes() if n.data in containers and state.in_degree(n) > 0}
    stack = list(keep)
    while stack:
        for edge in state.in_edges(stack.pop()):
            if edge.src not in keep:
                keep.add(edge.src)
                stack.append(edge.src)
    return keep


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
        # A lifted loop's body moves into a map's nested SDFG: its inner counters (QE's ``ijh`` per species) lift next round
        for round_number in range(MAX_ROUNDS):
            lifted_now = 0
            for sd in list(sdfg.all_sdfgs_recursive()):
                for loop in [r for r in sd.all_control_flow_regions() if isinstance(r, LoopRegion)]:
                    if not attached(loop, sd):
                        continue
                    match = self.match(sd, loop) or self.match_several_bumps(sd, loop)
                    if match is not None:
                        self.lift(sd, match)
                        lifted_now += 1
            lifted += lifted_now
            if not lifted_now:
                break
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
        temp = None
        if not recurrent:
            recurrent = self.copied_recurrence(located)
            temp = recurrent[0][4] if len(recurrent) == 1 else None
        if len(recurrent) != 1:
            return None  # no carried symbol, or several: no single scan
        region, bump, counter, rhs = recurrent[0][:4]
        step = step_of(counter, rhs)
        if step is None or counter == loop.loop_variable or not bump.data.is_unconditional():
            return None
        if counter not in sdfg.symbols or sdfg.symbols[counter] not in dtypes.INTEGER_TYPES:
            return None
        if sum(1 for edge in edges for name in edge.data.assignments if name == counter) != 1:
            return None
        # A bump inside an inner loop runs any number of times: its iteration adds up all of them (QE's ``ijh``)
        nested = False
        enclosing = region
        while enclosing is not loop:
            nested = nested or isinstance(enclosing, LoopRegion)
            enclosing = enclosing.parent_graph
        derived = self.derived_symbols(edges, counter, [bump]) - {temp}
        if temp is not None and (
            any(temp in state.free_symbols for state in loop.states())
            or sum(temp in names_in(v) for edge in edges for v in edge.data.assignments.values()) != 1
        ):
            return None  # the bumped value is read before the bump
        moving = derived | {counter}
        if names_in(step) & moving or self.conditions_read(loop) & moving:
            return None  # the increments are not computable without the counter
        kept = self.control_slice(sdfg, loop, moving)
        if kept is None:
            return None  # a test, bound or step reads data phase 1 cannot recompute without the counter
        states = list(loop.states())
        if not any(s.free_symbols & moving for s in states):
            return None  # the counter is only read after the loop: a reduction, not a scan
        cursor_arrays = self.cursor_arrays(states, counter, derived)
        if cursor_arrays is None:
            return None  # a nested SDFG sees the counter, or a derived symbol indexes an array
        every_iteration = region is loop and linear_chain(loop) is not None
        assigned = {name for edge in edges for name in edge.data.assignments} | {loop.loop_variable}
        if every_iteration and "[" not in step and not names_in(step) & assigned:
            return None  # ``c = c + K`` every iteration: the induction-variable substitution closes it
        if not cursor_arrays:
            return CounterMatch(
                loop, counter, step, bump.data, frozenset(derived), every_iteration, temp=temp, nested=nested, kept=kept
            )
        if derived or "[" in step:
            return None
        if symbolic.pystr_to_symbolic(step).is_number and (not nested or self.closable(loop, region, assigned)):
            return None  # a constant step: the induction-variable substitution or the compaction lift owns it
        if not self.segments_are_own(sdfg, loop, region, bump, counter, step, cursor_arrays):
            return None
        if not self.parallel_with_contract(loop, counter, cursor_arrays):
            return None  # phase 3 would stay in order: three passes for nothing
        nonnegative = symbolic.provably_nonnegative(symbolic.pystr_to_symbolic(step), sdfg.facts())
        if nested and not nonnegative:
            return None  # the check of the iterations' sums would not see one negative bump among them
        return CounterMatch(
            loop,
            counter,
            step,
            bump.data,
            frozenset(derived),
            every_iteration,
            cursor_arrays,
            nonnegative,
            temp,
            nested,
            kept,
        )

    def match_several_bumps(self, sdfg: SDFG, loop: LoopRegion) -> CounterMatch | None:
        """The match of one counter bumped at several sites, each ``c = c + s_k`` with ``s_k >= 0`` proven.

        Every access through the counter must lie in its iteration's segment ``(c, c + inc]``: the steps of the bumps
        that precede it on its path bound it from below, its own index from above (:meth:`several_segments_are_own`).
        """
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
        counter = recurrent[0][2] if len(recurrent) > 1 else None
        if counter is None or any(entry[2] != counter for entry in recurrent) or counter == loop.loop_variable:
            return None
        if counter not in sdfg.symbols or sdfg.symbols[counter] not in dtypes.INTEGER_TYPES:
            return None
        bumps = self.nonnegative_bumps(sdfg, counter, recurrent)
        if bumps is None or sum(1 for edge in edges for name in edge.data.assignments if name == counter) != len(bumps):
            return None
        if self.derived_symbols(edges, counter, [edge for region, edge, step in bumps]):
            return None
        assigned = {name for edge in edges for name in edge.data.assignments} | {loop.loop_variable}
        if all(
            symbolic.pystr_to_symbolic(step).is_number and self.closable(loop, region, assigned)
            for region, edge, step in bumps
        ):
            return None  # constant bumps no guard skips: the induction-variable substitution closes them
        if counter in self.conditions_read(loop):
            return None
        kept = self.control_slice(sdfg, loop, {counter})
        states = list(loop.states())
        if kept is None or not any(counter in {str(f) for f in s.free_symbols} for s in states):
            return None
        cursor_arrays = self.cursor_arrays(states, counter, set())
        if cursor_arrays is None:
            return None
        if cursor_arrays and not (
            self.several_segments_are_own(sdfg, loop, bumps, counter, cursor_arrays)
            and self.parallel_with_contract(loop, counter, cursor_arrays)
        ):
            return None
        first = bumps[0]
        return CounterMatch(
            loop, counter, str(first[2]), first[1].data, frozenset(), False, cursor_arrays, kept=kept, multi=True
        )

    def nonnegative_bumps(self, sdfg: SDFG, counter: str, recurrent: list) -> list | None:
        """``(region, edge, step)`` per bump ``c = c + s`` of an unconditional edge, or ``None`` unless every ``s`` is
        a plain expression free of the counter and proven ``>= 0`` (else a later segment could move back over an
        earlier one)."""
        bumps = [(region, edge, step_of(counter, rhs)) for region, edge, name, rhs in recurrent]
        facts = sdfg.facts()
        for region, edge, step in bumps:
            if step is None or "[" in step or counter in names_in(step) or not edge.data.is_unconditional():
                return None
            if not symbolic.provably_nonnegative(symbolic.pystr_to_symbolic(step), facts):
                return None
        return bumps

    def preceding_steps(self, state: SDFGState, loop: LoopRegion, bumps: list) -> Any | None:
        """The sum of the steps of the bumps every path to ``state`` runs in its iteration; ``None`` under an inner
        loop or a region that is not one chain."""
        total = sympy.Integer(0)
        block: Any = state
        while block is not loop:
            region = block.parent_graph
            if isinstance(region, ConditionalBlock):
                block = region
                continue
            if region is None or (isinstance(region, LoopRegion) and region is not loop):
                return None
            chain = linear_chain(region)
            if chain is None:
                return None
            for bump_region, edge, step in bumps:
                if bump_region is region and chain.index(edge.dst) <= chain.index(block):
                    total = total + symbolic.pystr_to_symbolic(step)
            block = region
        return total

    def several_segments_are_own(
        self, sdfg: SDFG, loop: LoopRegion, bumps: list, counter: str, arrays: frozenset[str]
    ) -> bool:
        """Whether every access to ``arrays`` lies in ``(c_in, c_in + inc]``: at ``c + d`` after bumps of steps
        ``P`` that surely ran, with ``d <= 0`` (at most what the iteration has bumped) and ``P + d >= 1``. With every
        step ``>= 0`` the segments of consecutive iterations are then disjoint."""
        facts = sdfg.facts()
        for state in loop.states():
            accesses = [n for n in state.data_nodes() if n.data in arrays]
            if not accesses:
                continue
            ran = self.preceding_steps(state, loop, bumps)
            if ran is None or any(state.entry_node(node) is not None for node in accesses):
                return False
            for node in accesses:
                sides = [e.data.get_dst_subset(e, state) for e in state.in_edges(node) if not e.data.is_empty()]
                sides += [e.data.get_src_subset(e, state) for e in state.out_edges(node) if not e.data.is_empty()]
                for subset in sides:
                    if not isinstance(subset, subsets.Range) or not any(
                        self.index_after_bumps(begin, end, counter, ran, facts) for begin, end, stride in subset.ranges
                    ):
                        return False
        return True

    def index_after_bumps(self, begin: Any, end: Any, counter: str, ran: Any, facts: Any) -> bool:
        """Whether ``[begin, end]``, through the counter ``c``, stays in ``[c - ran + 1, c]``."""
        c = symbolic.pystr_to_symbolic(counter)
        begin, end, c, ran = symbolic.equalize_symbols_across(begin, end, c, ran)
        if c not in begin.free_symbols | end.free_symbols:
            return False
        first, last = sympy.expand(begin - c), sympy.expand(end - c)
        if c in first.free_symbols or c in last.free_symbols:
            return False
        return symbolic.provably_le(1 - ran, first, facts) and symbolic.provably_le(last, 0, facts)

    def closable(self, loop: LoopRegion, region: ControlFlowRegion, assigned: set[str]) -> bool:
        """Whether a constant bump in ``region`` runs a number of times per iteration of ``loop`` that the
        induction-variable substitution closes: no guard between, and inner loops whose bounds the iteration does not
        change (a rectangular nest)."""
        between = []
        enclosing = region
        while enclosing is not loop:
            if isinstance(enclosing, ConditionalBlock):
                return False
            if isinstance(enclosing, LoopRegion):
                between.append(enclosing)
            enclosing = enclosing.parent_graph
        moving = set(assigned) | {inner.loop_variable for inner in between}
        for inner in between:
            bounds = (get_init_assignment(inner), get_loop_end(inner))
            if any(b is None or {str(f) for f in sympy.sympify(b).free_symbols} & moving for b in bounds):
                return False
        return True

    def copied_recurrence(self, located: list) -> list[tuple]:
        """``(region, bump, c, t_rhs, t)`` for each ``c = t`` whose ``t``, assigned once and in the same region, is
        ``t = c + s``: the frontend's spelling of ``c = c + s`` through a call such as ``max``."""
        sites: dict[str, list] = {}
        for region, edge in located:
            for name, rhs in edge.data.assignments.items():
                sites.setdefault(name, []).append((region, edge, rhs))
        found = []
        for name, assigned in sites.items():
            if len(assigned) != 1:
                continue
            region, edge, rhs = assigned[0]
            tree = ast.parse(rhs, mode="eval").body
            if not isinstance(tree, ast.Name) or len(sites.get(tree.id, ())) != 1:
                continue
            source_region, source, source_rhs = sites[tree.id][0]
            if source_region is not region or source is edge or name not in names_in(source_rhs):
                continue
            # ``t`` is computed on the path before ``c`` takes it, in the same iteration
            chain = linear_chain(region)
            if (
                chain is not None
                and source.data.is_unconditional()
                and chain.index(source.dst) <= chain.index(edge.src)
            ):
                found.append((region, edge, name, source_rhs, tree.id))
        return found

    def derived_symbols(self, edges: list, counter: str, bumps: list[Any]) -> set[str]:
        """The symbols the loop assigns from the counter, directly or through another such symbol."""
        derived: set[str] = set()
        changed = True
        while changed:
            changed = False
            for edge in edges:
                for name, rhs in edge.data.assignments.items():
                    if name == counter and any(edge is bump for bump in bumps):
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

    def control_slice(self, sdfg: SDFG, loop: LoopRegion, moving: set[str]) -> frozenset[str] | None:
        """The containers the body writes that a test, a loop bound or an assignment reads (QE's ``nh(j)`` copied into
        a scalar for the inner bound), with those their writers read. ``None`` when a writer reads the counter, writes
        a global or does more than tasklets and copies: phase 1 keeps only these writers."""
        written = written_data(list(loop.nodes()))
        read = self.conditions_read(loop)
        for region in loop.all_control_flow_regions():
            for edge in region.edges():
                for rhs in edge.data.assignments.values():
                    read |= names_in(rhs)
        needed = read & written
        while True:
            reached: set[str] = set()
            for state in loop.states():
                keep = writers(state, needed)
                for node in keep:
                    if isinstance(node, nodes.AccessNode):
                        reached.add(node.data)
                    elif not isinstance(node, nodes.Tasklet) or any(e.dst not in keep for e in state.out_edges(node)):
                        return None
                    if {str(f) for f in node.free_symbols} & moving:
                        return None
                    for edge in state.in_edges(node):
                        if {str(f) for f in edge.data.free_symbols} & moving:
                            return None
            grown = needed | (reached & written)
            if grown == needed:
                break
            needed = grown
        if any(not sdfg.arrays[name].transient for name in needed):
            return None
        return frozenset(needed)

    def cursor_arrays(self, states: list[SDFGState], counter: str, derived: set[str]) -> frozenset[str] | None:
        """The arrays an access in ``states`` indexes through the counter; ``None`` when a nested SDFG's symbol mapping
        reads the counter or a symbol derived from it, or an index reads a derived symbol (not modeled)."""
        moving = derived | {counter}
        arrays: set[str] = set()
        for state in states:
            for node in state.nodes():
                if isinstance(node, nodes.NestedSDFG) and any(
                    {str(f) for f in symbolic.pystr_to_symbolic(str(v)).free_symbols} & moving
                    for v in node.symbol_mapping.values()
                ):
                    return None
            # Each container at its own side of the memlet: a copy ``A[i] -> B[c]`` indexes ``B`` through ``c``
            for node in state.data_nodes():
                sides = [e.data.get_dst_subset(e, state) for e in state.in_edges(node) if not e.data.is_empty()]
                sides += [e.data.get_src_subset(e, state) for e in state.out_edges(node) if not e.data.is_empty()]
                for subset in sides:
                    names = {str(f) for f in subset.free_symbols} if subset is not None else set()
                    if names & derived:
                        return None
                    if names & moving:
                        arrays.add(node.data)
        # A read-only array the counter indexes (a gather) needs no segment
        return frozenset(arrays) & written_data(states)

    def segments_are_own(
        self,
        sdfg: SDFG,
        loop: LoopRegion,
        region: ControlFlowRegion,
        bump: Any,
        counter: str,
        step: str,
        arrays: frozenset[str],
    ) -> bool:
        """Whether every access to ``arrays`` in ``loop`` lies in its iteration's segment ``[c, c + s)``: under the
        region holding the bump, before or after it, through the counter in one dimension whose bounds, over the
        ranges of the loops between, stay in the segment."""
        chain = linear_chain(region)
        if chain is None:
            return False
        bumped_from = chain.index(bump.dst)
        c = symbolic.pystr_to_symbolic(counter)
        s = symbolic.pystr_to_symbolic(step)
        c, s = symbolic.equalize_symbols_across(c, s)
        facts = sdfg.facts()
        for state in loop.states():
            accesses = [n for n in state.data_nodes() if n.data in arrays]
            if not accesses:
                continue
            top, inner = state, []
            while top.parent_graph is not region:
                top = top.parent_graph
                if top is None or top is loop:
                    return False  # outside the region of the bump: an iteration that skips it would touch too
                if isinstance(top, LoopRegion):
                    inner.append(top)
            offset = s if chain.index(top) >= bumped_from else sympy.Integer(0)
            for node in accesses:
                if state.entry_node(node) is not None:
                    return False
                edges = [(e, e.data.get_dst_subset(e, state)) for e in state.in_edges(node)]
                edges += [(e, e.data.get_src_subset(e, state)) for e in state.out_edges(node)]
                for edge, subset in edges:
                    if edge.data.is_empty():
                        continue
                    if not isinstance(subset, subsets.Range) or not any(
                        self.dimension_in_segment(begin, end, c, s, offset, inner, facts)
                        for begin, end, stride in subset.ranges
                    ):
                        return False
        return True

    def dimension_in_segment(
        self, begin: Any, end: Any, c: Any, s: Any, offset: Any, inner: list[LoopRegion], facts: Any
    ) -> bool:
        """Whether ``[begin, end]``, an index through the counter ``c`` (``c_in + offset`` where it runs), stays in
        ``[c_in, c_in + s)`` over every value of the loops ``inner`` (innermost first)."""
        begin, end, c, s, offset = symbolic.equalize_symbols_across(begin, end, c, s, offset)
        if c not in begin.free_symbols | end.free_symbols:
            return False
        for nested in inner:
            stride = get_loop_stride(nested)
            low, high = get_init_assignment(nested), get_loop_end(nested)
            if stride is None or low is None or high is None or not (stride.is_number and stride > 0):
                return False
            variable, begin, end, low, high = symbolic.equalize_symbols_across(
                symbolic.pystr_to_symbolic(nested.loop_variable), begin, end, low, high
            )
            # An affine bound takes its extreme at one end of the loop's range; an empty range touches nothing
            slopes = (sympy.expand(begin).diff(variable), sympy.expand(end).diff(variable))
            if not all(slope.is_number for slope in slopes):
                return False
            begin = begin.subs(variable, low if slopes[0] >= 0 else high)
            end = end.subs(variable, high if slopes[1] >= 0 else low)
        first = sympy.expand(begin.subs(c, c + offset) - c)
        last = sympy.expand(end.subs(c, c + offset) - c)
        if c in first.free_symbols or c in last.free_symbols:
            return False
        return symbolic.provably_le(0, first, facts) and symbolic.provably_le(last, s - 1, facts)

    def parallel_with_contract(self, loop: LoopRegion, counter: str, arrays: frozenset[str]) -> bool:
        """Whether phase 3 lifts once the segment proof vouches for ``arrays`` and the counter is bound per iteration."""
        probe = LoopLift()
        probe.contract_arrays, probe.contract_reads = arrays, arrays
        closed = {counter: symbolic.pystr_to_symbolic(loop.loop_variable)}
        return ParallelizeLoops().accepts(loop, probe, closed_forms=closed)

    def parallelize(self, loop: LoopRegion, arrays: frozenset[str]) -> None:
        """Lift phase 3 now, on the segment proof: the dependence test cannot see it, so no later sweep would."""
        xform = LoopLift()
        xform.contract_arrays, xform.contract_reads = arrays, arrays
        parallelize = ParallelizeLoops()
        parallelize.expand_buffers = True
        if parallelize.lift(xform, loop, {}):
            parallelize.finish(loop.sdfg)

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
        if not match.cursor_arrays:
            return
        if match.nonnegative:
            self.parallelize(loop, match.cursor_arrays)
            return
        # A negative step moves a later segment back over an earlier one: checked once per run of the loop
        lowest = sdfg.add_scalar(MIN_PREFIX + loop.label, dtype, transient=True, find_new_name=True)[0]
        self.emit_min(scan, increments, lowest, shape)
        specialize_loop_under_condition(
            loop, f"{lowest} >= 0", lambda par_loop, region, owner: self.parallelize(par_loop, match.cursor_arrays)
        )

    def increments_loop(self, sdfg: SDFG, match: CounterMatch, increments: str, step: str, position: str) -> LoopRegion:
        """Phase 1: a copy of the loop without its dataflow and the counter, storing each iteration's increment."""
        loop = match.loop
        copied = copy.deepcopy(loop)
        copied.label = f"{loop.label}_increments"
        loop.parent_graph.add_node(copied, ensure_unique_name=True)
        for state in list(copied.states()):
            keep = writers(state, match.kept)
            state.remove_nodes_from([node for node in state.nodes() if node not in keep])
        source = match.temp or match.counter
        for region in list(copied.all_control_flow_regions()):
            for edge in list(region.edges()):
                assignments = edge.data.assignments
                if not assignments.keys() & (match.derived | {match.counter, source}):
                    continue
                kept: dict[str, str] = {}
                for name, rhs in assignments.items():
                    if name == source and match.multi:
                        kept[step] = f"{step} + ({step_of(match.counter, rhs)})"
                    elif name == source:
                        # Nested, each bump adds its step to the iteration's sum
                        kept[step] = f"{step} + ({match.step})" if match.nested else match.step
                    elif name not in match.derived and name != match.counter:
                        kept[name] = rhs
                edge.data.assignments = kept
                if source not in assignments or match.nested or match.multi:
                    continue
                store = region.add_state(f"{copied.label}_store")
                self.store(store, increments, step, position)
                region.add_edge(edge.src, store, edge.data)
                region.add_edge(store, edge.dst, InterstateEdge())
                region.remove_edge(edge)
        # Symbols of its own (``nh(j)`` copied for a bound): shared with phase 3, they would look live after it
        own = {name for edge in copied.all_interstate_edges() for name in edge.data.assignments} - {step}
        renames = {}
        for name in sorted(own & sdfg.symbols.keys()):
            renames[name] = sdfg.find_new_symbol(name)
            sdfg.add_symbol(renames[name], sdfg.symbols[name])
        copied.replace_dict(renames)
        if match.nested or match.multi:
            # The sum starts at 0 and is stored once the iteration's body is done
            start = copied.start_block
            zero = copied.add_state(f"{copied.label}_sum", is_start_block=True)
            copied.add_edge(zero, start, InterstateEdge(assignments={step: "0"}))
            last = next(block for block in copied.nodes() if copied.out_degree(block) == 0 and block is not zero)
            store = copied.add_state(f"{copied.label}_store")
            self.store(store, increments, step, position)
            copied.add_edge(last, store, InterstateEdge())
            return copied
        if match.every_iteration:
            return copied
        # An iteration that skips the bump adds nothing
        start = copied.start_block
        zero = copied.add_state(f"{copied.label}_zero", is_start_block=True)
        self.store(zero, increments, "0", position)
        copied.add_edge(zero, start, InterstateEdge())
        return copied

    def emit_min(self, state: SDFGState, increments: str, lowest: str, shape: list) -> None:
        """``lowest = min(increments)`` in ``state``, off the scan's read of the increments."""
        read = next(n for n in state.data_nodes() if n.data == increments)
        node = Reduce(name=f"{state.label}_min", wcr="lambda a, b: min(a, b)", axes=[0])
        node.add_in_connector("_in")
        node.add_out_connector("_out")
        state.add_node(node)
        span = subsets.Range([(0, shape[0] - 1, 1)])
        state.add_edge(read, None, node, "_in", Memlet(data=increments, subset=span))
        state.add_edge(
            node, "_out", state.add_write(lowest), None, Memlet(data=lowest, subset=subsets.Range([(0, 0, 1)]))
        )

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
