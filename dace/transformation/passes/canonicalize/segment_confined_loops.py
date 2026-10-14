# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Lift a loop over the segments of a pointer array when every iteration touches only its own segment.

``nfa_frontier`` runs one automaton component per iteration ``c``; component ``c`` owns the states
``comp_ptr[c] .. comp_ptr[c + 1]`` and writes nothing outside them::

    for c in range(C):
        base = comp_ptr[c]
        for k in range(start_ptr[c], start_ptr[c + 1]):
            s = start_idx[k]
            if enabled[s] == 0:
                enabled[s] = 1; frontier[base + n] = s; n += 1
        ...  # per symbol: walk frontier[base : base + n], children col_idx[row_ptr[s] : row_ptr[s + 1]]

``LoopToMap`` sees writes to ``enabled[s]`` and ``frontier[base + n]`` at data-dependent indices and keeps the loop.
They are disjoint across iterations because of facts about the DATA, which this pass establishes in two parts:

* **Checked once at entry** (a trapping guard, like
  :class:`~dace.transformation.passes.canonicalize.assume_symbols_nonnegative.AssumeSymbolConstraints`): the
  pointer arrays are monotone, an index array read over a segment of a secondary pointer stays inside the
  component (``start_idx[start_ptr[c] : start_ptr[c + 1]]``), and a CSR graph read from a member stays inside the
  component (``col_idx[row_ptr[s] : row_ptr[s + 1]]`` for ``s`` in the component).
* **Proven from the program**: a *member* is a value loaded through those facts, or loaded from a worklist that only
  ever stores members; every written index is a member, the iteration variable itself, or ``base + n`` for a
  counter ``n`` that provably stays below the component's width. A counter is bounded either by deduplication
  (each increment claims a member whose flag it finds clear and sets, and no flag is cleared between the counter's
  reset and the increment) or by its loop (at most one increment per trip of a loop bounded by such a counter).

Anything the pass cannot classify keeps the loop sequential.
"""

import ast
import copy
from typing import Any, NamedTuple

from dace import SDFG, InterstateEdge, Memlet, dtypes, symbolic
from dace.sdfg import nodes
from dace.sdfg import tasklet_utils as tutil
from dace.sdfg.state import ConditionalBlock, ControlFlowRegion, LoopRegion, SDFGState
from dace.transformation import pass_pipeline as ppl
from dace.transformation.passes.analysis import loop_analysis

#: What a lifted segment loop is, for a CPF reader.
SEGMENT_HINT = (
    "parallel -- segment-confined: iteration {var} reads and writes only inside {pointer}[{var}] .. "
    "{pointer}[{var} + 1]\nwhy: every written index is a member of the iteration's segment, the iteration variable, "
    "or a worklist slot below the segment's width; the data facts that make the members members are checked once "
    "at entry"
)


def load(text: str) -> tuple[str, str] | None:
    """``(array, index)`` of an assignment value that is one subscript load ``A[index]``, else ``None``."""
    try:
        expr = ast.parse(text.strip(), mode="eval").body
    except SyntaxError:
        return None
    if isinstance(expr, ast.Subscript) and isinstance(expr.value, ast.Name) and not isinstance(expr.slice, ast.Tuple):
        return expr.value.id, ast.unparse(expr.slice)
    return None


def without(expr, name: str) -> str | None:
    """``expr - name`` as text, subtracting the symbol object ``expr`` itself holds (a fresh symbol of the same
    name may carry other assumptions and not cancel); ``None`` when ``name`` does not occur in ``expr``."""
    symbol = next((s for s in expr.free_symbols if str(s) == name), None)
    return None if symbol is None else str(symbolic.simplify(expr - symbol))


def offset_from(index: str, var: str) -> int | None:
    """``k`` when ``index`` is ``var + k`` for an integer constant ``k``, else ``None`` -- also for a
    comparison or other boolean value (``max_b <= knob``), which has no difference to take."""
    try:
        diff = symbolic.simplify(symbolic.pystr_to_symbolic(index) - symbolic.pystr_to_symbolic(var))
        return int(diff)
    except (TypeError, ValueError):
        return None


class WorklistWrite(NamedTuple):
    """A write ``array[base + n]`` (or ``array[base + j]``, ``j < n``) with ``n`` the bounding counter."""

    state: SDFGState
    edge: Any
    array: str
    counter: str


class Range(NamedTuple):
    """An inner loop's iteration range, classified."""

    kind: str  # "segment", "row", "worklist", or "other"
    source: str  # the pointer array (segment / row), or the bounding counter (worklist)
    member: str | None  # the member a row range belongs to


class Proof(NamedTuple):
    """What the entry guard checks for a confined loop."""

    pointer: str  # the component pointer array: iteration c owns pointer[c] .. pointer[c + 1]
    secondary: set[str]  # further pointer arrays whose segments index arrays are read over
    index_facts: set[tuple[str, str]]  # (A, Q): A[Q[c] : Q[c + 1]] lies in the component
    csr_facts: set[tuple[str, str]]  # (A, R): A[R[s] : R[s + 1]] lies in the component for every member s


class SegmentLoop:
    """The analysis of one candidate loop; :meth:`prove` returns the :class:`Proof` or ``None``."""

    def __init__(self, sdfg: SDFG, loop: LoopRegion):
        self.sdfg = sdfg
        self.loop = loop
        self.var = loop.loop_variable
        self.assignments: dict[str, list[str]] = {}
        self.assigned_in: dict[str, list[ControlFlowRegion]] = {}
        for edge in loop.all_interstate_edges():
            for name, value in edge.data.assignments.items():
                self.assignments.setdefault(name, []).append(value)
                self.assigned_in.setdefault(name, []).append(edge.src.parent_graph)

    def resolve(self, expr, region: ControlFlowRegion):
        """``expr`` with every symbol bound once, in ``region`` itself, to an expression without loads replaced by
        that expression: canonical form names a write's index (``slice = base + n``) or a loop's bound
        (``n_match = n_front``) on the edge just before it."""
        for symbol in list(expr.free_symbols):
            name = str(symbol)
            values = self.assignments.get(name, [])
            if len(values) != 1 or self.assigned_in[name][0] is not region or load(values[0]) is not None:
                continue
            if "[" in values[0]:
                continue
            expr = expr.subs(symbol, symbolic.pystr_to_symbolic(values[0]))
        return symbolic.simplify(expr)

    # --- symbols --------------------------------------------------------------------------------------------

    def pointer_binding(self, symbol: str) -> tuple[str, int] | None:
        """``(P, k)`` when ``symbol`` is bound once, to ``P[var + k]``."""
        values = self.assignments.get(symbol, [])
        if len(values) != 1 or (loaded := load(values[0])) is None:
            return None
        offset = offset_from(loaded[1], self.var)
        return (loaded[0], offset) if offset in (0, 1) else None

    def counters(self) -> set[str]:
        """Symbols every assignment of which is ``0`` or ``itself + 1``."""
        found = set()
        for name, values in self.assignments.items():
            if all(v.strip() in ("0", "(0)") or offset_from(v, name) == 1 for v in values) and any(
                offset_from(v, name) == 1 for v in values
            ):
                found.add(name)
        return found

    def ranges(self, members: set[str], counters: set[str]) -> dict[str, Range]:
        """Every inner loop's variable, with its range classified."""
        classified = {}
        for region in self.loop.all_control_flow_regions():
            if region is self.loop or not isinstance(region, LoopRegion):
                continue
            start = loop_analysis.get_init_assignment(region)
            end = loop_analysis.get_loop_end(region)
            stride = loop_analysis.get_loop_stride(region)
            kind = Range("other", "", None)
            if start is not None and end is not None and stride == 1:
                low = str(start)
                high = str(self.resolve(symbolic.simplify(end + 1), region.parent_graph))
                low_ptr, high_ptr = self.pointer_binding(low), self.pointer_binding(high)
                if low_ptr and high_ptr and low_ptr[0] == high_ptr[0] and (low_ptr[1], high_ptr[1]) == (0, 1):
                    kind = Range("segment", low_ptr[0], None)
                elif low == "0" and high in counters:
                    kind = Range("worklist", high, None)
                else:
                    row = self.row_range(low, high, members)
                    if row is not None:
                        kind = Range("row", *row)
            classified[region.loop_variable] = kind
        return classified

    def row_range(self, low: str, high: str, members: set[str]) -> tuple[str, str] | None:
        """``(R, m)`` when ``low`` is bound to ``R[m]`` and ``high`` to ``R[m + 1]`` for a member ``m``."""
        lows, highs = self.assignments.get(low, []), self.assignments.get(high, [])
        if len(lows) != 1 or len(highs) != 1:
            return None
        lo, hi = load(lows[0]), load(highs[0])
        if lo is None or hi is None or lo[0] != hi[0] or lo[1] not in members:
            return None
        return (lo[0], lo[1]) if offset_from(hi[1], lo[1]) == 1 else None

    # --- containers -----------------------------------------------------------------------------------------

    def states(self) -> list[SDFGState]:
        return [b for b in self.loop.all_control_flow_blocks() if isinstance(b, SDFGState)]

    def written(self) -> set[str]:
        return {node.data for state in self.states() for node in state.data_nodes() if state.in_degree(node) > 0}

    def worklist_writes(self, counters: set[str], ranges: dict[str, "Range"]) -> list[WorklistWrite]:
        """``(state, edge, array, counter)`` of every write ``W[base + n]`` for a counter ``n``, or ``W[base + j]``
        for ``j`` a loop variable running below the counter ``n``."""
        found = []
        for state in self.states():
            for edge in state.edges():
                if not isinstance(edge.dst, nodes.AccessNode) or edge.data.data is None or edge.data.subset is None:
                    continue
                if len(edge.data.subset) != 1 or edge.data.subset.num_elements() != 1:
                    continue
                index = self.resolve(edge.data.subset[0][0], state.parent_graph)
                for symbol in map(str, index.free_symbols):
                    binding = self.pointer_binding(symbol)
                    if not binding or binding[1] != 0:
                        continue
                    rest = without(index, symbol)
                    if rest in counters:
                        found.append(WorklistWrite(state, edge, edge.data.data, rest))
                    elif rest in ranges and ranges[rest].kind == "worklist":
                        found.append(WorklistWrite(state, edge, edge.data.data, ranges[rest].source))
        return found

    # --- the proof ------------------------------------------------------------------------------------------

    def prove(self) -> Proof | None:
        if loop_analysis.get_init_assignment(self.loop) != 0 or loop_analysis.get_loop_stride(self.loop) != 1:
            return None
        if any(isinstance(n, nodes.NestedSDFG) for state in self.states() for n in state.nodes()):
            return None
        counters = self.counters()
        worklist_writes = self.worklist_writes(counters, self.ranges(set(), counters))
        bases = set()
        for write in worklist_writes:
            for symbol in map(str, self.resolve(write.edge.data.subset[0][0], write.state.parent_graph).free_symbols):
                binding = self.pointer_binding(symbol)
                if binding and binding[1] == 0:
                    bases.add(binding[0])
        if len(bases) != 1:
            return None
        pointer = bases.pop()
        worklists = {write.array for write in worklist_writes}
        base_symbols = {s for s in self.assignments if self.pointer_binding(s) == (pointer, 0)}

        # Members: a fixed point over the symbols' loads.
        members: set[str] = set()
        index_facts: set[tuple[str, str]] = set()
        csr_facts: set[tuple[str, str]] = set()
        changed = True
        while changed:
            changed = False
            ranges = self.ranges(members, counters)
            for name, values in self.assignments.items():
                if name in members or len(values) != 1 or (loaded := load(values[0])) is None:
                    continue
                array, index = loaded
                kind = ranges.get(index)
                fact = None
                if kind and kind.kind == "segment":
                    fact = ("index", (array, kind.source))
                elif kind and kind.kind == "row":
                    fact = ("csr", (array, kind.source))
                elif array in worklists:
                    sym = symbolic.pystr_to_symbolic(index)
                    for base in base_symbols:
                        rest = without(sym, base)
                        if rest in ranges and ranges[rest].kind == "worklist":
                            fact = ("worklist", None)
                if fact is None:
                    continue
                if fact[0] == "index":
                    index_facts.add(fact[1])
                elif fact[0] == "csr":
                    csr_facts.add(fact[1])
                members.add(name)
                changed = True
        ranges = self.ranges(members, counters)

        if not self.accesses_confined(members, counters, worklists, base_symbols, ranges):
            return None
        if not self.worklists_store_members(worklist_writes, members):
            return None
        bounded = self.bounded_counters(counters, ranges, members)
        if any(write.counter not in bounded for write in worklist_writes):
            return None
        secondary = {fact[1] for fact in index_facts}
        return Proof(pointer, secondary, index_facts, csr_facts)

    def accesses_confined(self, members, counters, worklists, base_symbols, ranges) -> bool:
        """Whether every access to a container the loop writes is confined to the iteration's segment."""
        written = self.written()
        for state in self.states():
            for edge in state.edges():
                name = edge.data.data
                if name is None or name not in written:
                    continue
                desc = self.sdfg.arrays[name]
                if desc.transient and desc.total_size == 1:
                    if any(
                        n.data == name for st in self.sdfg.states() if st not in self.states() for n in st.data_nodes()
                    ):
                        return False
                    continue
                subset = edge.data.subset
                if subset is None or len(subset) != 1 or subset.num_elements() != 1:
                    return False
                index = self.resolve(subset[0][0], state.parent_graph)
                text = str(index)
                if text in members or text == self.var:
                    continue
                if name in worklists and any(
                    without(index, base) in counters
                    or ranges.get(without(index, base) or "", Range("", "", None)).kind == "worklist"
                    for base in base_symbols
                ):
                    continue
                return False
        return True

    def worklists_store_members(self, worklist_writes, members) -> bool:
        """Whether every worklist write stores a member: a tasklet ``out = x`` for a member symbol ``x``."""
        for write in worklist_writes:
            tasklet = write.edge.src
            if not isinstance(tasklet, nodes.Tasklet) or write.state.in_degree(tasklet) > 0:
                return False
            try:
                body = ast.parse(tasklet.code.as_string).body
            except SyntaxError:
                return False
            if len(body) != 1 or not isinstance(body[0], ast.Assign) or not isinstance(body[0].value, ast.Name):
                return False
            if body[0].value.id not in members:
                return False
        return True

    def bounded_counters(self, counters, ranges, members) -> set[str]:
        """The counters that provably stay below the segment's width (see the module docstring)."""
        bounded = {n for n in counters if self.deduplicated(n, members)}
        changed = True
        while changed:
            changed = False
            for counter in counters - bounded:
                if self.one_per_trip(counter, ranges, bounded):
                    bounded.add(counter)
                    changed = True
        return bounded

    def increments(self, counter: str):
        return [
            edge
            for edge in self.loop.all_interstate_edges()
            if offset_from(edge.data.assignments.get(counter, "_"), counter) == 1
        ]

    def deduplicated(self, counter: str, members: set[str]) -> bool:
        """Every increment of ``counter`` sits in an arm guarded by ``flag[x] == 0`` that also sets ``flag[x]``,
        and no flag is cleared between the counter's reset and the increment."""
        flags = set()
        for edge in self.increments(counter):
            arm = edge.src.parent_graph
            block = arm.parent_graph if arm is not None else None
            if not isinstance(block, ConditionalBlock):
                return False
            condition = next((c for c, branch in block.branches if branch is arm), None)
            flag = self.cleared_flag_test(block, condition, members)
            if flag is None or not self.arm_sets(arm, flag, members):
                return False
            flags.add(flag)
        for flag in flags:
            if not self.clears_precede_resets(counter, flag):
                return False
        return True

    def cleared_flag_test(self, block: ConditionalBlock, condition, members) -> str | None:
        """The flag array ``F`` when ``condition`` is ``f == 0`` and ``f`` is loaded from ``F[x]`` for a member."""
        if condition is None:
            return None
        try:
            expr = ast.parse(condition.as_string, mode="eval").body
        except SyntaxError:
            return None
        if not (
            isinstance(expr, ast.Compare)
            and isinstance(expr.ops[0], ast.Eq)
            and isinstance(expr.left, ast.Name)
            and isinstance(expr.comparators[0], ast.Constant)
            and expr.comparators[0].value == 0
        ):
            return None
        scalar = expr.left.id
        for state in self.states():
            for node in state.data_nodes():
                if node.data != scalar:
                    continue
                for in_edge in state.in_edges(node):
                    writer = in_edge.src
                    for read in state.in_edges(writer) if isinstance(writer, nodes.Tasklet) else []:
                        if read.data.data and str(read.data.subset[0][0]) in members:
                            return read.data.data
        return None

    def arm_sets(self, arm: ControlFlowRegion, flag: str, members: set[str]) -> bool:
        """Whether ``arm`` writes a nonzero constant to ``flag[x]`` for a member ``x``."""
        for state in (b for b in arm.all_control_flow_blocks() if isinstance(b, SDFGState)):
            for edge in state.edges():
                if edge.data.data != flag or not isinstance(edge.dst, nodes.AccessNode):
                    continue
                if str(edge.data.subset[0][0]) not in members or not isinstance(edge.src, nodes.Tasklet):
                    continue
                value = constant_written(edge.src)
                if value is not None and value != 0:
                    return True
        return False

    def flag_clears(self, flag: str) -> list[SDFGState]:
        """The states that write zero to ``flag``."""
        return [
            state
            for state in self.states()
            for edge in state.edges()
            if edge.data.data == flag
            and isinstance(edge.dst, nodes.AccessNode)
            and isinstance(edge.src, nodes.Tasklet)
            and constant_written(edge.src) == 0
        ]

    def clears_precede_resets(self, counter: str, flag: str) -> bool:
        """Every increment of ``counter`` follows a reset of it, with no clear of ``flag`` in between, in the
        sequential order of the nearest region that holds both."""
        resets = [
            e
            for e in self.loop.all_interstate_edges()
            if counter in e.data.assignments and e.data.assignments[counter].strip() in ("0", "(0)")
        ]
        clears = self.flag_clears(flag)
        for increment in self.increments(counter):
            region, reset = nearest_reset(increment.src, resets)
            if region is None:
                return False
            order = sequential_order(region)
            if order is None:
                return False
            start = order[top_block(reset.dst, region)]
            here = order[top_block(increment.src, region)]
            if here < start:
                return False
            for clear in clears:
                if region_contains(region, clear):
                    position = order[top_block(clear, region)]
                    if start <= position <= here:
                        return False
        return True

    def one_per_trip(self, counter: str, ranges, bounded: set[str]) -> bool:
        """All increments of ``counter`` sit, once per trip, in one loop over ``0 .. n`` for a bounded ``n``."""
        increments = self.increments(counter)
        if not increments:
            return False
        loops = {innermost_loop(edge.src) for edge in increments}
        if len(loops) != 1:
            return False
        loop = loops.pop()
        if loop is None or loop is self.loop:
            return False
        kind = ranges.get(loop.loop_variable)
        if kind is None or kind.kind != "worklist" or kind.source not in bounded:
            return False
        # One increment per trip: no increment inside a loop nested in this one, and no path through two of them.
        return len(increments) == 1 and innermost_loop(increments[0].src) is loop


def constant_written(tasklet: nodes.Tasklet) -> int | None:
    """The integer constant a one-assignment tasklet writes, else ``None``."""
    try:
        body = ast.parse(tasklet.code.as_string).body
    except SyntaxError:
        return None
    if len(body) == 1 and isinstance(body[0], ast.Assign) and isinstance(body[0].value, ast.Constant):
        value = body[0].value.value
        return int(value) if isinstance(value, (int, float, bool)) else None
    return None


def innermost_loop(block) -> LoopRegion | None:
    region = block.parent_graph
    while region is not None and not isinstance(region, LoopRegion):
        region = region.parent_graph
    return region


def region_contains(region: ControlFlowRegion, block) -> bool:
    parent = block.parent_graph
    while parent is not None:
        if parent is region:
            return True
        parent = parent.parent_graph
    return False


def top_block(block, region: ControlFlowRegion):
    """The block of ``region``'s own graph that contains ``block``."""
    while block.parent_graph is not region:
        block = block.parent_graph
    return block


def nearest_reset(block, resets):
    """The innermost region around ``block`` that holds a reset edge in its own graph, with that edge."""
    region = block.parent_graph
    while region is not None:
        own = [e for e in resets if e in region.edges()]
        if own:
            return region, own[0]
        region = region.parent_graph
    return None, None


def sequential_order(region: ControlFlowRegion) -> dict | None:
    """The position of each block of ``region``'s graph in its one path from start to end, or ``None`` when
    its graph branches (a branch is a ``ConditionalBlock`` in canonical form, so a chain is expected)."""
    order = {}
    block = region.start_block
    while block is not None:
        if block in order:
            return None
        order[block] = len(order)
        successors = region.successors(block)
        if len(successors) > 1:
            return None
        block = successors[0] if successors else None
    return order if len(order) == region.number_of_nodes() else None


def add_entry_guard(sdfg: SDFG, loop: LoopRegion, proof: Proof) -> None:
    """Prepend, before ``loop``, loops of trapping guards that check the data facts ``proof`` relies on."""
    parent = loop.parent_graph
    count = str(symbolic.simplify(loop_analysis.get_loop_end(loop) + 1))

    def fresh(base: str) -> str:
        name = sdfg.find_new_symbol(base)
        sdfg.add_symbol(name, dtypes.int64)
        return name

    def guard_state(region: ControlFlowRegion, label: str, condition: str, reads: dict[str, str]) -> SDFGState:
        state = region.add_state(label, is_start_block=region.number_of_nodes() == 0)
        guard = tutil.add_abort_guard(state, f"check_{label}", condition, set(reads))
        for connector, element in reads.items():
            array = element.split("[", 1)[0]
            state.add_edge(state.add_read(array), None, guard, connector, Memlet(element))
        return state

    def component_loop(label: str) -> tuple[LoopRegion, str]:
        var = fresh(f"__seg_{label}")
        region = LoopRegion(f"seg_check_{label}", f"{var} < {count}", var, f"{var} = 0", f"{var} = {var} + 1")
        return region, var

    checks: list[LoopRegion] = []
    for pointer in sorted({proof.pointer} | proof.secondary):
        region, c = component_loop(f"monotone_{pointer}")
        guard_state(region, "monotone", "__lo > __hi", {"__lo": f"{pointer}[{c}]", "__hi": f"{pointer}[{c} + 1]"})
        checks.append(region)
    for array, secondary in sorted(proof.index_facts):
        region, c = component_loop(f"index_{array}")
        low, high, first, last = fresh("__seg_lo"), fresh("__seg_hi"), fresh("__seg_first"), fresh("__seg_last")
        entry = region.add_state("bind", is_start_block=True)
        k = fresh("__seg_k")
        inner = LoopRegion(f"seg_check_{array}_k", f"{k} < {last}", k, f"{k} = {first}", f"{k} = {k} + 1")
        region.add_node(inner)
        region.add_edge(
            entry,
            inner,
            InterstateEdge(
                assignments={
                    low: f"{proof.pointer}[{c}]",
                    high: f"{proof.pointer}[{c} + 1]",
                    first: f"{secondary}[{c}]",
                    last: f"{secondary}[{c} + 1]",
                }
            ),
        )
        guard_state(inner, "member", f"__x < {low} or __x >= {high}", {"__x": f"{array}[{k}]"})
        checks.append(region)
    for array, rows in sorted(proof.csr_facts):
        region, c = component_loop(f"csr_{array}")
        low, high = fresh("__seg_lo"), fresh("__seg_hi")
        entry = region.add_state("bind", is_start_block=True)
        s = fresh("__seg_s")
        members = LoopRegion(f"seg_check_{array}_s", f"{s} < {high}", s, f"{s} = {low}", f"{s} = {s} + 1")
        region.add_node(members)
        region.add_edge(
            entry,
            members,
            InterstateEdge(assignments={low: f"{proof.pointer}[{c}]", high: f"{proof.pointer}[{c} + 1]"}),
        )
        first, last, e = fresh("__seg_first"), fresh("__seg_last"), fresh("__seg_e")
        row_entry = members.add_state("bind_row", is_start_block=True)
        edges = LoopRegion(f"seg_check_{array}_e", f"{e} < {last}", e, f"{e} = {first}", f"{e} = {e} + 1")
        members.add_node(edges)
        members.add_edge(
            row_entry, edges, InterstateEdge(assignments={first: f"{rows}[{s}]", last: f"{rows}[{s} + 1]"})
        )
        guard_state(edges, "member", f"__x < {low} or __x >= {high}", {"__x": f"{array}[{e}]"})
        checks.append(region)

    # Chain the checks in front of the loop.
    predecessors = list(parent.in_edges(loop))
    previous = None
    for region in checks:
        parent.add_node(region)
        if previous is None:
            for edge in predecessors:
                parent.add_edge(edge.src, region, copy.deepcopy(edge.data))
                parent.remove_edge(edge)
            if parent.start_block is loop:
                parent.start_block = parent.node_id(region)
        else:
            parent.add_edge(previous, region, InterstateEdge())
        previous = region
    if previous is not None:
        parent.add_edge(previous, loop, InterstateEdge())


class SegmentConfinedLoops(ppl.Pass):
    """Lift every loop :class:`SegmentLoop` proves segment-confined, behind its entry guard."""

    CATEGORY: str = "Parallelization"

    def modifies(self) -> ppl.Modifies:
        return ppl.Modifies.CFG | ppl.Modifies.Nodes | ppl.Modifies.Symbols

    def should_reapply(self, modified: ppl.Modifies) -> bool:
        return False

    def apply_pass(self, sdfg: SDFG, pipeline_results: dict[str, Any]) -> int | None:
        """:returns: how many loops were lifted, or ``None`` if none were."""
        from dace.transformation.passes.parallelize_loops import ParallelizeLoops

        lifted = 0
        for region in list(sdfg.all_control_flow_regions()):
            if not isinstance(region, LoopRegion) or region.sdfg is not sdfg:
                continue
            proof = SegmentLoop(sdfg, region).prove()
            if proof is None:
                continue
            var, pointer = region.loop_variable, proof.pointer
            add_entry_guard(sdfg, region, proof)
            if ParallelizeLoops(propagate=False).parallelize_loop(sdfg, region, proven=True):
                for node, parent in sdfg.all_nodes_recursive():
                    if isinstance(node, nodes.MapEntry) and var in node.map.params:
                        node.specialization_hint = SEGMENT_HINT.format(var=var, pointer=pointer)
                lifted += 1
        return lifted or None
