# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Sharing one buffer between transient arrays whose uses never overlap."""

import re
from dataclasses import dataclass, field

import sympy

from dace import SDFG, Memlet, SDFGState, dtypes, properties, subsets, symbolic
from dace import data as dt
from dace.config import Config
from dace.ordered import OrderedSet
from dace.sdfg import nodes
from dace.sdfg.replace import replace_in_codeblock
from dace.sdfg.state import (
    AbstractControlFlowRegion,
    BreakBlock,
    ConditionalBlock,
    ContinueBlock,
    LoopRegion,
    ReturnBlock,
)
from dace.transformation import pass_pipeline as ppl
from dace.transformation.transformation import explicit_cf_compatible

#: Lifetimes a buffer may have to take part; Global and External ones belong to the program or the caller.
REUSABLE_LIFETIMES = (
    dtypes.AllocationLifetime.Scope,
    dtypes.AllocationLifetime.State,
    dtypes.AllocationLifetime.SDFG,
    dtypes.AllocationLifetime.Persistent,
)


@dataclass(slots=True)
class StateOrder:
    """Happens-before of the nodes directly in scope ``within`` of one state (the top level if None), every map scope
    below it collapsed onto its entry."""

    succ: dict[nodes.Node, OrderedSet] = field(default_factory=dict)
    pred: dict[nodes.Node, OrderedSet] = field(default_factory=dict)
    below: dict[nodes.Node, OrderedSet] = field(default_factory=dict)
    scope: dict[nodes.Node, nodes.EntryNode | None] = field(default_factory=dict)
    entry_of: dict[nodes.Node, nodes.EntryNode] = field(default_factory=dict)
    within: nodes.EntryNode | None = None

    @staticmethod
    def of(state: SDFGState, within: nodes.EntryNode | None = None) -> "StateOrder":
        order = StateOrder(scope=state.scope_dict(), within=within)
        order.entry_of = {state.exit_node(n): n for n in state.nodes() if isinstance(n, nodes.EntryNode)}
        for edge in state.edges():
            src, dst = order.top(edge.src), order.top(edge.dst)
            if src is not dst and src is not None and dst is not None:
                order.succ.setdefault(src, OrderedSet()).add(dst)
                order.pred.setdefault(dst, OrderedSet()).add(src)
        return order

    def top(self, node: nodes.Node) -> nodes.Node | None:
        """The node directly in ``within`` holding ``node``, None if ``node`` is outside ``within``."""
        node = self.entry_of.get(node, node)
        while self.scope[node] is not self.within:
            node = self.scope[node]
            if node is None:
                return None
        return node

    def descendants(self, node: nodes.Node) -> OrderedSet:
        """Every node reachable from ``node``, without ``node`` itself."""
        if node not in self.below:
            seen, stack = OrderedSet(), list(self.succ.get(node, ()))
            while stack:
                nxt = stack.pop()
                if nxt not in seen:
                    seen.add(nxt)
                    stack.extend(self.succ.get(nxt, ()))
            self.below[node] = seen
        return self.below[node]

    def before(self, first: list[nodes.AccessNode], second: list[nodes.AccessNode]) -> bool:
        """Whether every access through ``first`` completes before ``second`` is first written: each writer of a
        ``second`` node is reached from every reader of every ``first`` node (from the node itself if unread). A node
        inside a map stands for that whole map."""
        ends, starts = [], []
        for a in first:
            ends.extend(self.succ.get(a) or (a,) if self.scope[a] is None else (self.top(a),))
        for b in second:
            found = self.pred.get(b) if self.scope[b] is None else (self.top(b),)
            if not found:
                return False
            starts.extend(found)
        return all(w in self.descendants(end) for end in ends for w in starts)


@properties.make_properties
@explicit_cf_compatible
class TransientReuse(ppl.Pass):
    """
    Lets transient arrays whose values are never live together share one buffer, the largest of the group, by renaming
    them to it.

    Liveness runs over the control flow graph of each SDFG, loops and branches flattened, so one array may have several
    live ranges. A write of the whole array ends a range; a partial write does not. Two arrays interfere where both are
    live on entry to a state, both on exit, one all through it, or within it unless the accesses of the one dying or
    local to it all finish before the first write of the other. An array read where no path holds a write before it
    carries a value from an earlier run and shares nothing.

    A whole-array copy hands a dying array over to one born there: the two do not interfere and the copy goes.

    Only plain heap / GPU global arrays qualify, not read by control flow, not viewed or referenced, no WCR target and
    not ``may_alias``; one used inside a map scope only if it is allocated outside it (SDFG or Persistent lifetime, or also
    accessed outside every map in each state using it). In an SDFG nested in a parallel map only the stack arrays of a
    host thread share, within one run of the SDFG (:func:`sharing_storages`). A shared buffer has the same dtype, storage, pooling and strides as its members, is provably at least
    as large in every dimension, and is allocated once: Persistent if any member is, else SDFG lifetime. At the root
    SDFG every candidate sized only by symbols bound on entry becomes Persistent, shared or not. GPU arrays share only
    when all device work runs on one stream (``compiler.cuda.max_concurrent_streams == -1``), so device accesses run in
    program order.
    """

    CATEGORY: str = "Memory Footprint Reduction"

    verbose = properties.Property(
        dtype=bool,
        default=False,
        category="Diagnostics",
        desc="Print information about the memory reduction.",
    )

    def modifies(self) -> ppl.Modifies:
        return ppl.Modifies.Descriptors | ppl.Modifies.AccessNodes

    def should_reapply(self, modified: ppl.Modifies) -> bool:
        return modified & (ppl.Modifies.Nodes | ppl.Modifies.Memlets)

    def apply_pass(self, sdfg: SDFG, _) -> dict[int, dict[str, str]] | None:
        """
        :param sdfg: The SDFG to modify in place, nested SDFGs outside every map included.
        :return: Per ``cfg_id``, each renamed transient mapped to the buffer it now shares, or None if none was.
        """
        storages = [dtypes.StorageType.CPU_Heap]
        if Config.get("compiler", "cuda", "max_concurrent_streams") == -1:
            storages.append(dtypes.StorageType.GPU_Global)
        result: dict[int, dict[str, str]] = {}
        for sd in sdfg.all_sdfgs_recursive():
            shared = sharing_storages(sd, tuple(storages))
            if not shared:
                continue
            before = sum(desc.total_size_in_bytes for desc in sd.arrays.values()) if self.verbose else 0
            renamed = BufferSharing(sd, shared).run()
            if renamed:
                result[sd.cfg_id] = renamed
                if self.verbose:
                    after = sum(desc.total_size_in_bytes for desc in sd.arrays.values())
                    print(f"{sd.name}: {len(renamed)} transients reuse a buffer, {before} B -> {after} B")
        return result or None


def enclosing_schedules(sdfg: SDFG) -> list[dtypes.ScheduleType | None]:
    """The schedule of every scope around ``sdfg`` in the SDFGs enclosing it, ``None`` for one that is no map."""
    found = []
    while sdfg.parent_nsdfg_node is not None:
        entry = sdfg.parent.entry_node(sdfg.parent_nsdfg_node)
        while entry is not None:
            found.append(entry.map.schedule if isinstance(entry, nodes.MapEntry) else None)
            entry = sdfg.parent.entry_node(entry)
        sdfg = sdfg.parent_sdfg
    return found


def inside_parallel_map(sdfg: SDFG) -> bool:
    """Whether ``sdfg`` runs inside a map scope of some enclosing SDFG that is not ``Sequential``."""
    return any(schedule != dtypes.ScheduleType.Sequential for schedule in enclosing_schedules(sdfg))


def sharing_storages(sdfg: SDFG, global_storages: tuple) -> tuple:
    """The storages :class:`BufferSharing` may share in ``sdfg``. Each run of an SDFG nested in a host parallel map has
    its own frame, so its stack arrays may share one buffer as the arrays of a sequential SDFG do; a device thread's
    have no frame, and nothing global can be shared there: it would be shared with the iterations running at once."""
    schedules = enclosing_schedules(sdfg)
    if all(schedule == dtypes.ScheduleType.Sequential for schedule in schedules):
        return global_storages
    if all(schedule in (dtypes.ScheduleType.Sequential, *dtypes.CPU_SCHEDULES) for schedule in schedules):
        return (dtypes.StorageType.Register,)
    return ()


@dataclass(slots=True)
class Group:
    rep: str
    members: list[str]


@dataclass(slots=True)
class ControlFlow:
    """One SDFG's control flow as a plain graph: a vertex per state, per jump and per region's head and tail. A loop or
    a conditional may always be bypassed, which only adds paths."""

    succ: list[list[int]] = field(default_factory=list)
    vertex: dict[SDFGState, int] = field(default_factory=dict)
    start: int = 0
    stop: int = 0

    @staticmethod
    def of(sdfg: SDFG) -> "ControlFlow":
        flow = ControlFlow()
        flow.stop = flow.new()
        flow.start, tail = flow.flatten(sdfg, None)
        flow.succ[tail].append(flow.stop)
        return flow

    def new(self) -> int:
        self.succ.append([])
        return len(self.succ) - 1

    def flatten(self, region: AbstractControlFlowRegion, loop: tuple[int, int] | None) -> tuple[int, int]:
        """The head and tail of ``region``; ``loop`` is the head and tail of the loop around it."""
        head, tail = self.new(), self.new()
        if isinstance(region, ConditionalBlock):
            self.succ[head].append(tail)
            for _, branch in region.branches:
                first, last = self.flatten(branch, loop)
                self.succ[head].append(first)
                self.succ[last].append(tail)
            return head, tail
        end = tail
        if isinstance(region, LoopRegion):
            loop, end = (head, tail), head
            self.succ[head].append(tail)
        ends: dict = {}
        for block in region.nodes():
            first = last = None
            if isinstance(block, AbstractControlFlowRegion):
                first, last = self.flatten(block, loop)
            else:
                first = last = self.new()
                if isinstance(block, SDFGState):
                    self.vertex[block] = first
                elif isinstance(block, ContinueBlock):
                    self.succ[first].append(loop[0])
                elif isinstance(block, BreakBlock):
                    self.succ[first].append(loop[1])
                elif isinstance(block, ReturnBlock):
                    self.succ[first].append(self.stop)
            ends[block] = (first, last)
        self.succ[head].append(ends[region.start_block][0])
        for edge in region.edges():
            self.succ[ends[edge.src][1]].append(ends[edge.dst][0])
        for block in region.nodes():
            if region.out_degree(block) == 0 and not isinstance(block, (ContinueBlock, BreakBlock, ReturnBlock)):
                self.succ[ends[block][1]].append(end)
        return head, tail

    def postorder(self) -> list[int]:
        """Every vertex reachable from the start, each after its successors, the first successor last."""
        seen, order, stack = {self.start}, [], [(self.start, reversed(self.succ[self.start]))]
        while stack:
            vertex, rest = stack[-1]
            for nxt in rest:
                if nxt not in seen:
                    seen.add(nxt)
                    stack.append((nxt, reversed(self.succ[nxt])))
                    break
            else:
                order.append(vertex)
                stack.pop()
        return order


@dataclass(slots=True)
class BufferSharing:
    """One SDFG's candidates, their live ranges, and the greedy grouping into shared buffers."""

    sdfg: SDFG
    storages: tuple
    uses: dict[str, dict[SDFGState, list[nodes.AccessNode]]] = field(default_factory=dict)
    #: Candidates with an access node inside a map scope.
    scoped: OrderedSet = field(default_factory=OrderedSet)
    rank: dict[SDFGState, int] = field(default_factory=dict)
    orders: dict[SDFGState, StateOrder] = field(default_factory=dict)
    #: Per state, as a bit per candidate: live on entry and live on exit, both only once written on some path.
    live_in: dict[SDFGState, int] = field(default_factory=dict)
    live_out: dict[SDFGState, int] = field(default_factory=dict)
    bit: dict[str, int] = field(default_factory=dict)
    #: Per candidate, the states where it is accessed or live, which it may interfere in.
    active: dict[str, OrderedSet] = field(default_factory=dict)
    hoistable_names: OrderedSet | None = None
    #: Candidates a nested SDFG's connector is bound to: its descriptor must keep matching the array's.
    nested: OrderedSet = field(default_factory=OrderedSet)
    #: The SDFG's facts, derived once: nothing changes until the renaming at the end.
    facts: symbolic.Facts | None = None
    #: Per candidate, the candidates a whole-array copy connects it to.
    partners: dict[str, OrderedSet] = field(default_factory=dict)
    equalities: dict[tuple, bool] = field(default_factory=dict)
    bounds: dict[tuple, bool] = field(default_factory=dict)

    def run(self) -> dict[str, str]:
        self.facts = self.sdfg.facts()
        self.collect()
        groups: list[Group] = []
        for name in sorted(self.uses, key=lambda n: min(self.rank[s] for s in self.uses[n])):
            group = self.place(name, groups)
            if group is None:
                groups.append(Group(name, [name]))
        persistent = dtypes.AllocationLifetime.Persistent
        root = self.sdfg.parent_nsdfg_node is None
        for group in groups:
            desc = self.sdfg.arrays[group.rep]
            # At the root, symbols bound on entry are known at init: a buffer sized by them alone is allocated there.
            if desc.lifetime != persistent and root and self.hoistable(desc):
                desc.lifetime = persistent
            # Below the root the SDFG may be inlined into a parallel map, whose iterations each keep their own buffer.
            elif len(group.members) > 1 and desc.lifetime != persistent and root:
                desc.lifetime = dtypes.AllocationLifetime.SDFG
        renamed = {m: g.rep for g in groups for m in g.members if m != g.rep}
        if not renamed:
            return {}
        for state in self.sdfg.states():
            for node in state.nodes():
                if isinstance(node, nodes.AccessNode) and node.data in renamed:
                    node.data = renamed[node.data]
                # A Python tasklet may name a container in its code (``InlineTaskletConnectors``), not by connector
                elif isinstance(node, nodes.Tasklet) and node.language is dtypes.Language.Python:
                    named = re.findall(r"\w+", node.code.as_string)
                    code_repl = {
                        name: renamed[name]
                        for name in named
                        if name in renamed and name not in node.in_connectors and name not in node.out_connectors
                    }
                    if code_repl:
                        replace_in_codeblock(node.code, code_repl, node, self.sdfg)
            # Scope connectors passing a renamed array through are named after it, unless a scope already has that
            # name: connector names need not match their data, so another array may hold it.
            moved: dict[tuple[nodes.Node, str], bool] = {}

            def renames(node: nodes.Node, old: str) -> bool:
                if (node, old) not in moved:
                    new = renamed[old]
                    moved[(node, old)] = (
                        f"IN_{new}" not in node.in_connectors and f"OUT_{new}" not in node.out_connectors
                    )
                return moved[(node, old)]

            for edge in state.edges():
                old = edge.data.data
                if old not in renamed:
                    continue
                edge.data.data = renamed[old]
                src, dst = edge.src, edge.dst
                if (
                    isinstance(src, (nodes.EntryNode, nodes.ExitNode))
                    and edge.src_conn == f"OUT_{old}"
                    and renames(src, old)
                ):
                    edge.src_conn = f"OUT_{renamed[old]}"
                if (
                    isinstance(dst, (nodes.EntryNode, nodes.ExitNode))
                    and edge.dst_conn == f"IN_{old}"
                    and renames(dst, old)
                ):
                    edge.dst_conn = f"IN_{renamed[old]}"
            for (node, old), rename in moved.items():
                if not rename:
                    continue
                if f"IN_{old}" in node.in_connectors:
                    node.add_in_connector(f"IN_{renamed[old]}", node.in_connectors.pop(f"IN_{old}"), force=True)
                if f"OUT_{old}" in node.out_connectors:
                    node.add_out_connector(f"OUT_{renamed[old]}", node.out_connectors.pop(f"OUT_{old}"), force=True)
        for state in self.sdfg.states():
            for edge in state.edges():
                if (
                    isinstance(edge.src, nodes.AccessNode)
                    and isinstance(edge.dst, nodes.AccessNode)
                    and edge.src.data == edge.dst.data in renamed.values()
                    and not edge.data.is_empty()
                ):
                    edge.data = Memlet()
        for name in renamed:
            self.sdfg.remove_data(name, validate=False)
        return renamed

    def collect(self) -> None:
        sd = self.sdfg
        excluded: OrderedSet = OrderedSet(sd.constants_prop.keys())
        for region in sd.all_control_flow_regions():
            for code in region.get_meta_codeblocks():
                excluded.update(code.get_free_symbols())
            for edge in region.edges():
                excluded.update(edge.data.read_symbols())
        # Per scoped candidate, whether every state using it also holds it outside every map (allocated there).
        state_level: dict[str, bool] = {}
        for state in sd.states():
            scope = None
            outside, seen = OrderedSet(), OrderedSet()
            for node in state.data_nodes():
                desc = sd.arrays[node.data]
                if not self.candidate(desc):
                    continue
                scope = scope or state.scope_dict()
                if scope[node] is None:
                    outside.add(node.data)
                else:
                    self.scoped.add(node.data)
                for edge in state.all_edges(node):
                    other = edge.dst if edge.src is node else edge.src
                    if isinstance(other, nodes.AccessNode) and isinstance(other.desc(sd), (dt.View, dt.Reference)):
                        excluded.add(node.data)
                    if node.data not in self.nested and any(
                        isinstance(tree.edge.src, nodes.NestedSDFG) or isinstance(tree.edge.dst, nodes.NestedSDFG)
                        for tree in state.memlet_tree(edge).traverse_children(include_self=True)
                    ):
                        self.nested.add(node.data)
                self.uses.setdefault(node.data, {}).setdefault(state, []).append(node)
                seen.add(node.data)
            for name in seen:
                state_level[name] = state_level.get(name, True) and name in outside
            for edge in state.edges():
                if edge.data.wcr is not None and edge.data.data is not None:
                    excluded.add(edge.data.data)
        # An array used only inside maps may be allocated per scope, privately per map iteration.
        once = (dtypes.AllocationLifetime.SDFG, dtypes.AllocationLifetime.Persistent)
        excluded.update(n for n in self.scoped if not state_level[n] and sd.arrays[n].lifetime not in once)
        for name in excluded:
            self.uses.pop(name, None)
        # Sharing takes two, and only the root lifts a lone buffer.
        if len(self.uses) < 2 and sd.parent_nsdfg_node is not None:
            self.uses.clear()
            return
        self.liveness()
        for name, states in self.uses.items():
            for state, accessed in states.items():
                for node in accessed:
                    for edge in state.out_edges(node):
                        other = edge.dst
                        if isinstance(other, nodes.AccessNode) and other.data in self.uses and other.data != name:
                            self.partners.setdefault(name, OrderedSet()).add(other.data)
                            self.partners.setdefault(other.data, OrderedSet()).add(name)

    def liveness(self) -> None:
        """Live ranges of every candidate over the control flow graph, as bits per vertex: ``uses`` is read before any
        whole-array write of the state, ``kills`` written whole, ``writes`` written at all, ``exposed`` read before any
        write of the state. A value is live between a write that may reach a read and that read."""
        flow = ControlFlow.of(self.sdfg)
        self.bit = {name: 1 << i for i, name in enumerate(self.uses)}
        size = len(flow.succ)
        uses, kills, writes, exposed, touched = [0] * size, [0] * size, [0] * size, [0] * size, [0] * size
        for name, states in self.uses.items():
            bit = self.bit[name]
            for state, accessed in states.items():
                vertex = flow.vertex[state]
                use, kill, wrote, early = self.summary(state, name, accessed)
                touched[vertex] |= bit
                uses[vertex] |= bit if use else 0
                kills[vertex] |= bit if kill else 0
                writes[vertex] |= bit if wrote else 0
                exposed[vertex] |= bit if early else 0
        order = flow.postorder()
        position = {vertex: size - i for i, vertex in enumerate(order)}
        self.rank = {state: position.get(vertex, size) for state, vertex in flow.vertex.items()}
        pred: list[list[int]] = [[] for _ in range(size)]
        for vertex in order:
            for nxt in flow.succ[vertex]:
                pred[nxt].append(vertex)
        live, written = [0] * size, [0] * size
        changed = True
        while changed:
            changed = False
            for vertex in order:
                out = 0
                for nxt in flow.succ[vertex]:
                    out |= live[nxt]
                new = uses[vertex] | (out & ~kills[vertex])
                if new != live[vertex]:
                    live[vertex], changed = new, True
        changed = True
        while changed:
            changed = False
            for vertex in reversed(order):
                seen = writes[vertex]
                for prev in pred[vertex]:
                    seen |= written[prev]
                if seen != written[vertex]:
                    written[vertex], changed = seen, True
        carried = 0
        for vertex in order:
            before = 0
            for prev in pred[vertex]:
                before |= written[prev]
            carried |= exposed[vertex] & ~before
        for name, bit in list(self.bit.items()):
            if carried & bit:
                del self.bit[name], self.uses[name]
        self.active = {name: OrderedSet() for name in self.uses}
        names = list(self.bit.items())
        for state, vertex in flow.vertex.items():
            entered = 0
            for prev in pred[vertex]:
                entered |= written[prev]
            leaving = 0
            for nxt in flow.succ[vertex]:
                leaving |= live[nxt]
            self.live_in[state] = live[vertex] & entered
            self.live_out[state] = leaving & written[vertex]
            here = self.live_in[state] | self.live_out[state] | touched[vertex]
            if here:
                for name, bit in names:
                    if here & bit:
                        self.active[name].add(state)

    def summary(self, state: SDFGState, name: str, accessed: list[nodes.AccessNode]) -> tuple[bool, bool, bool, bool]:
        """Whether ``state`` reads ``name`` before writing all of it, writes all of it, writes it at all, reads it
        before any write. Inside a map the order within the state is unknown: any read there takes the old value too."""
        order = self.order(state)
        desc = self.sdfg.arrays[name]
        full = subsets.Range.from_array(desc)
        written, killers, read = [], [], []
        for node in accessed:
            into = [e for e in state.in_edges(node) if not e.data.is_empty()]
            if into:
                written.append(node)
                if order.scope[node] is None and any(self.full_write(state, e, full) for e in into):
                    killers.append(node)
            if any(not e.data.is_empty() for e in state.out_edges(node)):
                read.append(node)

        def after(node: nodes.AccessNode, sources: list[nodes.AccessNode]) -> bool:
            if order.scope[node] is not None:
                return False
            return node in sources or any(node in order.descendants(s) for s in sources)

        use = not all(after(node, killers) for node in read)
        # A map writing what it reads may write first: only the order inside it could tell.
        makers: OrderedSet = OrderedSet()
        for node in written:
            makers.update(order.pred.get(node, ()) if order.scope[node] is None else (order.top(node),))
        early = not all(
            after(node, written)
            or (order.scope[node] is None and any(s in makers for s in order.succ.get(node, ())))
            or (order.scope[node] is not None and order.top(node) in makers)
            for node in read
        )
        return use, bool(killers), bool(written), early

    def candidate(self, desc: dt.Data) -> bool:
        return (
            desc.transient
            and type(desc) is dt.Array
            and not desc.may_alias
            and desc.storage in self.storages
            and desc.lifetime in REUSABLE_LIFETIMES
            and desc.total_size != 1
        )

    def place(self, name: str, groups: list[Group]) -> Group | None:
        """Add ``name`` to the first group it fits: one holding a copy partner, the copy going with it, then one with
        the same shape."""
        shape = self.sdfg.arrays[name].shape
        partners = self.partners.get(name, ())

        def preference(group: Group) -> tuple[bool, bool]:
            return not any(m in partners for m in group.members), self.sdfg.arrays[group.rep].shape != shape

        for group in sorted(groups, key=preference):
            rep = self.joined_rep(group, name)
            if rep is not None and all(self.disjoint(name, m) for m in group.members):
                group.rep = rep
                group.members.append(name)
                return group
        return None

    def disjoint(self, a: str, b: str) -> bool:
        return all(self.ordered(state, a, b) for state in self.active[a] & self.active[b])

    def ordered(self, state: SDFGState, a: str, b: str) -> bool:
        """Whether one of ``a`` and ``b`` is over in ``state`` before the other is first written there."""
        in_a, in_b = self.live_in[state] & self.bit[a], self.live_in[state] & self.bit[b]
        out_a, out_b = self.live_out[state] & self.bit[a], self.live_out[state] & self.bit[b]
        order = self.order(state)
        uses_a, uses_b = self.uses[a].get(state), self.uses[b].get(state)
        if uses_a is None or uses_b is None:
            return False
        if not (in_b or out_a) and (order.before(uses_a, uses_b) or self.handed_over(state, uses_a, uses_b)):
            return True
        return not (in_a or out_b) and (order.before(uses_b, uses_a) or self.handed_over(state, uses_b, uses_a))

    def handed_over(self, state: SDFGState, first: list[nodes.AccessNode], second: list[nodes.AccessNode]) -> bool:
        """Whether the last access through ``first`` is a whole-array copy that is the first access through ``second``:
        in one buffer the copy reads and writes the same elements, and :meth:`run` drops it."""
        order = self.order(state)
        for edge in state.edges():
            source, target = edge.src, edge.dst
            if not (source in first and target in second) or edge.data.wcr is not None or edge.data.is_empty():
                continue
            if state.out_degree(source) != 1 or state.in_degree(target) != 1 or not self.whole_copy(state, edge):
                continue
            return all(n is source or source in order.descendants(n) for n in first) and all(
                n is target or n in order.descendants(target) for n in second
            )
        return False

    def whole_copy(self, state: SDFGState, edge) -> bool:
        desc, other = self.sdfg.arrays[edge.src.data], self.sdfg.arrays[edge.dst.data]
        read, written = edge.data.get_src_subset(edge, state), edge.data.get_dst_subset(edge, state)
        # A copy memlet without an other subset uses its one subset on both sides.
        read, written = read or written, written or read
        return (
            read is not None
            and len(desc.shape) == len(other.shape)
            and all(self.equal(x, y) for x, y in zip(desc.shape, other.shape))
            and read.covers_precise(subsets.Range.from_array(desc), self.facts)
            and written.covers_precise(subsets.Range.from_array(other), self.facts)
        )

    def order(self, state: SDFGState) -> StateOrder:
        if state not in self.orders:
            self.orders[state] = StateOrder.of(state)
        return self.orders[state]

    def full_write(self, state: SDFGState, edge, full: subsets.Range) -> bool:
        """Whether ``edge``, into an access node, stores every element of ``full``: the propagated memlet alone says
        which elements may be written."""
        if edge.data.is_empty() or edge.data.wcr is not None or edge.data.dynamic:
            return False
        for leaf in state.memlet_tree(edge).root().leaves():
            if leaf.data.dynamic or leaf.data.wcr is not None or isinstance(leaf.src, nodes.NestedSDFG):
                return False
            if isinstance(leaf.src, nodes.LibraryNode) and leaf.src.folds_onto_output():
                return False
            if not self.one_to_one(state, leaf):
                return False
        written = edge.data.get_dst_subset(edge, state)
        return written is not None and written.covers_precise(full, self.facts)

    def one_to_one(self, state: SDFGState, leaf) -> bool:
        """Whether the map parameters around ``leaf`` index its subset by unit steps, each in one axis alone."""
        subset = leaf.data.get_dst_subset(leaf, state)
        if not isinstance(subset, subsets.Range):
            return False
        params: OrderedSet = OrderedSet()
        entry = self.order(state).scope[self.order(state).entry_of.get(leaf.src, leaf.src)]
        while entry is not None:
            params.update(entry.map.params if isinstance(entry, nodes.MapEntry) else ())
            entry = self.order(state).scope[entry]
        seen: OrderedSet = OrderedSet()
        for begin, end, _ in subset.ranges:
            names = OrderedSet()
            for bound in (begin, end):
                bound = bound.expr if isinstance(bound, symbolic.SymExpr) else bound
                found = [s for s in sympy.sympify(bound).free_symbols if s.name in params]
                if len(found) > 1 or (found and abs(sympy.diff(bound, found[0])) != 1):
                    return False
                names.update(s.name for s in found)
            if len(names) > 1 or names & seen:
                return False
            seen |= names
        return True

    def joined_rep(self, group: Group, name: str) -> str | None:
        """The buffer ``group`` shares once ``name`` joins it, or None if ``name`` cannot join."""
        arrays = self.sdfg.arrays
        rep, new = arrays[group.rep], arrays[name]
        if not (
            rep.dtype == new.dtype
            and rep.storage == new.storage
            and rep.location == new.location
            and len(rep.shape) == len(new.shape)
            and rep.alignment == new.alignment
            and rep.pool == new.pool
            and all(self.equal(x, y) for x, y in zip(rep.strides, new.strides))
            and all(self.equal(x, y) for x, y in zip(rep.offset, new.offset))
            and self.equal(rep.start_offset, new.start_offset)
        ):
            return None
        persistent = dtypes.AllocationLifetime.Persistent
        if self.covers(rep, new) and (rep.lifetime == persistent or new.lifetime != persistent):
            chosen = group.rep
        elif self.covers(new, rep):
            chosen = name
        else:
            return None
        # A renamed member a nested SDFG is bound to would hand that connector a differently shaped array.
        shape = arrays[chosen].shape
        for m in [*group.members, name]:
            if m == chosen or m not in self.nested:
                continue
            if not (
                len(arrays[m].shape) == len(shape) and all(self.equal(x, y) for x, y in zip(arrays[m].shape, shape))
            ):
                return None
        if arrays[chosen].lifetime == persistent:
            return chosen
        if any(arrays[m].lifetime == persistent for m in [*group.members, name]) or not self.hoistable(arrays[chosen]):
            return None
        return chosen

    def equal(self, x, y) -> bool:
        """``symbolic.equal`` once per pair: few distinct extents recur across all candidates."""
        if (x, y) not in self.equalities:
            self.equalities[(x, y)] = symbolic.equal(x, y) is True
        return self.equalities[(x, y)]

    def at_least(self, x, y) -> bool:
        if (x, y) not in self.bounds:
            self.bounds[(x, y)] = symbolic.provably_nonnegative(symbolic.simplify(x - y), self.facts)
        return self.bounds[(x, y)]

    def covers(self, big: dt.Array, small: dt.Array) -> bool:
        return all(self.at_least(x, y) for x, y in zip((*big.shape, big.total_size), (*small.shape, small.total_size)))

    def hoistable(self, desc: dt.Array) -> bool:
        """Whether ``desc`` can be allocated on entry to the SDFG: its size reads only symbols bound on entry and
        never reassigned, and every extent is provably nonnegative."""
        if self.hoistable_names is None:
            sd = self.sdfg
            names = OrderedSet(str(s) for s in sd.free_symbols) | OrderedSet(sd.constants_prop.keys())
            for region in sd.all_control_flow_regions():
                for edge in region.edges():
                    names -= OrderedSet(edge.data.assignments.keys())
                if isinstance(region, LoopRegion) and region.loop_variable:
                    names.discard(region.loop_variable)
            self.hoistable_names = names
        facts = self.facts
        return all(
            str(s) in self.hoistable_names
            for extent in (*desc.shape, desc.total_size)
            for s in symbolic.pystr_to_symbolic(extent).free_symbols
        ) and all(symbolic.provably_nonnegative(extent, facts) for extent in desc.shape)
