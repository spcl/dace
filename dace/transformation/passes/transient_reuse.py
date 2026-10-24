# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Sharing one buffer between transient arrays whose uses never overlap."""

from dataclasses import dataclass, field

from dace import SDFG, SDFGState, dtypes, properties, subsets, symbolic
from dace import data as dt
from dace import graphlib as nx
from dace.config import Config
from dace.ordered import OrderedSet
from dace.sdfg import nodes
from dace.sdfg.state import ControlFlowBlock, LoopRegion
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
    """Happens-before of the top-level nodes of one state, every map scope collapsed onto its entry."""

    succ: dict[nodes.Node, OrderedSet] = field(default_factory=dict)
    pred: dict[nodes.Node, OrderedSet] = field(default_factory=dict)
    below: dict[nodes.Node, OrderedSet] = field(default_factory=dict)

    @staticmethod
    def of(state: SDFGState) -> "StateOrder":
        scope = state.scope_dict()
        entry_of = {state.exit_node(n): n for n in state.nodes() if isinstance(n, nodes.EntryNode)}

        def top(node: nodes.Node) -> nodes.Node:
            node = entry_of.get(node, node)
            while scope[node] is not None:
                node = scope[node]
            return node

        order = StateOrder()
        for edge in state.edges():
            src, dst = top(edge.src), top(edge.dst)
            if src is not dst:
                order.succ.setdefault(src, OrderedSet()).add(dst)
                order.pred.setdefault(dst, OrderedSet()).add(src)
        return order

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
        ``second`` node is reached from every reader of every ``first`` node (from the node itself if unread)."""
        writers = [w for b in second for w in self.pred.get(b, ())]
        if not writers or any(not self.pred.get(b) for b in second):
            return False
        return all(
            w in self.descendants(reader) for a in first for reader in (self.succ.get(a) or (a,)) for w in writers
        )


@properties.make_properties
@explicit_cf_compatible
class TransientReuse(ppl.Pass):
    """
    Lets transient arrays whose uses never overlap share one buffer, the largest of the group, by renaming them to it.

    Two transients overlap unless the top-level control flow blocks using them (a loop or a conditional counts as one
    block, throughout which its arrays are live) occupy disjoint ranges of the SDFG's block order, or both are used in
    one state only and every reader of the first precedes every writer of the second there. In a state inside a loop
    the latter also requires each to be fully written before it is read, so neither carries a value across iterations.

    Only plain heap / GPU global arrays outside every map qualify, not read by control flow, not viewed or referenced,
    no WCR target and not ``may_alias``. A shared buffer has the same dtype, storage and strides as its members, is
    provably at least as large in every dimension, and is allocated once: Persistent if any member is, else SDFG
    lifetime. GPU arrays share only when all device work runs on one stream
    (``compiler.cuda.max_concurrent_streams == -1``), so device accesses run in program order.
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
            if inside_map(sd):
                continue
            before = sum(desc.total_size_in_bytes for desc in sd.arrays.values()) if self.verbose else 0
            renamed = BufferSharing(sd, tuple(storages)).run()
            if renamed:
                result[sd.cfg_id] = renamed
                if self.verbose:
                    after = sum(desc.total_size_in_bytes for desc in sd.arrays.values())
                    print(f"{sd.name}: {len(renamed)} transients reuse a buffer, {before} B -> {after} B")
        return result or None


def inside_map(sdfg: SDFG) -> bool:
    """Whether ``sdfg`` runs inside a map scope of some enclosing SDFG."""
    while sdfg.parent_nsdfg_node is not None:
        if sdfg.parent.entry_node(sdfg.parent_nsdfg_node) is not None:
            return True
        sdfg = sdfg.parent_sdfg
    return False


def in_loop(state: SDFGState) -> bool:
    """Whether ``state`` is inside a loop region of its SDFG."""
    block = state.parent_graph
    while not isinstance(block, SDFG):
        if isinstance(block, LoopRegion):
            return True
        block = block.parent_graph
    return False


def top_block(state: SDFGState) -> ControlFlowBlock:
    """The block of the SDFG's root region that contains ``state``."""
    block = state
    while not isinstance(block.parent_graph, SDFG):
        block = block.parent_graph
    return block


@dataclass(slots=True)
class Group:
    rep: str
    members: list[str]


@dataclass(slots=True)
class BufferSharing:
    """One SDFG's candidates, their uses, and the greedy grouping into shared buffers."""

    sdfg: SDFG
    storages: tuple
    uses: dict[str, dict[SDFGState, list[nodes.AccessNode]]] = field(default_factory=dict)
    span: dict[str, tuple[int, int]] = field(default_factory=dict)
    orders: dict[SDFGState, StateOrder] = field(default_factory=dict)
    locals: dict[tuple[SDFGState, str], bool] = field(default_factory=dict)
    hoistable_names: OrderedSet | None = None
    #: Candidates a nested SDFG's connector is bound to: its descriptor must keep matching the array's.
    nested: set[str] = field(default_factory=set)
    #: The SDFG's facts, derived once: nothing changes until the renaming at the end.
    facts: symbolic.Facts | None = None

    def run(self) -> dict[str, str]:
        self.facts = self.sdfg.facts()
        self.collect()
        if len(self.uses) < 2:
            return {}
        groups: list[Group] = []
        for name in sorted(self.uses, key=lambda n: self.span.get(n, (0, 0))[0]):
            group = self.place(name, groups)
            if group is None:
                groups.append(Group(name, [name]))
        renamed = {m: g.rep for g in groups for m in g.members if m != g.rep}
        if not renamed:
            return {}
        for group in groups:
            desc = self.sdfg.arrays[group.rep]
            if len(group.members) > 1 and desc.lifetime != dtypes.AllocationLifetime.Persistent:
                desc.lifetime = dtypes.AllocationLifetime.SDFG
        for state in self.sdfg.states():
            for node in state.data_nodes():
                if node.data in renamed:
                    node.data = renamed[node.data]
            for edge in state.edges():
                if edge.data.data in renamed:
                    edge.data.data = renamed[edge.data.data]
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
        for state in sd.states():
            scope = None
            for node in state.data_nodes():
                desc = sd.arrays[node.data]
                if not self.candidate(desc):
                    continue
                scope = scope or state.scope_dict()
                if scope[node] is not None:
                    excluded.add(node.data)
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
            for edge in state.edges():
                if edge.data.wcr is not None and edge.data.data is not None:
                    excluded.add(edge.data.data)
        for name in excluded:
            self.uses.pop(name, None)
        if not nx.is_directed_acyclic_graph(sd.nx):
            return
        index = {block: i for i, block in enumerate(nx.topological_sort(sd.nx))}
        for name, states in self.uses.items():
            positions = [index[top_block(state)] for state in states]
            self.span[name] = (min(positions), max(positions))

    def candidate(self, desc: dt.Data) -> bool:
        return (
            desc.transient
            and type(desc) is dt.Array
            and not desc.may_alias
            and not desc.pool
            and desc.storage in self.storages
            and desc.lifetime in REUSABLE_LIFETIMES
            and desc.total_size != 1
        )

    def place(self, name: str, groups: list[Group]) -> Group | None:
        """Add ``name`` to the first group it fits, one with the same shape first."""
        shape = self.sdfg.arrays[name].shape
        for same_shape in (True, False):
            for group in groups:
                if (self.sdfg.arrays[group.rep].shape == shape) != same_shape:
                    continue
                rep = self.joined_rep(group, name)
                if rep is not None and all(self.disjoint(name, m) for m in group.members):
                    group.rep = rep
                    group.members.append(name)
                    return group
        return None

    def disjoint(self, a: str, b: str) -> bool:
        if a in self.span and b in self.span:
            (lo_a, hi_a), (lo_b, hi_b) = self.span[a], self.span[b]
            if hi_a < lo_b or hi_b < lo_a:
                return True
        uses_a, uses_b = self.uses[a], self.uses[b]
        if len(uses_a) != 1 or len(uses_b) != 1:
            return False
        state = next(iter(uses_a))
        if state not in uses_b:
            return False
        if in_loop(state) and not (self.state_local(state, a) and self.state_local(state, b)):
            return False
        order = self.order(state)
        return order.before(uses_a[state], uses_b[state]) or order.before(uses_b[state], uses_a[state])

    def order(self, state: SDFGState) -> StateOrder:
        if state not in self.orders:
            self.orders[state] = StateOrder.of(state)
        return self.orders[state]

    def state_local(self, state: SDFGState, name: str) -> bool:
        """Whether every read of ``name`` in ``state`` follows a write of the whole array in ``state``."""
        key = (state, name)
        if key not in self.locals:
            order = self.order(state)
            full = subsets.Range.from_array(self.sdfg.arrays[name])
            killers = [
                node
                for node in self.uses[name][state]
                if any(self.full_write(state, edge, full) for edge in state.in_edges(node))
            ]
            self.locals[key] = all(
                node in killers or any(node in order.descendants(k) for k in killers)
                for node in self.uses[name][state]
                if state.out_degree(node) > 0
            )
        return self.locals[key]

    def full_write(self, state: SDFGState, edge, full: subsets.Range) -> bool:
        if edge.data.is_empty() or edge.data.wcr is not None or edge.data.dynamic:
            return False
        if any(
            leaf.data.dynamic or isinstance(leaf.src, nodes.NestedSDFG)
            for leaf in state.memlet_tree(edge).root().leaves()
        ):
            return False
        written = edge.data.get_dst_subset(edge, state)
        return written is not None and written.covers_precise(full, self.facts)

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
            and all(symbolic.equal(x, y) is True for x, y in zip(rep.strides, new.strides))
            and all(symbolic.equal(x, y) is True for x, y in zip(rep.offset, new.offset))
            and symbolic.equal(rep.start_offset, new.start_offset) is True
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
        if any(
            m != chosen
            and m in self.nested
            and not (
                len(arrays[m].shape) == len(shape)
                and all(symbolic.equal(x, y) is True for x, y in zip(arrays[m].shape, shape))
            )
            for m in [*group.members, name]
        ):
            return None
        if arrays[chosen].lifetime == persistent:
            return chosen
        if any(arrays[m].lifetime == persistent for m in [*group.members, name]) or not self.hoistable(arrays[chosen]):
            return None
        return chosen

    def covers(self, big: dt.Array, small: dt.Array) -> bool:
        facts = self.facts
        return all(
            symbolic.provably_nonnegative(symbolic.simplify(x - y), facts)
            for x, y in zip((*big.shape, big.total_size), (*small.shape, small.total_size))
        )

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
