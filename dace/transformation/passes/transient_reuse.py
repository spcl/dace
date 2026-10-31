# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Sharing one buffer between transient arrays whose uses never overlap."""

from dataclasses import dataclass, field

from dace import SDFG, SDFGState, dtypes, properties, subsets, symbolic
from dace import data as dt
from dace import graphlib as nx
from dace.config import Config
from dace.ordered import OrderedSet
from dace.sdfg import nodes
from dace.sdfg.replace import replace_in_codeblock
from dace.sdfg.state import AbstractControlFlowRegion, ConditionalBlock, ControlFlowBlock, ControlFlowRegion, LoopRegion
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

    Two transients overlap unless, in the lowest control flow region holding the uses of both, they sit in different
    branches of a conditional or in disjoint ranges of the region's block order, or both are used in one state only
    and every reader of the first precedes every writer of the second there. In a region that may run more than once
    (inside a loop) each must also be fully written before it is read in every run, so neither is live across the back
    edge.

    Only plain heap / GPU global arrays qualify, not read by control flow, not viewed or referenced, no WCR target and
    not ``may_alias``; one used inside a map scope only if it is allocated outside it (SDFG or Persistent lifetime, or
    also accessed outside every map in each state using it), and never ordered within a state. A shared buffer has the
    same dtype, storage, pooling and strides as its members, is provably at least as large in every dimension, and is
    allocated once: Persistent if any member is, else SDFG lifetime. At the root SDFG every candidate sized only by
    symbols bound on entry becomes Persistent, shared or not. GPU arrays share only when all device work runs
    on one stream (``compiler.cuda.max_concurrent_streams == -1``), so device accesses run in program order.
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


@dataclass(slots=True)
class Group:
    rep: str
    members: list[str]


@dataclass(slots=True)
class BufferSharing:
    """One SDFG's candidates, their uses, and the greedy grouping into shared buffers. A state's path is the chain of
    blocks from the SDFG's root region down to it; a cyclic region counts as a loop and orders nothing."""

    sdfg: SDFG
    storages: tuple
    uses: dict[str, dict[SDFGState, list[nodes.AccessNode]]] = field(default_factory=dict)
    #: Candidates with an access node inside a map scope: their order within a state is not tracked.
    scoped: OrderedSet = field(default_factory=OrderedSet)
    paths: dict[SDFGState, tuple] = field(default_factory=dict)
    prefix: dict[str, tuple] = field(default_factory=dict)
    rank: dict[ControlFlowBlock, int] = field(default_factory=dict)
    #: Block positions in topological order per acyclic region; cyclic regions are in ``cyclic`` instead.
    positions: dict[ControlFlowRegion, dict[ControlFlowBlock, int]] = field(default_factory=dict)
    cyclic: OrderedSet = field(default_factory=OrderedSet)
    idoms: dict[ControlFlowRegion, dict] = field(default_factory=dict)
    orders: dict[SDFGState, StateOrder] = field(default_factory=dict)
    locals: dict[tuple[SDFGState, str], bool] = field(default_factory=dict)
    hoistable_names: OrderedSet | None = None
    #: Candidates a nested SDFG's connector is bound to: its descriptor must keep matching the array's.
    nested: OrderedSet = field(default_factory=OrderedSet)
    #: The SDFG's facts, derived once: nothing changes until the renaming at the end.
    facts: symbolic.Facts | None = None

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
            elif len(group.members) > 1 and desc.lifetime != persistent:
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
                    names = renamed.keys() - node.in_connectors.keys() - node.out_connectors.keys()
                    code_repl = {name: renamed[name] for name in names if name in node.code.as_string}
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
        self.order_blocks(sd, ())
        for name, states in self.uses.items():
            paths = [self.paths[s] for s in states]
            depth = 0
            while all(len(p) > depth and p[depth] is paths[0][depth] for p in paths):
                depth += 1
            self.prefix[name] = paths[0][:depth]

    def order_blocks(self, region: ControlFlowRegion, path: tuple) -> None:
        """Record every state's path and rank below ``region``, and each region's topological positions."""
        if isinstance(region, ConditionalBlock):
            blocks = region.nodes()
        elif nx.is_directed_acyclic_graph(region.nx):
            blocks = list(nx.topological_sort(region.nx))
            self.positions[region] = {block: i for i, block in enumerate(blocks)}
        else:
            blocks = region.nodes()
            self.cyclic.add(region)
        for block in blocks:
            self.rank[block] = len(self.rank)
            if isinstance(block, SDFGState):
                self.paths[block] = (*path, block)
            elif isinstance(block, AbstractControlFlowRegion):
                self.order_blocks(block, (*path, block))

    def region_at(self, path: tuple, depth: int) -> ControlFlowRegion:
        return path[depth - 1] if depth else self.sdfg

    def children(self, states, depth: int) -> dict[ControlFlowBlock, list[SDFGState]]:
        """``states`` grouped by the block at ``depth`` of their paths."""
        groups: dict[ControlFlowBlock, list[SDFGState]] = {}
        for state in states:
            groups.setdefault(self.paths[state][depth], []).append(state)
        return groups

    def looped(self, path: tuple, depth: int) -> bool:
        """Whether the region at ``depth`` of ``path`` may run more than once per run of the SDFG."""
        return any(isinstance(b, LoopRegion) or b in self.cyclic for b in (self.sdfg, *path[:depth]))

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
        path_a, path_b = self.prefix[a], self.prefix[b]
        depth = 0
        while depth < min(len(path_a), len(path_b)) and path_a[depth] is path_b[depth]:
            depth += 1
        looped = self.looped(path_a, depth)
        if depth == len(path_a) == len(path_b) and depth and isinstance(path_a[-1], SDFGState):
            state = path_a[-1]
            if a in self.scoped or b in self.scoped:
                return False
            if looped and not (self.state_local(state, a) and self.state_local(state, b)):
                return False
            order = self.order(state)
            return order.before(self.uses[a][state], self.uses[b][state]) or order.before(
                self.uses[b][state], self.uses[a][state]
            )
        region = self.region_at(path_a, depth)
        blocks_a, blocks_b = self.children(self.uses[a], depth), self.children(self.uses[b], depth)
        if isinstance(region, ConditionalBlock):
            if any(block in blocks_b for block in blocks_a):
                return False
        else:
            if region in self.cyclic:
                return False
            index = self.positions[region]
            pos_a, pos_b = [index[x] for x in blocks_a], [index[x] for x in blocks_b]
            if not (max(pos_a) < min(pos_b) or max(pos_b) < min(pos_a)):
                return False
        return not looped or (self.covered(a, list(self.uses[a]), depth) and self.covered(b, list(self.uses[b]), depth))

    def covered(self, name: str, states: list[SDFGState], depth: int) -> bool:
        """Whether every run of the region at ``depth`` holding ``states`` writes all of ``name`` before reading it."""
        groups = self.children(states, depth)
        region = self.region_at(self.paths[states[0]], depth)
        if isinstance(region, ConditionalBlock):
            return all(self.covered(name, sub, depth + 1) for sub in groups.values())
        if len(groups) == 1:
            block, sub = next(iter(groups.items()))
            if isinstance(block, SDFGState):
                return self.state_local(block, name)
            return self.covered(name, sub, depth + 1)
        if region in self.cyclic:
            return False
        first = min(groups, key=self.positions[region].__getitem__)
        return (
            isinstance(first, SDFGState)
            and self.state_local(first, name)
            and self.kills(first, name)
            and all(self.dominates(region, first, block) for block in groups if block is not first)
        )

    def dominates(self, region: ControlFlowRegion, first: ControlFlowBlock, block: ControlFlowBlock) -> bool:
        if region not in self.idoms:
            self.idoms[region] = nx.immediate_dominators(region.nx, region.start_block)
        idom = self.idoms[region]
        while block in idom and idom[block] is not block:
            block = idom[block]
            if block is first:
                return True
        return False

    def order(self, state: SDFGState) -> StateOrder:
        if state not in self.orders:
            self.orders[state] = StateOrder.of(state)
        return self.orders[state]

    def kills(self, state: SDFGState, name: str) -> bool:
        """Whether ``state`` writes all of ``name``."""
        full = subsets.Range.from_array(self.sdfg.arrays[name])
        return any(self.full_write(state, e, full) for node in self.uses[name][state] for e in state.in_edges(node))

    def state_local(self, state: SDFGState, name: str) -> bool:
        """Whether every read of ``name`` in ``state`` follows a write of the whole array in ``state``."""
        if name in self.scoped:
            return False
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
            and rep.pool == new.pool
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
        for m in [*group.members, name]:
            if m == chosen or m not in self.nested:
                continue
            if not (
                len(arrays[m].shape) == len(shape)
                and all(symbolic.equal(x, y) is True for x, y in zip(arrays[m].shape, shape))
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
