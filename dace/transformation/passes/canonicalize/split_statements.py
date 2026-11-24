# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Statement fission: split a loop/map body into one perfect nest per independent output *statement*."""

import copy
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from types import ModuleType
from typing import Any, NamedTuple

import sympy

from dace import SDFG, Memlet, dtypes, properties, subsets, symbolic
from dace import data as dt
from dace.optionals import required
from dace.sdfg import nodes
from dace.sdfg.graph import MultiConnectorEdge, SubgraphView
from dace.sdfg.state import ConditionalBlock, ControlFlowRegion, LoopRegion, SDFGState
from dace.transformation import pass_pipeline as ppl
from dace.transformation import transformation
from dace.transformation.passes.analysis import loop_analysis

#: An edge of a body state that carries a value.
ValueEdge = MultiConnectorEdge[Memlet]
#: ``(state, access node, edge)`` of one input read in an output's producer cone.
ConeRead = tuple[SDFGState, nodes.AccessNode, ValueEdge]
#: What the statement analysis walks: a nested SDFG's body, or a loop that is split in place.
Body = SDFG | LoopRegion
#: ``name -> [(state, node)]`` for every data node of a body.
StageIndex = dict[str, list[tuple[SDFGState, nodes.AccessNode]]]
#: Per-output cone reads, memoized across the checks of one split.
Cone = dict[str, list[ConeRead]]


def states_touch_view(states: Iterable[SDFGState], arrays: dict[str, dt.Data]) -> bool:
    """Whether any state names a ``View``: it owns no storage, so a clone cut from its binding
    edge is unbound, and the split refuses rather than try to carry the binding."""
    return any(isinstance(arrays[n.data], dt.View) for st in states for n in st.data_nodes() if n.data in arrays)


def neighbours_touch_view(state: SDFGState, node: nodes.Node) -> bool:
    """Whether an AccessNode adjacent to ``node`` is a ``View`` (the split rewires those edges)."""
    arrays = state.sdfg.arrays
    return any(
        isinstance(arrays[n.data], dt.View)
        for e in state.all_edges(node)
        for n in (e.src, e.dst)
        if isinstance(n, nodes.AccessNode) and n.data in arrays
    )


def is_opaque_code(node: nodes.Node, sdfg: SDFG) -> bool:
    """Whether ``node``'s effects are invisible to statement splitting."""
    if not isinstance(node, nodes.CodeNode) or isinstance(node, nodes.NestedSDFG):
        return False
    if isinstance(node, nodes.Tasklet) and node.language != dtypes.Language.Python:
        return True
    return node.has_side_effects(sdfg)


def has_opaque_code(sdfg: SDFG) -> bool:
    """Whether any node anywhere in ``sdfg`` is opaque to splitting (:func:`is_opaque_code`)."""
    return any(is_opaque_code(node, parent.sdfg) for node, parent in sdfg.all_nodes_recursive())


def _has_conditional(sdfg: SDFG) -> bool:
    """Whether ``sdfg`` contains a ``ConditionalBlock`` (recursively)."""
    return any(isinstance(cfg, ConditionalBlock) for cfg in sdfg.all_control_flow_regions(recursive=True))


def _has_interstate_assignments(sdfg: SDFG) -> bool:
    """Whether ``sdfg`` has any interstate edge carrying an assignment."""
    return any(e.data.assignments for e in sdfg.all_interstate_edges())


def value_edges(edges: Iterable[MultiConnectorEdge[Memlet]]) -> list[ValueEdge]:
    """The edges of ``edges`` that carry a VALUE, dropping the empty-memlet ordering edges."""
    return [e for e in edges if e.data is not None and not e.data.is_empty()]


def producer_edges(state: SDFGState, node: nodes.Node) -> list[ValueEdge]:
    """``node``'s in-edges that carry a VALUE, dropping the empty-memlet ordering edges."""
    return value_edges(state.in_edges(node))


def body_compute_states(loop: LoopRegion) -> list[SDFGState] | None:
    """The loop body's non-empty states, or ``None`` when the body is not a plain chain of them."""
    from dace.sdfg import utils as sdutil

    if any(not isinstance(blk, SDFGState) for blk in loop.nodes()):
        return None
    for e in loop.edges():
        if e.data.assignments or e.data.condition.as_string not in ("1", "True", "(1)"):
            return None
    # Execution order, not insertion order: which state wrote a value before another read it is
    # what :func:`split_order` reads off this list.
    states = [blk for blk in sdutil.dfs_topological_sort(loop, [loop.start_block]) if blk.nodes()]
    return states or None


def body_stage_index(body: Body, index: StageIndex) -> StageIndex:
    """``name -> [(state, node)]`` for every data node in ``body``, in walk order, filled into ``index`` on first use."""
    if not index:
        for st in body.states():
            for n in st.data_nodes():
                index.setdefault(n.data, []).append((st, n))
    return index


def staged_producer_edges(
    body: Body, state: SDFGState, node: nodes.Node, input_names: dict[str, None], stage_index: StageIndex | None = None
) -> list[tuple[SDFGState, ValueEdge]]:
    """``(state, edge)`` for every VALUE producer of ``node``, following one staged transient back into the state that
    wrote it."""
    own = [(state, e) for e in producer_edges(state, node)]
    if own or not isinstance(node, nodes.AccessNode) or node.data in input_names:
        return own
    sites = body_stage_index(body, stage_index if stage_index is not None else {})
    return [(st, e) for st, n in sites.get(node.data, ()) if st is not state for e in producer_edges(st, n)]


#: ``(per-state names, states per name, always-outside, per-loop condition names, loop conditions per
#: name, candidate positions, candidates no state or loop condition names)``.
LocalTransientIndex = tuple[
    dict[int, dict[str, None]],
    Counter[str],
    dict[str, None],
    dict[int, dict[str, None]],
    Counter[str],
    dict[str, int],
    list[str],
]


def local_transient_index(sdfg: SDFG) -> LocalTransientIndex:
    """One walk of ``sdfg`` answering :func:`loop_local_transients` for every loop it contains."""
    state_names: dict[int, dict[str, None]] = {}
    in_states: Counter[str] = Counter()
    for state in sdfg.states():
        names = dict.fromkeys(n.data for n in state.data_nodes())
        state_names[id(state)] = names
        in_states.update(names.keys())
    always: dict[str, None] = {}
    for e in sdfg.all_interstate_edges():
        always.update(dict.fromkeys(s for s in e.data.free_symbols if s in sdfg.arrays))
    cond_names: dict[int, dict[str, None]] = {}
    in_conditions: Counter[str] = Counter()
    for cfr in sdfg.all_control_flow_regions(recursive=True):
        if isinstance(cfr, ConditionalBlock):
            for cond, _ in cfr.branches:
                if cond is not None:
                    always.update(dict.fromkeys(s for s in cond.get_free_symbols() if s in sdfg.arrays))
        elif isinstance(cfr, LoopRegion):
            names = dict.fromkeys(s for s in cfr.loop_condition.get_free_symbols() if s in sdfg.arrays)
            cond_names[id(cfr)] = names
            in_conditions.update(names.keys())
    # Only these names can be local to any loop; the position keeps the answer in ``sdfg.arrays`` order.
    position = {nm: i for i, (nm, desc) in enumerate(sdfg.arrays.items()) if desc.transient and nm not in always}
    unnamed = [nm for nm in position if in_states[nm] == 0 and in_conditions[nm] == 0]
    return state_names, in_states, always, cond_names, in_conditions, position, unnamed


def loop_local_transients(loop: LoopRegion, sdfg: SDFG, index: LocalTransientIndex | None = None) -> dict[str, None]:
    """``sdfg``'s transients that nothing outside ``loop`` observes -- the loop's own temporaries."""
    state_names, in_states, _, cond_names, in_conditions, position, unnamed = index or local_transient_index(sdfg)
    inner: Counter[str] = Counter()
    for state in loop.states():
        names = state_names.get(id(state))
        if names is not None:
            inner.update(names.keys())
    own_cond = cond_names.get(id(loop), {})
    # A name every state outside the loop leaves alone is one the loop's states all hold, or one no state holds.
    local = [
        nm
        for nm, count in inner.items()
        if nm in position and in_states[nm] == count and in_conditions[nm] == (1 if nm in own_cond else 0)
    ]
    local.extend(unnamed)
    local.extend(nm for nm in own_cond if nm in position and in_states[nm] == 0 and in_conditions[nm] == 1)
    return dict.fromkeys(sorted(local, key=position.__getitem__))


@dataclass(slots=True)
class LoopSplitIndex:
    """The whole-SDFG indices the loop split reads, each built on first use for an unmutated ``sdfg``."""

    sdfg: SDFG
    use: tuple[dict[str, set[int]], set[str]] | None = None
    local: LocalTransientIndex | None = None
    rebinding: dict[str, list[LoopRegion]] | None = None

    def use_sites(self) -> tuple[dict[str, set[int]], set[str]]:
        if self.use is None:
            self.use = loop_analysis.symbol_use_sites(self.sdfg)
        return self.use

    def rebinding_loops(self) -> dict[str, list[LoopRegion]]:
        if self.rebinding is None:
            self.rebinding = loop_analysis.rebinding_loops(self.sdfg)
        return self.rebinding

    def local_transients(self, loop: LoopRegion) -> dict[str, None]:
        if self.local is None:
            self.local = local_transient_index(self.sdfg)
        return loop_local_transients(loop, self.sdfg, self.local)

    def clear(self) -> None:
        # Both key on id(), which CPython recycles once a split destroys the blocks.
        self.use = None
        self.local = None
        self.rebinding = None


def _output_dependency(sdfg: SDFG, out_name: str, input_names: dict[str, None]) -> dict[str, None]:
    """Inner array names that feed ``out_name``, excluding pure shared inputs."""
    deps: dict[str, None] = {}
    stage_index: dict = {}
    for state in sdfg.states():
        writers = [n for n in state.nodes() if isinstance(n, nodes.AccessNode) and n.data == out_name]
        seen: dict = {}
        stack = [(state, w) for w in writers]
        while stack:
            st, node = stack.pop()
            if (st, node) in seen:
                continue
            seen[(st, node)] = None
            if isinstance(node, nodes.AccessNode):
                if node.data in input_names:
                    continue
                deps[node.data] = None
            stack.extend((s2, e.src) for s2, e in staged_producer_edges(sdfg, st, node, input_names, stage_index))
    return deps


def output_input_reads(sdfg: SDFG, out_name: str, input_names: dict[str, None]) -> dict[str, None]:
    """Input-connector arrays whose values feed ``out_name``'s computation."""
    reads: dict[str, None] = {}
    stage_index: dict = {}
    for state in sdfg.states():
        writers = [n for n in state.nodes() if isinstance(n, nodes.AccessNode) and n.data == out_name]
        seen: dict = dict.fromkeys((state, w) for w in writers)
        stack = [
            (s2, e.src) for w in writers for s2, e in staged_producer_edges(sdfg, state, w, input_names, stage_index)
        ]
        while stack:
            st, node = stack.pop()
            if (st, node) in seen:
                continue
            seen[(st, node)] = None
            if isinstance(node, nodes.AccessNode) and node.data in input_names:
                reads[node.data] = None
                continue
            stack.extend((s2, e.src) for s2, e in staged_producer_edges(sdfg, st, node, input_names, stage_index))
    return reads


def rmw_confined(body: Body, in_names: dict[str, None], rmw: list[str], groups: list[dict[str, None]]) -> bool:
    """Whether every read-modify-write array is read ONLY by the group that writes it."""
    # A branch guard / index-symbol assignment is duplicated into EVERY clone and re-evaluated there against
    # already-updated data, and it is not dataflow, so the walk below cannot see which array it reads.
    if _has_conditional(body) or _has_interstate_assignments(body):
        return False
    # ``output_input_reads`` walks a single state at a time, so a producer chain that crosses states is invisible to it
    if body.num_states() != 1:
        return False
    for grp in groups:
        reads: dict[str, None] = {}
        for oc in grp:
            reads.update(output_input_reads(body, oc, in_names))
        if any(r in reads for r in rmw if r not in grp):
            return False
    return True


def rmw_stays_in_writer_group(node: nodes.NestedSDFG, rmw: list[str], groups: list[dict[str, None]]) -> bool:
    """:func:`rmw_confined` for a NestedSDFG body, whose inputs are its input connectors."""
    return rmw_confined(node.sdfg, dict.fromkeys(node.in_connectors), rmw, groups)


def edge_subset(edge: ValueEdge, name: str) -> subsets.Subset | None:
    """The subset ``edge`` touches of array ``name``, whichever end of the memlet carries it."""
    if edge.data is None or edge.data.is_empty():
        return None
    return edge.data.subset if edge.data.data == name else edge.data.other_subset


def subset_point(subset: subsets.Subset | None) -> list[symbolic.SymbolicType] | None:
    """The single element ``subset`` addresses, one expression per dimension, else ``None``."""
    if subset is None:
        return None
    point = []
    for rb, re, _ in subset.ndrange():
        rb = rb.expr if isinstance(rb, symbolic.SymExpr) else rb
        re = re.expr if isinstance(re, symbolic.SymExpr) else re
        if symbolic.simplify(re - rb) != 0:
            return None
        point.append(rb)
    return point


def access_offset(read_subset: subsets.Subset | None, write_subset: subsets.Subset | None) -> int | None:
    """Sign of ``read - write`` as a constant iteration offset, or ``None`` when undecidable."""
    read, write = subset_point(read_subset), subset_point(write_subset)
    if read is None or write is None or len(read) != len(write):
        return None
    for r, w in zip(read, write):
        try:
            # Same-named symbols from two memlets can be different INSTANCES (the dtype is not part
            # of symbol identity), and then ``i + 1 - i`` stays an unevaluated Add instead of 1 --
            # the offset reads as unknown and every split is refused. Equalize before subtracting.
            r, w = symbolic.equalize_symbols(r, w)
            diff = symbolic.simplify(r - w)
        except (TypeError, ValueError, AttributeError):
            return None
        if not isinstance(diff, sympy.Integer):
            return None
        if diff != 0:
            return 1 if diff > 0 else -1
    return 0


def iteration_direction(
    read_subset: subsets.Subset | None, write_subset: subsets.Subset | None, loop: LoopRegion
) -> int | None:
    """``+1`` when the store at ``write_subset`` that hits the element read at ``read_subset`` runs in a LATER
    iteration than the read, ``-1`` when it ran in an EARLIER one, ``None`` when undecidable."""
    stride = loop_analysis.get_loop_stride(loop)
    read, write = subset_point(read_subset), subset_point(write_subset)
    if stride is None or read is None or write is None or len(read) != len(write):
        return None
    stride = symbolic.simplify(stride)
    if not isinstance(stride, sympy.Integer) or stride == 0:
        return None
    distance = None
    for r, w in zip(read, write):
        try:
            r, w = symbolic.equalize_symbols(r, w)
            diff = symbolic.simplify(r - w)
            ivs = [s for s in w.free_symbols if str(s) == loop.loop_variable]
            coef = symbolic.simplify(sympy.diff(w, ivs[0])) if ivs else sympy.Integer(0)
        except (TypeError, ValueError, AttributeError):
            return None
        if not isinstance(diff, sympy.Integer) or not isinstance(coef, sympy.Integer):
            return None
        if coef == 0:
            if diff != 0:
                return None
            continue
        # The store at iteration ``i + d`` hits ``w + coef * d``, so the read element needs ``d = diff / coef``.
        if distance is not None and distance != diff / coef:
            return None
        distance = diff / coef
    if not distance:
        return None
    return 1 if distance * stride > 0 else -1


def output_read_edges(sdfg: SDFG, out_name: str, input_names: dict[str, None]) -> list[ConeRead]:
    """``(state, access_node, edge)`` for every input array read in ``out_name``'s producer cone."""
    found = []
    stage_index: dict = {}
    for state in sdfg.states():
        writers = [n for n in state.nodes() if isinstance(n, nodes.AccessNode) and n.data == out_name]
        seen: dict = dict.fromkeys((state, w) for w in writers)
        stack = [pair for w in writers for pair in staged_producer_edges(sdfg, state, w, input_names, stage_index)]
        while stack:
            st, edge = stack.pop()
            node = edge.src
            if isinstance(node, nodes.AccessNode) and node.data in input_names:
                found.append((st, node, edge))
                continue
            if (st, node) in seen:
                continue
            seen[(st, node)] = None
            stack.extend(staged_producer_edges(sdfg, st, node, input_names, stage_index))
    return found


def read_cone(body: Body, out_name: str, in_names: dict[str, None], memo: Cone | None = None) -> list[ConeRead]:
    """:func:`output_read_edges` memoised per output name."""
    if memo is None:
        return output_read_edges(body, out_name, in_names)
    if out_name not in memo:
        memo[out_name] = output_read_edges(body, out_name, in_names)
    return memo[out_name]


def write_subsets(body: Body, name: str) -> list[subsets.Subset]:
    """Every subset of ``name`` that ``body`` STORES to."""
    out: list[subsets.Subset] = []
    for state in body.states():
        for node in state.data_nodes():
            if node.data != name:
                continue
            out.extend(sub for sub in (edge_subset(e, name) for e in producer_edges(state, node)) if sub is not None)
    return out


def suppress_writes(body: SDFG, name: str) -> None:
    """Delete the stores to ``name`` inside ``body``, keeping every load of it."""
    from dace.sdfg import utils as sdutil

    for state in body.states():
        for node in [n for n in state.data_nodes() if n.data == name]:
            # Cutting one store's dangling path can take another access node of ``name`` with it (a copy between two).
            if node not in state.nodes():
                continue
            for e in producer_edges(state, node):
                sdutil.remove_edge_and_dangling_path(state, e)
            if node in state.nodes() and state.degree(node) == 0:
                state.remove_node(node)


def drop_dataless_access_nodes(body: SDFG) -> None:
    """Splice out what the pruning left with no live value traffic, keeping the order it carried."""
    while True:
        removed = False
        read_for_value = {
            n.data for state in body.states() for n in state.data_nodes() if value_edges(state.out_edges(n))
        }
        for state in body.states():
            for node in [n for n in state.data_nodes() if not value_edges(state.out_edges(n))]:
                if value_edges(state.in_edges(node)) and not (
                    body.arrays[node.data].transient and node.data not in read_for_value
                ):
                    continue
                # Only the EMPTY in-edges carry order worth keeping
                preds = [e.src for e in state.in_edges(node) if e.data.is_empty()]
                succs = [e.dst for e in state.out_edges(node)]
                state.remove_node(node)
                for src in preds:
                    for dst in succs:
                        state.add_nedge(src, dst, Memlet())
                removed = True
            # A CodeNode whose out-edges are all EMPTY computes a value nobody reads: the node is dead but its ORDERING
            # is not, so it is spliced like an access node rather than dropped.
            for producer in [
                n for n in state.nodes() if isinstance(n, nodes.CodeNode) and not value_edges(state.out_edges(n))
            ]:
                preds = [e.src for e in state.in_edges(producer) if e.data is not None and e.data.is_empty()]
                succs = [e.dst for e in state.out_edges(producer)]
                state.remove_node(producer)
                for src in preds:
                    for dst in succs:
                        state.add_nedge(src, dst, Memlet())
                removed = True
        if not removed:
            return


def rmw_analyzable(body: Body) -> bool:
    """Whether the cross-group read/write picture of ``body`` can be read off its dataflow at all."""
    return not _has_conditional(body) and not _has_interstate_assignments(body)


def iteration_distinct(subsets: Sequence[subsets.Subset | None], loop_var: str) -> bool:
    """Whether ``subsets`` name a DIFFERENT element on every iteration of ``loop_var``."""
    return bool(subsets) and all(sub is not None and loop_var in (str(s) for s in sub.free_symbols) for sub in subsets)


def carries_across_iterations(body: LoopRegion, name: str, in_names: dict[str, None], cone: Cone | None = None) -> bool:
    """Whether ``name``'s read-modify-write actually crosses iterations."""
    stores = write_subsets(body, name)
    if not iteration_distinct(stores, body.loop_variable):
        return True
    for state, node, edge in read_cone(body, name, in_names, cone):
        if node.data == name and {access_offset(edge_subset(edge, name), w) for w in stores} != {0}:
            return True
    return False


def sees_written_value(states: list[SDFGState], name: str, state: SDFGState, node: nodes.AccessNode) -> bool:
    """Whether the read at ``node`` sees ``name``'s store from the SAME iteration."""
    if producer_edges(state, node):
        return True
    if state not in states:
        return False
    before = states[: states.index(state)]
    return any(producer_edges(st, n) for st in before for n in st.data_nodes() if n.data == name)


def split_order(
    body: LoopRegion, in_names: dict[str, None], rmw: list[str], groups: list[dict[str, None]], cone: Cone | None = None
) -> list[dict[str, None]] | None:
    """``groups`` re-ordered so the split is legal, or ``None`` when no order is."""
    if len(groups) < 2 or not rmw_analyzable(body):
        return None
    states = body_compute_states(body) or []
    # ``before[i]`` are the groups that must run after group ``i``
    after: list[dict[int, None]] = [{} for _ in groups]
    constrained = False
    for name in rmw:
        writer = next((i for i, grp in enumerate(groups) if name in grp), None)
        if writer is None:
            return None
        stores: list | None = None
        for reader, grp in enumerate(groups):
            if reader == writer:
                continue
            for state, node, edge in [r for oc in grp for r in read_cone(body, oc, in_names, cone)]:
                if node.data != name:
                    continue
                if stores is None:
                    # Asked here rather than per name: a name no OTHER group reads imposes no order, so what its writes
                    # look like never comes up.
                    stores = write_subsets(body, name)
                    if not iteration_distinct(stores, body.loop_variable):
                        return None
                read = edge_subset(edge, name)
                offsets = {access_offset(read, w) for w in stores}
                if len(offsets) != 1 or None in offsets:
                    return None
                offset = offsets.pop()
                if offset == 0:
                    # Same element, same iteration: whether the read already sees the write decides.
                    first = writer if sees_written_value(states, name, state, node) else reader
                else:
                    # The element offset alone flips under a reversed subscript (``a[N - i]``) or a negative stride;
                    # what decides is whether the store hitting the read element runs before or after the read.
                    directions = {iteration_direction(read, w, body) for w in stores}
                    if len(directions) != 1 or None in directions:
                        return None
                    first = reader if required(directions.pop()) > 0 else writer
                after[first][writer if first == reader else reader] = None
                constrained = True
    if not constrained:
        return None
    order = topological_group_order(after)
    return None if order is None else [groups[i] for i in order]


def topological_group_order(after: list[dict[int, None]]) -> list[int] | None:
    """``after[i]`` naming the groups that must follow group ``i``, the groups in a legal order, or ``None`` when the
    constraints contradict each other."""
    pending = [0] * len(after)
    for succs in after:
        for j in succs:
            pending[j] += 1
    order: list[int] = []
    ready = [i for i, n in enumerate(pending) if n == 0]
    while ready:
        i = min(ready)
        ready.remove(i)
        order.append(i)
        for j in after[i]:
            pending[j] -= 1
            if pending[j] == 0:
                ready.append(j)
    return order if len(order) == len(after) else None


def merge_carried_groups(
    body: LoopRegion, groups: list[dict[str, None]], rmw: list[str], in_names: dict[str, None], cone: Cone | None = None
) -> list[dict[str, None]]:
    """``groups`` with the producer of a CARRIED value and everything that reads it merged into one."""
    merged = list(groups)
    for name in rmw:
        if iteration_distinct(write_subsets(body, name), body.loop_variable):
            continue
        writer = next((i for i, grp in enumerate(merged) if name in grp), None)
        if writer is None:
            continue
        readers = [
            i
            for i, grp in enumerate(merged)
            if i != writer and any(n.data == name for oc in grp for st, n, e in read_cone(body, oc, in_names, cone))
        ]
        if not readers:
            continue
        for i in readers:
            merged[writer].update(merged[i])
        merged = [grp for i, grp in enumerate(merged) if i not in readers]
    return merged


def independent_groups(body: Body, out_names: list[str], in_names: dict[str, None]) -> list[dict[str, None]]:
    """Partition ``out_names`` so two outputs share a group iff their producer cones overlap."""
    dep = {oc: _output_dependency(body, oc, in_names) for oc in out_names}
    parent = {oc: oc for oc in out_names}

    def find(x: str) -> str:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for i, a in enumerate(out_names):
        for b in out_names[i + 1 :]:
            if any(s in dep[b] for s in dep[a]):
                parent[find(a)] = find(b)
    groups: dict[str, dict[str, None]] = {}
    for oc in out_names:
        groups.setdefault(find(oc), {})[oc] = None
    return list(groups.values())


def plain_map_outputs(state: SDFGState, xit: nodes.MapExit) -> list[str] | None:
    """The distinct containers written through ``xit``, or ``None`` when one is written by a reduction."""
    out_names: list[str] = []
    for e in state.in_edges(xit):
        if e.data is None or e.data.data is None:
            continue
        if e.data.wcr is not None:
            return None
        if e.data.data not in out_names:
            out_names.append(e.data.data)
    return out_names


def rewrites_a_global_it_reads(
    cfg: SDFG, state: SDFGState, entry: nodes.MapEntry, xit: nodes.MapExit, inner: list[nodes.Node]
) -> bool:
    """Whether the map reads a non-transient array it also writes (``t = A[i]; A[i] = t + B[i]; C[i] = t*2``): a
    clone per output would recompute ``t`` after another clone overwrote ``A[i]``."""
    read_arrays = dict.fromkeys(
        e.data.data
        for e in state.in_edges(entry)
        if e.data is not None and e.data.data is not None and not cfg.arrays[e.data.data].transient
    )
    write_arrays = dict.fromkeys(
        e.data.data
        for e in state.in_edges(xit)
        if e.data is not None and e.data.data is not None and not cfg.arrays[e.data.data].transient
    )
    write_arrays.update(
        dict.fromkeys(
            n.data
            for n in inner
            if isinstance(n, nodes.AccessNode) and state.in_degree(n) > 0 and not cfg.arrays[n.data].transient
        )
    )
    return any(a in write_arrays for a in read_arrays)


def clonable_leaf_body(cfg: SDFG, scope: Any, inner: list[nodes.Node]) -> bool:
    """Whether a map body can be recomputed per output: no view, no nested map or NestedSDFG, no opaque code."""
    if any(
        isinstance(cfg.arrays[n.data], dt.View)
        for n in scope.nodes()
        if isinstance(n, nodes.AccessNode) and n.data in cfg.arrays
    ):
        return False
    if not inner or any(isinstance(n, (nodes.NestedSDFG, nodes.MapEntry, nodes.MapExit)) for n in inner):
        return False
    return not any(is_opaque_code(n, cfg) for n in inner)


def pruned_group_clone(
    node: nodes.NestedSDFG, grp: dict[str, None], simplify_cls: type[ppl.Pass]
) -> tuple[SDFG, list[str]] | None:
    """A copy of ``node``'s SDFG pruned to the outputs of ``grp``, and the input connectors it still reads."""
    # Pruned before it is wired: the surviving descriptors decide the connector set.
    clone_sdfg = copy.deepcopy(node.sdfg)
    for other in [c for c in node.out_connectors if c not in grp]:
        suppress_writes(clone_sdfg, other)
    drop_dataless_access_nodes(clone_sdfg)
    for arr in [c for c in clone_sdfg.arrays if c not in grp and c not in node.in_connectors]:
        desc = clone_sdfg.arrays[arr]
        if not desc.transient:
            desc.transient = True
    # Detached, the clone has no cfg list for SimplifyPass to root at.
    clone_sdfg.reset_cfg_list()
    simplify_cls().apply_pass(clone_sdfg, {})
    # Kept: the inputs the clone still moves a value through.
    used = {
        n.data for st in clone_sdfg.states() for n in st.data_nodes() if value_edges(st.in_edges(n) + st.out_edges(n))
    }
    kept_in = [c for c in node.in_connectors if c in clone_sdfg.arrays and c in used]
    if any(o not in clone_sdfg.arrays for o in grp):
        return None
    for leftover in [c for c in clone_sdfg.arrays if c not in kept_in and c not in grp]:
        clone_sdfg.arrays[leftover].transient = True
    return clone_sdfg, kept_in


def splittable_loop_states(loop: LoopRegion, sdfg: SDFG, index: LoopSplitIndex) -> list[SDFGState] | None:
    """The body's compute states, when every clone can recompute the body and sweep the same iterations: plain
    dataflow, no NestedSDFG, view or opaque code, bounds from symbols alone, and a counter nothing outside reads."""
    states = body_compute_states(loop)
    if states is None:
        return None
    body_nodes = [n for state in states for n in state.nodes()]
    # A NestedSDFG body is _replicate_components' shape: it splits in place, no outlining needed.
    if any(isinstance(n, nodes.NestedSDFG) for n in body_nodes):
        return None
    if any(is_opaque_code(n, sdfg) for n in body_nodes):
        return None
    if states_touch_view(states, sdfg.arrays):
        return None
    return states if clonable_counter(loop, sdfg, index) else None


def clonable_counter(loop: LoopRegion, sdfg: SDFG, index: LoopSplitIndex) -> bool:
    """Whether every clone of ``loop`` sweeps the same iterations: bounds from symbols alone, and a counter nothing
    outside reads."""
    if not loop.loop_variable or not loop.init_statement:
        return False
    if any(s in sdfg.arrays for s in loop.loop_condition.get_free_symbols()):
        return False
    init = loop_analysis.get_init_assignment(loop)
    if init is None or loop.loop_variable in symbolic.free_symbols_and_functions(init):
        return False
    # The outlining exports a counter read outside as an output every clone would have to write.
    return not used_outside(loop.loop_variable, loop, sdfg, index)


def used_outside(name: str, loop: LoopRegion, sdfg: SDFG, index: LoopSplitIndex) -> bool:
    use_sites, descriptor_symbols = index.use_sites()
    return loop_analysis.counter_used_outside_loop(
        name, loop, sdfg, use_sites, descriptor_symbols, index.rebinding_loops()
    )


class Access(NamedTuple):
    """One access of a guarded body: the site it happens at (body order), its container and subset."""

    site: int
    data: str
    subset: subsets.Subset | None


class GuardedBody(NamedTuple):
    """What :func:`guarded_body` reads off a loop body of states and guarded branches."""

    #: per output, its stores
    stores: dict[str, list[Access]]
    #: per output, the reads its stores' dataflow makes
    cone_reads: dict[str, list[Access]]
    #: reads of conditions and interstate edges, which every clone repeats
    control_reads: list[Access]


def chain_sites(region: ControlFlowRegion) -> list[tuple[Any, ControlFlowRegion]] | None:
    """``(site, region)`` for the blocks and interstate edges of ``region`` in execution order, descending into every
    branch; ``None`` unless every region is a chain of unconditional edges holding only states and conditional
    blocks."""
    from dace.sdfg import utils as sdutil

    if any(len(region.out_edges(b)) > 1 for b in region.nodes()):
        return None
    if any(not e.data.is_unconditional() for e in region.edges()):
        return None
    sites: list[tuple[Any, ControlFlowRegion]] = []
    for block in sdutil.dfs_topological_sort(region, [region.start_block]):
        if isinstance(block, SDFGState):
            sites.append((block, region))
        elif isinstance(block, ConditionalBlock):
            sites.append((block, region))
            for branch in block.sub_regions():
                inner = chain_sites(branch)
                if inner is None:
                    return None
                sites.extend(inner)
        else:
            return None
        sites.extend((e, region) for e in region.out_edges(block))
    return sites


def within(region: Any, outer: ControlFlowRegion) -> bool:
    while region is not None and region is not outer:
        region = region.parent_graph
    return region is outer


def branch_conditions(block: ConditionalBlock) -> list[properties.CodeBlock]:
    return [cond for cond, branch in block.branches if cond is not None]


def site_reads(site: Any, arrays: dict[str, dt.Data]) -> list[Memlet]:
    """The container reads of an interstate edge or of a conditional block's conditions."""
    from dace.sdfg.sdfg import memlets_in_ast

    if isinstance(site, ConditionalBlock):
        return [
            m for cond in branch_conditions(site) for m in memlets_in_ast(cond.code[0], arrays, include_scalars=True)
        ]
    return site.data.get_read_memlets(arrays, include_scalars=True)


def cone_reads(state: SDFGState, store: nodes.AccessNode, site: int, local: dict[str, None]) -> list[Access] | None:
    """The reads feeding ``store`` within ``state``; ``None`` when a loop-local temporary on the way is read before
    this state writes it (it would carry a value between iterations)."""
    found = []
    seen = {store}
    stack = [e.src for e in producer_edges(state, store)]
    while stack:
        node = stack.pop()
        if node in seen:
            continue
        seen.add(node)
        if isinstance(node, nodes.AccessNode):
            if node.data in local:
                if not producer_edges(state, node):
                    return None
            else:
                found.extend(
                    Access(site, node.data, e.data.get_src_subset(e, state)) for e in value_edges(state.out_edges(node))
                )
        stack.extend(e.src for e in producer_edges(state, node))
    return found


def guarded_body(loop: LoopRegion, sdfg: SDFG, index: LoopSplitIndex) -> GuardedBody | None:
    """The accesses of a loop body that holds a guarded branch, when each clone can recompute what its statement
    needs: states and conditional blocks only, plain dataflow, symbols each iteration assigns before it reads them
    and nothing after the loop reads; ``None`` otherwise."""
    sites = chain_sites(loop)
    if sites is None or not any(isinstance(site, ConditionalBlock) for site, region in sites):
        return None
    states = [site for site, region in sites if isinstance(site, SDFGState)]
    body_nodes = [n for state in states for n in state.nodes()]
    if any(isinstance(n, nodes.NestedSDFG) or is_opaque_code(n, sdfg) for n in body_nodes):
        return None
    if states_touch_view(states, sdfg.arrays) or not clonable_counter(loop, sdfg, index):
        return None
    local = index.local_transients(loop)
    assigned = loop_assigned(loop)
    body = GuardedBody({}, {}, [])
    # symbol -> (site, region) of its one assignment
    assigned_at: dict[str, tuple[int, ControlFlowRegion]] = {}
    for position, (site, region) in enumerate(sites):
        if isinstance(site, SDFGState):
            read_names = {str(sym) for sym in site.free_symbols}
        elif isinstance(site, ConditionalBlock):
            read_names = {str(sym) for cond in branch_conditions(site) for sym in cond.get_free_symbols()}
        else:
            read_names = set(site.data.read_symbols())
        # A symbol read where its iteration's assignment may not have run holds an earlier iteration's value
        if any(name not in assigned_at or not within(region, assigned_at[name][1]) for name in read_names & assigned):
            return None
        if isinstance(site, SDFGState):
            for node in site.data_nodes():
                stores = producer_edges(site, node)
                if not stores or node.data in local:
                    continue
                if any(e.data.wcr is not None for e in stores):
                    return None
                body.stores.setdefault(node.data, []).extend(
                    Access(position, node.data, e.data.get_dst_subset(e, site)) for e in stores
                )
                reads = cone_reads(site, node, position, local)
                if reads is None:
                    return None
                body.cone_reads.setdefault(node.data, []).extend(reads)
            continue
        reads = site_reads(site, sdfg.arrays)
        if any(m.data in local for m in reads):
            return None
        body.control_reads.extend(Access(position, m.data, m.subset) for m in reads)
        if not isinstance(site, ConditionalBlock):
            for name in site.data.assignments:
                if name in assigned_at or used_outside(name, loop, sdfg, index):
                    return None
                assigned_at[name] = (position, region)
    return body


def loop_assigned(loop: LoopRegion) -> set[str]:
    return {name for e in loop.all_interstate_edges() for name in e.data.assignments}


def guarded_groups(loop: LoopRegion, sdfg: SDFG, index: LoopSplitIndex) -> list[dict[str, None]] | None:
    """One group per output of a loop body with a guarded branch, in an order the clones may run one after the other,
    or ``None``. A clone repeats every guard and every symbol assignment, which is what expands a scalar temporary
    per statement; an output another clone reads must be read at the one element it stores, in the same iteration,
    either before every store of it (that clone runs first) or after every store (that clone runs after)."""
    body = guarded_body(loop, sdfg, index)
    if body is None or len(body.stores) < 2:
        return None
    outputs = sorted(body.stores, key=lambda name: min(a.site for a in body.stores[name]))
    after: list[dict[int, None]] = [{} for name in outputs]
    for writer, name in enumerate(outputs):
        stores = body.stores[name]
        for reader, other in enumerate(outputs):
            if reader == writer:
                continue
            reads = [a for a in body.control_reads + body.cone_reads[other] if a.data == name]
            if not reads:
                continue
            if not iteration_distinct([a.subset for a in stores], loop.loop_variable):
                return None
            if any({access_offset(r.subset, w.subset) for w in stores} != {0} for r in reads):
                return None
            first, last = min(a.site for a in stores), max(a.site for a in stores)
            if all(r.site < first for r in reads):
                after[reader][writer] = None
            elif all(r.site > last for r in reads):
                after[writer][reader] = None
            else:
                return None
    order = topological_group_order(after)
    return None if order is None else [{outputs[i]: None} for i in order]


def body_reads_and_writes(states: list[SDFGState], local: Any) -> tuple[dict[str, None], dict[str, None]] | None:
    """The containers the body reads and writes by value, or ``None`` on a WCR store or a loop-local temporary read
    before the body writes it (a carried value no single clone holds)."""
    # Empty memlets are ORDERING edges, so they neither read nor write: counting one as a write
    # turns a plain input into a read-modify-write output and refuses the whole split.
    reads: dict[str, None] = {}
    writes: dict[str, None] = {}
    for state in states:
        for n in state.data_nodes():
            if value_edges(state.out_edges(n)):
                reads[n.data] = None
            stores = producer_edges(state, n)
            if not stores:
                continue
            writes[n.data] = None
            # A WCR store is a reduction; it is not replicable per group.
            if any(e.data.wcr is not None for e in stores):
                return None
    # A private temporary read before anything writes it holds the previous iteration's value.
    for state in states:
        for n in state.data_nodes():
            if n.data in local and value_edges(state.out_edges(n)) and not sees_written_value(states, n.data, state, n):
                return None
    return reads, writes


@transformation.explicit_cf_compatible
@properties.make_properties
class SplitStatements(ppl.Pass):
    """Split a loop/map body into one perfect nest per independent output statement -- including statements inside ifs
    and gather/scatter accesses (per-output NestedSDFG replication)."""

    CATEGORY: str = "Canonicalization"

    split_maps = properties.Property(
        dtype=bool, default=False, desc="Also fission a straight-line multi-global-output map into one map per output."
    )

    split_loops = properties.Property(
        dtype=bool, default=True, desc="Distribute a straight-line multi-output loop into one loop per output."
    )

    def __init__(self, split_maps: bool = False, split_loops: bool = True) -> None:
        super().__init__()
        self.split_maps = split_maps
        self.split_loops = split_loops

    def modifies(self) -> ppl.Modifies:
        return ppl.Modifies.Everything

    def should_reapply(self, modified: ppl.Modifies) -> bool:
        return False

    def depends_on(self) -> list[type[ppl.Pass] | ppl.Pass]:
        return []

    def apply_pass(self, sdfg: SDFG, _pipeline_results: dict[str, Any]) -> int | None:
        count = 0
        # (1) Statements inside ifs + gather/scatter: replicate the blocking
        #     NestedSDFG once per independent output group so MapFission splits it.
        count += self._replicate_components(sdfg)
        # (1b) Straight-line map with >=2 global outputs -> one map per output
        #      (opt-in; leaves the canon pipeline unchanged).
        if self.split_maps:
            count += self._split_map_bodies(sdfg)
        # (1c) Straight-line LOOP with >=2 independent outputs -> one loop per output.
        if self.split_loops:
            count += self._split_loop_bodies(sdfg)
        return count or None

    # (1) Per-output replication of a MapFission-blocking NestedSDFG (conditional / indirection-symbol).

    def _replicate_components(self, sdfg: SDFG) -> int:
        from dace.transformation.passes.simplify import SimplifyPass

        count = 0
        # Innermost first: splitting a node retires its whole nested SDFG tree, the clones hold copies of it
        for nsdfg in reversed(list(sdfg.all_sdfgs_recursive())):
            for state in list(nsdfg.states()):
                for node in [n for n in state.nodes() if isinstance(n, nodes.NestedSDFG)]:
                    groups = self._independent_output_groups(state, node)
                    if groups is None or len(groups) < 2:
                        continue
                    self._split(nsdfg, state, node, groups, SimplifyPass, rmw_read_is_dead=True)
                    count += 1
        return count

    @staticmethod
    def _independent_output_groups(state: SDFGState, node: nodes.NestedSDFG) -> list[dict[str, None]] | None:
        """Partition ``node``'s output connectors into independent groups, or ``None`` to refuse."""
        out_conns = list(node.out_connectors)
        if len(out_conns) < 2:
            return None
        if states_touch_view(node.sdfg.states(), node.sdfg.arrays) or neighbours_touch_view(state, node):
            return None
        # No WCR on the boundary (it would not be replicable per group).
        for e in state.out_edges(node):
            if e.data is None or e.data.wcr is not None:
                return None
        # In-place read-modify-write: output arrays that are ALSO input connectors.
        rmw = [c for c in out_conns if c in node.in_connectors]
        # The same carry under two connector names (a privatized ``_nnr_rwout_a`` writing back the ``a`` an input
        # reads) is invisible to that test and to the per-group dependency analysis: refuse it outright.
        read_outer = {e.data.data for e in state.in_edges(node) if e.data is not None and e.dst_conn not in rmw}
        if any(
            e.data is not None and e.data.data in read_outer and e.src_conn not in node.in_connectors
            for e in state.out_edges(node)
        ):
            return None
        # A black-box body is not analyzable: ``_output_dependency`` reads the memlets, which
        # do not describe an opaque node's effects, and ``_split`` would then duplicate them.
        if has_opaque_code(node.sdfg):
            return None
        in_names = dict.fromkeys(node.in_connectors)
        result = independent_groups(node.sdfg, out_conns, in_names)
        if rmw and not rmw_stays_in_writer_group(node, rmw, result):
            return None
        return result

    @staticmethod
    def _split(
        parent_sdfg: SDFG,
        state: SDFGState,
        node: nodes.NestedSDFG,
        groups: list[dict[str, None]],
        simplify_cls: type[ppl.Pass],
        rmw_read_is_dead: bool = False,
        cut_other_stores: bool = False,
    ) -> None:
        """Clone ``node`` once per group, prune each, rewire, drop original."""
        in_edges = list(state.in_edges(node))
        out_edges = list(state.out_edges(node))
        for grp in groups:
            # Both the read and the write of an RMW array are dead in a group that does not write it.
            if rmw_read_is_dead:
                kept_in = [c for c in node.in_connectors if c not in node.out_connectors or c in grp]
            else:
                kept_in = list(node.in_connectors)
            clone_sdfg = copy.deepcopy(node.sdfg)
            # The clone is still detached, so its cfg list is empty and the passes below cannot find a root.
            clone_sdfg.reset_cfg_list()
            if cut_other_stores:
                for other in [c for c in node.out_connectors if c not in grp]:
                    suppress_writes(clone_sdfg, other)
                drop_dataless_access_nodes(clone_sdfg)
                simplify_cls().apply_pass(clone_sdfg, {})
                # Cutting a store leaves the inputs only that statement read with no reader, and an input connector
                # nothing reads makes the inliner materialise an access node the next pass to derive read sets from
                # memlets does not know about (``LoopToMap`` on TSVC ``s211``).
                used = {
                    n.data
                    for st in clone_sdfg.states()
                    for n in st.data_nodes()
                    if value_edges(st.in_edges(n) + st.out_edges(n))
                }
                kept_in = [c for c in kept_in if c in clone_sdfg.arrays and c in used]
            # dicts/sorted, not sets: these become the clone's connector dicts, which are
            # observable in validation and codegen order.
            clone = state.add_nested_sdfg(
                clone_sdfg,
                inputs=dict.fromkeys(kept_in),
                outputs=dict.fromkeys(sorted(grp)),
                symbol_mapping=copy.deepcopy(node.symbol_mapping),
            )
            for e in in_edges:
                if e.dst_conn is not None and e.dst_conn not in kept_in:
                    continue
                state.add_edge(e.src, e.src_conn, clone, e.dst_conn, copy.deepcopy(e.data))
            for e in out_edges:
                # ``src_conn is None`` is an ORDERING edge, not an output: every clone inherits it.
                if e.src_conn is not None and e.src_conn not in grp:
                    continue
                state.add_edge(clone, e.src_conn, e.dst, e.dst_conn, copy.deepcopy(e.data))
            for arr in [c for c in clone_sdfg.arrays if c not in grp and c not in kept_in]:
                desc = clone_sdfg.arrays[arr]
                if not desc.transient:
                    desc.transient = True
            simplify_cls().apply_pass(clone_sdfg, {})
        for e in in_edges + out_edges:
            state.remove_edge(e)
        state.remove_node(node)

    # (1b) Straight-line map statement fission (opt-in via split_maps).

    def _split_map_bodies(self, sdfg: SDFG) -> int:
        """Map analogue of the loop statement split: fission a STRAIGHT-LINE map writing >=2 global outputs into one map
        per output, so each is a self-contained statement."""
        from dace.sdfg.graph import SubgraphView
        from dace.transformation import helpers
        from dace.transformation.interstate import InlineSDFG
        from dace.transformation.passes.simplify import SimplifyPass

        count = 0
        for cfg in list(sdfg.all_sdfgs_recursive()):
            for state in list(cfg.states()):
                entries = [n for n in state.nodes() if isinstance(n, nodes.MapEntry) and state.entry_node(n) is None]
                for entry in entries:
                    if entry not in state.nodes():  # a prior split in this state removed/replaced it
                        continue
                    if self._split_one_map(cfg, state, entry, SimplifyPass, helpers, InlineSDFG, SubgraphView):
                        count += 1
        return count

    @staticmethod
    def _split_one_map(
        cfg: ControlFlowRegion,
        state: SDFGState,
        entry: nodes.MapEntry,
        simplify_cls: type[ppl.Pass],
        helpers: ModuleType,
        inline_cls: type[transformation.TransformationBase],
        subgraph_cls: type[SubgraphView[Any, Any]],
    ) -> bool:
        """Split one straight-line multi-output map into one flat map per output: nest the scope, clone it per output
        with :meth:`_split`, and inline back only the clones this call made."""
        xit = state.exit_node(entry)
        out_names = plain_map_outputs(state, xit)
        if out_names is None or len(out_names) < 2:
            return False
        scope = state.scope_subgraph(entry, include_entry=True, include_exit=True)
        inner = [n for n in scope.nodes() if n not in (entry, xit)]
        # A plain dataflow NestedSDFG body (``A[i]=..; B[i]=A[i]*2``) is inlined first, so its flat tasklets show the
        # write precedes the read. A conditional or indirection-symbol body is _replicate_components' job.
        if (
            len(inner) == 1
            and isinstance(inner[0], nodes.NestedSDFG)
            and not _has_conditional(inner[0].sdfg)
            and not _has_interstate_assignments(inner[0].sdfg)
        ):
            inline_cls.apply_to(cfg, nested_sdfg=inner[0], save=False, verify=False)
            scope = state.scope_subgraph(entry, include_entry=True, include_exit=True)
            inner = [n for n in scope.nodes() if n not in (entry, xit)]
        if rewrites_a_global_it_reads(cfg, state, entry, xit, inner) or not clonable_leaf_body(cfg, scope, inner):
            return False
        before = dict.fromkeys(n for n in state.nodes() if isinstance(n, nodes.NestedSDFG))
        nsdfg_node = helpers.nest_state_subgraph(cfg, state, subgraph_cls(state, list(scope.nodes())))
        groups = [dict.fromkeys([o]) for o in nsdfg_node.out_connectors if o in out_names]
        if len(groups) < 2:  # nesting coalesced the outputs onto one connector -- nothing to split
            return False
        SplitStatements._split(cfg, state, nsdfg_node, groups, simplify_cls)
        for clone in [n for n in state.nodes() if isinstance(n, nodes.NestedSDFG) and n not in before]:
            inline_cls.apply_to(cfg, nested_sdfg=clone, save=False, verify=False)
        return True

    # (1c) Straight-line LOOP statement fission: one loop per output.

    def _split_loop_bodies(self, sdfg: SDFG) -> int:
        """Loop analogue of :meth:`_split_map_bodies`: fission a STRAIGHT-LINE loop writing >=2 outputs into one LOOP
        per independent output."""
        from dace.sdfg.graph import SubgraphView
        from dace.transformation import helpers
        from dace.transformation.interstate import InlineMultistateSDFG
        from dace.transformation.passes.simplify import SimplifyPass

        count = 0
        for cfg in list(sdfg.all_sdfgs_recursive()):
            # Both indices are whole-SDFG walks; most loops are refused before reading either.
            index = LoopSplitIndex(cfg)
            for loop in [r for r in cfg.all_control_flow_regions() if isinstance(r, LoopRegion)]:
                decision = self.loop_output_groups(loop, cfg, index)
                if decision is None:
                    continue
                groups, ordered = decision
                if self._split_one_loop(
                    cfg, loop, groups, ordered, SimplifyPass, helpers, InlineMultistateSDFG, SubgraphView
                ):
                    count += 1
                    index.clear()
        return count

    @staticmethod
    def loop_output_groups(
        loop: LoopRegion, sdfg: SDFG, index: LoopSplitIndex
    ) -> tuple[list[dict[str, None]], bool] | None:
        """``(groups, ordered)`` for the loop, or ``None`` to refuse -- computed WITHOUT touching it."""
        states = splittable_loop_states(loop, sdfg, index)
        if states is None:
            guarded = guarded_groups(loop, sdfg, index)
            return None if guarded is None else (guarded, True)
        local = index.local_transients(loop)
        accesses = body_reads_and_writes(states, local)
        if accesses is None:
            return None
        reads, writes = accesses
        out_names = [w for w in writes if w not in local]
        in_names = dict.fromkeys(r for r in reads if r not in local)
        if len(out_names) < 2:
            return None
        groups = independent_groups(loop, out_names, in_names)
        if len(groups) < 2:
            return None
        rmw = [o for o in out_names if o in in_names]
        # Shared across the three analyses below: none of them touches ``loop``.
        cone: dict = {}
        groups = merge_carried_groups(loop, groups, rmw, in_names, cone)
        if len(groups) < 2:
            return None
        if not rmw or rmw_confined(loop, in_names, rmw, groups):
            minimal = SplitStatements._minimal_loops(loop, groups, rmw, in_names, cone)
            return None if minimal is None else (minimal, False)
        # The clones cannot stand as unordered siblings, but ONE order of the two loops may still reproduce the fused
        # loop exactly -- see :func:`split_order`.
        ordered = split_order(loop, in_names, rmw, groups, cone)
        return None if ordered is None else (ordered, True)

    @staticmethod
    def _minimal_loops(
        body: LoopRegion,
        groups: list[dict[str, None]],
        rmw: list[str],
        in_names: dict[str, None],
        cone: Cone | None = None,
    ) -> list[dict[str, None]] | None:
        """Coalesce the output groups into two loops: CARRIED and FREE."""
        recurrences = [n for n in rmw if carries_across_iterations(body, n, in_names, cone)]
        carried: dict[str, None] = {}
        free: dict[str, None] = {}
        for grp in groups:
            (carried if any(o in recurrences for o in grp) else free).update(grp)
        if not carried or not free:
            return None
        return [carried, free]

    @staticmethod
    def _split_one_loop(
        sdfg: SDFG,
        loop: LoopRegion,
        groups: list[dict[str, None]],
        ordered: bool,
        simplify_cls: type[ppl.Pass],
        helpers: ModuleType,
        inline_cls: type[transformation.TransformationBase],
        subgraph_cls: type[SubgraphView[Any, Any]],
    ) -> bool:
        """Outline ``loop``, clone it per group, inline the clones back; return whether it fired."""
        outer = helpers.nest_sdfg_subgraph(sdfg, subgraph_cls(loop.parent_graph, [loop]))
        node = next(n for n in outer.nodes() if isinstance(n, nodes.NestedSDFG))
        # The outlining decides the connector set itself
        covered = dict.fromkeys(o for grp in groups for o in grp)
        if dict.fromkeys(node.out_connectors) != covered:
            inline_cls.apply_to(sdfg, nested_sdfg=node, save=False, verify=False)
            return False
        before = dict.fromkeys(n for n in outer.nodes() if isinstance(n, nodes.NestedSDFG))
        if ordered:
            clones = SplitStatements._split_ordered(outer, node, groups, simplify_cls)
            if clones is None:
                inline_cls.apply_to(sdfg, nested_sdfg=node, save=False, verify=False)
                return False
        else:
            # ``rmw_read_is_dead``: the sibling split only runs once ``rmw_confined`` has proved a read-modify-write
            # array is read by nobody but its writer, so the read is dead in every other clone.
            SplitStatements._split(
                sdfg, outer, node, groups, simplify_cls, rmw_read_is_dead=True, cut_other_stores=True
            )
            clones = [n for n in outer.nodes() if isinstance(n, nodes.NestedSDFG) and n not in before]
        for clone in clones:
            inline_cls.apply_to(sdfg, nested_sdfg=clone, save=False, verify=False)
        return True

    @staticmethod
    def _split_ordered(
        state: SDFGState, node: nodes.NestedSDFG, groups: list[dict[str, None]], simplify_cls: type[ppl.Pass]
    ) -> list[nodes.NestedSDFG] | None:
        """Clone ``node`` once per group into CONSECUTIVE states; the clones, or ``None`` to refuse."""
        in_edges = list(state.in_edges(node))
        out_edges = list(state.out_edges(node))
        # A later group's clone needs its own access nodes, which can only be rebuilt from an AccessNode neighbour
        if any(not isinstance(e.src, nodes.AccessNode) for e in in_edges):
            return None
        if any(not isinstance(e.dst, nodes.AccessNode) for e in out_edges):
            return None
        # An ORDERING edge cannot be re-homed: the clones land in CONSECUTIVE states, so a copy in
        # a later state constrains nothing about the endpoint left behind in the first. Refuse.
        if any(e.data is not None and e.data.is_empty() for e in in_edges + out_edges):
            return None
        parent = state.parent_graph
        clones = []
        target = state
        for idx, grp in enumerate(groups):
            if idx:
                target = parent.add_state_after(target, label=f"{state.label}_split_{idx}")
            pruned = pruned_group_clone(node, grp, simplify_cls)
            if pruned is None:
                return None
            clone_sdfg, kept_in = pruned
            # dicts/sorted, not sets: these become the clone's connector dicts, which are
            # observable in validation and codegen order.
            clone = target.add_nested_sdfg(
                clone_sdfg,
                inputs=dict.fromkeys(kept_in),
                outputs=dict.fromkeys(sorted(grp)),
                symbol_mapping=copy.deepcopy(node.symbol_mapping),
            )
            for e in in_edges:
                if e.dst_conn not in kept_in:
                    continue
                src = e.src if target is state else target.add_access(e.src.data)
                target.add_edge(src, e.src_conn, clone, e.dst_conn, copy.deepcopy(e.data))
            for e in out_edges:
                if e.src_conn not in grp:
                    continue
                dst = e.dst if target is state else target.add_access(e.dst.data)
                target.add_edge(clone, e.src_conn, dst, e.dst_conn, copy.deepcopy(e.data))
            clones.append(clone)
        endpoints = dict.fromkeys([e.src for e in in_edges] + [e.dst for e in out_edges])
        for e in in_edges + out_edges:
            state.remove_edge(e)
        state.remove_node(node)
        # The first group's clone stays in ``state`` and rewires only ITS OWN edges, so an access
        # node that served only a later group is left behind with nothing attached.
        for n in endpoints:
            if n in state.nodes() and state.degree(n) == 0:
                state.remove_node(n)
        return clones


__all__ = ["SplitStatements"]
