# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Merge the two maps of an ``if/else`` over one range into one map that computes both and selects per element.

fv3_dycore picks its limiter by a scalar that does not change inside the step::

    if abs(ord) == 5:                        map (i, j, k): smt5[i, j, k] = bl * br < 0
    else:                                    map (i, j, k): smt5[i, j, k] = 3 * abs(b0) < abs(bl - br)

Kept as a conditional, the two maps sit in their own states and fuse with neither the maps producing ``bl``, ``br``,
``b0`` nor the ones reading ``smt5``, so every operand stays a full-size array. Merged, the branch is one map with a
select per output, which fuses like any element-wise map; the condition, made trivially true, is lifted by
``LiftTrivialIf``.

Contract, per ``ConditionalBlock`` with an ``if`` and an ``else`` branch:

* each branch is states without interstate assignments, exactly one of them non-empty, holding one top-level map
  whose body is tasklets and one-element access nodes only (no WCR), with every output written by a tasklet, and
  no copy beside it;
* the two maps have equal ranges (up to parameter names) and write the same arrays at the same subsets;
* the condition reads symbols, none a map parameter, and one-element containers neither branch writes;
* both bodies together have at most ``MAX_TASKLETS`` tasklets: each element computes both, so every access outside
  the other branch stays in range without the guard, and no tasklet divides integers;
* a map over the same range elsewhere in the block's region produces one of the branch inputs or reads one of its
  outputs, so the merged map has a fusion partner.
"""

import copy
import re
from dataclasses import dataclass
from typing import Any

from dace import SDFG, dtypes, properties, symbolic
from dace.memlet import Memlet
from dace.properties import CodeBlock
from dace.sdfg import nodes
from dace.sdfg.state import ConditionalBlock, ControlFlowRegion, SDFGState, SymbolResolver
from dace.sdfg.tasklet_utils import integer_division
from dace.transformation import pass_pipeline as ppl
from dace.transformation import transformation

#: Most tasklets the two merged bodies may hold together.
MAX_TASKLETS = 64


def branch_map(region: ControlFlowRegion) -> tuple[SDFGState, nodes.MapEntry] | None:
    """The one non-empty state of ``region`` and its single top-level map, when the contract admits them."""
    if any(e.data.assignments or not e.data.is_unconditional() for e in region.edges()):
        return None
    states = [b for b in region.nodes() if isinstance(b, SDFGState)]
    if len(states) != len(region.nodes()):
        return None
    full = [s for s in states if s.number_of_nodes() > 0]
    if len(full) != 1:
        return None
    state = full[0]
    entries = [n for n in state.nodes() if isinstance(n, nodes.MapEntry) and state.entry_node(n) is None]
    if len(entries) != 1:
        return None
    entry = entries[0]
    scope = state.scope_subgraph(entry)
    top = set(state.nodes()) - set(scope.nodes())
    if any(not isinstance(n, nodes.AccessNode) for n in top):
        return None
    # A copy between two of those access nodes runs only in its branch; the merged map would drop it.
    if any(e.src in top and e.dst in top for e in state.edges()):
        return None
    body = [n for n in scope.nodes() if n is not entry and n is not state.exit_node(entry)]
    if any(not isinstance(n, (nodes.Tasklet, nodes.AccessNode)) for n in body):
        return None
    if any(isinstance(n, nodes.AccessNode) and state.sdfg.arrays[n.data].total_size != 1 for n in body):
        return None
    if any(e.data.wcr is not None for e in scope.edges()):
        return None
    # A tasklet handed several elements picks one in its code: no memlet shows that index in range without the guard.
    if any(
        isinstance(e.dst, nodes.Tasklet) and not e.data.is_empty() and e.data.subset.num_elements() != 1
        for e in scope.edges()
    ):
        return None
    if any(
        not isinstance(e.src, nodes.Tasklet) for e in state.in_edges(state.exit_node(entry)) if not e.data.is_empty()
    ):
        return None
    return state, entry


def fusion_partner(block: ConditionalBlock, rng, reads: set[str], writes: set[str]) -> bool:
    """Whether a state of ``block``'s region holds a top-level map over ``rng`` (up to parameter names) that writes
    one of ``reads`` or reads one of ``writes``: computing both branches pays only when the merged map fuses."""
    for st in block.parent_graph.nodes():
        if not isinstance(st, SDFGState):
            continue
        for n in st.nodes():
            if not isinstance(n, nodes.MapEntry) or st.entry_node(n) is not None or n.map.range != rng:
                continue
            ins = {e.data.data for e in st.in_edges(n)}
            outs = {e.data.data for e in st.out_edges(st.exit_node(n))}
            if ins & writes or outs & reads:
                return True
    return False


def map_accesses(state: SDFGState, entry: nodes.MapEntry) -> set[tuple[str, str]]:
    """``(container, subset)`` of what the map at ``entry`` reads and writes, over its whole range."""
    edges = (*state.in_edges(entry), *state.out_edges(state.exit_node(entry)))
    return {(e.data.data, str(e.data.subset)) for e in edges if not e.data.is_empty()}


def safe_unguarded(block: ConditionalBlock, arms: list[tuple[SDFGState, nodes.MapEntry]]) -> bool:
    """Whether both maps may run on every path: what only one of them accesses is in range there without the guard (a
    guard can be what keeps it in range), and neither divides integers (a guard can keep the divisor from zero)."""
    (state, entry), (other, entry_other) = arms
    only_one = map_accesses(state, entry) ^ map_accesses(other, entry_other)
    facts = SymbolResolver().facts_at(block)
    for st, en in arms:
        scope = st.scope_subgraph(en).nodes()
        if any(isinstance(n, nodes.Tasklet) and integer_division(st, n) for n in scope):
            return False
        for e in (*st.in_edges(en), *st.out_edges(st.exit_node(en))):
            if e.data.is_empty() or (e.data.data, str(e.data.subset)) not in only_one:
                continue
            shape = st.sdfg.arrays[e.data.data].shape
            for (begin, end, _step), extent in zip(e.data.subset.ndrange(), shape):
                if not symbolic.provably_nonnegative(begin, facts):
                    return False
                if not symbolic.provably_nonnegative(extent - 1 - end, facts):
                    return False
    return True


def map_outputs(st: SDFGState, ex: nodes.MapExit, ren: dict) -> dict[str, tuple] | None:
    """Per written container, the exit's in-edge and its subset under ``ren``; ``None`` when one is written twice."""
    out = {}
    for e in st.in_edges(ex):
        if e.data.is_empty():
            continue
        inner = copy.deepcopy(e.data)
        inner.replace(ren)
        if e.data.data in out:
            return None
        out[e.data.data] = (e, str(inner.subset))
    return out


@dataclass(slots=True)
class BranchMerge:
    """Two branch maps :func:`merge_match` admits."""

    cond: CodeBlock
    region_t: ControlFlowRegion
    state: SDFGState
    entry: nodes.MapEntry
    other: SDFGState
    entry_f: nodes.MapEntry
    rename: dict
    cond_data: list[str]
    body_f: list[nodes.Node]
    exit_t: nodes.MapExit
    outs_t: dict[str, tuple]
    outs_f: dict[str, tuple]


def merge_match(sdfg: SDFG, block: ConditionalBlock, arrays: set[str]) -> BranchMerge | None:
    """The two arms' maps when they cover one range, write the same elements, and may run unguarded."""
    if len(block.branches) != 2 or block.branches[0][0] is None or block.branches[1][0] is not None:
        return None
    (cond, region_t), (_, region_f) = block.branches
    found_t, found_f = branch_map(region_t), branch_map(region_f)
    if found_t is None or found_f is None:
        return None
    (state, entry), (other, entry_f) = found_t, found_f
    params, params_f = entry.map.params, entry_f.map.params
    if len(params) != len(params_f):
        return None
    rename = {p: symbolic.pystr_to_symbolic(q) for p, q in zip(params_f, params) if p != q}
    range_f = copy.deepcopy(entry_f.map.range)
    range_f.replace(rename)
    if range_f != entry.map.range:
        return None
    cond_names = {str(s) for s in cond.get_free_symbols()}
    if cond_names & (set(params) | set(params_f)):
        return None
    # Data the condition reads: one-element containers neither branch writes, handed to each select
    cond_data = sorted(cond_names & arrays)
    if any(sdfg.arrays[n].total_size != 1 for n in cond_data):
        return None
    if any(n.data in cond_data and st.in_degree(n) > 0 for st in (state, other) for n in st.data_nodes()):
        return None
    body_t = state.scope_subgraph(entry, include_entry=False, include_exit=False).nodes()
    body_f = other.scope_subgraph(entry_f, include_entry=False, include_exit=False).nodes()
    if sum(isinstance(n, nodes.Tasklet) for n in [*body_t, *body_f]) > MAX_TASKLETS:
        return None
    exit_t, exit_f = state.exit_node(entry), other.exit_node(entry_f)

    outs_t, outs_f = map_outputs(state, exit_t, {}), map_outputs(other, exit_f, rename)
    if outs_t is None or outs_f is None or set(outs_t) != set(outs_f):
        return None
    if any(outs_t[k][1] != outs_f[k][1] for k in outs_t):
        return None
    reads = {e.data.data for st, en in ((state, entry), (other, entry_f)) for e in st.in_edges(en) if e.data.data}
    if not fusion_partner(block, entry.map.range, reads, set(outs_t)):
        return None
    if not safe_unguarded(block, [found_t, found_f]):
        return None
    return BranchMerge(cond, region_t, state, entry, other, entry_f, rename, cond_data, body_f, exit_t, outs_t, outs_f)


def copy_else_body(sdfg: SDFG, m: BranchMerge) -> dict[nodes.Node, nodes.Node]:
    """Copy the else body into the if map, its parameters renamed and its containers given fresh names; returns the
    copy of each else node."""
    state, entry, other, entry_f, rename = m.state, m.entry, m.other, m.entry_f, m.rename
    mapping: dict[nodes.Node, nodes.Node] = {}
    renamed: dict[str, str] = {}
    for n in m.body_f:
        new = copy.deepcopy(n)
        if isinstance(n, nodes.AccessNode):
            if n.data not in renamed:
                renamed[n.data] = sdfg.add_datadesc(n.data + "_else", copy.deepcopy(sdfg.arrays[n.data]), True)
            new.data = renamed[n.data]
        state.add_node(new)
        mapping[n] = new

    def moved(memlet: Memlet) -> Memlet:
        result = copy.deepcopy(memlet)
        result.replace(rename)
        if result.data in renamed:
            result.data = renamed[result.data]
        return result

    for e in other.edges():
        if e.src in mapping and e.dst in mapping:
            state.add_edge(mapping[e.src], e.src_conn, mapping[e.dst], e.dst_conn, moved(e.data))
    reads: dict[str, nodes.AccessNode] = {}
    for e in other.out_edges(entry_f):
        if e.dst not in mapping:
            continue
        if e.data.is_empty():
            state.add_nedge(entry, mapping[e.dst], Memlet())
            continue
        (outer,) = [o for o in other.in_edges(entry_f) if o.dst_conn == "IN_" + e.src_conn[4:]]
        name = outer.data.data
        if name not in reads:
            reads[name] = state.add_read(name)
        conn = entry.next_connector(name)
        entry.add_in_connector("IN_" + conn)
        entry.add_out_connector("OUT_" + conn)
        state.add_edge(reads[name], None, entry, "IN_" + conn, moved(outer.data))
        state.add_edge(entry, "OUT_" + conn, mapping[e.dst], e.dst_conn, moved(e.data))
    return mapping


def add_selects(sdfg: SDFG, m: BranchMerge, mapping: dict[nodes.Node, nodes.Node]) -> None:
    """Route each output's two values into fresh scalars and a select on the condition, into the if map's exit."""
    state, entry, cond, cond_data, exit_t = m.state, m.entry, m.cond, m.cond_data, m.exit_t
    code_cond = cond.as_string
    cond_conns = {name: f"__cond{i}" for i, name in enumerate(cond_data)}
    for name, conn in cond_conns.items():
        code_cond = re.sub(rf"\b{re.escape(name)}\b(?:\[[0,\s]*\])?", conn, code_cond)
    cond_inputs: dict[str, str] = {}
    for name in cond_data:
        outer = entry.next_connector(name)
        entry.add_in_connector("IN_" + outer)
        entry.add_out_connector("OUT_" + outer)
        state.add_edge(state.add_read(name), None, entry, "IN_" + outer, Memlet(f"{name}[0]"))
        cond_inputs[name] = "OUT_" + outer
    for name, (edge_t, _) in m.outs_t.items():
        edge_f = m.outs_f[name][0]
        dtype = sdfg.arrays[name].dtype
        picks = []
        for tag, src, conn in (("t", edge_t.src, edge_t.src_conn), ("f", mapping[edge_f.src], edge_f.src_conn)):
            scalar, _ = sdfg.add_scalar(
                f"__select_{name}_{tag}",
                dtype,
                transient=True,
                storage=dtypes.StorageType.Register,
                find_new_name=True,
            )
            node = state.add_access(scalar)
            state.add_edge(src, conn, node, None, Memlet(f"{scalar}[0]"))
            picks.append(node)
        select = state.add_tasklet(
            f"select_{name}",
            {"__t": None, "__f": None, **dict.fromkeys(cond_conns.values())},
            {"__o": None},
            f"__o = __t if ({code_cond}) else __f",
        )
        for data_name, conn in cond_conns.items():
            state.add_edge(entry, cond_inputs[data_name], select, conn, Memlet(f"{data_name}[0]"))
        state.add_edge(picks[0], None, select, "__t", Memlet(f"{picks[0].data}[0]"))
        state.add_edge(picks[1], None, select, "__f", Memlet(f"{picks[1].data}[0]"))
        state.add_edge(select, "__o", exit_t, edge_t.dst_conn, copy.deepcopy(edge_t.data))
        state.remove_edge(edge_t)


@properties.make_properties
@transformation.explicit_cf_compatible
class SelectBranchMaps(ppl.Pass):
    """Merge every ``if/else`` pair of same-range maps the module contract admits into one selecting map."""

    CATEGORY: str = "Canonicalization"

    def modifies(self) -> ppl.Modifies:
        return ppl.Modifies.Nodes | ppl.Modifies.Edges | ppl.Modifies.Descriptors | ppl.Modifies.CFG

    def should_reapply(self, modified: ppl.Modifies) -> bool:
        return False

    def apply_pass(self, sdfg: SDFG, _pipeline_results: dict[str, Any]) -> int | None:
        merged = 0
        for g in sdfg.all_sdfgs_recursive():
            arrays = set(g.arrays)
            for block in [b for b in g.all_control_flow_blocks() if isinstance(b, ConditionalBlock)]:
                merged += self.try_merge(g, block, arrays)
        return merged or None

    def try_merge(self, sdfg: SDFG, block: ConditionalBlock, arrays: set[str]) -> int:
        m = merge_match(sdfg, block, arrays)
        if m is None:
            return 0
        add_selects(sdfg, m, copy_else_body(sdfg, m))
        # The if branch now holds both; the else is unreachable and LiftTrivialIf removes it
        block.branches[0] = (CodeBlock("1"), m.region_t)
        return 1
