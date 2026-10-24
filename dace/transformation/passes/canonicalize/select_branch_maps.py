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
  whose body is tasklets and one-element access nodes only (no WCR), with every output written by a tasklet;
* the two maps have equal ranges (up to parameter names) and write the same arrays at the same subsets;
* the condition reads symbols, none a map parameter, and one-element containers neither branch writes;
* both bodies together have at most ``MAX_TASKLETS`` tasklets: each element computes both.
"""

import copy
import re
from typing import Any

from dace import SDFG, dtypes, properties, symbolic
from dace.memlet import Memlet
from dace.properties import CodeBlock
from dace.sdfg import nodes
from dace.sdfg.state import ConditionalBlock, ControlFlowRegion, SDFGState
from dace.transformation import pass_pipeline as ppl
from dace.transformation import transformation

#: Most tasklets the two merged bodies may hold together.
MAX_TASKLETS = 64


def branch_map(region: ControlFlowRegion) -> tuple[SDFGState, nodes.MapEntry] | None:
    """The one non-empty state of ``region`` and its single top-level map, when the contract admits them."""
    if any(e.data.assignments for e in region.edges()):
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
    body = [n for n in scope.nodes() if n is not entry and n is not state.exit_node(entry)]
    if any(not isinstance(n, (nodes.Tasklet, nodes.AccessNode)) for n in body):
        return None
    if any(isinstance(n, nodes.AccessNode) and state.sdfg.arrays[n.data].total_size != 1 for n in body):
        return None
    if any(e.data.wcr is not None for e in scope.edges()):
        return None
    if any(
        not isinstance(e.src, nodes.Tasklet) for e in state.in_edges(state.exit_node(entry)) if not e.data.is_empty()
    ):
        return None
    return state, entry


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
        if len(block.branches) != 2 or block.branches[0][0] is None or block.branches[1][0] is not None:
            return 0
        (cond, region_t), (_, region_f) = block.branches
        found_t, found_f = branch_map(region_t), branch_map(region_f)
        if found_t is None or found_f is None:
            return 0
        (state, entry), (other, entry_f) = found_t, found_f
        params, params_f = entry.map.params, entry_f.map.params
        if len(params) != len(params_f):
            return 0
        rename = {p: symbolic.pystr_to_symbolic(q) for p, q in zip(params_f, params) if p != q}
        range_f = copy.deepcopy(entry_f.map.range)
        range_f.replace(rename)
        if range_f != entry.map.range:
            return 0
        cond_names = {str(s) for s in cond.get_free_symbols()}
        if cond_names & (set(params) | set(params_f)):
            return 0
        # Data the condition reads: one-element containers neither branch writes, handed to each select
        cond_data = sorted(cond_names & arrays)
        if any(sdfg.arrays[n].total_size != 1 for n in cond_data):
            return 0
        if any(n.data in cond_data and st.in_degree(n) > 0 for st in (state, other) for n in st.data_nodes()):
            return 0
        body_t = state.scope_subgraph(entry, include_entry=False, include_exit=False).nodes()
        body_f = other.scope_subgraph(entry_f, include_entry=False, include_exit=False).nodes()
        if sum(isinstance(n, nodes.Tasklet) for n in [*body_t, *body_f]) > MAX_TASKLETS:
            return 0
        exit_t, exit_f = state.exit_node(entry), other.exit_node(entry_f)

        def outputs(st: SDFGState, ex: nodes.MapExit, ren: dict) -> dict[str, tuple] | None:
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

        outs_t, outs_f = outputs(state, exit_t, {}), outputs(other, exit_f, rename)
        if outs_t is None or outs_f is None or set(outs_t) != set(outs_f):
            return 0
        if any(outs_t[k][1] != outs_f[k][1] for k in outs_t):
            return 0

        # The else body, copied into the if map with its parameters renamed and its scalars given fresh names
        mapping: dict[nodes.Node, nodes.Node] = {}
        renamed: dict[str, str] = {}
        for n in body_f:
            new = copy.deepcopy(n)
            if isinstance(n, nodes.AccessNode):
                if n.data not in renamed:
                    renamed[n.data] = sdfg.add_datadesc(n.data + "_else", copy.deepcopy(sdfg.arrays[n.data]), True)
                new.data = renamed[n.data]
            state.add_node(new)
            mapping[n] = new

        def moved(memlet: Memlet) -> Memlet:
            m = copy.deepcopy(memlet)
            m.replace(rename)
            if m.data in renamed:
                m.data = renamed[m.data]
            return m

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

        # Each output: both values into fresh scalars, the select into the exit
        code_cond = cond.as_string
        cond_conns = {name: f"__cond{i}" for i, name in enumerate(cond_data)}
        for name, conn in cond_conns.items():
            code_cond = re.sub(rf"\b{re.escape(name)}\b", conn, code_cond)
        cond_inputs: dict[str, str] = {}
        for name in cond_data:
            outer = entry.next_connector(name)
            entry.add_in_connector("IN_" + outer)
            entry.add_out_connector("OUT_" + outer)
            state.add_edge(state.add_read(name), None, entry, "IN_" + outer, Memlet(f"{name}[0]"))
            cond_inputs[name] = "OUT_" + outer
        for name, (edge_t, _) in outs_t.items():
            edge_f = outs_f[name][0]
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
        # The if branch now holds both; the else is unreachable and LiftTrivialIf removes it
        block.branches[0] = (CodeBlock("1"), region_t)
        return 1
