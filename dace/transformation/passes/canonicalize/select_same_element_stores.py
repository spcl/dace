# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""If-convert a chain of branches that all store one element into one select.

``if c1: x[p] = A elif c2: x[p] = B`` becomes ``x[p] = A if c1 else (B if c2 else x[p])``: every arm computes its
value, and one store keeps the value of the first arm whose condition holds, or the old value when no arm runs and
the chain has no ``else``. The store is then unconditional, so statement distribution and the loop lift see a plain
statement instead of a branch.
"""

import copy
from typing import Any

from dace import SDFG, dtypes, properties
from dace.memlet import Memlet
from dace.properties import CodeBlock
from dace.sdfg import nodes
from dace.sdfg.state import ConditionalBlock, ControlFlowRegion, SDFGState
from dace.sdfg.tasklet_utils import integer_division
from dace.transformation import pass_pipeline as ppl
from dace.transformation import transformation
from dace.transformation.passes.canonicalize.select_branch_maps import sole_work_state

#: One arm of the chain: its condition (``None`` for the final ``else``) and its one state.
Arm = tuple[CodeBlock | None, SDFGState]


def chain_arms(block: ConditionalBlock) -> list[Arm] | None:
    """The arms of ``block`` in test order, an ``else`` that holds one more conditional block unfolded into its arms;
    an empty ``else`` contributes none. ``None`` for any other branch shape."""
    arms: list[Arm] = []
    for cond, region in block.branches:
        state = sole_work_state(region)
        if state is not None:
            arms.append((cond, state))
            continue
        blocks = region.nodes()
        if cond is None and all(isinstance(b, SDFGState) and b.number_of_nodes() == 0 for b in blocks):
            continue
        inner = nested_chain(region) if cond is None else None
        if inner is None:
            return None
        arms.extend(inner)
    return arms


def nested_chain(region: ControlFlowRegion) -> list[Arm] | None:
    """The arms of the one conditional block an ``else`` region holds beside empty states."""
    if any(e.data.assignments or not e.data.is_unconditional() for e in region.edges()):
        return None
    blocks = [b for b in region.nodes() if not (isinstance(b, SDFGState) and b.number_of_nodes() == 0)]
    if len(blocks) != 1 or not isinstance(blocks[0], ConditionalBlock):
        return None
    return chain_arms(blocks[0])


def arm_store(sdfg: SDFG, state: SDFGState) -> tuple[nodes.AccessNode, Any] | None:
    """The one element store of ``state``: its sink access node, written by one tasklet, and the edge; every other
    write goes to a transient scalar no other state names."""
    stores = [n for n in state.data_nodes() if state.in_degree(n) > 0 and state.out_degree(n) == 0]
    if len(stores) != 1 or state.in_degree(stores[0]) != 1:
        return None
    edge = state.in_edges(stores[0])[0]
    if not isinstance(edge.src, nodes.Tasklet) or edge.data.wcr is not None or edge.data.subset.num_elements() != 1:
        return None
    for n in state.data_nodes():
        if n is stores[0] or state.in_degree(n) == 0:
            continue
        desc = sdfg.arrays[n.data]
        if not desc.transient or desc.total_size != 1:
            return None
        if any(n.data == m.data for st in sdfg.states() if st is not state for m in st.data_nodes()):
            return None
    return stores[0], edge


def safe_unguarded(sdfg: SDFG, arms: list[Arm]) -> bool:
    """Whether every arm may run on every path: no integer division (a guard can keep a divisor from zero), conditions
    over symbols only, and no read indexed by a symbol a condition tests (a guard can keep that index in range)."""
    tested = {str(s) for cond, state in arms if cond is not None for s in cond.get_free_symbols()}
    if tested & set(sdfg.arrays):
        return False
    for cond, state in arms:
        if any(isinstance(n, nodes.Tasklet) and integer_division(state, n) for n in state.nodes()):
            return False
        for e in state.edges():
            if isinstance(e.src, nodes.AccessNode) and not e.data.is_empty():
                if {str(s) for s in e.data.subset.free_symbols} & tested:
                    return False
    return True


def select_code(conds: list[CodeBlock | None], has_else: bool) -> str:
    """``__o = __v0 if (c0) else (__v1 if (c1) else __old)``; the last value is the ``else`` arm's when there is one."""
    tail = f"__v{len(conds) - 1}" if has_else else "__old"
    tested = conds[:-1] if has_else else conds
    for index in reversed(range(len(tested))):
        tail = f"(__v{index} if ({tested[index].as_string}) else {tail})"
    return f"__o = {tail}"


@properties.make_properties
@transformation.explicit_cf_compatible
class SelectSameElementStores(ppl.Pass):
    """If-convert every branch chain whose arms all store the same one element (module docstring)."""

    CATEGORY: str = "Canonicalization"

    def modifies(self) -> ppl.Modifies:
        return ppl.Modifies.Nodes | ppl.Modifies.Edges | ppl.Modifies.Descriptors | ppl.Modifies.CFG

    def should_reapply(self, modified: ppl.Modifies) -> bool:
        return False

    def apply_pass(self, sdfg: SDFG, pipeline_results: dict[str, Any]) -> int | None:
        converted = 0
        for g in sdfg.all_sdfgs_recursive():
            # Outermost first: converting a chain retires the conditional blocks nested in its else arm
            for block in [b for b in g.all_control_flow_blocks() if isinstance(b, ConditionalBlock)]:
                if block.parent_graph is not None and block in block.parent_graph.nodes():
                    converted += self.convert(g, block)
        return converted or None

    def convert(self, sdfg: SDFG, block: ConditionalBlock) -> int:
        arms = chain_arms(block)
        # One guarded store is a masked statement other lifts match (a conditional reduction): only chains convert
        if arms is None or len(arms) < 2:
            return 0
        found = [arm_store(sdfg, state) for cond, state in arms]
        if any(f is None for f in found):
            return 0
        stores = [f for f in found if f is not None]
        name, element = stores[0][0].data, str(stores[0][1].data.subset)
        if any(n.data != name or str(e.data.subset) != element for n, e in stores):
            return 0
        if not safe_unguarded(sdfg, arms):
            return 0
        has_else = arms[-1][0] is None
        parent = block.parent_graph
        merged = parent.add_state(f"{block.label}_select", is_start_block=parent.start_block is block)
        select = merged.add_tasklet(
            f"select_{name}",
            {f"__v{i}": None for i in range(len(arms))} | ({} if has_else else {"__old": None}),
            {"__o": None},
            select_code([cond for cond, state in arms], has_else),
        )
        dtype = sdfg.arrays[name].dtype
        for index, (cond, state) in enumerate(arms):
            store, edge = stores[index]
            copies = {n: copy.deepcopy(n) for n in state.nodes() if n is not store}
            for n in copies.values():
                merged.add_node(n)
            for e in state.edges():
                if e.dst is not store:
                    merged.add_edge(copies[e.src], e.src_conn, copies[e.dst], e.dst_conn, copy.deepcopy(e.data))
            value, desc = sdfg.add_scalar(
                f"__select_{name}_{index}",
                dtype,
                transient=True,
                storage=dtypes.StorageType.Register,
                find_new_name=True,
            )
            access = merged.add_access(value)
            merged.add_edge(copies[edge.src], edge.src_conn, access, None, Memlet(f"{value}[0]"))
            merged.add_edge(access, None, select, f"__v{index}", Memlet(f"{value}[0]"))
        if not has_else:
            merged.add_edge(merged.add_read(name), None, select, "__old", Memlet(f"{name}[{element}]"))
        merged.add_edge(select, "__o", merged.add_write(name), None, copy.deepcopy(stores[0][1].data))
        for e in parent.in_edges(block):
            parent.add_edge(e.src, merged, copy.deepcopy(e.data))
        for e in parent.out_edges(block):
            parent.add_edge(merged, e.dst, copy.deepcopy(e.data))
        parent.remove_node(block)
        return 1
