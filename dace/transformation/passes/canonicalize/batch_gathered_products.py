# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Turn the guarded matrix product of a loop nest into ONE gathered batched product (GPU target)."""

import copy
from typing import Any, NamedTuple

import dace
from dace import SDFG, Memlet, properties, subsets, symbolic
from dace.libraries.blas.nodes.batched_matmul import BATCH_INDEX, BatchedMatMul
from dace.libraries.blas.nodes.gemm import Gemm
from dace.libraries.blas.nodes.matmul import MatMul
from dace.sdfg import nodes
from dace.sdfg.state import ControlFlowRegion, LoopRegion, SDFGState
from dace.transformation import pass_pipeline as ppl
from dace.transformation import transformation
from dace.transformation.passes.canonicalize.loop_to_stream_compaction import COUNT_DTYPE, LoopToStreamCompaction


class Operand(NamedTuple):
    """An operand pool or shared matrix, with the index expression one iteration reads it at (``None`` if shared)."""

    data: str
    index: Any


class Product(NamedTuple):
    nest: LoopRegion
    branch: ControlFlowRegion
    state: SDFGState
    node: nodes.LibraryNode
    operands: dict[str, Operand]
    result: nodes.AccessNode


def operand(state: SDFGState, node: nodes.LibraryNode, conn: str) -> Operand | None:
    """The pool read at one index, or the whole shared matrix, ``node`` reads on ``conn``."""
    edge = next((e for e in state.in_edges(node) if e.dst_conn == conn), None)
    if edge is None or not isinstance(edge.src, nodes.AccessNode) or edge.data.subset is None:
        return None
    desc = state.sdfg.arrays[edge.data.data]
    ranges = edge.data.subset.ranges
    full = [r == (0, extent - 1, 1) for r, extent in zip(ranges, desc.shape)]
    if len(ranges) == 2 and all(full):
        return Operand(edge.data.data, None)
    if len(ranges) == 3 and all(full[1:]) and ranges[0][0] == ranges[0][1]:
        return Operand(edge.data.data, ranges[0][0])
    return None


def bindings_only(blocks) -> bool:
    """Whether ``blocks`` only bind symbols: copies of them may run again without repeating any work."""
    return all(isinstance(b, SDFGState) and not b.number_of_nodes() for b in blocks)


def written_in(region: ControlFlowRegion) -> set[str]:
    return {n.data for st in region.states() for n in st.data_nodes() if st.in_degree(n) > 0}


def guarded_work_state(nest: LoopRegion) -> tuple[ControlFlowRegion, SDFGState] | None:
    """The one-branch guard of a compaction-shaped ``nest`` and the first state of that branch doing work, when
    everything else around it only binds symbols."""
    found = LoopToStreamCompaction().match_nest(nest)
    if found is None:
        return None
    levels, guard = found
    if not all(bindings_only((*level.preamble, *level.tail)) for level in levels):
        return None
    if len(guard.branches) != 1 or guard.branches[0][0] is None:
        return None
    branch = guard.branches[0][1]
    chain = LoopToStreamCompaction().body_chain(branch)
    if chain is None:
        return None
    at = next((i for i, b in enumerate(chain) if isinstance(b, SDFGState) and b.number_of_nodes()), None)
    if at is None or not bindings_only(chain[:at]):
        return None
    return branch, chain[at]


def match(nest: LoopRegion) -> Product | None:
    found = guarded_work_state(nest)
    if found is None:
        return None
    branch, state = found
    libs = [n for n in state.nodes() if not isinstance(n, nodes.AccessNode)]
    if len(libs) != 1 or not isinstance(libs[0], (MatMul, Gemm)):
        return None
    node = libs[0]
    if isinstance(node, Gemm) and (not symbolic.equal_valued(0, node.beta) or "_cin" in node.in_connectors):
        return None
    operands = {conn: operand(state, node, conn) for conn in ("_a", "_b")}
    if any(op is None for op in operands.values()) or all(op.index is None for op in operands.values()):
        return None
    (out,) = state.out_edges(node)
    result = out.dst
    sdfg = state.sdfg
    desc = sdfg.arrays[result.data]
    if not (isinstance(result, nodes.AccessNode) and desc.transient and len(desc.shape) == 2):
        return None
    if out.data.subset.ranges != [(0, extent - 1, 1) for extent in desc.shape]:
        return None
    if {op.data for op in operands.values()} & written_in(nest):
        return None
    inside = set(branch.states())
    if any(n.data == result.data for st in sdfg.states() if st not in inside for n in st.data_nodes()):
        return None
    return Product(nest, branch, state, node, operands, result)


def append_into(branch: ControlFlowRegion, state: SDFGState, cursor: str) -> None:
    """Bump ``cursor`` on the way into ``state``, the one path through ``branch``."""
    edges = branch.in_edges(state)
    if not edges:
        branch.add_state_before(state, f"{state.label}_append", is_start_block=True)
        edges = branch.in_edges(state)
    (edge,) = edges
    edge.data.assignments[cursor] = f"{cursor} + 1"


def read_from(branch: ControlFlowRegion, result: str, products: str, cursor: str) -> None:
    """Every access to ``result`` in ``branch`` becomes one to the ``cursor``-th product of ``products``."""
    at = (symbolic.pystr_to_symbolic(cursor),) * 2 + (1,)
    for state in branch.states():
        for edge in state.edges():
            ends = (edge.src, edge.dst)
            if edge.data.data == result:
                edge.data.data = products
                edge.data.subset = subsets.Range([at, *edge.data.subset.ranges])
            elif edge.data.other_subset is not None and any(
                isinstance(n, nodes.AccessNode) and n.data == result for n in ends
            ):
                edge.data.other_subset = subsets.Range([at, *edge.data.other_subset.ranges])
        for access in state.data_nodes():
            if access.data == result:
                access.data = products


def drop_after(branch: ControlFlowRegion, state: SDFGState) -> None:
    after = list(branch.bfs_nodes(state))[1:]
    for block in after:
        branch.remove_node(block)


def clone(product: Product, parent: ControlFlowRegion) -> tuple[LoopRegion, ControlFlowRegion, SDFGState]:
    memo: dict[int, Any] = {}
    nest = copy.deepcopy(product.nest, memo)
    parent.add_node(nest, ensure_unique_name=True)
    return nest, memo[id(product.branch)], memo[id(product.state)]


def rewrite(product: Product) -> None:
    sdfg = product.state.sdfg
    parent = product.nest.parent_graph
    node, result = product.node, product.result
    was_start = parent.start_block is product.nest
    names = {}
    for role in ("count", "fill", "use", "npairs"):
        names[role] = sdfg.find_new_symbol(f"batched_{role}")
        sdfg.add_symbol(names[role], COUNT_DTYPE)
    npairs = symbolic.pystr_to_symbolic(names["npairs"])

    # Count the guarded iterations
    counting, branch, state = clone(product, parent)
    drop_after(branch, state)
    state.remove_nodes_from(list(state.nodes()))
    append_into(branch, state, names["count"])

    # Gather the indices of each product's operands
    filling, branch, state = clone(product, parent)
    drop_after(branch, state)
    state.remove_nodes_from(list(state.nodes()))
    append_into(branch, state, names["fill"])
    indices = {}
    for conn, op in product.operands.items():
        if op.index is None:
            continue
        indices[conn], _ = sdfg.add_array(
            f"batched_index{conn}", [npairs], COUNT_DTYPE, transient=True, find_new_name=True
        )
        store = state.add_tasklet(f"store{conn}", {}, {"__out"}, f"__out = {symbolic.symstr(op.index)}")
        state.add_edge(
            store, "__out", state.add_write(indices[conn]), None, Memlet(f"{indices[conn]}[{names['fill']}]")
        )

    # One batched product of them all
    desc = sdfg.arrays[result.data]
    products, _ = sdfg.add_array(
        f"batched_{result.data}", [npairs, *desc.shape], desc.dtype, transient=True, find_new_name=True
    )
    batched_state = parent.add_state(f"{node.label}_batched")
    batched = BatchedMatMul(f"{node.label}_batched")
    batched.transA, batched.transB = (node.transA, node.transB) if isinstance(node, Gemm) else (False, False)
    batched.alpha = node.alpha if isinstance(node, Gemm) else 1
    batched_state.add_node(batched)
    for conn, op in product.operands.items():
        src = sdfg.arrays[op.data]
        batched_state.add_edge(batched_state.add_read(op.data), None, batched, conn, Memlet.from_array(op.data, src))
        if conn in indices:
            batched.add_in_connector(BATCH_INDEX[conn])
            batched_state.add_edge(
                batched_state.add_read(indices[conn]),
                None,
                batched,
                BATCH_INDEX[conn],
                Memlet.from_array(indices[conn], sdfg.arrays[indices[conn]]),
            )
    batched_state.add_edge(
        batched, "_c", batched_state.add_write(products), None, Memlet.from_array(products, sdfg.arrays[products])
    )

    # The original nest reads its product from the batch, in place of its own
    state = product.state
    for edge in state.in_edges(node):
        state.remove_edge(edge)
        if state.degree(edge.src) == 0:
            state.remove_node(edge.src)
    state.remove_node(node)
    if state.degree(result) == 0:
        state.remove_node(result)
    read_from(product.branch, result.data, products, names["use"])
    sdfg.remove_data(result.data, validate=False)
    append_into(product.branch, state, names["use"])

    # count -> fill -> batched product -> use, where the nest was
    entry = parent.add_state_before(product.nest, f"{node.label}_count", is_start_block=was_start)
    parent.remove_edge(next(iter(parent.edges_between(entry, product.nest))))
    parent.add_edge(entry, counting, dace.InterstateEdge(assignments={names["count"]: "-1"}))
    counted = {names["npairs"]: f"{names['count']} + 1", names["fill"]: "-1"}
    parent.add_edge(counting, filling, dace.InterstateEdge(assignments=counted))
    parent.add_edge(filling, batched_state, dace.InterstateEdge())
    parent.add_edge(batched_state, product.nest, dace.InterstateEdge(assignments={names["use"]: "-1"}))


@properties.make_properties
@transformation.explicit_cf_compatible
class BatchGatheredProducts(ppl.Pass):
    """Turn each guarded gathered matrix product of a loop nest into one ``BatchedMatMul`` over the guarded pairs."""

    CATEGORY: str = "Specialization"

    def modifies(self) -> ppl.Modifies:
        return ppl.Modifies.Everything

    def should_reapply(self, modified: ppl.Modifies) -> bool:
        return False

    def apply_pass(self, sdfg: SDFG, _: dict[str, Any]) -> int | None:
        rewritten = 0
        for sd in list(sdfg.all_sdfgs_recursive()):
            # Outermost first: a rewritten nest is not revisited, its copies are new blocks
            for region in list(sd.all_control_flow_regions()):
                if not isinstance(region, LoopRegion) or not region.loop_variable:
                    continue
                if isinstance(region.parent_graph, LoopRegion):
                    continue
                product = match(region)
                if product is not None:
                    rewrite(product)
                    rewritten += 1
        return rewritten or None
