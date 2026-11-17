# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""A map of matrix-vector products or dots becomes ONE batched product before the device offload.

``for b in map: y[b] = A[b] @ x[b]`` is a map whose body is one ``Gemv`` (or a ``MatMul`` of that shape), and
``s[b] = x[b, :] @ y[b, :]`` one whose body is a ``Dot``. The offload would keep such a map on the host, one vendor
call per iteration, or run each iteration's product on a thread block of our own code. The products are
independent, so the map becomes one ``BatchedMatMul`` over stacks of vectors: a vendor ``gemmStridedBatched`` when
every stacked operand is read at ``b`` plus a constant, else a ``gemmBatched`` over pointer arrays that gathers the
operands read at another expression of ``b`` through an index the device fills.
"""

from typing import Any

from dace import SDFG, data, dtypes, properties, subsets, symbolic
from dace.libraries.blas.nodes.batched_matmul import BATCH_INDEX, BatchedMatMul
from dace.libraries.blas.nodes.dot import Dot
from dace.libraries.blas.nodes.gemv import Gemv
from dace.libraries.blas.nodes.matmul import MatMul, _get_matmul_operands, specialization
from dace.memlet import Memlet
from dace.sdfg import nodes
from dace.sdfg.state import SDFGState
from dace.transformation import pass_pipeline as ppl
from dace.transformation.passes.offloading.taskloop import small_operands

#: Per kind of product, its connectors as ``BatchedMatMul``'s ``_a``, ``_b``, ``_c``, and whether ``_a`` is transposed
#: (``gemvt``: ``x @ A`` is ``A^T x``).
ROLES = {
    "gemv": (("_A", "_x", "_y"), False),
    "dot": (("_x", "_y", "_result"), False),
    "matmul_gemv": (("_a", "_b", "_c"), False),
    "matmul_dot": (("_a", "_b", "_c"), False),
    "matmul_gemvt": (("_b", "_a", "_c"), True),
}

#: The rank of each operand's product, ``_a``, ``_b``, ``_c``, by the rank of ``_c``.
RANKS = {1: (2, 1, 1), 0: (1, 1, 0)}


def kind_of(node: nodes.LibraryNode, state: SDFGState) -> str | None:
    if isinstance(node, Gemv):
        return "gemv"
    if isinstance(node, Dot):
        return None if node.conjugate else "dot"
    if isinstance(node, MatMul):
        try:
            found = specialization(*_get_matmul_operands(node, state, state.sdfg))
        except ValueError:
            return None
        return f"matmul_{found}" if f"matmul_{found}" in ROLES else None
    return None


def follows(expr, param) -> bool:
    return str(param) in symbolic.symlist(expr)


def stacked(memlet: Memlet, desc: data.Data, param, begin, end) -> tuple[subsets.Range, Any] | None:
    """The subset of every iteration's ``memlet`` together and the per-iteration pool index if it is gathered (else
    ``None``); ``None`` overall unless only the leading dimension follows ``param``, at one point, the rest whole."""
    ranges = memlet.subset.ranges
    # C-contiguous: a stack's elements lie one element's size apart, the stride the strided call walks
    if not ranges or not isinstance(desc, data.Array) or not desc.is_packed_c_strides():
        return None
    first, last, _ = ranges[0]
    if symbolic.equal(first, last) is not True or not follows(first, param):
        return None
    for (lo, hi, step), extent in zip(ranges[1:], desc.shape[1:]):
        if follows(lo, param) or follows(hi, param) or (lo, hi, step) != (0, extent - 1, 1):
            return None
    offset = symbolic.simplify(first - param)
    if not follows(offset, param):
        return subsets.Range([(begin + offset, end + offset, 1), *ranges[1:]]), None
    return subsets.Range([(0, desc.shape[0] - 1, 1), *ranges[1:]]), first


def batch(state: SDFGState, entry: nodes.MapEntry, node: nodes.LibraryNode) -> bool:
    """Replace ``map b { node }`` by one ``BatchedMatMul``; ``False`` if the map is not that."""
    sdfg = state.sdfg
    kind = kind_of(node, state)
    if kind is None or len(entry.map.params) != 1 or small_operands(node, state):
        return False
    (conn_a, conn_b, conn_c), trans = ROLES[kind]
    param = symbolic.pystr_to_symbolic(entry.map.params[0])
    begin, end, step = entry.map.range[0]
    if step != 1:
        return False
    inputs = {e.dst_conn: e for e in state.in_edges(node)}
    edges = inputs | {e.src_conn: e for e in state.out_edges(node)}
    # A product folding onto its output (a Gemv with nonzero beta reads ``_y``) is not batched
    if conn_c in inputs or set(edges) - {conn_a, conn_b, conn_c} or any(e.data.wcr is not None for e in edges.values()):
        return False
    plan = {}
    for role, conn in (("_a", conn_a), ("_b", conn_b), ("_c", conn_c)):
        edge = edges.get(conn)
        if edge is None or edge.data.is_empty():
            return False
        path = state.memlet_path(edge)
        outer = path[0].src if role != "_c" else path[-1].dst
        if not isinstance(outer, nodes.AccessNode):
            return False
        desc = sdfg.arrays[edge.data.data]
        if str(param) in edge.data.subset.free_symbols:
            found = stacked(edge.data, desc, param, begin, end)
            if found is None or (role == "_c" and found[1] is not None):
                return False
            plan[role] = (outer, edge, *found, len(edge.data.subset) - 1)
        elif role == "_c":
            return False  # every iteration writes the same output
        else:
            plan[role] = (outer, edge, edge.data.subset, None, len(edge.data.subset))
    rank = plan["_c"][4]
    if rank not in RANKS or (plan["_a"][4], plan["_b"][4], rank) != RANKS[rank]:
        return False

    product = BatchedMatMul(f"{node.label}_batched")
    product.transA = node.transA if isinstance(node, Gemv) else trans
    if isinstance(node, (Gemv, MatMul)):
        product.alpha, product.beta = node.alpha, node.beta
    state.add_node(product)
    trip = entry.map.range.num_elements()
    for role, (outer, edge, subset, index, _) in plan.items():
        memlet = Memlet(data=edge.data.data, subset=subset)
        if role == "_c":
            state.add_edge(product, role, outer, None, memlet)
            continue
        state.add_edge(outer, None, product, role, memlet)
        if index is not None:
            name, _ = sdfg.add_transient(f"{node.label}{role}_index", [trip], dtypes.int64, find_new_name=True)
            _, _, fill_exit = state.add_mapped_tasklet(
                f"{node.label}{role}_index",
                {entry.map.params[0]: f"{symbolic.symstr(begin)}:{symbolic.symstr(end + 1)}"},
                {},
                f"__out = {symbolic.symstr(index)}",
                {"__out": Memlet(f"{name}[{symbolic.symstr(param - begin)}]")},
                external_edges=True,
            )
            product.add_in_connector(BATCH_INDEX[role])
            state.add_edge(
                state.out_edges(fill_exit)[0].dst, None, product, BATCH_INDEX[role], Memlet(f"{name}[0:{trip}]")
            )
    exit_node = state.exit_node(entry)
    around = {e.src for e in state.in_edges(entry)} | {e.dst for e in state.out_edges(exit_node)}
    state.remove_nodes_from([entry, exit_node, node])
    # Only the map's own access nodes may be left unconnected; others in the state are not this rewrite's
    state.remove_nodes_from([n for n in around if state.degree(n) == 0])
    return True


@properties.make_properties
class BatchMapProducts(ppl.Pass):
    """Batch every top-level map whose body is one matrix-vector product or dot into one ``BatchedMatMul``."""

    CATEGORY: str = "Device Specialization"

    def modifies(self) -> ppl.Modifies:
        return ppl.Modifies.Scopes | ppl.Modifies.Descriptors | ppl.Modifies.Memlets | ppl.Modifies.Nodes

    def should_reapply(self, modified: ppl.Modifies) -> bool:
        return False

    def depends_on(self):
        return set()

    def apply_pass(self, sdfg: SDFG, _pipeline_results: dict[str, Any]) -> int | None:
        batched = 0
        for state in list(sdfg.states()):
            children = state.scope_children()
            for entry in list(children[None]):
                if not isinstance(entry, nodes.MapEntry) or entry.map.schedule == dtypes.ScheduleType.Sequential:
                    continue
                body = [n for n in children[entry] if n is not state.exit_node(entry)]
                if len(body) == 1 and isinstance(body[0], nodes.LibraryNode) and batch(state, entry, body[0]):
                    batched += 1
        return batched or None
