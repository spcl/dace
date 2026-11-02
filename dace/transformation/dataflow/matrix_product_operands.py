# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Operations on the operands of a matrix product that the product can perform as it reads them: a gather (as an index
connector, see ``dace.libraries.blas.indexed``) and a transposition (as ``transA`` or ``transB``). Both remove the
operation and its intermediate array."""

from dace import SDFG, SDFGState, nodes
from dace.libraries.blas import Gemm, Gemv, MatMul
from dace.libraries.blas.blas_helpers import matrix_view
from dace.libraries.blas.indexed import index_connector, operand_edges
from dace.libraries.blas.nodes.batched_matmul import BatchedMatMul
from dace.libraries.linalg import Transpose
from dace.libraries.standard import Gather
from dace.properties import make_properties
from dace.sdfg import utils as sdutil
from dace.transformation import transformation

#: The matrix-product nodes, and the operand connectors whose transposition each has as a property
PRODUCTS = (MatMul, Gemm, Gemv, BatchedMatMul)
TRANSPOSED = {MatMul: {"_a": "transA", "_b": "transB"}, Gemm: {"_a": "transA", "_b": "transB"}, Gemv: {"_A": "transA"}}


def _only_feeds(state: SDFGState, sdfg: SDFG, producer: nodes.Node, array: nodes.AccessNode, product) -> str | None:
    """The connector of ``product`` that ``array`` feeds, if it is a transient that only carries the whole output of
    ``producer`` to that connector; None otherwise."""
    desc = sdfg.arrays[array.data]
    if not desc.transient or state.in_degree(array) != 1 or state.out_degree(array) != 1:
        return None
    if state.in_edges(array)[0].src is not producer:
        return None
    if any(n.data == array.data for s in sdfg.states() for n in s.data_nodes() if n is not array):
        return None
    edge = state.out_edges(array)[0]
    if edge.dst is not product or edge.data.subset is None:
        return None
    if edge.data.subset.size() != list(desc.shape) or any(r[0] != 0 for r in edge.data.subset):
        return None
    return edge.dst_conn


def _bypass(state: SDFGState, sdfg: SDFG, producer: nodes.Node, array: nodes.AccessNode) -> None:
    """Removes ``producer`` and the intermediate ``array``, and its descriptor."""
    state.remove_node(producer)
    state.remove_node(array)
    sdfg.remove_data(array.data, validate=False)


@make_properties
@transformation.explicit_cf_compatible
class GatherIntoMatrixProduct(transformation.SingleStateTransformation):
    """A gather of the rows of a matrix-product operand, ``(x[rows]) @ w``, as an index connector of the product: the
    product reads ``x`` through ``rows`` instead of the gathered copy."""

    gather = transformation.PatternNode(Gather)
    array = transformation.PatternNode(nodes.AccessNode)
    product = transformation.PatternNode(nodes.LibraryNode)

    @classmethod
    def expressions(cls):
        return [sdutil.node_path_graph(cls.gather, cls.array, cls.product)]

    def can_be_applied(self, graph: SDFGState, expr_index: int, sdfg: SDFG, permissive: bool = False) -> bool:
        if not isinstance(self.product, PRODUCTS) or self.gather.axis != 0:
            return False
        connector = _only_feeds(graph, sdfg, self.gather, self.array, self.product)
        if connector is None or index_connector(connector) in self.product.in_connectors:
            return False
        # The product's view of the operand, and of the array it gathers from, keep the gathered dimension first
        edges = self.gather.validate(sdfg, graph)
        _, gathered = matrix_view(graph.out_edges(self.array)[0].data.subset)
        _, source = matrix_view(edges["_inp"].data.subset)
        index_size = edges["_index"].data.subset.size()
        return gathered[:1] == [0] and source[:1] == [0] and len(gathered) == len(source) + len(index_size) - 1

    def apply(self, graph: SDFGState, sdfg: SDFG):
        edges = self.gather.validate(sdfg, graph)
        connector = graph.out_edges(self.array)[0].dst_conn
        self.product.add_in_connector(index_connector(connector))
        for gather_connector, product_connector in (("_inp", connector), ("_index", index_connector(connector))):
            e = edges[gather_connector]
            graph.add_edge(e.src, e.src_conn, self.product, product_connector, e.data)
        _bypass(graph, sdfg, self.gather, self.array)


@make_properties
@transformation.explicit_cf_compatible
class TransposeIntoMatrixProduct(transformation.SingleStateTransformation):
    """A transposed matrix-product operand, ``a @ w.T``, as the product's ``transA`` or ``transB``: the product reads
    ``w`` instead of the transposed copy."""

    transpose = transformation.PatternNode(Transpose)
    array = transformation.PatternNode(nodes.AccessNode)
    product = transformation.PatternNode(nodes.LibraryNode)

    @classmethod
    def expressions(cls):
        return [sdutil.node_path_graph(cls.transpose, cls.array, cls.product)]

    def can_be_applied(self, graph: SDFGState, expr_index: int, sdfg: SDFG, permissive: bool = False) -> bool:
        transposed = TRANSPOSED.get(type(self.product))
        if transposed is None:
            return False
        connector = _only_feeds(graph, sdfg, self.transpose, self.array, self.product)
        if connector not in transposed or index_connector(connector) in self.product.in_connectors:
            return False
        if isinstance(self.product, MatMul):
            # A product with a batched operand specializes to BatchedMatMul, which does not take transA or transB
            for e in operand_edges(graph, self.product):
                if e.dst_conn in ("_a", "_b") and len(matrix_view(e.data.subset)[0]) > 2:
                    return False
        return True

    def apply(self, graph: SDFGState, sdfg: SDFG):
        connector = graph.out_edges(self.array)[0].dst_conn
        flag = TRANSPOSED[type(self.product)][connector]
        setattr(self.product, flag, not getattr(self.product, flag))
        (e,) = graph.in_edges(self.transpose)
        graph.add_edge(e.src, e.src_conn, self.product, connector, e.data)
        _bypass(graph, sdfg, self.transpose, self.array)
