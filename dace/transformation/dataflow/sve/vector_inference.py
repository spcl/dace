from samples.fpga.jacobi_fpga_systolic import T
from tests import dynamic_sdfg_functions_test
from networkx import DiGraph
from dace.memlet import Memlet
import itertools
from collections import defaultdict
from copy import copy
from math import isfinite
from dace.sdfg.utils import dfs_topological_sort, find_input_arraynode
from dace.sdfg.graph import MultiConnectorEdge, SubgraphView
import dace
from dace import SDFG, SDFGState
import dace.sdfg.nodes as nodes
from collections import defaultdict
import itertools
import dace.transformation.dataflow.sve.infer_types as infer_types
import copy
import dace.dtypes as dtypes
from dace.sdfg.scope import ScopeSubgraphView
import dace.data as data
from typing import *


class VectorInferenceException(Exception):
    def __init__(self, msg):
        super().__init__(msg)


class InferenceNode():
    Scalar = 0
    Vector = 1
    Unknown = -1

    def __init__(self, belongs_to):
        self.belongs_to = belongs_to
        self.inferred = InferenceNode.Unknown

    def is_inferred(self):
        return self.inferred != InferenceNode.Unknown

    def infer_as(self, inf_type: int):
        if inf_type != InferenceNode.Scalar and inf_type != InferenceNode.Vector:
            raise ValueError('Can only make node into Vector or Scalar')

        if self.is_inferred() and self.inferred != inf_type:
            # Node has already been inferred, and it is again inferred to a different type
            raise VectorInferenceException(
                f'Inference failed: re-assigning {self.inferred} -> {inf_type}')

        self.inferred = inf_type


# TODO: Implement modifying the subset.
class VectorInferenceGraph(DiGraph):
    def __init__(self, sdfg: SDFG, state: SDFGState, subgraph: SubgraphView,
                 param: str, vec_len):
        """
            Builds a vector inference graph for a Map to infer vectorizable Tasklet connectors
            and AccessNodes in polynomial time.

            :param sdfg: The SDFG where the Map resides.
            :param state: The state where the Map resides.
            :param subgraph: The subgraph of the Map without the entry and exit node.
            :param param: The loop param of the vectorized dimension.
            :param vec_len: The vector length that should be used when creating a `dtypes.vector`.     
        """
        super().__init__()
        self.sdfg = sdfg
        self.state = state
        self.subgraph = subgraph
        self.inf: infer_types.TypeInferenceDict = infer_types.infer_connector_types(
            sdfg, state, subgraph)
        self.param = param
        self.vec_len = vec_len
        self.conn_to_node: DefaultDict[Union[Tuple[nodes.Tasklet, str, bool],
                                             nodes.AccessNode]] = defaultdict(
                                                 lambda: None)

        self._build()
        self._detect_constraints()

    def set_constraint(self, conn: Union[Tuple[nodes.Tasklet, str, bool],
                                         nodes.AccessNode], infer_type: int):
        """
            Allows to manually specify a constraint either on a Tasklet connector
            by providing a tuple `(node, connector, is_input)` or a Scalar AccessNode.
            Should be done before calling `infer()`.
        """
        self.conn_to_node[conn].infer_as(infer_type)

    def get_constraint(
            self, conn: Union[Tuple[nodes.Tasklet, str, bool],
                              nodes.AccessNode]) -> int:
        """
            Allows to obtain the inferred constraint for a Tasklet connector or AccessNode.
            Should be done after calling `infer()`.
        """
        return self.conn_to_node[conn].inferred

    def _forward(self, node: InferenceNode):
        # Only vector constraints are only propagated forwards
        if node.inferred != InferenceNode.Vector:
            return
        for _, dst in self.out_edges(node):
            dst.infer_as(InferenceNode.Vector)
            self._forward(dst)

    def _backward(self, node: InferenceNode):
        # Only scalar constraints are only propagated backwards
        if node.inferred != InferenceNode.Scalar:
            return
        for src, _ in self.in_edges(node):
            src.infer_as(InferenceNode.Scalar)
            self._backward(src)

    def infer(self):
        """
            Infers by propagating the constraints through the graph.
        """
        # Propagate constraints forwards from source and backwards from sink
        for node in [n for n in self.nodes() if self.in_degree(n) == 0]:
            self._forward(node)
        for node in [n for n in self.nodes() if self.out_degree(n) == 0]:
            self._backward(node)

        # Make everything else scalar
        for node in self.nodes():
            if not node.is_inferred():
                node.infer_as(InferenceNode.Scalar)

    def _get_output_subsets(self, node: nodes.Tasklet) -> Dict[str, Set[str]]:
        """
            Computes for each output connector the set of input connectors for which
            if at least one of them is a vector, the output becomes a vector.
            :param node: The Tasklet to infer
            :returns: A dictionary for output connector -> set of inputs.
        """
        non_pointer_in_conns = [
            conn for conn in node.in_connectors
            if not isinstance(self.inf[(node, conn, True)], dtypes.pointer)
        ]
        non_pointer_out_conns = [
            conn for conn in node.out_connectors
            if not isinstance(self.inf[(node, conn, False)], dtypes.pointer)
        ]

        unit_outputs = {}

        # Turn each non-pointer input into a vector for once
        for inp in non_pointer_in_conns:
            # Dictionary that stores the in and out connector `dtypes` for the type inference
            in_dict = infer_types.TypeInferenceDict()

            for conn in node.in_connectors:
                in_dict[(node, conn, True)] = self.inf[(node, conn, True)]

            # Only set the pointer out connectors
            for conn in node.out_connectors:
                if isinstance(self.inf[(node, conn, False)], dtypes.pointer):
                    in_dict[(node, conn, False)] = self.inf[(node, conn, False)]

            # Toggle the "unit" vector input
            in_dict[(node, inp,
                     True)] = self._as_type(self.inf[(node, inp, True)],
                                            InferenceNode.Vector)

            # Infer the outputs
            infer_types.infer_tasklet_connectors(self.sdfg, self.state, node,
                                                 in_dict)

            outp = {}

            # Detect and store the output connector types as the output combination
            for conn in non_pointer_out_conns:
                if isinstance(in_dict[(node, conn, False)], dtypes.vector):
                    outp[conn] = InferenceNode.Vector
                else:
                    outp[conn] = InferenceNode.Scalar

            # Add to the list of combinations
            unit_outputs[inp] = outp

        relation = {}

        # Infer the input->output relation for each output connector
        for outp in non_pointer_out_conns:
            relation[outp] = set()
            for inp in non_pointer_in_conns:
                if unit_outputs[inp][outp] == InferenceNode.Vector:
                    # The input causes the output to become vector
                    relation[outp].add(inp)

        return relation

    def _try_add_edge(self, src, dst):
        """
            Adds an edge only if both source and destination are not `None`.
            This is used when building the graph and some connector might be dropped.
        """
        if src is not None and dst is not None:
            self.add_edge(src, dst)

    def _build(self):
        """
            Builds the vector inference graph.
        """
        # Create all necessary nodes
        for node in dfs_topological_sort(self.subgraph):
            if isinstance(node, nodes.Tasklet):
                non_pointer_in_conns = [
                    conn for conn in node.in_connectors
                    if not isinstance(self.inf[(node, conn,
                                                True)], dtypes.pointer)
                ]
                non_pointer_out_conns = [
                    conn for conn in node.out_connectors
                    if not isinstance(self.inf[(node, conn,
                                                False)], dtypes.pointer)
                ]

                # Create a node for every non-pointer input connector
                in_nodes = {}
                for conn in non_pointer_in_conns:
                    n = InferenceNode((node, conn, True))
                    self.conn_to_node[(node, conn, True)] = n
                    in_nodes[conn] = n
                    self.add_node(n)

                # Create a node for every non-pointer output connector
                out_nodes = {}
                for conn in non_pointer_out_conns:
                    n = InferenceNode((node, conn, False))
                    self.conn_to_node[(node, conn, False)] = n
                    out_nodes[conn] = n
                    self.add_node(n)

                # Connect the inputs of every union to its corresponding output
                for out, inputs in self._get_output_subsets(node).items():
                    for inp in inputs:
                        self.add_edge(in_nodes[inp], out_nodes[out])

            elif isinstance(node, nodes.AccessNode):
                desc = node.desc(self.sdfg)
                if isinstance(desc, data.Scalar):
                    # Only create nodes for Scalar AccessNodes (they can get a vector dtype)
                    n = InferenceNode(node)
                    self.conn_to_node[node] = n
                    self.add_node(n)

            else:
                # Some other node occurs in the graph, not supported
                raise VectorInferenceException(
                    'Only Tasklets and AccessNodes are supported')

        # Create edges based on connectors
        for node in dfs_topological_sort(self.subgraph):
            if isinstance(node, nodes.Tasklet):
                for e in self.state.in_edges(node):
                    if isinstance(e.src, nodes.Tasklet):
                        self._try_add_edge(
                            self.conn_to_node[(e.src, e.src_conn, False)],
                            self.conn_to_node[(node, e.dst_conn, True)])
                    elif isinstance(e.src, nodes.AccessNode):
                        self._try_add_edge(
                            self.conn_to_node[e.src],
                            self.conn_to_node[(node, e.dst_conn, True)])
            elif isinstance(node, nodes.AccessNode):
                for e in self.state.in_edges(node):
                    if isinstance(e.src, nodes.Tasklet):
                        self._try_add_edge(
                            self.conn_to_node[(e.src, e.src_conn, False)],
                            self.conn_to_node[node])
                    elif isinstance(e.src, nodes.AccessNode):
                        # TODO: What does that mean?
                        self._try_add_edge(self.conn_to_node[e.src],
                                           self.conn_to_node[node])

    def _as_type(self, dtype: dace.typeclass, inf_type: int) -> dace.typeclass:
        """
            Turns a typeclass into a scalar or vector.
        """
        if isinstance(dtype, dtypes.pointer):
            raise ValueError('Pointer was provided')
        elif isinstance(dtype, dtypes.vector):
            if inf_type == InferenceNode.Vector:
                return dtype
            else:
                raise VectorInferenceException('Cannot make vector into scalar')
        else:
            if inf_type == InferenceNode.Vector:
                return dtypes.vector(dtype, self.vec_len)
            else:
                return dtype

    def _carries_vector_data(self, edge: MultiConnectorEdge[Memlet]) -> bool:
        if edge.data.data is None:
            return False
        if edge.data.subset.num_elements() != 1:
            return False
        if not self.param in edge.data.subset.free_symbols:
            return False
        return True

    def _carries_scalar_data(self, edge: MultiConnectorEdge[Memlet]) -> bool:
        if edge.data.data is None:
            return False
        if edge.data.subset.num_elements() != 1:
            return False
        if self.param in edge.data.subset.free_symbols:
            return False
        return True

    def _detect_constraints(self):
        """
            Detects scalar/vector constraints on the graph based on the following two rules:

            * Reads/writes containing the loop param are Vectors
            * Reads/writes from/to an Array access node without loop param is always a Scalar
        """
        for node in dfs_topological_sort(self.subgraph):
            if isinstance(node, nodes.Tasklet):
                for edge in self.state.in_edges(node):
                    if self._carries_vector_data(edge):
                        # In connector must be vector since Memlet carries vector data
                        self.conn_to_node[(node, edge.dst_conn,
                                           True)].infer_as(InferenceNode.Vector)
                    elif self._carries_scalar_data(edge):
                        # Reading a scalar (with no loop param) from an Array
                        # AccessNode is always a scalar
                        if isinstance(edge.src,
                                      nodes.AccessNode) and isinstance(
                                          edge.src.desc(self.sdfg), data.Array):
                            self.conn_to_node[(node, edge.dst_conn,
                                               True)].infer_as(
                                                   InferenceNode.Scalar)

                for edge in self.state.out_edges(node):
                    if self._carries_vector_data(edge):
                        # Out connector must be vector since Memlet carries vector data
                        self.conn_to_node[(node, edge.src_conn,
                                           False)].infer_as(
                                               InferenceNode.Vector)
                    elif self._carries_scalar_data(edge):
                        # Writing a scalar (with no loop param) from an Array
                        # AccessNode is always a scalar
                        if isinstance(edge.dst,
                                      nodes.AccessNode) and isinstance(
                                          edge.dst.desc(self.sdfg), data.Array):
                            self.conn_to_node[(node, edge.src_conn,
                                               False)].infer_as(
                                                   InferenceNode.Scalar)

    def apply(self):
        """
            Applies the inference on the graph by making the suitable connectors vectors.
        """
        for node in self.nodes():
            if isinstance(node.belongs_to, nodes.AccessNode):
                t = node.belongs_to.desc(self.sdfg).dtype
                node.belongs_to.desc(self.sdfg).dtype = self._as_type(
                    t, node.inferred)
            else:
                n, c, i = node.belongs_to
                if i:
                    n.in_connectors[c] = self._as_type(
                        self.inf[node.belongs_to], node.inferred)
                else:
                    n.out_connectors[c] = self._as_type(
                        self.inf[node.belongs_to], node.inferred)


def infer_vectors(sdfg: SDFG,
                  state: SDFGState,
                  subgraph: SubgraphView,
                  param: str,
                  vec_len,
                  apply: bool = True) -> VectorInferenceGraph:
    """
        Builds a vector inference graph for a Map to infer vectorizable Tasklet connectors
        and AccessNodes in polynomial time. Applies the changes on the SDFG if `apply` is `True`.

        :raises VectorInferenceException: If some constraints are violated and inference was not successful.
        :param sdfg: The SDFG where the Map resides.
        :param state: The state where the Map resides.
        :param subgraph: The subgraph of the Map without the entry and exit node.
        :param param: The loop param of the vectorized dimension.
        :param vec_len: The vector length that should be used when creating a `dtypes.vector`.
        :returns: The inference graph for analysis.
    """
    graph = VectorInferenceGraph(sdfg, state, subgraph, param, vec_len)
    graph.infer()
    if apply:
        graph.apply()
    return graph
