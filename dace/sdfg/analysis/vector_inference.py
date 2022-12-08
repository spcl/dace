from enum import Enum, Flag
from networkx import DiGraph
from dace.memlet import Memlet
from collections import defaultdict
from dace.sdfg.utils import dfs_topological_sort
from dace.sdfg.graph import MultiConnectorEdge, SubgraphView
import dace
from dace import SDFG, SDFGState
import dace.sdfg.nodes as nodes
from collections import defaultdict
import dace.transformation.dataflow.sve.infer_types as infer_types
import dace.dtypes as dtypes
import dace.data as data
from typing import *
import dace.symbolic as symbolic


class VectorInferenceFlags(Flag):
    # Allows stride vector loads/stores instead of just contiguous ones
    Allow_Stride = 1


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
        """
        Infers the internal node either as `Scalar` or `Vector`.
        Re-inferring it to a different type will raise a `VectorInferenceException`.
        """
        if inf_type != InferenceNode.Scalar and inf_type != InferenceNode.Vector:
            raise ValueError('Can only make node into Vector or Scalar')

        if self.is_inferred() and self.inferred != inf_type:
            # Node has already been inferred, and it is again inferred to a different type
            # Provide a meaningful exception message
            exp_str = 'Vector' if self.inferred == InferenceNode.Vector else 'Scalar'
            inf_str = 'Vector' if inf_type == InferenceNode.Vector else 'Scalar'
            if isinstance(self.belongs_to, nodes.AccessNode):
                raise VectorInferenceException(
                    f'Violating constraint @ {self.belongs_to} (expected: {exp_str}, inferred: {inf_str})')
            else:
                node, conn, inp = self.belongs_to
                in_out = 'input' if inp else 'output'
                raise VectorInferenceException(
                    f'Violating constraint @ {self.belongs_to[0]} at {in_out} connector "{conn}"  (expected: {exp_str}, inferred as: {inf_str})'
                )

        self.inferred = inf_type


class VectorInferenceGraph(DiGraph):
    # Default propagation mode, where vectors propagate forwards and scalars propagate backwards.
    Propagate_Default = 0

    # Inverted propagation mode, where vectors propagate backwards and scalars propagate forwards.
    Propagate_WCR = 1

    def __init__(self,
                 sdfg: SDFG,
                 state: SDFGState,
                 map_entry: nodes.MapEntry,
                 vec_len,
                 initial_constraints: Dict[Union[Tuple[nodes.Tasklet, str, bool], nodes.AccessNode], int] = None,
                 flags: VectorInferenceFlags = None):
        """
            Builds a vector inference graph for a Map to infer vectorizable Tasklet connectors
            and AccessNodes in polynomial time.

            :param sdfg: The SDFG where the Map resides.
            :param state: The state where the Map resides.
            :param map_entry: The entry node of the Map.
            :param vec_len: The vector length that should be used when creating a `dtypes.vector`.
            :param initial_constraints: A dictionary mapping from a connector specified using `(node, name, is_input)`
                                        or an `AccessNode` to either `InferenceNode.Scalar` or `InferenceNode.Vector`.
            :param flags: Additional flags to limit the vectorization.
        """
        super().__init__()
        self.sdfg = sdfg
        self.state = state

        self.subgraph = state.scope_subgraph(map_entry, include_entry=False, include_exit=False)

        self.subgraph_with_scope = state.scope_subgraph(map_entry)

        self.map = map_entry.map

        # Infer connectors on the entire subgraph (including the entry and exit)
        self.inf: infer_types.TypeInferenceDict = infer_types.infer_connector_types(sdfg, state,
                                                                                    self.subgraph_with_scope)

        # Use the innermost loop param
        self.param = self.map.params[-1]

        self.vec_len = vec_len

        # Stores a mapping from SDFG nodes/connectors to InferenceNode's
        # Used when constructing the internal inference graph
        self.conn_to_node = DefaultDict[Union[Tuple[nodes.Tasklet, str, bool], nodes.AccessNode],
                                        InferenceNode](lambda: None)

        self.flags = flags

        self._build()
        self._detect_constraints()

        if initial_constraints is not None:
            for n, t in initial_constraints.items():
                self.set_constraint(n, t)

    def set_constraint(self, conn: Union[Tuple[nodes.Tasklet, str, bool], nodes.AccessNode], infer_type: int):
        """
            Allows to manually specify a constraint either on a Tasklet connector
            by providing a tuple `(node, connector, is_input)` or a Scalar AccessNode.
            Should be done before calling `infer()`.
        """
        self.conn_to_node[conn].infer_as(infer_type)

    def get_constraint(self, conn: Union[Tuple[nodes.Tasklet, str, bool], nodes.AccessNode]) -> int:
        """
            Allows to obtain the inferred constraint for a Tasklet connector or AccessNode.
            Should be done after calling `infer()`.
        """
        inf = self.conn_to_node.get(conn)
        if inf is None:
            return InferenceNode.Unknown
        else:
            return inf.inferred

    def _forward(self, node: InferenceNode):
        if node.inferred == InferenceNode.Unknown:
            # Nothing to propagate
            return
        for _, dst, data in self.out_edges(node, data=True):
            # In default mode, vector constraints are propagated forwards
            if data['mode'] == VectorInferenceGraph.Propagate_Default and node.inferred == InferenceNode.Vector:
                dst.infer_as(InferenceNode.Vector)
            # In WCR mode, scalar constraints are propagated forwards
            if data['mode'] == VectorInferenceGraph.Propagate_WCR and node.inferred == InferenceNode.Scalar:
                dst.infer_as(InferenceNode.Scalar)

            self._forward(dst)

    def _backward(self, node: InferenceNode):
        if node.inferred == InferenceNode.Unknown:
            # Nothing to propagate
            return
        for src, _, data in self.in_edges(node, data=True):
            # In default mode, scalar constraints are propagated backwards
            if data['mode'] == VectorInferenceGraph.Propagate_Default and node.inferred == InferenceNode.Scalar:
                src.infer_as(InferenceNode.Scalar)
            # In WCR mode, vector constraints are propagated backwards
            if data['mode'] == VectorInferenceGraph.Propagate_WCR and node.inferred == InferenceNode.Vector:
                src.infer_as(InferenceNode.Vector)

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

        if self.flags is None or VectorInferenceFlags.Allow_Stride not in self.flags:
            # Make sure no stride vector memlets occur
            for edge, _ in self.subgraph_with_scope.all_edges_recursive():
                src_type = InferenceNode.Unknown
                if isinstance(edge.src, nodes.Tasklet):
                    src_type = self.get_constraint((edge.src, edge.src_conn, False))
                elif isinstance(edge.src, nodes.AccessNode):
                    src_type = self.get_constraint(edge.src)

                dst_type = InferenceNode.Unknown
                if isinstance(edge.dst, nodes.Tasklet):
                    dst_type = self.get_constraint((edge.dst, edge.dst_conn, True))
                elif isinstance(edge.dst, nodes.AccessNode):
                    dst_type = self.get_constraint(edge.dst)

                if src_type == InferenceNode.Vector or dst_type == InferenceNode.Vector:
                    if not edge.data.get_stride(self.sdfg, self.map) in [0, 1]:
                        raise VectorInferenceException(f'Found stride vector Memlet at {edge}')

        return False

    def _get_output_subsets(self, node: nodes.Tasklet) -> Dict[str, Set[str]]:
        """
            Computes for each output connector the set of input connectors for which
            if at least one of them is a vector, the output becomes a vector.

            :param node: The Tasklet to infer
            :return: A dictionary for output connector -> set of inputs.
        """
        non_pointer_in_conns = [
            conn for conn in node.in_connectors if not isinstance(self.inf[(node, conn, True)], dtypes.pointer)
        ]
        non_pointer_out_conns = [
            conn for conn in node.out_connectors if not isinstance(self.inf[(node, conn, False)], dtypes.pointer)
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
            in_dict[(node, inp, True)] = self._as_type(self.inf[(node, inp, True)], InferenceNode.Vector)

            # Infer the outputs
            infer_types.infer_tasklet_connectors(self.sdfg, self.state, node, in_dict)

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

    def _try_add_edge(self, src, dst, mode):
        """
            Adds an edge only if both source and destination are not `None`.
            This is used when building the graph and some connector might be dropped.
        """
        if src is not None and dst is not None:
            self.add_edge(src, dst, mode=mode)

    def _get_propagation_mode(self, edge: MultiConnectorEdge):
        """
            Determines the propagation mode of an SDFG edge.
        """
        if edge.data.wcr is None:
            return VectorInferenceGraph.Propagate_Default
        else:
            return VectorInferenceGraph.Propagate_WCR

    def _build(self):
        """
            Builds the vector inference graph.
        """
        # Create all necessary nodes
        for node in dfs_topological_sort(self.subgraph):
            if isinstance(node, nodes.Tasklet):
                non_pointer_in_conns = [
                    conn for conn in node.in_connectors if not isinstance(self.inf[(node, conn, True)], dtypes.pointer)
                ]
                non_pointer_out_conns = [
                    conn for conn in node.out_connectors
                    if not isinstance(self.inf[(node, conn, False)], dtypes.pointer)
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
                        self.add_edge(in_nodes[inp], out_nodes[out], mode=VectorInferenceGraph.Propagate_Default)

            elif isinstance(node, nodes.AccessNode):
                desc = node.desc(self.sdfg)
                if isinstance(desc, data.Scalar):
                    # Only create nodes for Scalar AccessNodes (they can get a vector dtype)
                    n = InferenceNode(node)
                    self.conn_to_node[node] = n
                    self.add_node(n)

            else:
                # Some other node occurs in the graph, not supported
                raise VectorInferenceException('Only Tasklets and AccessNodes are supported')

        # Create edges based on connectors
        for node in dfs_topological_sort(self.subgraph):
            if isinstance(node, nodes.Tasklet):
                for e in self.state.in_edges(node):
                    if isinstance(e.src, nodes.Tasklet):
                        self._try_add_edge(self.conn_to_node[(e.src, e.src_conn, False)],
                                           self.conn_to_node[(node, e.dst_conn, True)], self._get_propagation_mode(e))
                    elif isinstance(e.src, nodes.AccessNode):
                        self._try_add_edge(self.conn_to_node[e.src], self.conn_to_node[(node, e.dst_conn, True)],
                                           self._get_propagation_mode(e))
            elif isinstance(node, nodes.AccessNode):
                for e in self.state.in_edges(node):
                    if isinstance(e.src, nodes.Tasklet):
                        self._try_add_edge(self.conn_to_node[(e.src, e.src_conn, False)], self.conn_to_node[node],
                                           self._get_propagation_mode(e))
                    elif isinstance(e.src, nodes.AccessNode):
                        # TODO: What does that mean?
                        self._try_add_edge(self.conn_to_node[e.src], self.conn_to_node[node],
                                           self._get_propagation_mode(e))

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
        if edge.data.get_stride(self.sdfg, self.map) == 0:
            return False
        if edge.data.wcr is not None:
            return False
        return True

    def _carries_scalar_data(self, edge: MultiConnectorEdge[Memlet]) -> bool:
        if edge.data.data is None:
            return False
        if edge.data.subset.num_elements() != 1:
            return False
        if self.param in edge.data.subset.free_symbols:
            return False
        if edge.data.get_stride(self.sdfg, self.map) != 0:
            return False
        if edge.data.wcr is not None:
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
                        self.conn_to_node[(node, edge.dst_conn, True)].infer_as(InferenceNode.Vector)
                    elif self._carries_scalar_data(edge):
                        # Reading a scalar (with no loop param) from an Array
                        # AccessNode is always a scalar
                        src_node = self.state.memlet_path(edge)[0].src
                        if isinstance(src_node, nodes.AccessNode) and isinstance(src_node.desc(self.sdfg), data.Array):
                            self.conn_to_node[(node, edge.dst_conn, True)].infer_as(InferenceNode.Scalar)

                for edge in self.state.out_edges(node):
                    if self._carries_vector_data(edge):
                        # Out connector must be vector since Memlet carries vector data
                        self.conn_to_node[(node, edge.src_conn, False)].infer_as(InferenceNode.Vector)
                    elif self._carries_scalar_data(edge):
                        # Writing a scalar (with no loop param) to an Array
                        # AccessNode is always a scalar
                        dst_node = self.state.memlet_path(edge)[-1].dst
                        if isinstance(dst_node, nodes.AccessNode) and isinstance(dst_node.desc(self.sdfg), data.Array):
                            self.conn_to_node[(node, edge.src_conn, False)].infer_as(InferenceNode.Scalar)

    def _vectorize_subset(self, edge: MultiConnectorEdge[Memlet]):
        """
            Vectorize the subset of the memlet on an edge (if possible).
        """
        # Check that the edge stores vector data
        if not self._carries_vector_data(edge):
            return

        # Possibly multidimensional subset, find the dimension where the param occurs
        vec_dim = None
        loop_sym = symbolic.pystr_to_symbolic(self.param)
        for dim, sub in enumerate(edge.data.subset):
            if loop_sym in symbolic.pystr_to_symbolic(sub[0]).free_symbols:
                if vec_dim is None:
                    vec_dim = dim
                else:
                    # Param occurs in multiple dimensions
                    # TODO: Requires flattening!
                    return

        stride = edge.data.get_stride(self.sdfg, self.map)

        # Update the subset using the stride and the vector length on the correct dimension
        sub = edge.data.subset[vec_dim]
        edge.data.subset[vec_dim] = (sub[0], sub[1] + stride * self.vec_len, stride)

    def apply(self):
        """
            Applies the inference on the graph by making the suitable connectors vectors.
            Also sets the dtypes accordingly.
        """
        infer_types.apply_connector_types(self.inf)
        for node in self.nodes():
            if isinstance(node.belongs_to, nodes.AccessNode):
                t = node.belongs_to.desc(self.sdfg).dtype
                node.belongs_to.desc(self.sdfg).dtype = self._as_type(t, node.inferred)
            else:
                n, c, i = node.belongs_to
                if i:
                    n.in_connectors[c] = self._as_type(self.inf[node.belongs_to], node.inferred)

                    if node.inferred == InferenceNode.Vector:
                        # Update the subset for all incoming edges
                        # (if they carry vector information)
                        for e in self.state.in_edges_by_connector(n, c):
                            self._vectorize_subset(e)
                else:
                    n.out_connectors[c] = self._as_type(self.inf[node.belongs_to], node.inferred)

                    # Update the subset for all outgoing edges
                    # (if they carry vector information)
                    if node.inferred == InferenceNode.Vector:
                        for e in self.state.out_edges_by_connector(n, c):
                            self._vectorize_subset(e)


def infer_vectors(sdfg: SDFG,
                  state: SDFGState,
                  map_entry: nodes.MapEntry,
                  vec_len,
                  initial_constraints: Dict[Union[Tuple[nodes.Tasklet, str, bool], nodes.AccessNode], int] = None,
                  flags: VectorInferenceFlags = None,
                  apply: bool = True) -> VectorInferenceGraph:
    """
        Builds a vector inference graph for a Map to infer vectorizable Tasklet connectors
        and AccessNodes in polynomial time. Applies the changes on the SDFG if `apply` is `True`.

        :raises VectorInferenceException: If some constraints are violated and inference was not successful.
        :param sdfg: The SDFG where the Map resides.
        :param state: The state where the Map resides.
        :param map_entry: The entry node of the Map.
        :param vec_len: The vector length that should be used when creating a `dtypes.vector`.
        :param initial_constraints: A dictionary mapping from a connector specified using `(node, name, is_input)`
                                    or an `AccessNode` to either `InferenceNode.Scalar` or `InferenceNode.Vector`.
        :param flags: Additional flags to limit the vectorization (e. g. allow stride loads).
        :param apply: Whether to apply the vectorization or not.
        :return: The inference graph for analysis.
    """
    graph = VectorInferenceGraph(sdfg, state, map_entry, vec_len, initial_constraints, flags)
    graph.infer()
    if apply:
        graph.apply()
    return graph
