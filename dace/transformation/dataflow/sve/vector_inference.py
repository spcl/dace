from tests import dynamic_sdfg_functions_test
from networkx import DiGraph
from dace.memlet import Memlet
import itertools
from collections import defaultdict
from copy import copy
from math import isfinite
from dace.sdfg.utils import dfs_topological_sort, find_input_arraynode
from dace.sdfg.graph import SubgraphView
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


def power_set(iterable):
    s = list(iterable)
    return itertools.chain.from_iterable(
        itertools.combinations(s, r) for r in range(len(s) + 1))


class InferenceNode():
    Scalar = 0
    Vector = 1
    Unknown = -1

    def __init__(self, belongs_to):
        self.belongs_to = belongs_to
        self.inferred = InferenceNode.Unknown

    def is_inferred(self):
        return self.inferred != InferenceNode.Unknown

    def infer_as(self, to: int):
        if to != InferenceNode.Scalar and to != InferenceNode.Vector:
            raise ValueError('Can only make node into Vector or Scalar')

        if self.is_inferred() and self.inferred != to:
            # Node has already been inferred, and it is again inferred to a different type
            raise VectorInferenceException(
                f'Inference failed: re-assigning {self.inferred} -> {to}')

        self.inferred = to


class VectorInferenceGraph(DiGraph):
    def __init__(self, sdfg: SDFG, state: SDFGState, subgraph: SubgraphView,
                 param: str, vector_length):
        super().__init__()
        self.sdfg = sdfg
        self.state = state
        self.subgraph = subgraph
        self.inf = infer_types.infer_connector_types(sdfg, state, subgraph)
        self.param = param
        self.vector_length = vector_length
        self.conn_to_node = defaultdict(lambda: None)

        self._build()
        self._detect_constraints()

    def add_constraint(self, conn, infer_as: int):
        self.conn_to_node[conn].infer_as(infer_as)

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
        # Propagate constraints forwards from source and backwards from sink
        for node in [n for n in self.nodes() if self.in_degree(n) == 0]:
            self._forward(node)
        for node in [n for n in self.nodes() if self.out_degree(n) == 0]:
            self._backward(node)

        # Make everything else scalar
        for node in self.nodes():
            if not node.is_inferred():
                node.infer_as(InferenceNode.Scalar)

    def _get_output_input_unions(self, combinations):
        if len(combinations) == 0:
            return {}

        in_cons = combinations[0][0].keys()
        out_cons = combinations[0][1].keys()

        # For every output, find the union of inputs that produces the correct result
        # (according to the given combinations)
        in_power_set = [p for p in power_set(in_cons)]
        connections = {}
        for conn in out_cons:
            # Which inputs do we have to OR to always obtain the right output?
            for cand in in_power_set:
                # Candidate consisting of c_i's
                # Check if the OR of just these candidates always produces the expected
                # output for the current out connector
                failed = False
                # Check every in->out mapping
                for inp, out in combinations:
                    # Expected output
                    expected = out[conn]

                    # Computed output
                    measured = InferenceNode.Scalar
                    for c in list(cand):
                        measured |= inp[c]

                    if measured != expected:
                        # Does not match the required output
                        failed = True
                        break

                if not failed:
                    # Union satisifes all in->out mappings
                    connections[conn] = cand
                    break

        if len(connections.keys()) != len(out_cons):
            raise VectorInferenceException(
                'At least one output can not be represented as union of inputs')

        return connections

    def _get_tasklet_combinations(self, node: nodes.Tasklet):
        """
            Tries out every possible input combination of scalar and vector for
            inputs in the Tasklet, that are not already inferred as pointers and
            returns a mappings as list of tuple `(in, out)`, where `in` is a dictionary
            mapping from an input connector name to either `InferenceNode.Scalar` or
            `InferenceNode.Vector` and `out` is analogous the dictionary for the
            inferred output connectors.

            Only connectors that are not already pointers will occur as keys.
        """
        combinations = []

        non_pointer_in_conns = [
            conn for conn in node.in_connectors
            if not isinstance(self.inf[(node, conn, True)], dtypes.pointer)
        ]
        non_pointer_out_conns = [
            conn for conn in node.out_connectors
            if not isinstance(self.inf[(node, conn, False)], dtypes.pointer)
        ]

        # Try out every combination of scalar and vector
        for comb in itertools.product(
            [InferenceNode.Scalar, InferenceNode.Vector],
                repeat=len(non_pointer_in_conns)):
            # Dictionaries that store the mapping from the connector name
            # to either `InferenceNode.Scalar` or `InferenceNode.Vector`
            inp = {}
            outp = {}

            # Dictionary that stores the in and out connector `dtypes` for the type inference
            in_dict = infer_types.TypeInferenceDict()

            # Set the in and out connectors that are pointers (they don't change)
            # Just so `infer_tasklet_connectors` has all inputs and outputs properly defined
            for conn in node.in_connectors:
                if isinstance(self.inf[(node, conn, True)], dtypes.pointer):
                    in_dict[(node, conn, True)] = self.inf[(node, conn, True)]
            for conn in node.out_connectors:
                if isinstance(self.inf[(node, conn, False)], dtypes.pointer):
                    in_dict[(node, conn, False)] = self.inf[(node, conn, False)]

            # Setup the non-pointer inputs according to the current combination
            for num, conn in enumerate(non_pointer_in_conns):
                # Input combination is just the combination currently used
                inp[conn] = comb[num]

                # Use the base type and turn it into vector or keep it scalar
                # (depending on the current combination)
                # Set the dtype for the inference
                base = self.inf[(node, conn, True)]
                in_dict[(node, conn, True)] = self._as_type(base, comb[num])

            # Infer the outputs
            infer_types.infer_tasklet_connectors(self.sdfg, self.state, node,
                                                 in_dict)

            # Detect and store the output connector types as the output combination
            for conn in non_pointer_out_conns:
                if isinstance(in_dict[(node, conn, False)], dtypes.vector):
                    outp[conn] = InferenceNode.Vector
                else:
                    outp[conn] = InferenceNode.Scalar

            # Add to the list of combinations
            combinations.append((inp, outp))

        # Return all pairs
        return combinations

    def _try_add_edge(self, src, dst):
        if src is not None and dst is not None:
            self.add_edge(src, dst)

    def _build(self):
        self.conn_to_node = defaultdict(lambda: None)

        # Create all necessary nodes
        for node in dfs_topological_sort(self.subgraph):
            if isinstance(node, nodes.Tasklet):
                # For a Tasklet compute its input to output combinations
                combinations = self._get_tasklet_combinations(node)
                if len(combinations) == 0:
                    continue

                # Create a node for every non-pointer input connector
                ins = combinations[0][0].keys()
                in_nodes = {}
                for conn in ins:
                    n = InferenceNode((node, conn, True))
                    self.conn_to_node[(node, conn, True)] = n
                    in_nodes[conn] = n
                    self.add_node(n)

                # Create a node for every non-pointer output connector
                outs = combinations[0][1].keys()
                out_nodes = {}
                for conn in outs:
                    n = InferenceNode((node, conn, False))
                    self.conn_to_node[(node, conn, False)] = n
                    out_nodes[conn] = n
                    self.add_node(n)

                # Connect the inputs of every union to its corresponding output
                for out, inputs in self._get_output_input_unions(
                        combinations).items():
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

    def _as_type(self, dtype, type):
        if isinstance(dtype, dtypes.pointer):
            raise ValueError('Pointer was provided')
        elif isinstance(dtype, dtypes.vector):
            if type == InferenceNode.Vector:
                return dtype
            else:
                raise VectorInferenceException('Cannot make vector into scalar')
        else:
            if type == InferenceNode.Vector:
                return dtypes.vector(dtype, self.vector_length)
            else:
                return dtype

    def _carries_vector_data(self, edge) -> bool:
        if edge.data.data is None:
            return False
        if edge.data.subset.num_elements() != 1:
            return False
        if not self.param in edge.data.subset.free_symbols:
            return False
        return True

    def _carries_scalar_data(self, edge) -> bool:
        if edge.data.data is None:
            return False
        if edge.data.subset.num_elements() != 1:
            return False
        if self.param in edge.data.subset.free_symbols:
            return False
        return True

    def _detect_constraints(self):
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
        for node in self.nodes():
            if isinstance(node.belongs_to, nodes.AccessNode):
                type = node.belongs_to.desc(self.sdfg).dtype
                node.belongs_to.desc(self.sdfg).dtype = self._as_type(
                    type, node.inferred)
            else:
                n, c, i = node.belongs_to
                if i:
                    n.in_connectors[c] = self._as_type(
                        self.inf[node.belongs_to], node.inferred)
                else:
                    n.out_connectors[c] = self._as_type(
                        self.inf[node.belongs_to], node.inferred)
