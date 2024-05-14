"""Automatic Differentiation of SDFGStates.
   This module exposes the add_backward_pass method that can be used to add a backward pass to an
   SDFGState.
"""
import ast
import astunparse
import collections
import copy
import logging
import numbers
from typing import List, Tuple, Set, Dict, Union, Deque, cast, Optional, Callable, Sequence

import dace
import dace.sdfg.nodes as nodes
import dace.transformation.transformation as xf
import sympy as sp
from dace import dtypes, data as dt

from dace.frontend.operations import detect_reduction_type
from dace.sdfg import SDFG, SDFGState, graph as dgraph, state as dstate, utils as dutils, infer_types
from dace.memlet import Memlet

from dace.autodiff.base_abc import (BackwardContext, BackwardResult, AutoDiffException, find_backward_implementation)

from dace.autodiff.utils import cast_consts_to_type
from dace.util import find_str_not_in_set
from dace.libraries.onnx.forward_implementation_abc import ONNXForward
from dace.libraries.onnx.nodes.onnx_op import ONNXOp

from dace.codegen.targets.cpp import is_write_conflicted_with_reason

ReverseNodeReturnType = Tuple[nodes.Node, BackwardResult]

log = logging.getLogger(__name__)


def init_grad(data: str, sdfg: SDFG, current_state: SDFGState):
    """
    Add a state where `data` is initialized with zero.

    :param data: the data to initialize
    :param sdfg: the SDFG to add the state to
    :param current_state: the current state; the initialization will be done before this state
    """
    arr = sdfg.arrays[data]

    state = sdfg.add_state_before(current_state, label="init_" + data)

    scalar = 0
    if dtypes.can_access(dtypes.ScheduleType.CPU_Multicore, arr.storage):
        cuda = False
    elif dtypes.can_access(dtypes.ScheduleType.GPU_Default, arr.storage):
        cuda = True
    else:
        raise ValueError(f"Unsupported storage {arr.storage}")

    if isinstance(arr, (dt.Array, dt.Scalar)):
        state.add_mapped_tasklet(
            "_init_" + data + "_", {
                "i{}".format(i): "0:{}".format(shape)
                for i, shape in enumerate(arr.shape)
            }, {},
            "__out = {}".format(scalar),
            {"__out": dace.Memlet.simple(data, ", ".join("i{}".format(i) for i in range(len(arr.shape))))},
            schedule=dtypes.ScheduleType.GPU_Device if cuda else dtypes.ScheduleType.Default,
            external_edges=True)
    elif type(arr) is dt.View:
        # not need to initialize: the viewed array will always be visited
        # (since a view can never be a required grad), and thus the viewed array will be initialized.
        pass
    else:
        raise AutoDiffException("Unsupported data descriptor {}".format(arr))


def _strings_to_symbols(strings: Set[str]) -> Set[sp.Symbol]:
    return {sp.symbols(string) for string in strings}


def _symbols_to_strings(symbs: Set[sp.Symbol]) -> Set[str]:
    return {str(symb) for symb in symbs}


def generate_grad_connector_names(existing_connectors: Set[str], forward_connector_names: List[str]) -> Dict[str, str]:
    """ Choose connector names for the gradients of all forward connectors.

        :param existing_connectors: existing connectors on the node.
        :param forward_connector_names: the list of connectors to generate names for.
        :returns: a mapping from entries in ``forward_connector_names`` to names for those entries.
    """

    # copy
    existing_connectors = set(existing_connectors)

    names = {}
    for n in sorted(forward_connector_names):
        result = find_str_not_in_set(existing_connectors, n + "_gradient")
        names[n] = result
        existing_connectors.add(result)

    return names


def is_initialization_state(state: SDFGState) -> bool:
    """ Check if state is an initialization state, i.e. it initializes one or more arrays with zero values
    """
    for n in state.data_nodes():
        if len(state.out_edges(n)) > 0:
            return False
    return True


def code_to_exprs(code: str, inputs: Set[str], outputs: Set[str]) -> Dict[str, sp.Expr]:
    """ Convert a python string to a set of (simplified) symbolic sympy expressions. Currently, this
        supports only code consisting of assignment statements.

        :param code: the code to convert
        :param inputs: the inputs (i.e. the defined variables) for the code
        :param outputs: the outputs to generate simplified expressions for
        :return: map from outputs to symbolic expressions
    """

    inputs = list(inputs)
    outputs = list(outputs)

    code_fn = """
def symbolic_execution({}):
    # define functions from cmath.h
    from sympy import exp, log
    def log2(x):
        return log(x, 2)
    def log10(x):
        return log(x, 10)
    from sympy import sin, cos, tan, asin, acos, atan, sinh, cosh, tanh, asinh, acosh, atanh
    from sympy import sin, cos, tan, asin, acos, atan, sinh, cosh, tanh, asinh, acosh, atanh
    from sympy import Pow as pow, sqrt
    from sympy import sign, floor, ceiling as ceil, Abs as abs, Abs as fabs
    from sympy import Max as max, Min as min
    from sympy import Max as fmax, Min as fmin
    from sympy import erf
{}
    return {}
    """
    code_fn = code_fn.format(
        ", ".join(inputs),
        "\n".join("    " + line.strip() for line in code.split("\n")),
        ", ".join(outputs),
    )

    try:
        # need to have dace so things like `dace.float32(1)` work
        temp_globals = {'dace': dace}
        exec(code_fn, temp_globals)

        # no idea why, but simply calling symbolic_execution doesn't work
        results = temp_globals["symbolic_execution"](*[sp.symbols(inp) for inp in inputs])

        if len(outputs) > 1:
            return dict(zip(outputs, results))
        else:
            return {outputs[0]: results}
    except Exception as e:
        raise AutoDiffException(
            "Exception occured while attempting to symbolically execute code:\n{}".format(code)) from e


def _is_int_value(value, target_value: int) -> bool:
    if isinstance(value, numbers.Integral):
        return value == target_value

    if len(value.free_symbols) > 0 or int(value) != target_value:
        return False

    return True


def _add_through_connector(node: Union[nodes.MapEntry, nodes.MapExit]):
    i = 1
    while ("IN_{}".format(i) in node.in_connectors or "OUT_{}".format(i) in node.out_connectors):
        i += 1
    assert node.add_in_connector("IN_{}".format(i))
    assert node.add_out_connector("OUT_{}".format(i))
    return "IN_{}".format(i), "OUT_{}".format(i)


def _invert_map_connector(conn):
    if conn.startswith("IN"):
        return "OUT" + conn[2:]
    elif conn.startswith("OUT"):
        return "IN" + conn[3:]
    else:
        raise AutoDiffException("Could not parse map connector '{}'".format(conn))


def _has_inplace_operation(state: SDFGState) -> bool:
    """Returns true if state has any inplace operations
    Note that this method is currently much stronger than required; some of the constraints can be
    loosened in the future.
    """

    sdfg = state.parent

    # check that each data descriptor has at most one access nodes
    seen_accesses: Set[str] = set()
    for node in state.nodes():
        if isinstance(node, nodes.AccessNode):
            if node.data in seen_accesses:
                return True
            seen_accesses.add(node.data)

    # Edges with scalar memlets can be used to connect two code nodes together. If this feature is
    # used, it should be done using a new scalar every time.
    # When a scalar is used in a code -> code edge, it should also have an AccessNode that refers to it.
    seen_scalars = set()
    for edge in state.edges():
        memlet_data = edge.data.data
        if (isinstance(sdfg.arrays[memlet_data], dt.Scalar) and isinstance(edge.src, nodes.CodeNode)
                and isinstance(edge.dst, nodes.CodeNode)):
            if memlet_data in seen_scalars or memlet_data in seen_accesses:
                return True
            seen_scalars.add(memlet_data)
    return False


def _walk_up_memlet_tree_through_view_nodes(
        sdfg, forward_state, start_name) -> Tuple[Union[dt.Scalar, dt.Array], str, Deque[Tuple[str, dt.Data, Memlet]]]:
    """ Starting from the (singular) access node for ``start_name`` in ``forward_state``, walk up the
        memlet path until a non-view node is reached

        :param sdfg: the forward sdfg
        :param forward_state: the forward state
        :param start_name: the name of the array to start at
        :return: the descriptor at the root of the path, the name at the root of the path, the list of
                 array names, view data descriptor and memlets encountered along the path.
    """
    forwarded_name = start_name
    view_nodes_to_clone: Deque[Tuple[str, dt.Data, Memlet]] = collections.deque()
    if isinstance(sdfg.arrays[start_name], dt.View):
        # this is complicated slightly by views: we need to walk up the memlet path until we reach a
        # non-view access node. We then need to replicate the sequence of views in the backward SDFG.
        query = [n for n in forward_state.nodes() if isinstance(n, nodes.AccessNode) and n.data == start_name]
        if len(query) != 1:
            raise AutoDiffException(f"Could not find access node to forward with data {start_name}")
        current_node = query[0]
        while isinstance(sdfg.arrays[current_node.data], dt.View):

            in_edges = forward_state.in_edges(current_node)
            if len(in_edges) != 1:
                raise AutoDiffException(
                    f"Expected view node with in degree 1, got {len(in_edges)} for view node {current_node}")
            if not isinstance(in_edges[0].src, nodes.AccessNode):
                raise AutoDiffException(
                    f"Expected view node {current_node} to be connected to access node, got {in_edges[0].src}"
                    f" (of type {type(in_edges[0].src)})")
            view_nodes_to_clone.append((current_node.data, sdfg.arrays[current_node.data], in_edges[0].data))
            current_node = in_edges[0].src
            forwarded_name = current_node.data

    return sdfg.arrays[forwarded_name], forwarded_name, view_nodes_to_clone


def _path_src_node_in_subgraph(edge: dgraph.MultiConnectorEdge, subgraph: dstate.StateSubgraphView):
    path_src = subgraph.memlet_path(edge)[0].src
    return path_src in subgraph.nodes()


class BackwardPassGenerator:
    """ Class that holds the state for one backward pass creation.

        See autodiff.py, _reverse_NestedSDFG and pytorch.py for examples of usage.

        :param state: the forward pass to differentiate should be in this state
        :param given_gradients: the outputs that gradients must be provided for (i.e. access nodes will be created for
               these)
        :param required_gradients: the inputs to generate gradients for
        :param backward_sdfg: the sdfg the backward pass will be contained in. If it is the same as the forward_sdfg,
                              outputs must be a list containing a single scalar.
        :param backward_state: the state which the backward pass should be added to (must be added to `backward_sdfg`
                               before calling this method).
        :param zero_non_transients: Whether non-transient gradient buffers should be zero initialized in the backward
                                    SDFG.
        :param array_grad_map: A mapping from array name to the gradient array name. May be passed when certain
                               mappings already exist.
        :param conflicted_gradient_buffers: A list of forward pass value names for which multiple backward passes will
                                            be computed, and thus gradients should be computed with
                                            write-conflict-resolution.
    """

    def __init__(
            self,
            *,
            sdfg: SDFG,
            state: SDFGState,
            given_gradients: Sequence[Union[nodes.AccessNode, str]],
            required_gradients: Sequence[Union[nodes.AccessNode, str]],
            backward_sdfg: SDFG,  # this can be the same as SDFG
            backward_state: SDFGState,
            zero_non_transients: bool,
            array_grad_map: Optional[Dict[str, str]] = None,
            conflicted_gradient_buffers: Optional[Set[str]] = None):

        if backward_state not in backward_sdfg.nodes():
            raise AutoDiffException("Expected to find backward_state in backward_sdfg")

        def str_to_access(data: str, source: str) -> nodes.AccessNode:
            """
            Given a string containing the name of the accessed array, return the AccessNode in the state
            that points to this array.
            If there are multiple AccessNodes, the behaviour will depend on whether we want
            an output or input AccessNode.
            Input: We will return the first occurance of this node in the state and make sure there are 
                only outgoing edges from this node
            Output: We will return the last occurance of this node in the state 
                where the node only has incoming edges.
            """
            matches = [node for node in state.nodes() if isinstance(node, nodes.AccessNode) and node.data == data]
            # Unused in model
            if len(matches) == 0:
                return None
            if len(matches) == 1:
                # there is only a single AccessNode with this name
                return matches[0]
            # len(matches) > 1
            else:
                # There are multiple occurances of the same AccessNode
                if source == "inputs":
                    # we return the first node with this data
                    input_node: nodes.AccessNode = matches[0]

                    # There should not be any incoming edges for this node since
                    in_edges = state.in_edges(input_node)
                    assert len(in_edges) == 0

                    return input_node

                if source == "outputs":
                    # Go through the list of matches in reverse
                    for output_node in reversed(matches):
                        # We want the first node that has at least one incoming edge to it
                        # This represents the last time the output data was modified
                        in_edges = state.in_edges(output_node)
                        if len(in_edges) > 0:
                            return output_node

                    raise AutoDiffException(
                        f"The specified output {data} was not written to by any AccessNode in this state")

                raise AutoDiffException(f"There are multiple nodes with data {data} "
                                        f" but the source (inputs or outputs) was not specified correctly")

        given_gradients = [
            n if isinstance(n, nodes.AccessNode) else str_to_access(n, "outputs") for n in given_gradients
        ]
        required_gradients = [
            n if isinstance(n, nodes.AccessNode) else str_to_access(n, "inputs") for n in required_gradients
        ]
        required_gradients = [n for n in required_gradients if n is not None]

        self.given_gradients = given_gradients
        self.required_gradients = required_gradients

        self.input_names = {n.data for n in required_gradients}
        self.output_names = {n.data for n in given_gradients}

        self.sdfg = sdfg
        self.forward_state = state
        self.strategy = "recompute_all"
        self.backward_sdfg = backward_sdfg
        self.backward_state: SDFGState = backward_state

        #: arrays descs for the gradients
        self.backward_grad_arrays: Dict[str, dt.Array] = {}

        #: arrays descs for inputs that are required from the forward pass
        self.backward_input_arrays: Dict[str, dt.Array] = {}

        #: mapping from forward node -> backward node, and forward map -> backward map
        self.reverse_map: Dict[nodes.Node, Union[nodes.Node, nodes.Map]] = {}

        #: for replicated nodes from the fwd pass: mapping from fwd node to
        #: replicated node
        self.replicated_map: Dict[nodes.Node, nodes.Node] = {}

        #: mapping from forward_node -> BackwardResult for that node
        self.result_map: Dict[nodes.Node, BackwardResult] = {}

        #: mapping from forward name to gradient name for arrays
        self.array_grad_map: Dict[str, str] = array_grad_map or {}

        self.conflicted_gradient_buffers: Set[str] = conflicted_gradient_buffers or set()

        # checks if backward has already been applied
        self._applied = False
        self.zero_non_transients = zero_non_transients

        for outp in self.given_gradients:
            if outp not in self.forward_state:
                raise AutoDiffException("Could not find output {} in state {}".format(outp, self.forward_state))

        for inp in self.required_gradients:
            if inp not in self.forward_state:
                raise AutoDiffException("Could not find input {} in state {}".format(inp, self.forward_state))

        # check for inplace operations (i.e. duplicated access nodes)
        # if _has_inplace_operation(self.forward_state):
        #     raise AutoDiffException(
        #         "Inplace operations are currently not supported in autodiff")

        if sdfg is backward_sdfg:
            # this only makes sense if the output is a single scalar.
            if len(given_gradients) != 1:
                raise AutoDiffException("When the forward sdfg is the same as the backward sdfg, outputs must be a"
                                        "single scalar")
            if not _is_int_value(sdfg.arrays[given_gradients[0].data].total_size, 1):
                raise AutoDiffException("When the forward sdfg is the same as the backward sdfg, outputs must be a"
                                        "single scalar")
            self.separate_sdfgs = False
        else:
            self.separate_sdfgs = True

        self.completion_hooks: List[Callable[[BackwardPassGenerator], None]] = []

    def _expand_nodes(self, subgraph: dstate.StateSubgraphView) -> bool:
        """ Expand all library nodes in the graph to pure implementations. Returns whether something was expanded
        """
        expanded_something = False
        for node, state in subgraph.all_nodes_recursive():
            if isinstance(state, dstate.StateSubgraphView):
                state = state.graph

            # check if the node exists in the backward implementation repository
            if find_backward_implementation(state.parent, state, node) is not None:
                continue

            # only check others if we didn't break out of the above loop
            if isinstance(node, ONNXOp):
                impls = ONNXForward.registered_implementations(node.schema.name)

                # order the implementations so that implementations containing "pure" are tried first
                impls = [i for name, i in impls if "pure" in name] + [i for name, i in impls if "pure" not in name]
                for impl in impls:
                    if impl.forward_can_be_applied(node, state, self.sdfg):
                        # try to apply the expansion
                        class Expansion(xf.ExpandTransformation):
                            environments = impl.environments if hasattr(impl, "environments") else []
                            _expansion_result = None

                            @classmethod
                            def expansion(cls, node, state, sdfg):
                                return impl.forward(node, state, sdfg)

                            @staticmethod
                            def annotates_memlets() -> bool:
                                return True

                        Expansion._match_node = xf.PatternNode(type(node))
                        Expansion.apply_to(state.parent, verify=False, _match_node=node)
                        expanded_something = True
                        break

            # This could later on be changed to check if the expansion is differentiable and if not, move
            # on to the next expansion. For now we will just apply the first one that matches, prioritizing ones that
            # have "pure" in the name
            if isinstance(node, nodes.LibraryNode) and not isinstance(node, ONNXOp):
                # try to select an expansion
                if hasattr(node, "implementations"):
                    implementations = node.implementations

                    pure_candidates = [name for name, impl in sorted(implementations.items()) if "pure" in name]
                    if len(pure_candidates) > 0:
                        expansion = pure_candidates[0]
                    else:
                        expansion = node.implementation
                else:
                    expansion = node.implementation

                node.implementation = expansion
                node.expand(state.parent, state)
                expanded_something = True

        return expanded_something

    def _disambiguate_direction_dependent_views(self):
        """ Consider the following subgraph:
            (A) -- y --> (n) -- x --> (C)
            In dace, if B is a View node and A and C are access nodes, and y and x both have data set to A.data and
            B.data respectively, the semantics of the graph depend on the order in which it is executed, i.e. reversing
            the subgraph doesn't perform as expected anymore. To disambiguate this case, we set y.data to the View's
            data.
        """

        for n in self.forward_state.nodes():
            if isinstance(n, nodes.AccessNode) and type(n.desc(self.sdfg)) is dt.View:
                in_edges = self.forward_state.in_edges(n)
                out_edges = self.forward_state.out_edges(n)

                if len(in_edges) == 1 and len(out_edges) == 1:
                    A = in_edges[0].src
                    y = in_edges[0].data
                    C = out_edges[0].dst
                    x = out_edges[0].data
                    if (isinstance(A, nodes.AccessNode) and isinstance(C, nodes.AccessNode) and y.data == A.data
                            and x.data == C.data):

                        # flip the memlet
                        y.subset, y.other_subset = y.other_subset, y.subset
                        y.data = n.data
                        y.try_initialize(self.sdfg, self.forward_state, in_edges[0])

    def backward(self) -> Tuple[BackwardResult, Dict[str, dt.Array], Dict[str, dt.Array]]:
        """ Generate the backward pass in backward_state.

            :return: tuple of:
                     * the backward result (see :class:`~dace.autodiff.backward_implementation.BackwardResult`)
                     * dict of data descriptors for the gradients (i.e. the outputs of the backward pass)
                     * dict of data descriptors of required outputs from the forward pass. These need to be added to the
                       parent SDFG of the backward pass.
        """

        if self._applied:
            raise AutoDiffException("Backward may only be called once. Instantiate a new BackwardPassGenerator.")

        forward_subgraph = self._find_subgraph_to_differentiate()

        # expand until there is nothing left to expand
        while self._expand_nodes(forward_subgraph):
            # Nodes have been expanded again on the expanded graph; recalculate the forward graph
            forward_subgraph = self._find_subgraph_to_differentiate()

        # check that all edges are float
        for edge, parent_subgraph in forward_subgraph.all_edges_recursive():
            if isinstance(parent_subgraph, SDFGState):
                parent_sdfg = parent_subgraph.parent
            elif isinstance(parent_subgraph, dstate.StateSubgraphView):
                parent_sdfg = parent_subgraph.graph.parent
            elif isinstance(parent_subgraph, SDFG):
                # if there are any fancy things on the interstate edges we should probably throw an error
                continue
            else:
                raise AutoDiffException("Unexpected subgraph structure")

            if edge.data.data:
                edge_type = parent_sdfg.arrays[edge.data.data].dtype
                if edge_type not in [dace.float16, dace.float32, dace.float64]:
                    raise AutoDiffException(
                        f"Expected Subgraph to differentiate to only contain float edges, but data {edge.data}"
                        f" on edge {edge} has type {edge_type}")

        self._disambiguate_direction_dependent_views()

        # recursively reverse the subgraph
        self._reverse_subgraph(forward_subgraph)

        self._applied = True

        # in some cases (accessnode -> accessnode), the descriptors for the gradients of the function outputs are not
        # added yet. Add them now

        for given_grad in sorted(self.given_gradients, key=lambda k: k.data):
            if self.array_grad_name(given_grad.data) not in self.backward_sdfg.arrays:
                self._add_gradient_data_descriptor(given_grad.data)

        # execute hooks
        for hook in self.completion_hooks:
            hook(self)

        # prepare the output
        required_grad_names = {name.data: self.array_grad_name(name.data) for name in self.required_gradients}
        given_grad_names = {name.data: self.array_grad_name(name.data) for name in self.given_gradients}

        # set mapping from gradient name to whether it should be zeroed out on
        # initialization
        zero_init: Dict[str, bool] = {}
        for node, bres in self.result_map.items():
            for zname, zinit in bres.zero_init.items():
                # Reverse lookup
                cname = next(k for k, v in bres.required_grad_names.items() if v == zname)
                for e in forward_subgraph.in_edges_by_connector(node, cname):
                    zero_init[e.data.data] = zinit
                for e in forward_subgraph.out_edges_by_connector(node, cname):
                    zero_init[e.data.data] = zinit

        result = BackwardResult(required_grad_names=required_grad_names,
                                given_grad_names=given_grad_names,
                                zero_init=zero_init)
        return result, self.backward_grad_arrays, self.backward_input_arrays

    def _find_subgraph_to_differentiate(self) -> dstate.StateSubgraphView:
        """ Determine which nodes we need to reverse; this forms the subgraph we will differentiate:
            we do a reverse BFS and a forward BFS, then take the intersection of nodes found.

            To calculate the gradients for a node x in ``required_gradients``, we need to sum up consider the gradient
            contributions from every node y where x is used as an input. We thus first do a forward BFS. Also, the
            gradient contributions of all nodes that are not connected by a path to a ``given_gradient`` node are
            implicitly zero. Thus, we take the intersection of the two BFSs.
        """
        forward_nodes = {n for e in self.forward_state.bfs_edges(self.required_gradients) for n in [e.src, e.dst]}
        backward_nodes = {
            n
            for e in self.forward_state.bfs_edges(self.given_gradients, reverse=True)
            for n in [e.src, e.dst]
        }

        forward_subgraph = dstate.StateSubgraphView(self.forward_state,
                                                    list(forward_nodes.intersection(backward_nodes)))
        return forward_subgraph

    def array_grad_name(self, forward_name: str) -> str:
        """ Return the gradient name of a name from the forward pass """
        if forward_name not in self.array_grad_map:
            self.array_grad_map[forward_name] = \
                find_str_not_in_set(set(self.backward_sdfg.arrays), "gradient_" + forward_name)

        return self.array_grad_map[forward_name]

    def _init_grad(self, data: str):
        desc = self.backward_sdfg.arrays[data]
        # No need to initialize if gradients point to outputs
        if not self.zero_non_transients and not desc.transient:
            return

        init_grad(data, self.backward_sdfg, self.backward_state)

    def _reverse_subgraph(self, subgraph: dstate.StateSubgraphView):
        """ Reverse a given subgraph. All nodes in the subgraph will be reversed. """
        from dace.libraries.onnx.nodes.onnx_op import ONNXSum
        # a reversed topological sort is a topological sort on the reverse graph
        for node in reversed(list(dutils.dfs_topological_sort(subgraph, subgraph.source_nodes()))):

            try:
                # output names on the forward node
                # (for which the gradient will be connected as an input on the reverse node)
                given_gradients = [
                    edge.src_conn for edge in subgraph.out_edges(node) if _path_src_node_in_subgraph(edge, subgraph)
                ]

                # input names on the forward node that gradients should be generated for
                required_gradients = [
                    edge.dst_conn for edge in subgraph.in_edges(node) if _path_src_node_in_subgraph(edge, subgraph)
                ]

                reversed_node, backward_result = self._get_reverse_node(node, given_gradients, required_gradients)

                self.reverse_map[node] = reversed_node
                self.result_map[node] = backward_result

                # connect the required inputs of the reverse node:
                # the gradients ...
                self._connect_given_gradients(subgraph, node)
                # ... and any required input values from the forward pass

                ####################################
                # Determine which forward inputs we need to connect.
                # these are the in_connectors on the reverse node, minus what has already been connected.
                already_connected = {e.dst_conn for e in self.backward_state.in_edges(reversed_node)}
                required_inputs = set(reversed_node.in_connectors).difference(already_connected)
                required_inputs = {c: c for c in required_inputs}
                self._connect_forward_inputs(node, reversed_node, required_inputs)

                if isinstance(node, nodes.AccessNode):

                    # this means we are writing out a grad to an array.
                    # initialize the gradient if it hasn't been initialized already (this can also happen in
                    # _connect_given_gradients
                    array_grad_name = self.array_grad_name(node.data)
                    if array_grad_name not in self.backward_sdfg.arrays:
                        # this grad hasn't been written before: initialize it
                        self._add_gradient_data_descriptor(node.data)

                    # we need to make all incoming gradients sum
                    if self.backward_state.in_degree(reversed_node) > 1:
                        summation_node = ONNXSum(f"sum_{array_grad_name}")

                        grad_desc = self.backward_sdfg.arrays[array_grad_name]
                        cuda = False
                        # connect each incoming edge to the summation node
                        for i, edge in enumerate(self.backward_state.in_edges(reversed_node)):

                            intermediate_desc = copy.deepcopy(grad_desc)

                            intermediate_desc.transient = True
                            intermediate_name = self.backward_sdfg.add_datadesc(f"{array_grad_name}_contribution_{i}",
                                                                                intermediate_desc,
                                                                                find_new_name=True)
                            access_intermediate = self.backward_state.add_access(intermediate_name)

                            for mte in self.backward_state.memlet_tree(edge):
                                mte.data.data = intermediate_name
                            new_edge = self.backward_state.add_edge(edge.src, edge.src_conn, access_intermediate, None,
                                                                    edge.data)
                            self._set_wcr_sum_if_needed(new_edge)
                            summation_node.add_in_connector(f"data_0__{i}")
                            self.backward_state.add_edge(access_intermediate, None, summation_node, f"data_0__{i}",
                                                         self.backward_sdfg.make_array_memlet(intermediate_name))
                            self.backward_state.remove_edge(edge)

                        self.backward_state.add_edge(summation_node, "sum", reversed_node, None,
                                                     self.backward_sdfg.make_array_memlet(array_grad_name))

                        if dtypes.can_access(dtypes.ScheduleType.CPU_Multicore, grad_desc.storage):
                            pass
                        elif dtypes.can_access(dtypes.ScheduleType.GPU_Default, grad_desc.storage):
                            summation_node.schedule = dtypes.ScheduleType.GPU_Default
                        else:
                            raise ValueError(f"Unsupported storage {grad_desc.storage}")
                    elif self.backward_state.in_degree(reversed_node) == 1:
                        self._set_wcr_sum_if_needed(self.backward_state.in_edges(reversed_node)[0])

            except AutoDiffException as e:
                raise AutoDiffException("Failed at node {}: {}".format(node, str(e))) from e

    def _set_wcr_sum_if_needed(self, edge: dgraph.MultiConnectorEdge):
        """ Set the WCR to sum for all edges along the path of edge, if needed.

            :param edge: the root edge to start from
        """
        inverse_array_grad_map = {v: k for k, v in self.array_grad_map.items()}

        add_wcr = False

        # this method assumes that the memlet tree is iterated from the root backwards
        for path_edge in self.backward_state.memlet_tree(edge):
            data_name = path_edge.data.data
            if data_name in inverse_array_grad_map and inverse_array_grad_map[
                    data_name] in self.conflicted_gradient_buffers:
                add_wcr = True
                # NOTE even though init_grad is called below, the gradient
                # buffer will not actually be zeroed when
                # self.zero_non_transients is False (this is checked in
                # self._init_grad)
                break

            # set the wcr to sum temporarily so that the following works
            old_wcr = path_edge.data.wcr
            path_edge.data.wcr = "lambda x, y: x + y"
            if is_write_conflicted_with_reason(self.backward_state, path_edge):
                # if we have a write conflict, we need WCR
                add_wcr = True
            path_edge.data.wcr = old_wcr

            # count the amount of in edges per connector
            connector_in_edges = collections.defaultdict(int)
            for _, _, _, dst_conn, _ in self.backward_state.in_edges(path_edge.dst):
                connector_in_edges[dst_conn] += 1

            more_than_one_edge_to_connector = any(v > 1 for v in connector_in_edges.values())

            if more_than_one_edge_to_connector:
                add_wcr = True

        if add_wcr:
            for tree_edge in self.backward_state.memlet_tree(edge):
                tree_edge.data.wcr = "lambda x, y: x + y"
            self._init_grad(edge.data.data)

    def _add_gradient_data_descriptor(self, data_name: str):
        """ Add the data descriptor for the gradient for `data_name`.
            :param data_name: the name of the forward descriptor.
        """
        grad_name = self.array_grad_name(data_name)

        if grad_name in self.backward_sdfg.arrays:
            raise AutoDiffException(f"descriptor for gradient of {data_name} ({grad_name}) already exists")

        array = self.sdfg.arrays[data_name]

        if not isinstance(array, (dt.Scalar, dt.Array, dt.View)):
            raise AutoDiffException("Unsupported data descriptor {}".format(array))

        cloned_datadesc = copy.deepcopy(array)

        # only the grads of the inputs and the outputs are not transient
        cloned_datadesc.transient = data_name not in self.input_names and data_name not in self.output_names

        self.backward_grad_arrays[grad_name] = cloned_datadesc
        self.backward_sdfg.arrays[grad_name] = copy.deepcopy(cloned_datadesc)

    def _connect_given_gradients(self, subgraph: dstate.StateSubgraphView, forward_node):
        """ Connect the gradients of the outputs of forward_node as inputs to the corresponding reverse node. """

        for edge in subgraph.out_edges(forward_node):
            if not _path_src_node_in_subgraph(edge, subgraph):
                # skip connecting edges for which we don't need to generate grads.
                continue

            src_node, output_conn, dest_node, input_conn, memlet = edge
            if detect_reduction_type(memlet.wcr) not in [
                    None,
                    dtypes.ReductionType.Sum,
            ]:
                raise AutoDiffException("Unsupported reduction type {} on memlet".format(
                    detect_reduction_type(memlet.wcr)))

            memlet = copy.deepcopy(memlet)

            # remove the WCR since these are now read edges
            memlet.wcr = None

            grad_name = self.array_grad_name(memlet.data)
            if grad_name not in self.backward_sdfg.arrays:
                # this grad hasn't been written before: initialize it
                self._add_gradient_data_descriptor(memlet.data)
            memlet.data = grad_name

            self.backward_state.add_edge(
                self.reverse_map[dest_node],
                self._lookup_required_grad_name(dest_node, input_conn),
                self.reverse_map[forward_node],
                self._lookup_given_grad_name(forward_node, output_conn),
                memlet,
            )

    def _connect_forward_inputs(self, forward_node: nodes.Node, backward_node: nodes.Node, required_inputs: Dict[str,
                                                                                                                 str]):
        """ Connect the reversed node of `forward_node` to all required non-gradient inputs.

            There are non-trivial points to handle:
            1. When we read an input from an accessnode in the forward pass, we need to route through maps in the
               backward pass.
            2. In some cases, we need to save the value of a connector to an array so that the backward pass can
               read it.
            
            Currently we have initial support two strategies: store-all and recompute all.

            :param forward_node: the forward node.
            :param backward_node: the backward node. This must not necessarily be a reversed node.
            :param required_inputs: the inputs to connect to the backward node. These inputs must exist on the forward
                                    node. The dict maps the fwd pass connector we require to the connector that we
                                    should connect to.
        """

        if set(required_inputs).difference(forward_node.in_connectors):
            missing_connectors = \
                set(required_inputs).difference(forward_node.in_connectors)
            raise ValueError(f"Can't connect connectors"
                             f" {missing_connectors} to {backward_node} "
                             f"because they don't exist on the corresponding "
                             f"forward node {forward_node}")

        # note we use forward state here: we might need to connect inputs that are not in the
        # forward pass
        input_edges_to_connect = (edge for edge in self.forward_state.in_edges(forward_node)
                                  if edge.dst_conn in required_inputs)

        for edge in input_edges_to_connect:
            # boolean to decide if the source of this edge needs to be replicated
            replicate_node = False

            # boolean to decide if the connection to the replicated node is required
            # this is set to False if the connection has already been established
            connect_replicated_node = True
            edge_src = edge.src
            next_required_inputs: Dict[Optional[str], Optional[str]]
            replicated_edge_src: nodes.Node
            replicated_edge_src_conn: str
            if isinstance(edge_src, nodes.MapEntry):
                # in the map case, the map must already exist in the bwd pass
                # (the following function call internally asserts this)
                replicated_edge_src = self._find_backward_entry_node_for_map_entry(edge_src)
                new_in_conn, new_out_conn = _add_through_connector(replicated_edge_src)

                replicated_edge_src_conn = new_out_conn

                # the inverse connector of edge.src_conn should be connected
                # to the new through connector we made
                next_required_inputs = {_invert_map_connector(edge.src_conn): new_in_conn}

            elif isinstance(edge_src, nodes.AccessNode):
                is_base_level = self.forward_state.scope_dict()[edge_src] is None
                data_name = edge_src.data
                data_desc = copy.deepcopy(edge_src.desc(self.sdfg))
                if self.separate_sdfgs:
                    # need to copy over the descriptor from the forward pass
                    if data_name not in self.backward_sdfg.arrays:
                        self.backward_sdfg.add_datadesc(data_name, data_desc)

                if isinstance(data_desc, dt.View):
                    # View is not yet supported
                    raise AutoDiffException("Data Views are not yet supported for AD in DaCe.")
                elif not is_base_level:
                    # this is a temporary value that needs to be stored
                    # replicate the node now since we need to connect it to the new AccessNode
                    replicate_node = False
                    replicated_edge_src_conn = edge.src_conn

                    if edge_src in self.replicated_map:
                        replicated_edge_src = self.replicated_map[edge_src]
                    else:
                        # node has not been replicated yet: do it now
                        replicated_edge_src = copy.deepcopy(edge_src)
                        self.backward_state.add_node(replicated_edge_src)
                        self.replicated_map[edge_src] = replicated_edge_src

                    # add the connection from the repliacted node to the tasklet
                    new_edge_data = copy.deepcopy(edge.data)
                    self.backward_state.add_edge(replicated_edge_src, replicated_edge_src_conn, backward_node,
                                                 required_inputs[edge.dst_conn], new_edge_data)
                    connect_replicated_node = False

                    # get the sink edge going into this AccessNode
                    sink_edge = self.forward_state.in_edges(edge_src)

                    assert len(sink_edge) == 1
                    sink_edge = sink_edge[0]

                    # get the AccessNode that the temporary values are coming from
                    edge_list = self.forward_state.memlet_path(sink_edge)
                    assert len(edge_list) > 0
                    source_edge = edge_list[0]
                    original_accessnode = source_edge.src
                    assert isinstance(original_accessnode, nodes.AccessNode)

                    # check if this AccessNode has been overwritten
                    overwritten, recomputable = self._check_node_overwrite(edge_src)
                    if overwritten:
                        # we lost values that are necessary for the backward pass
                        if self.strategy == "store_all":
                            # we need to modify the forward pass to store these neccessary values
                            new_store_accessnode, memlets = self._store_data(source_edge)
                            self._connect_temporary_to_accessnode(new_store_accessnode, replicated_edge_src,
                                                                  memlet_list, sink_edge)
                        elif self.strategy == "recompute_all":
                            replicate_node = False
                            connect_replicated_node = False
                    else:
                        # the data has not been overwritten
                        # we just need to connect the original AccessNode in the backward state
                        # first, get the memlets
                        memlet_list = []

                        # for each map in the path
                        for e in edge_list:
                            edge_src = e.src
                            if isinstance(edge_src, nodes.MapEntry):
                                memlet_list.append(e.data)
                        # replicate the original access node
                        replicated_original_accessnode = copy.deepcopy(original_accessnode)
                        self.backward_state.add_node(replicated_original_accessnode)
                        # connect the nodes
                        self._connect_temporary_to_accessnode(replicated_original_accessnode, replicated_edge_src,
                                                              memlet_list, sink_edge)
                else:
                    # base-case: we have reached a base level AccessNode.
                    # check if this AccessNode has been overwritten
                    overwritten, recomputable = self._check_node_overwrite(edge_src)
                    if overwritten:
                        # we lost values that are necessary for the backward pass
                        if self.strategy == "store_all":
                            # we need to modify the forward pass to store these neccessary values
                            new_store_accessnode, memlets = self._store_data(edge)

                            # replicate the new store node from the forward state
                            replicated_new_store_accessnode = copy.deepcopy(new_store_accessnode)
                            self.backward_state.add_node(replicated_new_store_accessnode)
                            # we will traverse the memlets in the reverse order
                            # that they were added from the forward pass
                            last_memlet = memlets.pop()
                            last_memlet = copy.deepcopy(last_memlet)
                            # add the new edge
                            self.backward_state.add_edge(replicated_new_store_accessnode, None, backward_node,
                                                         required_inputs[edge.dst_conn], last_memlet)

                            # set the boolean to False to avoid adding the connection again
                            connect_replicated_node = False

                            # extract the edge to the new store node to get the memlet path
                            new_edge = self.backward_state.out_edges(replicated_new_store_accessnode)

                            # sanity check: there should be only one edge on this node
                            assert len(new_edge) == 1
                            new_edge = new_edge[0]

                            # get the memlet path to update the memlets accordingly
                            edge_list = self.backward_state.memlet_path(new_edge)

                            # modify the memlets of the backward connections to the maps
                            for e in edge_list:
                                edge_src = e.src
                                if isinstance(edge_src, nodes.MapEntry):
                                    memlet_data = memlets.pop()
                                    memlet_data = copy.deepcopy(memlet_data)
                                    e.data = memlet_data

                            # sanity check: there should be the same number of connections
                            assert len(memlets) == 0

                        elif self.strategy == "recompute_all":
                            replicate_node = False
                            connect_replicated_node = False
                            # if we can recompute this value
                            if recomputable:
                                # call a function that inserts a recomputation nested-sdfg into the map nest
                                self._recompute_data(edge)
                                connector_to_clean = required_inputs[edge.dst_conn]
                                self._clean_after_recomputation(edge, connector_to_clean)
                            else:
                                # throw an exception in case a value can't be recomputed and the recompute all strategy was used
                                raise AutoDiffException(
                                    f"Attempting to recompute the node {edge_src.data}, but this node is not recomputable."
                                )
                            pass
                    else:
                        # if not, nothing to do
                        # this value must be forwarded.
                        replicate_node = True
                        if data_name not in self.backward_input_arrays:
                            self.backward_input_arrays[data_name] = data_desc

                        if self.separate_sdfgs:
                            # because we need to forward this, the descriptor
                            # is no longer transient
                            data_desc.transient = False

                # No further recusrive calls are required
                # in this branch; next_required_inputs stays empty
                next_required_inputs = {}
            elif isinstance(edge_src, nodes.Tasklet):
                replicate_node = True
                # in the tasklet case, we need to connect all inputs
                next_required_inputs = {c: c for c in edge_src.in_connectors}
            else:
                raise AutoDiffException("Unsupported node")

            if replicate_node:
                replicated_edge_src_conn = edge.src_conn

                if edge_src in self.replicated_map:
                    replicated_edge_src = self.replicated_map[edge_src]
                else:
                    # node has not been replicated yet: do it now
                    replicated_edge_src = copy.deepcopy(edge_src)
                    self.backward_state.add_node(replicated_edge_src)
                    self.replicated_map[edge_src] = replicated_edge_src

            if connect_replicated_node:
                new_edge_data = copy.deepcopy(edge.data)
                if isinstance(edge.src, nodes.CodeNode) and isinstance(edge.dst, nodes.CodeNode):
                    # code->code edges have a small special case:
                    # we need to copy the descriptor
                    data_name = new_edge_data.data
                    data_desc = copy.deepcopy(self.sdfg.arrays[data_name])
                    if self.separate_sdfgs:
                        self.backward_sdfg.add_datadesc(data_name, data_desc)
                    else:
                        new_data_name = self.backward_sdfg.add_datadesc(data_name, data_desc, find_new_name=True)
                        new_edge_data.data = new_data_name

                # add the new edge
                self.backward_state.add_edge(replicated_edge_src, replicated_edge_src_conn, backward_node,
                                             required_inputs[edge.dst_conn], new_edge_data)

            if next_required_inputs:
                # if there are any required inputs on the new node, we need to
                # recursively call
                self._connect_forward_inputs(edge.src, replicated_edge_src, next_required_inputs)

    def _connect_temporary_to_accessnode(self, source_node: nodes.AccessNode, sink_node: nodes.AccessNode,
                                         memlets: List[Memlet], forward_sink_edge: dgraph.MultiConnectorEdge):
        """
        Connect the source node to the sink node (both in the backawrd state) through a set of maps using the parameter memelets.
        We use the forward_sink_edge to track which maps to make this connection through.
        :param source_node: the source node of the new memlet path
        :param sink_node: the sink node of the new memlet path
        :param memlets: the set of memlets to use for the edges in the path
        :param forward_sink_edge: the sink edge connecting the original nodes in the forward state
        """
        # get the memlet path from the forward state
        edge_list = self.forward_state.memlet_path(forward_sink_edge)
        assert len(edge_list) > 0

        # we will iterate and connect parent -> child
        child_node = sink_node
        child_node_in_connector = None

        # ietarte through the maps in the path in reverse
        for edge in reversed(edge_list):
            edge_src = edge.src
            if isinstance(edge_src, nodes.MapEntry):
                # get the correponding map exist
                map_exit = self._find_map_exist_for_map_entry(map_entry=edge_src, state=self.forward_state)

                # use the lookup table to get the map entry in the backward state corresponding to this map exist in the forward state
                # sanity check: this map entry should already exist in the backward state
                assert map_exit in self.reverse_map
                bwd_map_entry = self.reverse_map[map_exit]

                # get a new connector id
                next_conn = bwd_map_entry.next_connector()

                # add a new in connector to the mapexit
                parent_node_in_connector = "IN_stored_" + source_node.data + "_" + next_conn
                assert bwd_map_entry.add_in_connector(parent_node_in_connector)

                # add a new out connector to the mapexit
                pranet_node_out_connector = "OUT_stored_" + source_node.data + "_" + next_conn
                assert bwd_map_entry.add_out_connector(pranet_node_out_connector)

                memlet_data = copy.deepcopy(memlets.pop())

                # add the edge with the corresponding memlet
                self.backward_state.add_edge(bwd_map_entry, pranet_node_out_connector, child_node,
                                             child_node_in_connector, memlet_data)

                child_node = bwd_map_entry
                child_node_in_connector = parent_node_in_connector

        # there should be the same number of memlets through the new path
        assert len(memlets) == 0
        # make the final memlet
        memlet_data = Memlet.from_array(source_node.data, self.sdfg.arrays[source_node.data])
        # add the final connection to the source node
        self.backward_state.add_edge(source_node, None, child_node, child_node_in_connector, memlet_data)

    def _clean_after_recomputation(self, edge: dgraph.MultiConnectorEdge, connector_to_clean: str):
        """
        In the case of the recomputation of a base-level AccessNode, 
        we will only know whether this node can be recomputed after adding a path from the tasklet that required the computation
        to the maps serrounding this tasklet.
        If recomputation is applied, this path of memlets is no longer required and needs to be removed.
        :param edge: the edge leading to the first backward map containing the first out edge in the path to clean
        :param connector_to_clean: name of the out connector of the first out edge in the path to clean
        """
        # get the map in the backward pass
        bwd_map = self._find_backward_entry_node_for_map_entry(edge.dst)
        assert isinstance(bwd_map, nodes.MapEntry)

        # find the starting edge of the path we want to delete
        out_edges = self.backward_state.out_edges(bwd_map)
        starting_edge = None
        for e in out_edges:
            if e.dst_conn == connector_to_clean:
                starting_edge = e
                break

        assert starting_edge
        starting_edge.src.remove_in_connector(connector_to_clean)

        # clean up
        while starting_edge:
            next_edge = None
            # get the next edge
            next_map_edges = self.backward_state.out_edges(starting_edge.dst)
            for e in next_map_edges:
                if isinstance(e.dst, nodes.MapEntry):
                    if e.dst_conn == connector_to_clean:
                        next_edge = e
                        break
                else:
                    if e.src_conn == connector_to_clean.replace("IN", "OUT"):
                        self.backward_state.remove_edge(e)
                        e.src.remove_out_connector(connector_to_clean.replace("IN", "OUT"))
                        break

            starting_edge.src.remove_out_connector(starting_edge.src_conn)
            starting_edge.dst.remove_in_connector(starting_edge.dst_conn)
            self.backward_state.remove_edge(starting_edge)

            starting_edge = next_edge

    def _recompute_data(self, edge: dgraph.MultiConnectorEdge):
        """
        Given an edge leading from a base-level AccessNode to a map in the forward state,
        add an sdfg to recompute the values of this node to the backward state.
        :param edge: the edge connecting the AccessNode to recompute data from to a map node.
        """
        replicate_nodes = {}
        # treat the case where the recomputation can be merged into the gradient maps
        # get the subgraph neccessary to calculate the AccessNode itself
        subgraph: dstate.StateSubgraphView = self._get_computation_subgraph(edge.src)

        # check if this is the case
        mergeable = self._check_if_recomputation_is_mergeable(edge, subgraph)
        if mergeable:
            # get the maps from the backward pass to modify
            edge_list = self.forward_state.memlet_path(edge)
            backward_maps = []
            for e in edge_list:
                if isinstance(e.src, nodes.MapEntry):
                    bwd_map_entry = self._find_backward_entry_node_for_map_entry(e.src)
                    backward_maps.append(bwd_map_entry)

            map_index = 0
            # for each map in the forward pass
            for nd in subgraph.nodes():
                if isinstance(nd, nodes.MapEntry):
                    # get the equivelent node in the backward pass
                    fwd_map_entry: nodes.MapEntry = nd
                    bwd_map_entry: nodes.MapEntry = backward_maps[map_index]

                    # add all of the connectors of the forward map to the backward map
                    for connector in fwd_map_entry.in_connectors:
                        bwd_map_entry.add_in_connector(f"{connector}_recomputation")

                    for connector in fwd_map_entry.out_connectors:
                        bwd_map_entry.add_out_connector(f"{connector}_recomputation")

                    # replicate and add all of the edges coming into this map
                    in_edges = self.forward_state.in_edges(fwd_map_entry)
                    for e in in_edges:
                        # replicate the edge src
                        replicated_edge_src_dst_con = f"{e.dst_conn}_recomputation" if e.dst_conn else None
                        replicated_edge_src_src_con = f"{e.src_conn}_recomputation" if e.src_conn else None
                        if map_index == 0:
                            # and the nodes they are coming from if necessary
                            if e.src in replicate_nodes:
                                replicated_edge_src = replicate_nodes[e.src]
                            else:
                                # node has not been replicated yet: do it now
                                replicated_edge_src = copy.deepcopy(e.src)
                                self.backward_state.add_node(replicated_edge_src)
                                replicate_nodes[e.src] = replicated_edge_src
                        else:
                            replicated_edge_src = backward_maps[map_index - 1]

                        memlet_data = copy.deepcopy(e.data)
                        # add a new edge between the backward map and the new replicated node
                        self.backward_state.add_edge(replicated_edge_src, replicated_edge_src_src_con, bwd_map_entry,
                                                     replicated_edge_src_dst_con, memlet_data)
                    map_index += 1

            next_level = []
            node = fwd_map_entry
            node_bwd = bwd_map_entry
            # we go level by level through the content of the map nest
            while node:
                # we start with all the edges coming out of the last map
                node_out_edges = self.forward_state.out_edges(node)
                for e in node_out_edges:
                    # we will reuse the same memlet for the recomputation
                    memlet_data = copy.deepcopy(e.data)
                    # rename the connectors to reflect that this was added for recomputation
                    replicated_edge_src_dst_con = f"{e.dst_conn}_recomputation" if e.dst_conn else None
                    replicated_edge_src_src_con = f"{e.src_conn}_recomputation" if e.src_conn else None
                    if isinstance(e.dst, nodes.MapExit):
                        # if we got to the map exit,
                        # we need to connect the output of the recomputation
                        # to the tasklet that required the values in the backward pass

                        # first, we get the target taskelt and its connector
                        tasklet = edge_list[-1].dst
                        tasklet_conn = edge_list[-1].dst_conn
                        assert isinstance(tasklet, nodes.Tasklet)

                        # get the replicated tasklet from the backward pass
                        assert tasklet in self.reverse_map
                        bwd_tasklet = self.reverse_map[tasklet]

                        # we want to first remove the last assign tasklet
                        # this was previously added to assign the calculated value to the correct position in the array
                        # since we will use the value directly, we will remove the assign tasklet
                        assign_tsaklet = node_bwd

                        # sanity check
                        assert isinstance(assign_tsaklet, nodes.Tasklet)
                        assert assign_tsaklet in self.backward_state.nodes()
                        assert "assign" in assign_tsaklet.name

                        # get the edge from the AccessNode coming to the assign tasklet
                        # this will be the edge that is connected to the reversed tasklet
                        assign_tasklet_in_edge = self.backward_state.in_edges(assign_tsaklet)
                        assert len(assign_tasklet_in_edge) == 1
                        assign_tasklet_in_edge = assign_tasklet_in_edge[0]
                        self.backward_state.remove_edge(assign_tasklet_in_edge)

                        # remove the tasklet
                        self.backward_state.remove_node(assign_tsaklet)

                        # add the new edge between the final AccessNode and the reversed tasklet
                        last_accessnode = assign_tasklet_in_edge.src
                        assert isinstance(last_accessnode, nodes.AccessNode)

                        memlet_data = assign_tasklet_in_edge.data
                        assert tasklet_conn in bwd_tasklet.in_connectors
                        self.backward_state.add_edge(last_accessnode, None, bwd_tasklet, tasklet_conn, memlet_data)
                    else:
                        # the general case, we are replicating the content of the map nest
                        # replicate the edge dst if not already replicated
                        if e.dst in replicate_nodes:
                            replicated_edge_dst = replicate_nodes[e.dst]
                        else:
                            # node has not been replicated yet: do it now
                            replicated_edge_dst = copy.deepcopy(e.dst)
                            # change the connectors for recomputation
                            self._modify_connectors_for_recomputation(replicated_edge_dst)
                            self.backward_state.add_node(replicated_edge_dst)
                            replicate_nodes[e.dst] = replicated_edge_dst

                        # add a new edge between the two nodes in the backward state
                        self.backward_state.add_edge(node_bwd, replicated_edge_src_src_con, replicated_edge_dst,
                                                     replicated_edge_src_dst_con, memlet_data)

                        # add the node for the next level only if it has not already been explored
                        if e.dst not in next_level: next_level.append(e.dst)

                node = next_level.pop() if next_level else None
                assert not node or node in replicate_nodes
                node_bwd = replicate_nodes[node] if node else None
        else:
            raise AutoDiffException(f"Recomputation of the node {edge.src} is not yet supported")

    def _modify_connectors_for_recomputation(self, node: nodes.Node):
        """
        Given a node in the graph, modify all the connectors to indicate that this node was added for recomputation.
        Additionally, if the node is a tasklet, we also modify the tasklet code to refelect this change.
        :param node: the node to modify the connectors for
        """
        # for an AccessNode, there are no connectors to be modified
        if isinstance(node, nodes.AccessNode):
            return

        all_connectors = node.out_connectors.copy()
        all_connectors.update(node.in_connectors)

        for con in list(all_connectors):
            new_con = f"{con}_recomputation"
            if con in node.in_connectors:
                node.remove_in_connector(con)
                assert node.add_in_connector(new_con)
            else:
                node.remove_out_connector(con)
                assert node.add_out_connector(new_con)

            # if this node is a tasklet we need to modify the content of the code
            if isinstance(node, nodes.Tasklet):
                node.code.as_string = node.code.as_string.replace(con, new_con)

    def _check_if_recomputation_is_mergeable(self, edge: dgraph.MultiConnectorEdge,
                                             subgraph: dstate.StateSubgraphView) -> bool:
        """
        Given an edge leading from a base-level AccessNode to a map in the forward state,
        Check if the computation of this AccessNode can be merged with the maps where
        this node will be used in the backward pass. 
        The constraints of this function are too strong and can be relaxed.
        :param edge: the edge connecting the AccessNode to recompute data from to a map node.
        """
        # if the forward tasklet is surrounded by the same number of maps with the same indicies
        # get the path of the AccessNode and store the maps until we reach the tasklet
        edge_list = self.forward_state.memlet_path(edge)
        successor_maps: List[nodes.MapEntry] = []
        for e in edge_list:
            if isinstance(e.src, nodes.MapEntry):
                successor_maps.append(e.src)

        mergeable = True

        # check if the number of maps in the subgraph matches
        for nd in subgraph.nodes():
            if isinstance(nd, nodes.MapEntry):
                # make sure the two maps match in terms of ranges
                if len(successor_maps) > 0:
                    s_map = successor_maps.pop()
                else:
                    # different number of maps
                    # for now, we return false
                    mergeable = False
                    break
                if s_map.map.range != nd.map.range:
                    # map ranges are not the same
                    # for now, we return false
                    mergeable = False
                    break

        if len(successor_maps) != 0:
            # different number of maps
            # for now, we return false
            mergeable = False

        return mergeable

    def _get_computation_subgraph(self, node: nodes.AccessNode) -> SDFG:
        """
        Given an access node get the subgraph from the forward state that writes to this access node
        """
        # reverse bfs from the accesss node
        backward_nodes = {n for e in self.forward_state.bfs_edges(node, reverse=True) for n in [e.src, e.dst]}
        forward_nodes = {n for n in self.forward_state.nodes()}
        # intersection with all the nodes in the forward state
        forward_subgraph = dstate.StateSubgraphView(self.forward_state,
                                                    list(forward_nodes.intersection(backward_nodes)))
        return forward_subgraph

    def _store_data(self, edge: dgraph.MultiConnectorEdge) -> Tuple[nodes.AccessNode, List[Memlet]]:
        """
        Given an edge leading from a base-level AccessNode to a map in the forward state,
        add a path from the connector for this AccessNode to store its values for all iterations.
        This can increase the dimension of the array. i.e. the size of the stored array is
        greater or equal to the size of the original array.
        
        :param edge: the edge connecting the AccessNode to save data from to a map node.
        :return: the new AccessNode which contains the stored data, 
                 a list of memlets connecting an assign tasklet to this new AccessNode.
        """
        # get the AccessNode we want to save data from
        node: nodes.AccessNode = edge.src

        # make sure this is indeed an AccessNode
        assert isinstance(node, nodes.AccessNode)

        # get all the maps in the path from the accessnode to the desired tasklet
        edge_list = self.forward_state.memlet_path(edge)

        # there should be at least two edges in the path
        assert len(edge_list) > 1

        # get the last map in the path.
        last_edge = edge_list[-1]
        # this is the map that contains the connector for the value we want to store
        last_map: nodes.MapEntry = last_edge.src

        # make sure this is indeed an MapEntry
        assert isinstance(last_map, nodes.MapEntry)

        last_map_connector = last_edge.src_conn

        # create the assign tasklet and add it to the forward state
        assign_tasklet_node_in_connector = "in_stored_" + last_map_connector
        assign_tasklet_node_out_connector = "out_stored_" + last_map_connector
        assign_tasklet_node = nodes.Tasklet(
            label=f"__store_{node.data}_assign_",
            inputs={assign_tasklet_node_in_connector},
            outputs={assign_tasklet_node_out_connector},
            code=f"{assign_tasklet_node_out_connector} = {assign_tasklet_node_in_connector}",
        )
        self.forward_state.add_node(assign_tasklet_node)

        # create the memlet for the assignement
        # this will be the same as the memlet going to the tasklet
        assign_memlet_data = copy.deepcopy(last_edge.data)

        # add the new edge from the last map to the new assign tasklet
        self.forward_state.add_edge(last_map, last_map_connector, assign_tasklet_node, assign_tasklet_node_in_connector,
                                    assign_memlet_data)

        # create and add the new access node that contains the saved values
        new_store_node_name = "stored_" + node.data

        # first, get the shape of the new array
        shape_list = []
        # we will also need the starting range of the maps in the path
        start_range = []
        # and the names of the parameters of the maps in the path
        param_list = []

        for e in edge_list:
            edge_src = e.src
            if isinstance(edge_src, nodes.MapEntry):
                for rng in edge_src.map.range.ranges:
                    # the range contains the last index in the loop
                    # while we want the size so we add 1
                    shape_list.append(rng[1] + 1)
                    start_range.append(rng[0])
                for par in edge_src.map.params:
                    param_list.append(par)

        # add the array descriptor and AccessNode to the forward state
        original_desc = node.desc(self.forward_state)
        new_store_node = self.forward_state.add_array(
            name=new_store_node_name,
            shape=shape_list,
            dtype=original_desc.dtype,
            transient=True,
        )

        # we will save the memlets we create an return them
        # this is useful to make the connections for the backward state
        memlets_stack = []

        # connect the assign tasklet to the corresponding mapexists iterativly
        # until we reach the final AccessNode
        parent_node = assign_tasklet_node
        parent_node_connector = assign_tasklet_node_out_connector

        # the parameters to add for the current memlet in the loop
        # at first we will use all of the parameters
        params_to_add = param_list

        # for each map in the path
        for e in reversed(edge_list):
            edge_src = e.src
            if isinstance(edge_src, nodes.MapEntry):
                # get the corresponding map exit
                child_node = self._find_map_exist_for_map_entry(map_entry=edge_src, state=self.forward_state)
                # get a new connector id
                next_conn = child_node.next_connector()

                # add a new in connector to the mapexit
                child_node_in_connector = "IN_stored_" + node.data + "_" + next_conn
                assert child_node.add_in_connector(child_node_in_connector)

                # add a new out connector to the mapexit
                child_node_out_connector = "OUT_stored_" + node.data + "_" + next_conn
                assert child_node.add_out_connector(child_node_out_connector)

                memlet_data = Memlet(
                    expr=
                    f"{new_store_node.data}[{','.join([f'{param_list[i]}' if param_list[i] in params_to_add else f'{start_range[i]}:{shape_list[i]}' for i in range(len(param_list))])}]"
                )
                # save the memlet for later
                memlets_stack.append(memlet_data)

                # add the edge with the corresponding memlet
                self.forward_state.add_edge(parent_node, parent_node_connector, child_node, child_node_in_connector,
                                            memlet_data)

                # prepare values for the next iteration
                parent_node = child_node
                parent_node_connector = child_node_out_connector

                # remove the parameters seen in the current map
                # since they will become out of scope in the next iteration
                params_to_add = [element for element in params_to_add if element not in edge_src.params]

        # sanity check: since we are out of scope for all maps, there shouldn't be any parameters left
        assert len(params_to_add) == 0

        # get the memlet data for the connection between the last map exit and the new store AccessNode
        memlet_data = Memlet(
            expr=
            f"{new_store_node.data}[{','.join([f'{start_range[i]}:{shape_list[i]}' for i in range(len(param_list))])}]")
        memlets_stack.append(memlet_data)

        # connect the last map exist to the newly created store node
        # add the new edge
        self.forward_state.add_edge(
            parent_node,  # the last MapExist
            parent_node_connector,  # the last MapExist connector
            new_store_node,  # the new store AccessNode
            None,
            memlet_data)

        return new_store_node, memlets_stack

    def _check_node_overwrite(self, node: nodes.AccessNode) -> Tuple[bool, bool]:
        """
        Given an AccessNode from the forward state, check if the data of this node has changed.
        We look at all the AccessNodes with the same data that occur after the 'node' parameter
        if any of them has an incoming edge, return the node has been overwritten.
        
        :param edge: the AccessNode to perform the check for.
        :return: a tuple of wether this node has been overwritten, and if it can be recomputed
        """
        overwritten = False
        decided = False
        recomputable = False
        # get all the AccessNodes with the same data
        matches = [nd for nd in self.forward_state.nodes() if isinstance(nd, nodes.AccessNode) and nd.data == node.data]

        # there needs to be at least one occurance which is the node passed as a parameter
        assert len(matches) > 0 and node in matches

        # if there is only one occurance of this data, it will not be overwritten later in the graph
        if len(matches) == 1:
            overwritten = False
            decided = True

        # get the index of the parameter node
        index = matches.index(node)

        # if the parameter node is the last occurance in the forward state
        # return False
        if len(matches) - 1 == index:
            overwritten = False
            decided = True

        # if we haven't already confirmed that this node has not been overwritten
        if not decided:
            # iterate through all the successor occurances
            for nd in matches[index + 1:]:
                # check if this node has an incoming edge
                if len(self.forward_state.in_edges(nd)) > 0:
                    overwritten = True

        if overwritten:
            # we only check if the node is recomputable if it has been overwritten
            # iterate through all the predecessor occurances
            for nd in matches[:index + 1]:
                # check if this node has an incoming edge
                if len(self.forward_state.in_edges(nd)) > 0:
                    recomputable = True
        # if the loop didn't return True, all of the successor occurances are read only
        return overwritten, recomputable

    def _lookup_required_grad_name(self, node: nodes.Node, connector: str) -> str:
        if node not in self.result_map:
            raise AutoDiffException("Attempted to access gradient of {}"
                                    " before the backward node was created".format(node))
        return self.result_map[node].required_grad_names[connector]

    def _lookup_given_grad_name(self, node: nodes.Node, connector: str) -> str:
        if node not in self.result_map:
            raise AutoDiffException("Attempted to access gradient of {}"
                                    " before the backward node was created".format(node))
        return self.result_map[node].given_grad_names[connector]

    def _find_backward_entry_node_for_map_entry(self, entry_node: nodes.MapEntry) -> nodes.MapExit:
        """Find the entry node in the backward pass corresponding to the exit node opened by
        `entry_node` (where `entry_node` is a node from the forward pass).
        """
        src_candidates = [
            cast(nodes.MapExit, node) for node in self.backward_state.nodes()
            if isinstance(node, nodes.MapEntry) and node.map == self.reverse_map[entry_node.map]
        ]
        if len(src_candidates) != 1:
            # this shouldn't happen; if we are within a scope, the exit nodes
            # for the scope should already exist in the backward pass
            raise AutoDiffException("Invalid graph")

        return src_candidates[0]

    def _find_map_exist_for_map_entry(self, map_entry: nodes.MapEntry, state: SDFGState) -> nodes.MapExit:
        """
        Find the map exist that corresponds to the input map entry
        """
        src_candidates = [
            node for node in state.nodes() if isinstance(node, nodes.MapExit) and node.map == map_entry.map
        ]
        if len(src_candidates) != 1:
            # this shouldn't happen; if we are within a scope, the exit nodes
            # for the scope should already exist in the backward pass
            raise AutoDiffException("Invalid graph")

        return src_candidates[0]

    def _get_reverse_node(self, node, given_gradients, required_gradients) -> ReverseNodeReturnType:
        """ Add the reverse node for a node from the forward pass to the backward pass, and return it.

            Resolution order:
            1) check for methods on this class
            2) check the backward pass repository

            :param node: node on the forward pass
            :param given_gradients: output names on the forward node (for which the gradient will be connected as
                                           an input on the reverse node)
            :param required_gradients: input name on the forward node that the gradient should be generated for
            :return: the reversed node and gradient names for the connectors
        """
        log.debug("Reversing {}".format(node))

        # (1)
        if hasattr(self, "_reverse_" + type(node).__name__):
            return getattr(self, "_reverse_" + type(node).__name__)(node, given_gradients, required_gradients)

        # (2)
        impl = find_backward_implementation(self.sdfg, forward_state=self.forward_state, node=node)
        if impl is not None:
            backward_node, backward_result = impl.backward(forward_node=node,
                                                           context=BackwardContext(
                                                               forward_state=self.forward_state,
                                                               forward_sdfg=self.sdfg,
                                                               backward_state=self.backward_state,
                                                               backward_sdfg=self.backward_sdfg,
                                                               backward_generator=self,
                                                           ),
                                                           given_gradients=given_gradients,
                                                           required_gradients=required_gradients)
            if isinstance(backward_node, nodes.CodeNode):
                backward_node.schedule = node.schedule
            if isinstance(backward_node,
                          nodes.NestedSDFG) and backward_node.schedule is not dtypes.ScheduleType.Default:
                infer_types._set_default_schedule_types(backward_node.sdfg, backward_node.schedule, True)
                infer_types._set_default_storage_types(backward_node.sdfg, backward_node.schedule)
            return backward_node, backward_result

        raise AutoDiffException("Unable to differentiate node type {}. Either add a pure forward implementation "
                                "or a backward implementation to progress.".format(type(node)))

    def _reverse_NestedSDFG(
        self,
        node: nodes.NestedSDFG,
        given_gradients: List[str],
        required_gradients: List[str],
    ) -> ReverseNodeReturnType:
        # check that the nested SDFG only has one state
        state_to_diff: SDFGState
        if len(node.sdfg.nodes()) != 1:
            # however we make an exception for initialization states; these are ignored
            is_init_state = [(state, is_initialization_state(state)) for state in node.sdfg.nodes()]
            num_non_init_states = sum(b for _, b in is_init_state)
            if num_non_init_states > 1:
                raise AutoDiffException(
                    "A nested SDFG may consist of at most one state (with the "
                    "exception of initalization states), found {} states".format(num_non_init_states))
            state_to_diff = [state for state, b in is_init_state if not b][0]
        else:
            state_to_diff = node.sdfg.nodes()[0]

        reverse_sdfg = dace.SDFG(node.sdfg.name + "_backward")
        backward_state = reverse_sdfg.add_state()
        # recursive call
        gen = BackwardPassGenerator(sdfg=node.sdfg,
                                    state=state_to_diff,
                                    given_gradients=given_gradients,
                                    required_gradients=required_gradients,
                                    backward_sdfg=reverse_sdfg,
                                    backward_state=backward_state,
                                    zero_non_transients=True)
        backward_result, _, backward_input_arrays = gen.backward()

        # we need to defer add edges until after the arrays have been added because creation of the nested
        # sdfg fails otherwise
        deferred_edges = []

        inputs = set(backward_result.given_grad_names[name] for name in sorted(given_gradients))
        # loop through the arrays that we need from the forward pass
        for name, desc in sorted(backward_input_arrays.items()):
            # if the name is not already passed to the reverse SDFG node ...
            if name not in required_gradients and name not in node.in_connectors:
                # ... this array needs to be forwarded out of the forward SDFG (i.e. it is an intermediate value)
                # 1) add it to the current SDFG, and to self.backward_input_arrays
                # 2) add an out connector to the forward nested SDFG, add a write node to the current state, and an edge
                #    from the output to there
                # 3) add a read node to the backward state, and an edge into it

                desc, forwarded_name, _ = _walk_up_memlet_tree_through_view_nodes(node.sdfg, state_to_diff, name)

                # (1)
                new_name = find_str_not_in_set(set(self.sdfg.arrays), forwarded_name + "_forwarded")
                if new_name in self.sdfg.arrays or new_name in self.backward_input_arrays:
                    raise AutoDiffException(
                        "Attempted to create array with name '{}', but it already existed".format(new_name))

                self.sdfg.add_datadesc(new_name, copy.deepcopy(desc))
                self.backward_input_arrays[new_name] = copy.deepcopy(desc)

                if self.separate_sdfgs:
                    to_add = copy.deepcopy(desc)
                    to_add.transient = False
                    self.backward_sdfg.add_datadesc(new_name, to_add)

                # (2)
                node.sdfg.arrays[forwarded_name].transient = False
                assert node.add_out_connector(forwarded_name)
                write = self.forward_state.add_write(new_name)
                self.forward_state.add_edge(node, forwarded_name, write, None, self.sdfg.make_array_memlet(new_name))

                # (3)
                read = self.backward_state.add_read(new_name)
                deferred_edges.append(
                    dict(u=read,
                         u_connector=None,
                         v_connector=forwarded_name,
                         memlet=self.backward_sdfg.make_array_memlet(new_name)))
                inputs.add(forwarded_name)
            else:
                inputs.add(name)

        outputs = set(backward_result.required_grad_names[name] for name in required_gradients)

        for inp in inputs:
            reverse_sdfg.arrays[inp].transient = False
        for outp in outputs:
            reverse_sdfg.arrays[outp].transient = False

        # actually create the sdfg and return it
        nsdfg = self.backward_state.add_nested_sdfg(
            reverse_sdfg,
            None,
            inputs=inputs,
            outputs=outputs,
        )

        for edge_args in deferred_edges:
            edge_args["v"] = nsdfg
            self.backward_state.add_edge(**edge_args)

        return nsdfg, BackwardResult(required_grad_names=backward_result.required_grad_names,
                                     given_grad_names=backward_result.given_grad_names)

    def _reverse_AccessNode(
        self,
        node: nodes.AccessNode,
        given_gradients: List[str],
        required_gradients: List[str],
    ) -> ReverseNodeReturnType:
        rev = nodes.AccessNode(self.array_grad_name(node.data))
        self.backward_state.add_node(rev)
        required_grad_names = {None: None}
        given_grad_names = {None: None}

        if "views" in node.in_connectors:
            required_grad_names = {"views": "views"}
        if "views" in node.out_connectors:
            given_grad_names = {"views": "views"}

        return rev, BackwardResult(required_grad_names=required_grad_names, given_grad_names=given_grad_names)

    def _reverse_MapEntry(
        self,
        node: nodes.MapEntry,
        given_gradients: List[str],
        required_gradients: List[str],
    ) -> ReverseNodeReturnType:

        required_grad_names = {n: _invert_map_connector(n) for n in required_gradients}
        given_grad_names = {n: _invert_map_connector(n) for n in given_gradients}
        result = BackwardResult(required_grad_names=required_grad_names, given_grad_names=given_grad_names)
        rev = nodes.MapExit(self.reverse_map[node.map])

        for _, conn in sorted(given_grad_names.items()):
            assert rev.add_in_connector(conn)

        for _, conn in sorted(required_grad_names.items()):
            assert rev.add_out_connector(conn)

        self.backward_state.add_node(rev)
        return rev, result

    def _reverse_MapExit(
        self,
        node: nodes.MapExit,
        given_gradients: List[str],
        required_gradients: List[str],
    ):
        self.reverse_map[node.map] = copy.deepcopy(node.map)

        rev = nodes.MapEntry(self.reverse_map[node.map])
        for conn in sorted(node.in_connectors):
            assert rev.add_in_connector(conn)

        for conn in sorted(node.out_connectors):
            assert rev.add_out_connector(conn)

        self.backward_state.add_node(rev)
        # yapf: disable
        return (
            rev,
            BackwardResult(required_grad_names={
                n: _invert_map_connector(n)
                for n in required_gradients
            },
                given_grad_names={
                    n: _invert_map_connector(n)
                    for n in given_gradients
                }),
        )
        # yapf: enable

    def _reverse_Tasklet(
        self,
        tasklet: nodes.Tasklet,
        given_gradients: List[str],
        required_gradients: List[str],
    ) -> ReverseNodeReturnType:

        if tasklet.language is not dtypes.Language.Python:
            raise AutoDiffException("Expected tasklet with language Python, got language {}".format(tasklet.language))

        # tasklets should have scalar inputs (can be relaxed)
        for _, _, _, _, memlet in self.forward_state.in_edges(tasklet):
            try:
                _is_int_value(memlet.subset.num_elements(), 1)
            except AutoDiffException as e:
                raise AutoDiffException("Autodiff only supported for tasklets with scalar inputs and outputs") from e

        for _, _, _, _, memlet in self.forward_state.out_edges(tasklet):
            try:
                _is_int_value(memlet.subset.num_elements(), 1)
            except AutoDiffException as e:
                raise AutoDiffException("Autodiff only supported for tasklets with scalar inputs and outputs") from e

        code_str = tasklet.code.as_string
        output_exprs = code_to_exprs(code_str, tasklet.in_connectors, tasklet.out_connectors)

        # for each output that an input is used in, there will be an entry for the expression of the
        # grad in this list in the final code snippet. When we generate the final code for the
        # reverse tasklet, we need to add them all up.
        rev_code = collections.defaultdict(list)

        # the outputs of the reversed nodes are the grads of inputs of the original node
        rev_outputs = set()
        rev_inputs = set()

        result = BackwardResult(required_grad_names={}, given_grad_names={})

        # symbol generator to use for CSE
        symbol_generator = sp.numbered_symbols()

        code = ""

        for output_conn in sorted(given_gradients):

            # for each output_conn...
            for inp in sorted(required_gradients):
                # ...add the code to generate {inp}_grad

                if inp not in result.required_grad_names:
                    # pick a name for the gradient
                    rev_output_grad_name = find_str_not_in_set(rev_outputs, inp + "_gradient")
                    result.required_grad_names[inp] = rev_output_grad_name
                    rev_outputs.add(rev_output_grad_name)
                else:
                    rev_output_grad_name = result.required_grad_names[inp]

                output_expr = output_exprs[output_conn]

                # symbolically differentiate the output w.r.t inp
                diff_expr = output_expr.diff(sp.symbols(inp))

                # do common subexpression elimination
                sub_expressions, diff_expr = sp.cse(diff_expr, symbols=symbol_generator)

                diff_expr = diff_expr[0]

                if diff_expr.atoms(sp.Derivative):
                    # the final result contains a call to sp.Derivative
                    raise AutoDiffException("Unable to symbolically differentiate expression: {}".format(
                        diff_expr.expr))

                if output_conn not in result.given_grad_names:
                    # pick a name for the input gradient
                    rev_input_grad_name = find_str_not_in_set(rev_inputs, output_conn + "_gradient")
                    result.given_grad_names[output_conn] = rev_input_grad_name
                else:
                    rev_input_grad_name = result.given_grad_names[output_conn]

                input_symbols = diff_expr.free_symbols\
                    .union(s for _, e in sub_expressions for s in e.free_symbols)\
                    .difference(e for e, _ in sub_expressions)

                rev_inputs |= _symbols_to_strings(input_symbols) | {rev_input_grad_name}

                diff_code_str = "{input} * ({diff_expr})".format(input=rev_input_grad_name, diff_expr=str(diff_expr))
                # small hack: our heaviside is lowercase
                diff_code_str = diff_code_str.replace("Heaviside", "heaviside")

                diff_code_str = astunparse.unparse(SympyCleaner().visit(ast.parse(diff_code_str)))

                sub_expression_code_strs = "\n".join(f"{target} = {expression}"
                                                     for target, expression in sub_expressions)

                # get the the final type of the gradient: this is just the type of the input connector we creating the
                # gradient for
                cands = list(self.forward_state.in_edges_by_connector(tasklet, inp))
                if len(cands) != 1:
                    raise AutoDiffException(f"Unexpected graph structure, could not find input edge for connector {inp}"
                                            f" on tasklet {tasklet}")

                converted_code = cast_consts_to_type(diff_code_str, self.sdfg.arrays[cands[0].data.data].dtype)
                converted_code = converted_code.replace("\n", " ")

                converted_sub_expressions = cast_consts_to_type(sub_expression_code_strs,
                                                                self.sdfg.arrays[cands[0].data.data].dtype)

                code += converted_sub_expressions + "\n"
                rev_code[rev_output_grad_name].append(converted_code)

        for output, exprs in sorted(rev_code.items()):
            code += "\n" + output + " = " + " + ".join(exprs)
        rev = nodes.Tasklet("_" + tasklet.label + "_reverse_",
                            inputs=rev_inputs,
                            outputs=rev_outputs,
                            code=code,
                            debuginfo=tasklet.debuginfo)
        self.backward_state.add_node(rev)
        return rev, result


class SympyCleaner(ast.NodeTransformer):

    def visit_Name(self, node):
        if node.id == "pi":
            return ast.copy_location(ast.parse("(3.141592653589)").body[0], node)
        else:
            return self.generic_visit(node)
