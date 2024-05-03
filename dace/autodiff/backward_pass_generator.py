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

from dace.autodiff.base_abc import (BackwardContext, BackwardResult,
                                      AutoDiffException,
                                      find_backward_implementation)

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
            "__out = {}".format(scalar), {
                "__out":
                dace.Memlet.simple(
                    data, ", ".join("i{}".format(i)
                                    for i in range(len(arr.shape))))
            },
            schedule=dtypes.ScheduleType.GPU_Device
            if cuda else dtypes.ScheduleType.Default,
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


def generate_grad_connector_names(
        existing_connectors: Set[str],
        forward_connector_names: List[str]) -> Dict[str, str]:
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


def code_to_exprs(code: str, inputs: Set[str],
                  outputs: Set[str]) -> Dict[str, sp.Expr]:
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
        results = temp_globals["symbolic_execution"](
            *[sp.symbols(inp) for inp in inputs])

        if len(outputs) > 1:
            return dict(zip(outputs, results))
        else:
            return {outputs[0]: results}
    except Exception as e:
        raise AutoDiffException(
            "Exception occured while attempting to symbolically execute code:\n{}"
            .format(code)) from e


def _is_int_value(value, target_value: int) -> bool:
    if isinstance(value, numbers.Integral):
        return value == target_value

    if len(value.free_symbols) > 0 or int(value) != target_value:
        return False

    return True


def _add_through_connector(node: Union[nodes.MapEntry, nodes.MapExit]):
    i = 1
    while ("IN_{}".format(i) in node.in_connectors
           or "OUT_{}".format(i) in node.out_connectors):
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
        raise AutoDiffException(
            "Could not parse map connector '{}'".format(conn))


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
        if (isinstance(sdfg.arrays[memlet_data], dt.Scalar)
                and isinstance(edge.src, nodes.CodeNode)
                and isinstance(edge.dst, nodes.CodeNode)):
            if memlet_data in seen_scalars or memlet_data in seen_accesses:
                return True
            seen_scalars.add(memlet_data)
    return False


def _walk_up_memlet_tree_through_view_nodes(
    sdfg, forward_state, start_name
) -> Tuple[Union[dt.Scalar, dt.Array], str, Deque[Tuple[str, dt.Data,
                                                        Memlet]]]:
    """ Starting from the (singular) access node for ``start_name`` in ``forward_state``, walk up the
        memlet path until a non-view node is reached

        :param sdfg: the forward sdfg
        :param forward_state: the forward state
        :param start_name: the name of the array to start at
        :return: the descriptor at the root of the path, the name at the root of the path, the list of
                 array names, view data descriptor and memlets encountered along the path.
    """
    forwarded_name = start_name
    view_nodes_to_clone: Deque[Tuple[str, dt.Data,
                                     Memlet]] = collections.deque()
    if isinstance(sdfg.arrays[start_name], dt.View):
        # this is complicated slightly by views: we need to walk up the memlet path until we reach a
        # non-view access node. We then need to replicate the sequence of views in the backward SDFG.
        query = [
            n for n in forward_state.nodes()
            if isinstance(n, nodes.AccessNode) and n.data == start_name
        ]
        if len(query) != 1:
            raise AutoDiffException(
                f"Could not find access node to forward with data {start_name}"
            )
        current_node = query[0]
        while isinstance(sdfg.arrays[current_node.data], dt.View):

            in_edges = forward_state.in_edges(current_node)
            if len(in_edges) != 1:
                raise AutoDiffException(
                    f"Expected view node with in degree 1, got {len(in_edges)} for view node {current_node}"
                )
            if not isinstance(in_edges[0].src, nodes.AccessNode):
                raise AutoDiffException(
                    f"Expected view node {current_node} to be connected to access node, got {in_edges[0].src}"
                    f" (of type {type(in_edges[0].src)})")
            view_nodes_to_clone.append(
                (current_node.data, sdfg.arrays[current_node.data],
                 in_edges[0].data))
            current_node = in_edges[0].src
            forwarded_name = current_node.data

    return sdfg.arrays[forwarded_name], forwarded_name, view_nodes_to_clone


def _path_src_node_in_subgraph(edge: dgraph.MultiConnectorEdge,
                               subgraph: dstate.StateSubgraphView):
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
            raise AutoDiffException(
                "Expected to find backward_state in backward_sdfg")

        def str_to_access(data: str, source: str) -> nodes.AccessNode:
            matches = [
                node for node in state.nodes()
                if isinstance(node, nodes.AccessNode) and node.data == data
            ]
            # Unused in model
            if len(matches) == 0:
                return None
            if len(matches) > 1:
                # There are multiple occurances of the same AccessNode
                # Return the last one
                # TODO: generalize this?
                # raise AutoDiffException(
                #     "Expected to find exactly one node with data"
                #     " '{}' in {}, but found {}".format(data, source,
                #                                        len(matches)))
                return matches[-1]
            return matches[0]

        given_gradients = [
            n if isinstance(n, nodes.AccessNode) else str_to_access(n, "outputs")
            for n in given_gradients
        ]
        required_gradients = [
            n if isinstance(n, nodes.AccessNode) else str_to_access(n, "inputs")
            for n in required_gradients
        ]
        required_gradients = [n for n in required_gradients if n is not None]

        self.given_gradients = given_gradients
        self.required_gradients = required_gradients

        self.input_names = {n.data for n in required_gradients}
        self.output_names = {n.data for n in given_gradients}

        self.sdfg = sdfg
        self.forward_state = state
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

        self.conflicted_gradient_buffers: Set[
            str] = conflicted_gradient_buffers or set()

        # checks if backward has already been applied
        self._applied = False
        self.zero_non_transients = zero_non_transients

        for outp in self.given_gradients:
            if outp not in self.forward_state:
                raise AutoDiffException(
                    "Could not find output {} in state {}".format(
                        outp, self.forward_state))

        for inp in self.required_gradients:
            if inp not in self.forward_state:
                raise AutoDiffException(
                    "Could not find input {} in state {}".format(
                        inp, self.forward_state))

        # check for inplace operations (i.e. duplicated access nodes)
        # if _has_inplace_operation(self.forward_state):
        #     raise AutoDiffException(
        #         "Inplace operations are currently not supported in autodiff")

        if sdfg is backward_sdfg:
            # this only makes sense if the output is a single scalar.
            if len(given_gradients) != 1:
                raise AutoDiffException(
                    "When the forward sdfg is the same as the backward sdfg, outputs must be a"
                    "single scalar")
            if not _is_int_value(
                    sdfg.arrays[given_gradients[0].data].total_size, 1):
                raise AutoDiffException(
                    "When the forward sdfg is the same as the backward sdfg, outputs must be a"
                    "single scalar")
            self.separate_sdfgs = False
        else:
            self.separate_sdfgs = True

        self.completion_hooks: List[Callable[[BackwardPassGenerator],
                                             None]] = []

    def _expand_nodes(self, subgraph: dstate.StateSubgraphView) -> bool:
        """ Expand all library nodes in the graph to pure implementations. Returns whether something was expanded
        """
        expanded_something = False
        for node, state in subgraph.all_nodes_recursive():
            if isinstance(state, dstate.StateSubgraphView):
                state = state.graph

            # check if the node exists in the backward implementation repository
            if find_backward_implementation(state.parent, state,
                                            node) is not None:
                continue

            # only check others if we didn't break out of the above loop
            if isinstance(node, ONNXOp):
                impls = ONNXForward.registered_implementations(
                    node.schema.name)

                # order the implementations so that implementations containing "pure" are tried first
                impls = [i for name, i in impls if "pure" in name
                         ] + [i for name, i in impls if "pure" not in name]
                for impl in impls:
                    if impl.forward_can_be_applied(node, state, self.sdfg):
                        # try to apply the expansion
                        class Expansion(xf.ExpandTransformation):
                            environments = impl.environments if hasattr(
                                impl, "environments") else []
                            _expansion_result = None

                            @classmethod
                            def expansion(cls, node, state, sdfg):
                                return impl.forward(node, state, sdfg)

                            @staticmethod
                            def annotates_memlets() -> bool:
                                return True

                        Expansion._match_node = xf.PatternNode(type(node))
                        Expansion.apply_to(state.parent,
                                           verify=False,
                                           _match_node=node)
                        expanded_something = True
                        break

            # This could later on be changed to check if the expansion is differentiable and if not, move
            # on to the next expansion. For now we will just apply the first one that matches, prioritizing ones that
            # have "pure" in the name
            if isinstance(node,
                          nodes.LibraryNode) and not isinstance(node, ONNXOp):
                # try to select an expansion
                if hasattr(node, "implementations"):
                    implementations = node.implementations

                    pure_candidates = [
                        name for name, impl in sorted(implementations.items())
                        if "pure" in name
                    ]
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
            if isinstance(
                    n, nodes.AccessNode) and type(n.desc(self.sdfg)) is dt.View:
                in_edges = self.forward_state.in_edges(n)
                out_edges = self.forward_state.out_edges(n)

                if len(in_edges) == 1 and len(out_edges) == 1:
                    A = in_edges[0].src
                    y = in_edges[0].data
                    C = out_edges[0].dst
                    x = out_edges[0].data
                    if (isinstance(A, nodes.AccessNode)
                            and isinstance(C, nodes.AccessNode)
                            and y.data == A.data and x.data == C.data):

                        # flip the memlet
                        y.subset, y.other_subset = y.other_subset, y.subset
                        y.data = n.data
                        y.try_initialize(self.sdfg, self.forward_state,
                                         in_edges[0])

    def backward(
        self
    ) -> Tuple[BackwardResult, Dict[str, dt.Array], Dict[str, dt.Array]]:
        """ Generate the backward pass in backward_state.

            :return: tuple of:
                     * the backward result (see :class:`~dace.autodiff.backward_implementation.BackwardResult`)
                     * dict of data descriptors for the gradients (i.e. the outputs of the backward pass)
                     * dict of data descriptors of required outputs from the forward pass. These need to be added to the
                       parent SDFG of the backward pass.
        """

        if self._applied:
            raise AutoDiffException(
                "Backward may only be called once. Instantiate a new BackwardPassGenerator."
            )

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
            if self.array_grad_name(
                    given_grad.data) not in self.backward_sdfg.arrays:
                self._add_gradient_data_descriptor(given_grad.data)

        # execute hooks
        for hook in self.completion_hooks:
            hook(self)

        # prepare the output
        required_grad_names = {
            name.data: self.array_grad_name(name.data)
            for name in self.required_gradients
        }
        given_grad_names = {
            name.data: self.array_grad_name(name.data)
            for name in self.given_gradients
        }

        # set mapping from gradient name to whether it should be zeroed out on
        # initialization
        zero_init: Dict[str, bool] = {}
        for node, bres in self.result_map.items():
            for zname, zinit in bres.zero_init.items():
                # Reverse lookup
                cname = next(k for k, v in bres.required_grad_names.items()
                             if v == zname)
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
        forward_nodes = {
            n
            for e in self.forward_state.bfs_edges(self.required_gradients)
            for n in [e.src, e.dst]
        }
        backward_nodes = {
            n
            for e in self.forward_state.bfs_edges(self.given_gradients,
                                                  reverse=True)
            for n in [e.src, e.dst]
        }

        forward_subgraph = dstate.StateSubgraphView(
            self.forward_state,
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
        for node in reversed(
                list(
                    dutils.dfs_topological_sort(subgraph,
                                                subgraph.source_nodes()))):

            try:
                # output names on the forward node
                # (for which the gradient will be connected as an input on the reverse node)
                given_gradients = [
                    edge.src_conn for edge in subgraph.out_edges(node)
                    if _path_src_node_in_subgraph(edge, subgraph)
                ]

                # input names on the forward node that gradients should be generated for
                required_gradients = [
                    edge.dst_conn for edge in subgraph.in_edges(node)
                    if _path_src_node_in_subgraph(edge, subgraph)
                ]

                reversed_node, backward_result = self._get_reverse_node(
                    node, given_gradients, required_gradients)

                self.reverse_map[node] = reversed_node
                self.result_map[node] = backward_result

                # connect the required inputs of the reverse node:
                # the gradients ...
                self._connect_given_gradients(subgraph, node)
                # ... and any required input values from the forward pass

                ####################################
                # Determine which forward inputs we need to connect.
                # these are the in_connectors on the reverse node, minus what has already been connected.
                already_connected = {
                    e.dst_conn
                    for e in self.backward_state.in_edges(reversed_node)
                }
                required_inputs = set(
                    reversed_node.in_connectors).difference(already_connected)
                required_inputs = {c: c for c in required_inputs}
                self._connect_forward_inputs(node, reversed_node,
                                             required_inputs)

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
                        for i, edge in enumerate(
                                self.backward_state.in_edges(reversed_node)):

                            intermediate_desc = copy.deepcopy(grad_desc)

                            intermediate_desc.transient = True
                            intermediate_name = self.backward_sdfg.add_datadesc(
                                f"{array_grad_name}_contribution_{i}",
                                intermediate_desc,
                                find_new_name=True)
                            access_intermediate = self.backward_state.add_access(
                                intermediate_name)

                            for mte in self.backward_state.memlet_tree(edge):
                                mte.data.data = intermediate_name
                            new_edge = self.backward_state.add_edge(
                                edge.src, edge.src_conn, access_intermediate,
                                None, edge.data)
                            self._set_wcr_sum_if_needed(new_edge)
                            summation_node.add_in_connector(f"data_0__{i}")
                            self.backward_state.add_edge(
                                access_intermediate, None, summation_node,
                                f"data_0__{i}",
                                self.backward_sdfg.make_array_memlet(
                                    intermediate_name))
                            self.backward_state.remove_edge(edge)

                        self.backward_state.add_edge(
                            summation_node, "sum", reversed_node, None,
                            self.backward_sdfg.make_array_memlet(
                                array_grad_name))

                        if dtypes.can_access(dtypes.ScheduleType.CPU_Multicore,
                                             grad_desc.storage):
                            pass
                        elif dtypes.can_access(dtypes.ScheduleType.GPU_Default,
                                               grad_desc.storage):
                            summation_node.schedule = dtypes.ScheduleType.GPU_Default
                        else:
                            raise ValueError(
                                f"Unsupported storage {grad_desc.storage}")
                    elif self.backward_state.in_degree(reversed_node) == 1:
                        self._set_wcr_sum_if_needed(
                            self.backward_state.in_edges(reversed_node)[0])

            except AutoDiffException as e:
                raise AutoDiffException("Failed at node {}: {}".format(
                    node, str(e))) from e

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
            for _, _, _, dst_conn, _ in self.backward_state.in_edges(
                    path_edge.dst):
                connector_in_edges[dst_conn] += 1

            more_than_one_edge_to_connector = any(
                v > 1 for v in connector_in_edges.values())

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
            raise AutoDiffException(
                f"descriptor for gradient of {data_name} ({grad_name}) already exists"
            )

        array = self.sdfg.arrays[data_name]

        if not isinstance(array, (dt.Scalar, dt.Array, dt.View)):
            raise AutoDiffException(
                "Unsupported data descriptor {}".format(array))

        cloned_datadesc = copy.deepcopy(array)

        # only the grads of the inputs and the outputs are not transient
        cloned_datadesc.transient = data_name not in self.input_names and data_name not in self.output_names

        self.backward_grad_arrays[grad_name] = cloned_datadesc
        self.backward_sdfg.arrays[grad_name] = copy.deepcopy(cloned_datadesc)

    def _connect_given_gradients(self, subgraph: dstate.StateSubgraphView,
                                 forward_node):
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
                raise AutoDiffException(
                    "Unsupported reduction type {} on memlet".format(
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

    def _connect_forward_inputs(self, forward_node: nodes.Node,
                                backward_node: nodes.Node,
                                required_inputs: Dict[str, str]):
        """ Connect the reversed node of `forward_node` to all required non-gradient inputs.

            There are non-trivial points to handle:
            1. When we read an input from an accessnode in the forward pass, we need to route through maps in the
               backward pass.
            2. In some cases, we need to save the value of a connector to an array so that the backward pass can
               read it.
               For now, this is only supported when the node is at the "top level" of the SDFG, since it's quite
               difficult to handle otherwise (you have to decide whether to recompute or to store the value, and you
               have to store the value once for every iteration in the map)

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
        input_edges_to_connect = (
            edge for edge in self.forward_state.in_edges(forward_node)
            if edge.dst_conn in required_inputs)

        for edge in input_edges_to_connect:

            edge_src = edge.src
            next_required_inputs: Dict[Optional[str], Optional[str]]
            replicated_edge_src: nodes.Node
            replicated_edge_src_conn: str
            
            if isinstance(edge_src, nodes.MapEntry):
                # in the map case, the map must already exist in the bwd pass
                # (the following function call internally asserts this)
                replicated_edge_src = self._find_backward_entry_node_for_map_entry(
                    edge_src)
                new_in_conn, new_out_conn = _add_through_connector(
                    replicated_edge_src)

                replicated_edge_src_conn = new_out_conn

                # the inverse connector of edge.src_conn should be connected
                # to the new through connector we made
                next_required_inputs = {
                    _invert_map_connector(edge.src_conn): new_in_conn
                }

            else:
                replicated_edge_src_conn = edge.src_conn

                if edge_src in self.replicated_map:
                    replicated_edge_src = self.replicated_map[edge_src]
                else:
                    # node has not been replicated yet: do it now
                    replicated_edge_src = copy.deepcopy(edge_src)
                    self.backward_state.add_node(replicated_edge_src)

                if isinstance(edge_src, nodes.AccessNode):
                    is_base_level = self.forward_state.scope_dict(
                    )[edge_src] is None
                    data_name = edge_src.data
                    data_desc = copy.deepcopy(edge_src.desc(self.sdfg))
                    if self.separate_sdfgs:
                        # need to copy over the descriptor from the forward pass
                        if data_name not in self.backward_sdfg.arrays:
                            self.backward_sdfg.add_datadesc(
                                data_name, data_desc)

                    if isinstance(data_desc, dt.View):
                        next_required_inputs = {None: None}
                    elif not is_base_level:
                        # in case this is not the base level
                        # we check if this is a temporary value representing a base level AccessNode
                        # TODO: change this by using memlet_path
                        base_level_access_node = self.check_temporary_access_node(edge_src)
                        
                        # if the data can be accessed without storing or recomputation
                        if base_level_access_node:
                            base_level_data_name = base_level_access_node.data
                            base_level_data_desc = copy.deepcopy(base_level_access_node.desc(self.sdfg))
                            if base_level_data_name not in self.backward_input_arrays:
                                self.backward_input_arrays[base_level_data_name] = base_level_data_desc

                            if self.separate_sdfgs:
                                # because we need to forward this, the descriptor
                                # is no longer transient
                                base_level_data_desc.transient = False
                            
                            # get the starting edge
                            # we know there is only one incoming edge to this access node
                            fwd_node_edge = self.forward_state.in_edges(edge_src)[0]
                            parent_node = self.backward_state.scope_dict()[backward_node]
                            child_node = replicated_edge_src
                            
                            edge_list = self.forward_state.memlet_path(fwd_node_edge)
                            
                            # iterate in both directions
                            while parent_node and edge_list:
                                # switch to the new connection
                                fwd_node_edge = edge_list.pop()
                                parent_node_conn = fwd_node_edge.src_conn
                                child_node_conn = fwd_node_edge.dst_conn
                                memlet_data = copy.deepcopy(fwd_node_edge.data)
                                
                                # add the necessary connectors 
                                if parent_node_conn:
                                    parent_node.add_out_connector(parent_node_conn)
                                if child_node_conn:
                                    child_node.add_in_connector(child_node_conn)
                                # add the new edge
                                self.backward_state.add_edge(parent_node,
                                                            parent_node_conn,
                                                            child_node,
                                                            child_node_conn,
                                                            memlet_data)

                                child_node = parent_node
                                parent_node = self.backward_state.scope_dict()[parent_node]
                                
                            # make sure we went through all of the path expect for the base level access node
                            # the base level access node will be added later through backward_input_arrays
                            assert len(edge_list) == 1 and not parent_node
                            
                            # child node now points to the last map in the nest
                            # This map should have as input the base level access node
                            
                            last_edge = edge_list.pop()
                            last_map_in_forward = last_edge.dst
                            last_map_in_backward = child_node
                            last_connector = last_edge.dst_conn
                            last_memlet_data = copy.deepcopy(last_edge.data)
                            
                            
                            assert last_edge.dst_conn
                            # add the destination connector 
                            last_map_in_backward.add_in_connector(last_connector)
                            # replicate the access node
                            if edge_src in self.replicated_map:
                                replicated_base_level_access_node = self.replicated_map[base_level_access_node]
                            else:
                                # node has not been replicated yet: do it now
                                replicated_base_level_access_node = copy.deepcopy(base_level_access_node)
                                self.backward_state.add_node(replicated_base_level_access_node)
                                    
                            
                            # add final edge
                            self.backward_state.add_edge(replicated_base_level_access_node,
                                                            None,
                                                            last_map_in_backward,
                                                            last_connector,
                                                            last_memlet_data)
                        # no recursive call necessary
                        next_required_inputs = {}
                    else:
                        # base-case: we have reached a base level AccessNode.
                        # this value must be forwarded.
                        if data_name not in self.backward_input_arrays:
                            self.backward_input_arrays[data_name] = data_desc

                        if self.separate_sdfgs:
                            # because we need to forward this, the descriptor
                            # is no longer transient
                            data_desc.transient = False

                        # because this is the base-case there is no recursive call
                        # in this branch; next_required_inputs stays empty
                        next_required_inputs = {
                            c: c
                            for c in edge_src.in_connectors
                        }
                elif isinstance(edge_src, nodes.Tasklet):
                    # in the tasklet case, we need to connect all inputs
                    next_required_inputs = {
                        c: c
                        for c in edge_src.in_connectors
                    }
                else:
                    raise AutoDiffException("Unsupported node")

            new_edge_data = copy.deepcopy(edge.data)
            if isinstance(edge_src, nodes.CodeNode) and isinstance(
                    edge.dst, nodes.CodeNode):
                # code->code edges have a small special case:
                # we need to copy the descriptor
                data_name = new_edge_data.data
                data_desc = copy.deepcopy(self.sdfg.arrays[data_name])
                if self.separate_sdfgs:
                    self.backward_sdfg.add_datadesc(data_name, data_desc)
                else:
                    new_data_name = self.backward_sdfg.add_datadesc(
                        data_name, data_desc, find_new_name=True)
                    new_edge_data.data = new_data_name

            # add the new edge
            self.backward_state.add_edge(replicated_edge_src,
                                         replicated_edge_src_conn,
                                         backward_node,
                                         required_inputs[edge.dst_conn],
                                         new_edge_data)

            if next_required_inputs:
                # if there are any required inputs on the new node, we need to
                # recursively call
                self._connect_forward_inputs(edge_src, replicated_edge_src,
                                             next_required_inputs)
        
        
    def check_temporary_access_node(self, node: nodes.Node) -> nodes.AccessNode:
        """
        Check if the input AccessNode is a temporary value from a base level AccessNode and returns it
        """
        if not isinstance(node, nodes.AccessNode):
           raise AutoDiffException(
                f"Attempted to check access of a non-Access node {node}"
                )
        # Check if the node itself is at base level
        is_base_level = self.forward_state.scope_dict(
                    )[node] is None
        
        # TODO: Do we want this behaviour
        if is_base_level:
            return node
        
        # node is not at base level
        # check if it is a temporary that represents a base level node
        # i.e. if it has a single edge in, a single edge out and a memlet to a base level AccessNode
        
        # get all the in-edges to this node
        in_edges = self.forward_state.in_edges(node)
        # get all the out-edges from this node
        out_edges = self.forward_state.out_edges(node)
        
        # TODO: is this rule too strict? why limit the out edges
        # TODO: is there a case where the in-edges are zero but the node is not base level?
        if len(in_edges) != 1 or len(out_edges) != 1:
            return None
        
        in_edge = in_edges[0]
        
        data = in_edge.data
        # TODO: can the data of an in-edge not be a memlet?
        if isinstance(data, Memlet):
            access_node_name = data.data

            # TODO: is there a better way to get an access node by its name?
            for nd in self.forward_state.nodes():
                if isinstance(nd, nodes.AccessNode) and nd.data == access_node_name:
                    return nd
        
        
    def _lookup_required_grad_name(self, node: nodes.Node, connector: str) -> str:
        if node not in self.result_map:
            raise AutoDiffException(
                "Attempted to access gradient of {}"
                " before the backward node was created".format(node))
        return self.result_map[node].required_grad_names[connector]

    def _lookup_given_grad_name(self, node: nodes.Node, connector: str) -> str:
        if node not in self.result_map:
            raise AutoDiffException(
                "Attempted to access gradient of {}"
                " before the backward node was created".format(node))
        return self.result_map[node].given_grad_names[connector]

    def _find_backward_entry_node_for_map_entry(
            self, entry_node: nodes.MapEntry) -> nodes.MapExit:
        """Find the entry node in the backward pass corresponding to the exit node opened by
        `entry_node` (where `entry_node` is a node from the forward pass).
        """
        src_candidates = [
            cast(nodes.MapExit, node) for node in self.backward_state.nodes()
            if isinstance(node, nodes.MapEntry)
            and node.map == self.reverse_map[entry_node.map]
        ]
        if len(src_candidates) != 1:
            # this shouldn't happen; if we are within a scope, the exit nodes
            # for the scope should already exist in the backward pass
            raise AutoDiffException("Invalid graph")

        return src_candidates[0]

    def _get_reverse_node(self, node, given_gradients,
                          required_gradients) -> ReverseNodeReturnType:
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
            return getattr(self, "_reverse_" + type(node).__name__)(
                node, given_gradients, required_gradients)

        # (2)
        impl = find_backward_implementation(self.sdfg,
                                            forward_state=self.forward_state,
                                            node=node)
        if impl is not None:
            backward_node, backward_result = impl.backward(
                forward_node=node,
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
            if isinstance(
                    backward_node, nodes.NestedSDFG
            ) and backward_node.schedule is not dtypes.ScheduleType.Default:
                infer_types._set_default_schedule_types(
                    backward_node.sdfg, backward_node.schedule, True)
                infer_types._set_default_storage_types(backward_node.sdfg,
                                                       backward_node.schedule)
            return backward_node, backward_result

        raise AutoDiffException(
            "Unable to differentiate node type {}. Either add a pure forward implementation "
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
            is_init_state = [(state, is_initialization_state(state))
                             for state in node.sdfg.nodes()]
            num_non_init_states = sum(b for _, b in is_init_state)
            if num_non_init_states > 1:
                raise AutoDiffException(
                    "A nested SDFG may consist of at most one state (with the "
                    "exception of initalization states), found {} states".
                    format(num_non_init_states))
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

        inputs = set(backward_result.given_grad_names[name]
                     for name in sorted(given_gradients))
        # loop through the arrays that we need from the forward pass
        for name, desc in sorted(backward_input_arrays.items()):
            # if the name is not already passed to the reverse SDFG node ...
            if name not in required_gradients and name not in node.in_connectors:
                # ... this array needs to be forwarded out of the forward SDFG (i.e. it is an intermediate value)
                # 1) add it to the current SDFG, and to self.backward_input_arrays
                # 2) add an out connector to the forward nested SDFG, add a write node to the current state, and an edge
                #    from the output to there
                # 3) add a read node to the backward state, and an edge into it

                desc, forwarded_name, _ = _walk_up_memlet_tree_through_view_nodes(
                    node.sdfg, state_to_diff, name)

                # (1)
                new_name = find_str_not_in_set(set(self.sdfg.arrays),
                                               forwarded_name + "_forwarded")
                if new_name in self.sdfg.arrays or new_name in self.backward_input_arrays:
                    raise AutoDiffException(
                        "Attempted to create array with name '{}', but it already existed"
                        .format(new_name))

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
                self.forward_state.add_edge(
                    node, forwarded_name, write, None,
                    self.sdfg.make_array_memlet(new_name))

                # (3)
                read = self.backward_state.add_read(new_name)
                deferred_edges.append(
                    dict(
                        u=read,
                        u_connector=None,
                        v_connector=forwarded_name,
                        memlet=self.backward_sdfg.make_array_memlet(new_name)))
                inputs.add(forwarded_name)
            else:
                inputs.add(name)

        outputs = set(backward_result.required_grad_names[name]
                      for name in required_gradients)

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

        return nsdfg, BackwardResult(
            required_grad_names=backward_result.required_grad_names,
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

        return rev, BackwardResult(required_grad_names=required_grad_names,
                                   given_grad_names=given_grad_names)

    def _reverse_MapEntry(
        self,
        node: nodes.MapEntry,
        given_gradients: List[str],
        required_gradients: List[str],
    ) -> ReverseNodeReturnType:

        required_grad_names = {
            n: _invert_map_connector(n)
            for n in required_gradients
        }
        given_grad_names = {
            n: _invert_map_connector(n)
            for n in given_gradients
        }
        result = BackwardResult(required_grad_names=required_grad_names,
                                given_grad_names=given_grad_names)
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
            raise AutoDiffException(
                "Expected tasklet with language Python, got language {}".
                format(tasklet.language))

        # tasklets should have scalar inputs (can be relaxed)
        for _, _, _, _, memlet in self.forward_state.in_edges(tasklet):
            try:
                _is_int_value(memlet.subset.num_elements(), 1)
            except AutoDiffException as e:
                raise AutoDiffException(
                    "Autodiff only supported for tasklets with scalar inputs and outputs"
                ) from e

        for _, _, _, _, memlet in self.forward_state.out_edges(tasklet):
            try:
                _is_int_value(memlet.subset.num_elements(), 1)
            except AutoDiffException as e:
                raise AutoDiffException(
                    "Autodiff only supported for tasklets with scalar inputs and outputs"
                ) from e

        code_str = tasklet.code.as_string
        output_exprs = code_to_exprs(code_str, tasklet.in_connectors,
                                     tasklet.out_connectors)

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
                    rev_output_grad_name = find_str_not_in_set(
                        rev_outputs, inp + "_gradient")
                    result.required_grad_names[inp] = rev_output_grad_name
                    rev_outputs.add(rev_output_grad_name)
                else:
                    rev_output_grad_name = result.required_grad_names[inp]

                output_expr = output_exprs[output_conn]

                # symbolically differentiate the output w.r.t inp
                diff_expr = output_expr.diff(sp.symbols(inp))

                # do common subexpression elimination
                sub_expressions, diff_expr = sp.cse(diff_expr,
                                                    symbols=symbol_generator)

                diff_expr = diff_expr[0]

                if diff_expr.atoms(sp.Derivative):
                    # the final result contains a call to sp.Derivative
                    raise AutoDiffException(
                        "Unable to symbolically differentiate expression: {}".
                        format(diff_expr.expr))

                if output_conn not in result.given_grad_names:
                    # pick a name for the input gradient
                    rev_input_grad_name = find_str_not_in_set(
                        rev_inputs, output_conn + "_gradient")
                    result.given_grad_names[output_conn] = rev_input_grad_name
                else:
                    rev_input_grad_name = result.given_grad_names[output_conn]

                input_symbols = diff_expr.free_symbols\
                    .union(s for _, e in sub_expressions for s in e.free_symbols)\
                    .difference(e for e, _ in sub_expressions)

                rev_inputs |= _symbols_to_strings(input_symbols) | {
                    rev_input_grad_name
                }

                diff_code_str = "{input} * ({diff_expr})".format(
                    input=rev_input_grad_name, diff_expr=str(diff_expr))
                # small hack: our heaviside is lowercase
                diff_code_str = diff_code_str.replace("Heaviside", "heaviside")

                diff_code_str = astunparse.unparse(SympyCleaner().visit(
                    ast.parse(diff_code_str)))

                sub_expression_code_strs = "\n".join(
                    f"{target} = {expression}"
                    for target, expression in sub_expressions)

                # get the the final type of the gradient: this is just the type of the input connector we creating the
                # gradient for
                cands = list(
                    self.forward_state.in_edges_by_connector(tasklet, inp))
                if len(cands) != 1:
                    raise AutoDiffException(
                        f"Unexpected graph structure, could not find input edge for connector {inp}"
                        f" on tasklet {tasklet}")

                converted_code = cast_consts_to_type(
                    diff_code_str, self.sdfg.arrays[cands[0].data.data].dtype)
                converted_code = converted_code.replace("\n", " ")

                converted_sub_expressions = cast_consts_to_type(
                    sub_expression_code_strs,
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
            return ast.copy_location(
                ast.parse("(3.141592653589)").body[0], node)
        else:
            return self.generic_visit(node)
