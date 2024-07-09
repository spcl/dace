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
import re
from typing import List, Tuple, Set, Dict, Union, Deque, cast, Optional, Callable, Sequence
import numpy as np
import dace
from dace.properties import CodeBlock
import dace.sdfg.nodes as nodes
import dace.transformation.transformation as xf
import sympy as sp
from dace import dtypes, subsets, data as dt

from dace.frontend.operations import detect_reduction_type
from dace.sdfg import SDFG, SDFGState, graph as dgraph, state as dstate, utils as dutils, infer_types
from dace.sdfg.state import LoopRegion, ControlFlowRegion

from dace.sdfg.analysis import cfg
from dace.memlet import Memlet
from dace.transformation.interstate.loop_detection import (DetectLoop, find_for_loop)
from dace.autodiff.base_abc import (BackwardContext, BackwardResult, AutoDiffException, find_backward_implementation)

from dace.autodiff.utils import cast_consts_to_type
from dace.util import find_str_not_in_set
from dace.libraries.onnx.forward_implementation_abc import ONNXForward
from dace.libraries.onnx.nodes.onnx_op import ONNXOp

from dace.codegen.targets.cpp import is_write_conflicted_with_reason

ReverseNodeReturnType = Tuple[nodes.Node, BackwardResult]

log = logging.getLogger(__name__)


def init_grad(data: str, sdfg: SDFG, current_state: SDFGState) -> SDFGState:
    """
    Add a state where `data` is initialized with zero.

    :param data: the data to initialize
    :param sdfg: the SDFG to add the state to
    :param current_state: the current state; the initialization will be done before this state
    :return: the newly created initialization state
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
            "_init_" + data + "_", {"i{}".format(i): "0:{}".format(shape)
                                    for i, shape in enumerate(arr.shape)}, {},
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

    return state


def _symbols_to_strings(symbs: Set[sp.Symbol]) -> Set[str]:
    return {str(symb) for symb in symbs}


def generate_grad_connector_names(existing_connectors: Set[str], forward_connector_names: List[str]) -> Dict[str, str]:
    """ Choose connector names for the gradients of all forward connectors.

        :param existing_connectors: existing connectors on the node.
        :param forward_connector_names: the list of connectors to generate names for.
        :return: a mapping from entries in ``forward_connector_names`` to names for those entries.
    """

    # copy
    existing_connectors = set(existing_connectors)

    names = {}
    for n in sorted(forward_connector_names):
        result = find_str_not_in_set(existing_connectors, n + "_gradient")
        names[n] = result
        existing_connectors.add(result)

    return names


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


# TODO: use this function if the compare memlet assignement
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


def _path_src_node_in_subgraph(edge: dgraph.MultiConnectorEdge, subgraph: dstate.StateSubgraphView):
    path_src = subgraph.memlet_path(edge)[0].src
    return path_src in subgraph.nodes()


class BackwardPassGenerator:
    """ Class that holds the states for one backward pass creation.

        See autodiff.py, _reverse_NestedSDFG and pytorch.py for examples of usage.
        :param given_gradients: the outputs that gradients must be provided for (i.e. access nodes will be created for
               these)
        :param required_gradients: the inputs to generate gradients for
        :param backward_sdfg: the sdfg the backward pass will be contained in. If it is the same as the forward_sdfg,
                              outputs must be a list containing a single scalar.
        :param zero_non_transients: Whether non-transient gradient buffers should be zero initialized in the backward
                                    SDFG.
        :param array_grad_map: A mapping from array name to the gradient array name. May be passed when certain
                               mappings already exist.
        :param conflicted_gradient_buffers: A list of forward pass value names for which multiple backward passes will
                                            be computed, and thus gradients should be computed with
                                            write-conflict-resolution.
        :param overwrite_strategy: The strategy to use to provide overwritten values from the forward pass to the backward pass. 
                                    Should be either: store_all or recompute_all.
    """
    def __init__(
        self,
        *,
        sdfg: SDFG,
        given_gradients: Sequence[Union[nodes.AccessNode, str]],
        required_gradients: Sequence[Union[nodes.AccessNode, str]],
        backward_sdfg: SDFG,  # This can be the same as sdfg
        zero_non_transients: bool,
        array_grad_map: Optional[Dict[str, str]] = None,
        conflicted_gradient_buffers: Optional[Set[str]] = None,
        overwrite_strategy: str = "store_all",
    ):

        self.sdfg = sdfg
        self.strategy = overwrite_strategy
        self.backward_sdfg = backward_sdfg

        given_gradients = [
            n if isinstance(n, nodes.AccessNode) else self._str_to_access(n, "outputs") for n in given_gradients
        ]
        required_gradients = [
            n if isinstance(n, nodes.AccessNode) else self._str_to_access(n, "inputs") for n in required_gradients
        ]
        required_gradients = [n for n in required_gradients if n is not None]

        self.given_gradients = given_gradients
        self.required_gradients = required_gradients

        self.given_gradients_data = {n.data for n in given_gradients}
        self.required_gradients_data = {n.data for n in required_gradients}

        self.input_names = {n.data for n in required_gradients}
        self.output_names = {n.data for n in given_gradients}

        #: Arrays descs for the gradients
        self.backward_grad_arrays: Dict[str, dt.Array] = {}

        #: Arrays descs for inputs that are required from the forward pass
        self.backward_input_arrays: Dict[str, dt.Array] = {}

        #: Mapping from forward node -> backward node, and forward map -> backward map
        self.reverse_map: Dict[nodes.Node, Union[nodes.Node, nodes.Map]] = {}

        #: Mapping from forward state -> backward state
        self.reversed_states_map: Dict[SDFGState, SDFGState] = {}

        #: Mapping from forward LoopRegion -> backward LoopRegion
        self.reversed_loops_map: Dict[LoopRegion, LoopRegion] = {}

        #: Mapping from forward state -> backward state for loop states
        self.reversed_loop_states_map: Dict[nodes.Node, nodes.Node] = {}

        #: Mapping between states and their views that indicate what to AD
        self.states_view_map: Dict[SDFGState, dstate.StateSubgraphView] = {}

        #: Mapping between states and their views that indicate what to AD
        self.loop_states_view_map: Dict[SDFGState, dstate.StateSubgraphView] = {}

        #: Mapping between states and their necessary initialization states
        self.init_states: Dict[SDFGState, SDFGState] = {}

        #: Mapping from forward_node -> BackwardResult for that node
        self.result_map: Dict[nodes.Node, BackwardResult] = {}

        #: Mapping from access nodes required from the forward pass, and their stored access nodes
        self.stored_data_map: Dict[Tuple[SDFGState, nodes.AccessNode], Tuple[nodes.AccessNode, List[Memlet]]] = {}

        #: Mapping from forward name to gradient name for arrays
        self.array_grad_map: Dict[str, str] = array_grad_map or {}

        # Topological orderning of the states
        self.state_order = self.sdfg.states()
        self.conflicted_gradient_buffers: Set[str] = conflicted_gradient_buffers or set()

        # Variable to check if backward has already been applied
        self._applied = False
        self.zero_non_transients = zero_non_transients

        # Sanity-check: the outputs need to be present in at least one of the states of the sdfg
        for outp in self.given_gradients:
            found = False
            for state in self.state_order:
                if outp in state:
                    found = True
                    break
            if not found:
                raise AutoDiffException(f"Could not find output {outp} in any of the SDFG states")

        # Sanity-check: the inputs need to be present in at least one of the states of the sdfg
        for inp in self.required_gradients:
            found = False
            for state in self.state_order:
                if inp in state:
                    found = True
                    break
            if not found:
                raise AutoDiffException(f"Could not find input {inp} in any of the SDFG states")

        if sdfg is backward_sdfg:
            # TODO: why is this condition necessary
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

    def backward(self) -> Tuple[BackwardResult, Dict[str, dt.Array], Dict[str, dt.Array]]:
        """ Generate the backward pass in backward_sdfg.
        """
        return self.reverse_sdfg()

    def reverse_sdfg(self):
        """
        Go through all the state of the SDFG and reverse each one of them. 
        Connect these newly reversed states with the appropriate conditions so that the gradients flow correctly.
        :return: tuple of:
            * the backward result (see :class:`~dace.autodiff.backward_implementation.BackwardResult`)
            * dict of data descriptors for the gradients (i.e. the outputs of the backward pass)
            * dict of data descriptors of required outputs from the forward pass. These need to be added to the
            parent SDFG of the backward pass.
        """

        if self._applied:
            raise AutoDiffException("Backward may only be called once. Instantiate a new BackwardPassGenerator.")

        self._remove_unnecessary_conditional_regions()

        # If any of the inputs is overwritten before it is ever used,
        # we will return zero for its gradients and skip AD for it.
        # Identify what gradients follow this pattern and remove them from the gradients list if necessary
        self._identify_overwritten_inputs()

        # Create state views mapping and exapnd all the SDFG nodes
        self._create_stateviews_mapping()

        # Reverse each state in the graph
        self._reverse_states()

        # Connect the new reversed states to the other states correctly
        self._connect_reversed_state()

        # Fill the interstate edges with the correct conditions
        self._fill_interstate_edge_conditions()

        # Zero out the necessary gradients
        # These are the overwritten inputs and inputs that don't contribute to the desired output
        self._zero_out_required_gradients()

        # In some cases (accessnode -> accessnode), the descriptors for the gradients of the function outputs are not
        # added yet. Add them now.
        for given_grad in sorted(self.given_gradients, key=lambda k: k.data):
            if self.array_grad_name(given_grad.data) not in self.backward_sdfg.arrays:
                self._add_gradient_data_descriptor(given_grad.data)

        # Execute completion hooks
        for hook in self.completion_hooks:
            hook(self)

        # Prepare the output
        required_grad_names = {name.data: self.array_grad_name(name.data) for name in self.required_gradients}
        given_grad_names = {name.data: self.array_grad_name(name.data) for name in self.given_gradients}

        # Set mapping from gradient name to whether it should be zeroed out on initialization
        zero_init: Dict[str, bool] = {}
        for node, bres in self.result_map.items():
            forward_state = self._get_node_state(node=node)
            for zname, zinit in bres.zero_init.items():
                # Reverse lookup
                cname = next(k for k, v in bres.required_grad_names.items() if v == zname)

                for e in forward_state.in_edges_by_connector(node, cname):
                    zero_init[e.data.data] = zinit
                for e in forward_state.out_edges_by_connector(node, cname):
                    zero_init[e.data.data] = zinit

        self._applied = True
        result = BackwardResult(required_grad_names=required_grad_names,
                                given_grad_names=given_grad_names,
                                zero_init=zero_init)
        return result, self.backward_grad_arrays, self.backward_input_arrays

    def _remove_unnecessary_conditional_regions(self):
        """
        """
        for region in self.sdfg.all_control_flow_regions():
            if isinstance(region, SDFG) or isinstance(region, LoopRegion):
                continue
            if isinstance(region, ControlFlowRegion):
                # Make sure this region has only one state
                nodes = dutils.dfs_topological_sort(region, region.source_nodes())
                for node in nodes:

                    # Add this state to the parent graph of the control region
                    parent_graph = region.parent_graph
                    parent_graph.add_node(node)

                    # The ebd state needs to be connected to the out edges of the region
                    if region.out_degree(node) == 0:

                        # Get all the out edges of the region
                        out_edges = parent_graph.out_edges(region)

                        # Replicate all the edges to the state
                        for edge in out_edges:
                            src, dst, data = edge
                            parent_graph.add_edge(src=node, dst=dst, data=data)

                    # The start states needs to be connected to the in edges of the region
                    if region.in_degree(node) == 0:

                        # Get all the in edges of the region
                        in_edges = parent_graph.in_edges(region)

                        # Replicate all the edges to the state
                        for edge in in_edges:
                            src, dst, data = edge
                            parent_graph.add_edge(src=src, dst=node, data=data)

                    # All the states need to be connected just as they were inside the region
                    node_in_edges = region.in_edges(node)
                    for edge in node_in_edges:
                        src, dst, data = edge
                        assert src in parent_graph
                        parent_graph.add_edge(src=src, dst=node, data=data)

                # Remove the uncessary region from the graph
                parent_graph.remove_node(region)

    def _identify_overwritten_inputs(self):
        """
        Go through all the inputs we want to differentiate the output w.r.t,
        if any of them are overwritten to before they are used in the SDFG,
        the resulting gradient for this input is zero. 
        Any changes to this input of the function will be overwritten immediatly.
        In this method we identify which inputs are overwritten to and save them
        in the overwritten_inputs list. 
        We also save the memlet of this overwrite to later only zero-out the correct indices
        in the case where not the whole array has been overwritten.
        To identify which which inputs have been overwritten:
            - We get the first AccessNode to this data in the SDFG
            - If this first occurance has any incoming edges (i.e. write)
            - Save this data to later be zeroed-out by the _zero_out_required_gradients method
        """

        self.overwritten_inputs: List[str, Memlet] = []

        # For each input we to differentiate with respect to
        for required_gradient in self.required_gradients_data:
            # Get the first instance of this input node in the SDFG
            first_occurance = None

            # This assumes topological sort of states
            for state in self.sdfg.states():
                for node in state.all_nodes_recursive():
                    if isinstance(node, nodes.AccessNode) and node.data == required_gradient:
                        first_occurance = node
                        first_occurance_state = state
                        break

                # If we already found the AccessNode, break
                if first_occurance:
                    break

            # If the data is used in the SDFG
            # We will deal with the case where the data is not used
            # or not used in the critical computation graph in the
            # _zero_out_required_gradients method
            if first_occurance:
                in_edges = first_occurance_state.in_edges(first_occurance)

                # Check if the first occurance is a write
                if len(in_edges) > 0:

                    # This is an AN so there only should be one incoming edge
                    assert len(in_edges) == 1
                    memlet = in_edges[0].data

                    # Save this data and the overwrite memlet to the list
                    self.overwritten_inputs.append((first_occurance.data, memlet))

                    # If this is a complete overwrite of the input, we don't want to do AD for it
                    comparison_result = self._compare_memlet_accesses_to_array_size(first_occurance.data, memlet)
                    if comparison_result and comparison_result == 0:
                        # The whole array is overwritten at the start of the program
                        # Remove this from the required gradients list
                        self.required_gradients_data.remove(first_occurance.data)
                        node = [n for n in self.required_gradients if n.data == first_occurance.data]
                        assert len(node) == 1
                        self.required_gradients.remove(node[0])

    def _create_stateviews_mapping(self):
        """
        Maps each state in the SDFG to the views that indicates what to differentiate
        """
        self._find_subgraph_to_differentiate()

        # Expand until there is nothing left to expand
        while self._expand_nodes():
            # Nodes have been expanded again on the expanded graph; recalculate the forward graph
            self._find_subgraph_to_differentiate()

    def _reverse_states(self):
        """
        Go through all the state of the forward SDFG, reverse them and add them to the backward SDFG.
        """
        # For reversal we want to iterate through the states in reverse topological order
        for state in reversed(self.state_order):
            # Get all the views of this state
            assert state in self.states_view_map
            state_subgraph_views = [self.states_view_map[state]]

            # In case this is a state loop
            state_subgraph_loop_view = []
            if state in self.loop_states_view_map:
                loop_view = self.loop_states_view_map[state]
                state_subgraph_loop_view.append(loop_view)

            all_views = state_subgraph_views + state_subgraph_loop_view
            for state_subgraph_view in state_subgraph_views:

                # Make sure this state has not already been reversed
                assert state not in self.reversed_states_map

                # Create the new reversed state label
                if state_subgraph_view in state_subgraph_loop_view:
                    reversed_state_label = f"{state.label}_loop_reversed" if state.label else None
                else:
                    reversed_state_label = f"{state.label}_reversed" if state.label else None

                # Create new state for reversal
                # At the moment we add all states to the backward_sdfg directly
                # This will later be modified when connecting the states
                reversed_state = self.backward_sdfg.add_state(label=reversed_state_label)

                # Add the new state to the reversed map dict
                if state_subgraph_view in state_subgraph_loop_view:
                    self.reversed_loop_states_map[state] = reversed_state
                else:
                    self.reversed_states_map[state] = reversed_state

                # Check that all edges are float or booleans
                self.check_edges_type_in_state(state_subgraph_view)

                self._disambiguate_direction_dependent_views(state)

                # Recursively reverse the subgraph
                self._reverse_subgraph(forward_state=state, backward_state=reversed_state, subgraph=state_subgraph_view)

        # We also reverse all the LoopRegions in the graph
        for node in self.sdfg.nodes():
            if not isinstance(node, LoopRegion):
                continue
            self._reverse_loop_region(node)

    def _connect_reversed_state(self):
        """
        Go through all the state of the forward SDFG and connect the equivelent backward state as necessary.
        All the incoming edges of a state in the forward SDFG will result in outgoing edges in the backward SDFG.
        """

        for state in self.state_order:
            # All states should be reversed already
            assert state in self.reversed_states_map
            backward_state = self.reversed_states_map[state]

            # Check if a gradient initialization state has been added
            if backward_state in self.init_states:
                # In this case we want to connect any in edges to the initialization instead
                backward_state = self.init_states[backward_state]

            # Get all the out edges of the forward state
            parent_graph = state.parent_graph
            state_out_edges = parent_graph.out_edges(state)

            # If there are no outgoing connections
            if len(state_out_edges) == 0:
                # This is an end-state and it needs to be connected to its reversed state
                # we do this only if the backward sdfg is the same as the forward one
                if parent_graph == self.sdfg and not self.separate_sdfgs:
                    self.backward_sdfg.add_edge(src=state, dst=backward_state, data=dace.InterstateEdge())

            # Get all the in connections of the forward state
            forward_state_in_edges = parent_graph.in_edges(state)

            # Get the backward state again
            # We need to do this in case the state is linked to an initialization state
            # For outgoing edges, we connect the actual state not its initialization
            backward_state = self.reversed_states_map[state]

            for edge in forward_state_in_edges:
                # Each incoming edge to a forward state will add an outgoing edge to a backward state
                fwd_src = edge.src
                if isinstance(fwd_src, SDFGState):
                    bwd_src = self.reversed_states_map[fwd_src]
                elif isinstance(fwd_src, LoopRegion):
                    bwd_src = self.reversed_loops_map[fwd_src]

                graph = bwd_src.parent_graph
                # Again, we need to check if we should make the connection to the state itself
                # or to its initialization if it exists
                if bwd_src in self.init_states:
                    bwd_src = self.init_states[bwd_src]
                graph.add_edge(src=backward_state, dst=bwd_src, data=dace.InterstateEdge())

        # Connect all the loops
        for loop in self.reversed_loops_map.keys():

            # Get the loop parent
            parent_graph = loop.parent_graph

            # Get the reversed loop
            reversed_loop = self.reversed_loops_map[loop]

            # Get all the out edges of the forward state
            loop_out_edges = parent_graph.out_edges(loop)

            # If there are no outgoing connections
            if len(loop_out_edges) == 0:
                # This is an end-region and it needs to be connected to its reversed region
                # We do this only if the backward sdfg is the same as the forward one
                if parent_graph == self.sdfg and not self.separate_sdfgs:
                    self.backward_sdfg.add_edge(src=state, dst=backward_state, data=dace.InterstateEdge())

            # Get all the in edges
            loop_in_edges = parent_graph.in_edges(loop)

            for edge in loop_in_edges:

                # A loop region could be connected to a state or another loop region
                fwd_src = edge.src
                if isinstance(fwd_src, SDFGState):
                    bwd_src = self.reversed_states_map[fwd_src]
                elif isinstance(fwd_src, LoopRegion):
                    bwd_src = self.reversed_loops_map[fwd_src]

                # Again, we need to check if we should make the connection to the state itself
                # or to its initialization if it exists
                if bwd_src in self.init_states:
                    bwd_src = self.init_states[bwd_src]

                # Get the graph to add the edge to
                if isinstance(parent_graph, LoopRegion):
                    bwd_parent_graph = self.reversed_loops_map[parent_graph]
                else:
                    bwd_parent_graph = self.backward_sdfg

                bwd_parent_graph.add_edge(src=reversed_loop, dst=bwd_src, data=dace.InterstateEdge())

    def _fill_interstate_edge_conditions_in_scope(self, graph):
        """
        Get all the nodes within this graph in topological order,
        Connect the states and call the function recusivly on the nested scopes. 
        """
        # A dictionary that keeps track of the conditions necessary to reach a state in the forward passs
        conditions_map: dict[SDFGState, str] = {}

        # Iterate through all the nodes in topological order
        nodes = dutils.dfs_topological_sort(graph, graph.source_nodes())
        for node in nodes:
            if isinstance(node, SDFG) or isinstance(node, LoopRegion):
                # if this is not a reversed loop region
                if not node in self.reversed_loops_map:
                    continue
                self._fill_interstate_edge_conditions_in_scope(node)
                continue

            assert isinstance(node, SDFGState)
            forward_state = node
            parent_graph = forward_state.parent_graph

            # if this is not a reversed state
            if not node in self.reversed_states_map:
                continue

            # We will iterate through all the incoming edges to the forward state
            edges_list = parent_graph.in_edges(forward_state)

            # If there are none, this is a start state
            # If there is only one incoming edge, no condition necessary
            if len(edges_list) < 2:
                conditions_map[forward_state] = "1"
                continue

            # A list of the conditions on all the in edges for this state
            in_edges_conditions: List[str] = []
            for edge in edges_list:
                # Get the src state
                src_state = edge.src

                # Get the condition to get to the source state in the forward pass
                src_state_condition = conditions_map[src_state]

                # Add the condition in the current edge
                current_edge_condition = edge.data.condition.as_string

                # New backward edge condition
                new_bwd_edge_condition = f"({src_state_condition}) and {current_edge_condition}" if current_edge_condition != "1" else src_state_condition
                bwd_edge = self._get_bcakward_state_edge(edge)

                # Add the condition to the edge
                bwd_edge.data.condition = CodeBlock(new_bwd_edge_condition)

                # If there is a special case for the first iteration of the backward state
                if forward_state in self.loop_states_view_map:

                    # Get the corresponding edge between the loop states
                    bwd_loop_edge = self._get_bcakward_loop_state_edge(edge)

                    # Add the same condition to the edge
                    bwd_loop_edge.data.condition = CodeBlock(new_bwd_edge_condition)

                # Add the forward condition to the list to update the conditions_map dict
                if new_bwd_edge_condition != "1":
                    # Only add the condition if it exists
                    in_edges_conditions.append(new_bwd_edge_condition)

            # Update the conditions mapping
            # This will be the logical or of all the saved conditions
            # because we can reach this state by taking any of the incoming edges
            if len(in_edges_conditions) == 0:
                condition_for_state = "1"
            else:
                condition_for_state = in_edges_conditions[0]
                for i in range(1, len(in_edges_conditions)):
                    condition_for_state += f" or {in_edges_conditions[i]}"

            # Since we are doing topological sort before iterating
            # This node should not have been updated before
            assert forward_state not in conditions_map
            conditions_map[forward_state] = condition_for_state

    def _fill_interstate_edge_conditions(self):
        """
        Go through all of the states in the forward graph and fill the necessary conditions in the backward states.
        Each edge in the backward SDFG will be the logical AND between the equivelent edge in the forward SDFG and 
        all of the conditions that are necessary to get to this state in the forward pass.
        """
        self._fill_interstate_edge_conditions_in_scope(self.sdfg)

        # Iterate through all the loop regions and connect the loop states if necessary
        for loop in self.sdfg.all_control_flow_regions():
            # Only iterate over loop regions
            if not isinstance(loop, LoopRegion):
                continue
            # Get the start state
            loop_start_state = loop.start_block
            if not isinstance(loop_start_state, SDFGState):
                # This would be the case for perfectly nested loops
                # Nothing to do in this case
                continue

            if not loop_start_state in self.reversed_loop_states_map:
                # There are no extra states to connect
                continue

            # If there are loop states to connect
            # Prepare the condition for the new state
            loop_it = loop.loop_variable
            reversed_loop = self.reversed_loops_map[loop]
            start, _ = self._extract_loop_region_info(reversed_loop)

            # We only want the loop state to execute
            # in the first iteration of the reversed loop
            first_state_condition = f"{loop_it} == {start}"
            first_state_condition = CodeBlock(first_state_condition)

            leftover_loop_state = self.reversed_loop_states_map[loop_start_state]

            # Get the reversed loop start state
            reversed_loop_start_state = self.reversed_states_map[loop_start_state]

            # In case there are initializations
            if reversed_loop_start_state in self.init_states:
                reversed_loop_start_state = self.init_states[reversed_loop_start_state]

            # Add a state to the reversed loop region
            new_start_state = reversed_loop.add_state_before(reversed_loop_start_state,
                                                             is_start_block=True,
                                                             condition=first_state_condition)

            # The condition for this interstate edge should be all iterations expect the fist
            leftover_iterations_condition = f"not {first_state_condition.as_string}"

            # In case there are initializations
            if leftover_loop_state in self.init_states:
                leftover_loop_state = self.init_states[leftover_loop_state]

            # Add a connection between this new start state and the first iteration state
            reversed_loop.add_edge(src=new_start_state,
                                   dst=leftover_loop_state,
                                   data=dace.InterstateEdge(condition=leftover_iterations_condition))

    def _zero_out_required_gradients(self):
        """
        Zero-out gradients that either didn't contribute to the desired output,
        or were overwritten and identified by the _identify_overwritten_inputs method.
        """
        gradients_to_zero_out: List[str, Memlet] = []
        # First, we zero-out all the required gradients that didn't contribute at all to the output
        for required_gradient in self.required_gradients_data:
            gradient_name = self.array_grad_name(required_gradient)
            added = False
            for node, _ in self.backward_sdfg.all_nodes_recursive():
                if isinstance(node, nodes.AccessNode) and node.data == gradient_name:
                    added = True
                    break
            if not added:
                gradients_to_zero_out.append((required_gradient, None))

        # Fuse the lists of gradients to remove
        gradients_to_zero_out += self.overwritten_inputs

        # We zero out the gradients that were already identified (if any)
        if len(gradients_to_zero_out) == 0:
            return

        # Get all the end-states of the backward SDFG
        bwd_sdfg_exit_nodes = []
        for node in self.backward_sdfg.nodes():
            if len(self.backward_sdfg.out_edges(node)) == 0:
                bwd_sdfg_exit_nodes.append(node)
        assert bwd_sdfg_exit_nodes

        # Create the state that we will be adding the initialization to
        zero_out_state = self.backward_sdfg.add_state_after(bwd_sdfg_exit_nodes[0], label="__gradient_zero_out_state__")

        # Connect all the leftover exit nodes to this new state
        bwd_sdfg_exit_nodes = bwd_sdfg_exit_nodes[1:]
        for exit_node in bwd_sdfg_exit_nodes:
            self.backward_sdfg.add_edge(src=exit_node, dst=zero_out_state, data=dace.InterstateEdge())

        # Add the initialization maps per data
        for data, memlet in gradients_to_zero_out:
            # Zero out each gradient
            self._zero_out_gradient(zero_out_state, data, memlet)

    def _zero_out_gradient(self, init_state: SDFGState, data: str, memlet: Memlet = None):
        """
        Add a zero assignment map to the data array in init_state that zeros-out
        the elements that were previously accesses by memlet. 
        By default we zero-out all the values of the array.
        """

        # Get the original array
        array_desc = self.sdfg.arrays[data]
        new_array_desc = copy.deepcopy(array_desc)
        new_array = self.array_grad_name(data)

        # Add the new array to the sdfg
        self.backward_sdfg.add_datadesc(new_array, new_array_desc)

        # If memlet was specified, only assign to the specified values
        if memlet:
            memlet.data = new_array
        else:
            memlet = dace.Memlet.simple(new_array, ", ".join("i{}".format(i) for i in range(len(new_array_desc.shape))))

        scalar = 0
        if dtypes.can_access(dtypes.ScheduleType.CPU_Multicore, new_array_desc.storage):
            cuda = False
        elif dtypes.can_access(dtypes.ScheduleType.GPU_Default, new_array_desc.storage):
            cuda = True
        else:
            raise ValueError(f"Unsupported storage {new_array_desc.storage}")

        if isinstance(new_array_desc, (dt.Array, dt.Scalar)):
            init_state.add_mapped_tasklet(
                "_init_" + data + "_",
                {"i{}".format(i): "0:{}".format(shape)
                 for i, shape in enumerate(new_array_desc.shape)}, {},
                "__out = {}".format(scalar), {"__out": memlet},
                schedule=dtypes.ScheduleType.GPU_Device if cuda else dtypes.ScheduleType.Default,
                external_edges=True)
        elif type(new_array_desc) is dt.View:
            # No need to initialize: the viewed array will always be visited
            # (since a view can never be a required grad), and thus the viewed array will be initialized.
            pass
        else:
            raise AutoDiffException("Unsupported data descriptor {}".format(new_array_desc))

    def _find_subgraph_to_differentiate(self) -> None:
        """ 
        Determine which nodes we need to reverse; this forms the subgraph we will differentiate:
        we do a reverse BFS and a forward BFS, then take the intersection of nodes found. 
        In the case where a state is within a loop, this may result in different subgraphs
        depending on the loop iteration.

        To calculate the gradients for a node x in ``required_gradients``, we need to sum up the gradient
        contributions from every node y where x is used as an input. We thus first do a forward BFS. Also, the
        gradient contributions of all nodes that are not connected by a path to a ``given_gradient`` node are
        implicitly zero. Thus, we take the intersection of the two BFSs.
        """
        forward_nodes: set[nodes.Node] = set()
        backward_nodes: set[nodes.Node] = set()
        required_gradients_all_states = {n.data for n in self.required_gradients}
        given_gradients_all_states = {n.data for n in self.given_gradients}

        # Do the forward BFS iterativly
        for state in self.state_order:
            state_required_gradients = []

            # Get all the access nodes that are neccessary for AD and are present in this state
            for node in state:
                if isinstance(node, nodes.AccessNode) and node.data in required_gradients_all_states:
                    state_required_gradients.append(node)

            forward_nodes = forward_nodes.union(
                {n
                 for e in state.bfs_edges(state_required_gradients) for n in [e.src, e.dst]})

            # Update the list of required gradients to use for states
            for node in forward_nodes:
                if isinstance(node, nodes.AccessNode) and node.data not in required_gradients_all_states:
                    # We want all of the forward AccessNodes that made it to the intersection
                    required_gradients_all_states.add(node.data)

        # Do the backward BFS iterativly
        for state in reversed(self.state_order):
            state_given_gradients: List[nodes.AccessNode] = []

            for node in state:
                if isinstance(node, nodes.AccessNode) and node.data in given_gradients_all_states:
                    state_given_gradients.append(node)

            backward_nodes = {n for e in state.bfs_edges(state_given_gradients, reverse=True) for n in [e.src, e.dst]}
            nodes_list = list(forward_nodes.intersection(backward_nodes))
            state_subgraph = dstate.StateSubgraphView(state, nodes_list)

            state_subgraph = self._add_missing_nested_sdfg_connectors_to_view(state=state,
                                                                              state_subgraph=state_subgraph,
                                                                              view_nodes=nodes_list)

            # Add mapping
            self.states_view_map[state] = state_subgraph

            # In the case where this state is within a for loop
            if self._state_within_loop(state):
                # Other elements that are not within state_subgraph will need to be reversed
                # We create a separate mapping for these elements

                # Get all the access nodes that are used in the previous view
                subgraph_an = [node.data for node in state_subgraph.nodes() if isinstance(node, nodes.AccessNode)]

                # For each access node in this view
                for state_node in state:
                    if isinstance(state_node, nodes.AccessNode) and state_node.data in subgraph_an:
                        state_given_gradients.append(state_node)

                # Do reverse BFS starting from this new set of nodes
                backward_nodes = {
                    n
                    for e in state.bfs_edges(state_given_gradients, reverse=True) for n in [e.src, e.dst]
                }
                view_nodes = list(forward_nodes.intersection(backward_nodes))
                # Get the intersection with the forward nodes
                # Note that these don't change even in the case of a for loop
                loop_state_subgraph = dstate.StateSubgraphView(state, view_nodes)

                loop_state_subgraph = self._add_missing_nested_sdfg_connectors_to_view(
                    state=state, state_subgraph=loop_state_subgraph, view_nodes=view_nodes)

                # If the two views are different
                # Here we only check if the number of nodes is the same
                # Since states_view_map[state] is a subset of loop_states_view_map[state]
                if len(state_subgraph) != len(loop_state_subgraph):
                    self.loop_states_view_map[state] = loop_state_subgraph

            # Update the list of given gradients to use for states
            for node in backward_nodes:
                if isinstance(node, nodes.AccessNode) and node.data not in given_gradients_all_states:
                    # We want all of the backward AccessNodes that made it to the intersection
                    given_gradients_all_states.add(node.data)

    def array_grad_name(self, forward_name: str) -> str:
        """ Return the gradient name of a name from the forward pass """
        if forward_name not in self.array_grad_map:
            self.array_grad_map[forward_name] = \
                find_str_not_in_set(set(self.backward_sdfg.arrays), "gradient_" + forward_name)

        return self.array_grad_map[forward_name]

    def _add_gradient_data_descriptor(self, data_name: str):
        """ 
        Add the data descriptor for the gradient for `data_name`.
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

    def _state_within_loop(self, forward_state: SDFGState) -> bool:
        """
        Check if this state will be executed several times within a loop.
        We check if any of the parents of this state is a loop region.
        """
        parent = forward_state.parent_graph
        while parent is not None:
            if isinstance(parent, LoopRegion):
                return True
            parent = parent.parent_graph
        return False

    def _reverse_loop_conditional(self, loop: LoopRegion) -> str:
        """
        Given a loop region as a parameter, create the conditional for the reversed
        version of this loop. 
        """

        # Get the loop iterator
        it = loop.loop_variable

        # Get the loop start
        start, _ = self._extract_loop_region_info(loop)

        # Get the stride sign
        stride_sign = self._get_stride_sign(loop)

        # Reverse the conditional to end at the start of the original loop
        # This will be incremented or decremented depending on the stride
        if stride_sign > 0:
            reversed_condition = f"{it} > {start}-1"
        else:
            reversed_condition = f"{it} < {start}+1"

        return reversed_condition

    def _reverse_loop_initial_statement(self, loop: LoopRegion) -> str:
        """
        Given a loop region as a parameter, create the initialization statement for the reversed
        version of this loop.
        """
        # Get the loop iterator
        it = loop.loop_variable

        stride_sign = self._get_stride_sign(loop)

        # Get the loop end
        _, end = self._extract_loop_region_info(loop)

        # Reverse the initialization to start from the end of the forward loop
        # This will be incremented or decremented depending on the stride
        if stride_sign > 0:
            init_expr = f"{it} = {end}-1"
        else:
            init_expr = f"{it} = {end}+1"

        return init_expr

    def _reverse_loop_update_statement(self, loop: LoopRegion) -> str:
        """
        Given a loop region as a parameter, create the update statement for the reversed
        version of this loop.
        """

        # Get the original update statement
        fwd_update = loop.update_statement.as_string

        stride_sign = self._get_stride_sign(loop)

        # If the stride is positive
        if stride_sign > 0:
            update_statement = fwd_update.replace("+", "-")
        else:
            # If the stride is negative
            update_statement = fwd_update.replace("-", "+")

        return update_statement

    def _get_stride_sign(self, loop: LoopRegion) -> int:
        """
        Check if the stride for this loop is positive or negative.
        returns: 1 if the stride is positive and -1 if it is negative
        """
        if loop.update_statement is None:
            raise AutoDiffException("While loops are not yet supported in DaCe AD")
        update_statement = loop.update_statement.as_string
        if "-" in update_statement:
            return -1
        if "+" in update_statement:
            return 1

        # unsupported loop structure
        raise AutoDiffException(f"Expected the loop region {loop.label} to have a regular update statement."
                                f" Instead got: {update_statement}")

    def _extract_loop_region_info(self, loop: LoopRegion):
        """
        Use regular expression matching to extract the start and end of the loop region.
        We only treat regular for-loops with incrementation and decrementation updates.
        """

        # Extract the loop iterator
        it = loop.loop_variable

        # Extract the end of the loop from the conditional statement
        conditional = loop.loop_condition.as_string

        stride_sign = self._get_stride_sign(loop)

        # If the stride is positive
        if stride_sign > 0:
            conditional_expression = fr".*{it} < .*"
        else:
            # If the stride is negative
            conditional_expression = fr".*{it} > .*"

        # Match the conditional using regular expressions
        matches = re.search(conditional_expression, conditional)
        assert matches
        expression = matches.group()
        matches = re.search(conditional_expression[:-2], conditional)
        assert matches
        expression_to_remove = matches.group()
        end = expression.replace(expression_to_remove, "")

        # Remove parenthesis and space
        end = end.replace("(", "")
        end = end.replace(")", "")
        end = end.replace(" ", "")

        # Get the start from the initialization code
        init_code = loop.init_statement.as_string
        matches = re.search(fr".*{it} = .*", init_code)
        assert matches
        expression = matches.group()
        matches = re.search(fr"{it} =", init_code)
        assert matches
        expression_to_remove = matches.group()
        start = expression.replace(expression_to_remove, "")

        # Remove parenthesis and space
        start = start.replace("(", "")
        start = start.replace(")", "")
        start = start.replace(" ", "")

        return start, end

    def _match_loop_region(self, fwd_loop: LoopRegion) -> LoopRegion:
        """
        Create the backward LoopRegion and fill it with the reversal of the forward LoopRegion.
        """

        init_expr = self._reverse_loop_initial_statement(fwd_loop)
        reversed_condition = self._reverse_loop_conditional(fwd_loop)
        update_statement = self._reverse_loop_update_statement(fwd_loop)

        # Create the label
        reversed_label = f"{fwd_loop.label}_reversed"

        # Create the loop object and return it
        reversed_loop = LoopRegion(label=reversed_label,
                                   initialize_expr=init_expr,
                                   condition_expr=reversed_condition,
                                   update_expr=update_statement,
                                   loop_var=fwd_loop.loop_variable)

        return reversed_loop

    def _reverse_loop_region(self, loop: LoopRegion):
        """
        Given a LoopRegion as a parameter, reverse it, add the loop states that belong in this region.
        """

        # Create the reversed loop region
        reversed_loop = self._match_loop_region(fwd_loop=loop)
        self.reversed_loops_map[loop] = reversed_loop

        # Add the reversed loop directly
        parent_graph = self._get_reversed_parent_graph(loop)
        parent_graph.add_node(reversed_loop)

        # Add all the loop nodes to the graph and recursivly reverse child loop regions
        for node in loop.nodes():
            if isinstance(node, LoopRegion):

                # This node shouldn't be reversed already since we're going top-down
                assert node not in self.reversed_loops_map
                self._reverse_loop_region(node)
            elif isinstance(node, SDFGState):

                # Get the backward_node
                bwd_node = self.reversed_states_map[node]

                # Add the initialization states if any
                if bwd_node in self.init_states:
                    first_init_state = self.init_states[bwd_node]
                    decendant_nodes = self.backward_sdfg.bfs_nodes(first_init_state)

                    prev_node = None
                    # Iterate through the decendants and add them
                    for d_node in decendant_nodes:

                        # Remove node from the backward SDFG
                        self.backward_sdfg.remove_node(d_node)

                        # Add it to the loop region
                        reversed_loop.add_node(d_node)

                        if prev_node:
                            reversed_loop.add_edge(src=prev_node, dst=d_node, data=dace.InterstateEdge())

                        # Prepare for the next iteration
                        prev_node = d_node
                else:
                    # No initialization, we just add this state
                    # Remove from the backward SDFG
                    self.backward_sdfg.remove_node(bwd_node)

                    # Add it to the loop region
                    reversed_loop.add_node(bwd_node)

                # Also add loop states if any
                if node in self.reversed_loop_states_map:
                    # Get the backward_node
                    bwd_node = self.reversed_loop_states_map[node]

                    # Add the initialization states if any
                    if bwd_node in self.init_states:
                        first_init_state = self.init_states[bwd_node]
                        decendant_nodes = self.backward_sdfg.bfs_nodes(first_init_state)

                        # Iterate through the decendants and add them
                        prev_node = None
                        for d_node in decendant_nodes:

                            # Remove node from the backward SDFG
                            self.backward_sdfg.remove_node(d_node)

                            # Add it to the loop region
                            reversed_loop.add_node(d_node)

                            if prev_node:
                                reversed_loop.add_edge(src=prev_node, dst=d_node, data=dace.InterstateEdge())

                            # Prepare for the next iteration
                            prev_node = d_node

    def _add_missing_nested_sdfg_connectors_to_view(self, state: SDFGState, state_subgraph: dstate.StateSubgraphView,
                                                    view_nodes: List[nodes.Node]):
        """
        """
        # There is a special case for NestedSDFGs that we need to fix
        # in the case where a NestedSDFG has an inout connector,
        # but we only care about one of those connectors for the sake of AD
        # we need to add the missing connector for correctness
        # TODO: this is only a problem if the said connector is written to
        #       inside the NestedSDFG
        # Iterate over the nested SDFGs in the view
        for g in state_subgraph.nodes():
            if isinstance(g, nodes.NestedSDFG):

                inout_connoctors = set(g.in_connectors).intersection(set(g.out_connectors))
                # If there are any inout connectors
                if len(inout_connoctors) > 0:
                    out_connectors = {edge.src_conn: edge for edge in state.out_edges(g)}
                    in_connectors = {edge.dst_conn: edge for edge in state.in_edges(g)}
                    view_out_connectors = {edge.src_conn: edge for edge in state_subgraph.out_edges(g)}
                    view_in_connectors = {edge.dst_conn: edge for edge in state_subgraph.in_edges(g)}
                    for con in inout_connoctors:
                        # Check if it is missing in the out or in connectors of the view
                        if con in view_out_connectors and con not in view_in_connectors:
                            # Get the equivelent in node and connector
                            edge = in_connectors[con]
                            assert isinstance(edge.src, nodes.AccessNode)
                            view_nodes.append(edge.src)
                        if con not in view_out_connectors and con in view_in_connectors:
                            # Add the corresponding edge to the view
                            edge = out_connectors[con]
                            assert isinstance(edge.dst, nodes.AccessNode)
                            view_nodes.append(edge.dst)

        return dstate.StateSubgraphView(state, view_nodes)

    def _compare_memlet_accesses_to_array_size(self, data_name: str, memlet: Memlet) -> int:
        """
        Compare the memlet range with the size of the array to see if the array is being overwritten.
        """

        total_size = self.backward_sdfg.arrays[data_name].total_size
        try:
            if total_size > memlet.num_accesses:
                return 1
            elif memlet.num_accesses == total_size:
                return 0

            # Something is wrong here raise an exception
            raise AutoDiffException(f"Memlet {memlet} has more accesses than the size of the data {data_name}")

        # If the comparison can not be made, return None
        except TypeError as e:
            return None

    def _get_reversed_parent_graph(self, forward_node: nodes.Node):
        """
        Given a node in the SDFG, get the reversed parent of this node.
        """
        fwd_parent_graph = forward_node.parent_graph

        if fwd_parent_graph == self.sdfg:
            parent_graph = self.backward_sdfg
        elif isinstance(fwd_parent_graph, SDFGState):
            parent_graph = self.reversed_states_map[fwd_parent_graph]
        elif isinstance(fwd_parent_graph, LoopRegion):
            parent_graph = self.reversed_loops_map[fwd_parent_graph]

        return parent_graph

    def _get_bcakward_loop_state_edge(self, forward_edge: dace.InterstateEdge) -> dace.InterstateEdge:
        """
        Given an edge from the forward pass, return the equivelent edge in the backward pass
        """
        # Get the source and destination states
        forward_state_src = forward_edge.src
        forward_state_dst = forward_edge.dst

        # Get the equivelent states in the backward pass
        assert forward_state_src in self.reversed_loop_states_map
        assert forward_state_dst in self.reversed_loop_states_map

        # Note that the source will become the destination
        backward_state_dst = self.reversed_loop_states_map[forward_state_src]
        backward_state_src = self.reversed_loop_states_map[forward_state_dst]

        # Each one of these in edges needs to have an equivelent
        # out edge in the backward part of the SDFG
        bwd_edge = None
        connection_state = backward_state_dst

        # Check if this state has a preceeding initialization state
        if backward_state_dst in self.init_states:
            # In this case the edge will be to this initialization state
            connection_state = self.init_states[backward_state_dst]

        # Find the equivelent edge in the backward SDFG
        for b_edge in self.backward_sdfg.in_edges(connection_state):
            if b_edge.src == backward_state_src:
                bwd_edge = b_edge
                break

        if not bwd_edge:
            raise AutoDiffException(f"Can't find the equivelent edge of {forward_edge} in the backward pass")

        return bwd_edge

    def _get_bcakward_state_edge(self, forward_edge: dace.InterstateEdge) -> dace.InterstateEdge:
        """
        Given an edge from the forward pass, return the equivelent edge in the backward pass
        """
        # Get the source and destination states
        forward_state_src = forward_edge.src
        forward_state_dst = forward_edge.dst

        # Get the equivelent states in the backward pass
        assert forward_state_src in self.reversed_states_map
        assert forward_state_dst in self.reversed_states_map

        # Note that the src will become the destination
        backward_state_dst = self.reversed_states_map[forward_state_src]
        backward_state_src = self.reversed_states_map[forward_state_dst]

        # Each one of these in edges needs to have an equivelent
        # out edge in the backward part of the SDFG
        bwd_edge = None
        connection_state = backward_state_dst

        # Check if this state has a preceeding initialization state
        if backward_state_dst in self.init_states:
            # In this case the edge will be to this initialization state
            connection_state = self.init_states[backward_state_dst]

        # Find the equivelent edge in the backward SDFG
        for b_edge in connection_state.parent_graph.in_edges(connection_state):
            if b_edge.src == backward_state_src:
                bwd_edge = b_edge
                break

        if not bwd_edge:
            raise AutoDiffException(f"Can't find the equivelent edge of {forward_edge} in the backward pass")

        return bwd_edge

    def _str_to_access(self, data: str, source: str) -> nodes.AccessNode:
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
        matches = [(node, state) for state in self.sdfg.states() for node in state.nodes()
                   if isinstance(node, nodes.AccessNode) and node.data == data]
        # Unused in model
        if len(matches) == 0:
            return None

        # there is only a single AccessNode with this name
        if len(matches) == 1:
            return matches[0][0]

        # len(matches) > 1
        else:
            # There are multiple occurances of the same AccessNode
            if source == "inputs":
                # We return the first node with this data
                input_node: nodes.AccessNode = matches[0][0]
                input_node_state: SDFGState = matches[0][1]

                # There should not be any incoming edges for this node since it is the first one
                in_edges = input_node_state.in_edges(input_node)
                assert len(in_edges) == 0

                return input_node

            if source == "outputs":
                # Go through the list of matches in reverse
                for output_node, output_node_state in reversed(matches):
                    # We want the first node that has at least one incoming edge to it
                    # This represents the last time the output data was modified
                    in_edges = output_node_state.in_edges(output_node)
                    if len(in_edges) > 0:
                        return output_node

                raise AutoDiffException(
                    f"The specified output {data} was not written to by any AccessNode in this state")

            raise AutoDiffException(f"There are multiple nodes with data {data} "
                                    f" but the source (inputs or outputs) was not specified correctly")

    def _expand_nodes(self) -> bool:
        """ 
        Expand all library nodes in the sdfg to pure implementations. Returns whether something was expanded
        """
        expanded_something = False
        for node, parent_graph in self.sdfg.all_nodes_recursive():
            if isinstance(parent_graph, dstate.StateSubgraphView):
                parent_graph = parent_graph.graph

            # Check if the node exists in the backward implementation repository
            if find_backward_implementation(parent_graph.parent_graph, parent_graph, node) is not None:
                continue

            # Only check others if we didn't break out of the above loop
            if isinstance(node, ONNXOp):
                impls = ONNXForward.registered_implementations(node.schema.name)

                # Order the implementations so that implementations containing "pure" are tried first
                impls = [i for name, i in impls if "pure" in name] + [i for name, i in impls if "pure" not in name]
                for impl in impls:
                    if impl.forward_can_be_applied(node, parent_graph, self.sdfg):
                        # Try to apply the expansion
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
                        Expansion.apply_to(parent_graph.parent, verify=False, _match_node=node)
                        expanded_something = True
                        break

            # This could later on be changed to check if the expansion is differentiable and if not, move
            # on to the next expansion. For now we will just apply the first one that matches, prioritizing ones that
            # have "pure" in the name
            if isinstance(node, nodes.LibraryNode) and not isinstance(node, ONNXOp):
                # Try to select an expansion
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
                node.expand(parent_graph.parent, parent_graph)
                expanded_something = True

        return expanded_something

    def _disambiguate_direction_dependent_views(self, state: SDFGState):
        """ 
        Consider the following subgraph:
        (A) -- y --> (B) -- x --> (C)
        In dace, if B is a View node and A and C are access nodes, and y and x both have data set to A.data and
        B.data respectively, the semantics of the graph depend on the order in which it is executed, i.e. reversing
        the subgraph doesn't perform as expected anymore. To disambiguate this case, we set y.data to the View's
        data.
        :param state: the state to disambiguate views in
        """
        for n in state.nodes():
            if isinstance(n, nodes.AccessNode) and type(n.desc(self.sdfg)) is dt.View:
                in_edges = state.in_edges(n)
                out_edges = state.out_edges(n)

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
                        y.try_initialize(self.sdfg, state, in_edges[0])

    def _get_node_state(self, node: nodes.Node) -> SDFGState:
        """
        Return the SDFG state that contains this node
        """
        matches = []
        for state in self.sdfg.states():
            if node in state.nodes():
                matches.append(state)

        assert len(matches) == 1
        return matches[0]

    def check_edges_type_in_state(self, subgraph: dstate.StateSubgraphView):
        """
        Check if all the edges in this state are of type float, int, or boolean.
        """
        for edge, parent_subgraph in subgraph.all_edges_recursive():
            if isinstance(parent_subgraph, SDFGState):
                parent_sdfg = parent_subgraph.parent
            elif isinstance(parent_subgraph, dstate.StateSubgraphView):
                parent_sdfg = parent_subgraph.graph.parent
            elif isinstance(parent_subgraph, SDFG) or isinstance(parent_subgraph, LoopRegion):
                # if there are any fancy things on the interstate edges we should probably throw an error
                continue
            else:
                raise AutoDiffException("Unexpected subgraph structure")

            if edge.data.data:
                edge_type = parent_sdfg.arrays[edge.data.data].dtype
                if edge_type in [dace.string]:
                    raise AutoDiffException(
                        f"Expected Subgraph to differentiate to only contain float, int, and bool edges, but data {edge.data}"
                        f" on edge {edge} has type {edge_type}")

    def _connect_conditional_map_exist(self, forward_state: SDFGState, backward_state: SDFGState,
                                       backward_map_exit: nodes.MapExit, fwd_tasklet: nodes.Tasklet) -> SDFGState:
        """
        """

        assert len(backward_map_exit.in_connectors) == 0

        # Add the in and out connectors for the zero-out operation
        assert backward_map_exit.add_in_connector("IN_zero_out")
        assert backward_map_exit.add_out_connector("OUT_zero_out")

        # Get the memlet data for the edge from the tasklet to the map exist
        tasklet_out_edge = forward_state.out_edges(fwd_tasklet)
        assert len(tasklet_out_edge) == 1
        tasklet_out_edge = tasklet_out_edge[0]
        tasklet_memlet_path = forward_state.memlet_path(tasklet_out_edge)
        assert len(tasklet_memlet_path) == 2

        # Copy the memlet and change the data name
        memlet_data = copy.deepcopy(tasklet_memlet_path[0].data)
        memlet_data.data = self.array_grad_map[memlet_data.data]

        # Get the reversed tasklet
        bwd_tasklet = self.reverse_map[fwd_tasklet]

        # Connect this map exist to the tasklet
        backward_state.add_edge(bwd_tasklet, "__zero_out_conn__", backward_map_exit, "IN_zero_out", memlet_data)

        # Replicate the target accedd node and connect it
        fwd_target_an: nodes.AccessNode = tasklet_memlet_path[-1].dst
        assert isinstance(fwd_target_an, nodes.AccessNode)
        assert fwd_target_an in self.reverse_map
        bwd_target_an = self.reverse_map[fwd_target_an]

        replicated_bwd_target_an = copy.deepcopy(bwd_target_an)
        backward_state.add_node(replicated_bwd_target_an)

        an_memlet_data: nodes.AccessNode = copy.deepcopy(tasklet_memlet_path[1].data)
        an_memlet_data.data = self.array_grad_map[an_memlet_data.data]
        backward_state.add_edge(backward_map_exit, "OUT_zero_out", replicated_bwd_target_an, None, an_memlet_data)

        new_backward_state = self.backward_sdfg.add_state_after(backward_state)
        return new_backward_state

    def _differentiate_code_symbolically(
        self,
        code_str: str,
        forward_state: SDFGState,
        tasklet: nodes.Tasklet,
        given_gradients: List[str],
        required_gradients: List[str],
    ):
        """
        """
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

            # special case for conditional tasklets with constant assignement
            if len(required_gradients) == 0:
                # for this we need to assing a zero to the gradient output
                # pick a name for the input gradient
                rev_input_grad_name = find_str_not_in_set(rev_inputs, output_conn + "_gradient")
                result.given_grad_names[output_conn] = rev_input_grad_name

                # zero out the gradient
                code = f"\n{rev_input_grad_name} = 0.0"
                rev_outputs = {}
                rev_inputs = {rev_input_grad_name}

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
                # if the expression is a constant assignement, we need to cast the float to the sympy equivalent
                if type(output_expr) in [np.float64, np.float32, np.float16]:
                    output_expr = sp.Float(output_expr)

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
                cands = list(forward_state.in_edges_by_connector(tasklet, inp))
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

        return code, rev_inputs, rev_outputs, result

    def _extract_conitional_expressions(self, tasklet_node: nodes.Tasklet):
        """
        Given a conditional tasklet node, extract the if and else expressions and return them with the conditional.
        The else statement could be None in case there is only an if statement. The current supported formats are the following:
        1 - if cond:
                out = expression_1
        which would return ("out = expression_1", None, "if cond")
        2- out = expression_1 if cond else expression 2
        """

        tasklet_code = tasklet_node.code.as_string

        # check which type of assignment this is
        if ":" in tasklet_code:
            # get the conditional input connector through regular expression matching
            matches = re.search(r"if (.)*:", tasklet_code)
            assert matches
            conditional = matches.group()

            # remove the conditional from the code to get the expression
            if_statement = tasklet_code.replace(conditional, "")
            if_statement = if_statement.replace("\n", "")

            # remove indentation
            if_statement = if_statement[3:]

            # extract the in connector only
            conditional = conditional.replace(":", "")
            conditional = conditional.replace("if ", "")
            assert conditional in tasklet_node.in_connectors

            else_statement = None

            # match the out connector
            matches = re.search(r"^(.)* =", if_statement)
            assert matches
            out_connector = matches.group()

            # remove the assingment from the if statement
            if_statement = if_statement.replace(out_connector, "")

            # extract the out connector only
            out_connector = out_connector[1:].replace(" =", "")

        else:
            # get the conditional input connector through regular expression matching
            matches = re.search(r"if (.)* else", tasklet_code)
            assert matches
            conditional = matches.group()

            # extract the in connector only
            conditional = conditional.replace("if ", "")
            conditional = conditional.replace(" else", "")

            assert conditional in tasklet_node.in_connectors

            # get the if statement by matching what comes before the if until we encounter a paranthesis or =
            matches = re.search(r"= \((.)* if", tasklet_code)
            if not matches:
                # try without the parenthesis
                matches = re.search(r"= (.)* if", tasklet_code)
                assert matches

            if_statement = matches.group()

            # extract the in statement only
            if_statement = if_statement.replace("= (", "")
            if_statement = if_statement.replace(" if", "")

            # get the else statement by matching the else and what comes after it until we encounter a paranthesis
            matches = re.search(r"else (.)*\)", tasklet_code)
            assert matches
            else_statement = matches.group()

            # extract the in statement only
            else_statement = else_statement.replace("else ", "")

            # remove the last closing parenthesis if it exists
            if else_statement.endswith(")"):
                else_statement = else_statement[:-1]

            # match the out connector
            matches = re.search(r"^(.)* =", tasklet_code)
            assert matches
            out_connector = matches.group()

            # extract the in statement only
            out_connector = out_connector.replace(" =", "")

        # sanity check this should be in the out connectors of the tasklet
        assert out_connector in tasklet_node.out_connectors

        # create the return expressions
        if_expression = f"{out_connector} = {if_statement}"
        else_expression = f"{out_connector} = {else_statement}" if else_statement else None

        return if_expression, else_expression, conditional

    def _conditional_tasklet(self, tasklet_node: nodes.Tasklet):
        """
        Checks if this tasklet contains a conditional. 
        This only happens in conditional array assignements and requires special treatement in reversing the graph.
        TODO: how to more accuratly check this?
        """
        # sanity check
        assert isinstance(tasklet_node, nodes.Tasklet)

        # get the code string and check if there is an if
        return "if" in tasklet_node.code.as_string

    def _conditional_nested_sdfg(self, forward_state: SDFGState, node: nodes.NestedSDFG):
        """
        Checks if this tasklet contains a conditional. 
        This only happens in conditional array assignements and requires special treatement in reversing the graph.
        """
        # sanity check
        assert isinstance(node, nodes.NestedSDFG)

        # get the incoming edges to the sdfg
        in_edges = forward_state.in_edges(node)

        # check if any of the incoming edges are boolean edges
        for edge in in_edges:
            if self.sdfg.arrays[edge.data.data].dtype == dace.bool:
                return True

        # get the code string and check if there is an if
        return False

    def _extract_conditional_array_assignement_block(self, forward_state: SDFGState, tasklet_node: nodes.Node,
                                                     subgraph: dstate.SubgraphView):
        """
        Given a conditional tasklet, check if this is a conditional array assignement of the type
        A[A>=0 and A<=5] = cst. At the moment the function only supports constant assignements. 
        """
        try:

            assert isinstance(tasklet_node, nodes.Tasklet) or isinstance(tasklet_node, nodes.NestedSDFG)
            # get the AccessNode containing the boolean values for this assignement
            tasklet_in_edges = forward_state.in_edges(tasklet_node)
            tasklet_boolan_edge = None
            single_boolean_edge_found = False
            for edge in tasklet_in_edges:
                edge_type = self.sdfg.arrays[edge.data.data].dtype
                if edge_type == dace.bool:
                    # sanitity check
                    if single_boolean_edge_found:
                        # we expect there to be a single AccessNode where the booleans come from
                        raise AutoDiffException()
                    tasklet_boolan_edge = edge
                    single_boolean_edge_found = True

            assert tasklet_boolan_edge
            tasklet_in_memlet_path = forward_state.memlet_path(tasklet_boolan_edge)
            # the first element in the path is the boolean AN
            bools_an = tasklet_in_memlet_path[0].src
            assert isinstance(bools_an, nodes.AccessNode)

            # save all the nodes in the path to the assignement block list
            conditional_assingement_block_nodes = {
                n
                for e in forward_state.bfs_edges(bools_an, reverse=True) for n in [e.src, e.dst]
            }

            # if any of the nodes in the block are required for gradient tracking
            nodes_to_keep_tracking: set[nodes.Node] = self._get_gradient_nodes_to_track(
                forward_state=forward_state, block_nodes=conditional_assingement_block_nodes, subgraph=subgraph)
            for node in nodes_to_keep_tracking:
                # we get the reverse bfs of this node and remove it from block nodes to avoid skipping these nodes
                node_subgraph = {n for e in forward_state.bfs_edges(node, reverse=True) for n in [e.src, e.dst]}

                # add the node itself
                node_subgraph.add(node)
                conditional_assingement_block_nodes = conditional_assingement_block_nodes.difference(node_subgraph)

        except Exception as e:
            # if this is not the structure we are expecting, fail
            raise AutoDiffException(f"The boolean datatype in edges is limited to conditional array assingements."
                                    f" This stucture is not supported.") from e

        return conditional_assingement_block_nodes

    def _get_gradient_nodes_to_track(self, forward_state: SDFGState, block_nodes: List[nodes.Node],
                                     subgraph: dstate.SubgraphView):
        """
        When extracting the block for a conditional assingement, we need to make sure we keep tracking
        the required gradient accessnodes. 
        This function checks all the required access nodes that are in the conditional block.
        At the moment this is just the target access node. 
        TODO: extend this to check for all the required gradient access nodes 
        """
        nodes_to_track: List[nodes.AccessNode] = []
        # TODO: get all the nodes used below the target accessnode, this would have to extend to multiple states too
        # at the moment we know that the target access node itself should be tracked
        gradinet_nodes = [n.data for n in self.required_gradients]
        gradinet_nodes += [n.data for n in self.given_gradients]

        # get the subgraph difference
        difference = set(subgraph.nodes()).difference(set(block_nodes))

        # go through all the access nodes in the conditional block
        for node in block_nodes:
            if not isinstance(node, nodes.AccessNode):
                continue

            # we always want to track the gradient nodes
            if node.data in gradinet_nodes:
                nodes_to_track.append(node)
                continue
            # if this access node has multiple edges and any of them are outside the block

            node_out_edges = forward_state.out_edges(node)
            if len(node_out_edges) > 1:
                for edge in node_out_edges:
                    if edge.dst in difference:
                        nodes_to_track.append(node)
            data = node.data

            # search for this array in the graph difference
            for d_node in difference:
                if not isinstance(d_node, nodes.AccessNode):
                    continue
                if d_node.data == data:
                    nodes_to_track.append(node)
        return nodes_to_track

    def _get_state_given_gradients(self, forward_state: SDFGState):
        """
        Given
        """
        given_gradients_all_states = {n.data for n in self.given_gradients}
        # go through the states in reverse topologicla order
        for state in reversed(self.state_order):
            if state == forward_state:
                return given_gradients_all_states
            backward_nodes: set[nodes.Node] = set()

            state_given_gradients = []
            # get all the access nodes that are neccessary for AD and are present in this state
            for node in state:
                if isinstance(node, nodes.AccessNode) and node.data in given_gradients_all_states:
                    state_given_gradients.append(node)

            backward_nodes = backward_nodes.union(
                {n
                 for e in state.bfs_edges(state_given_gradients, reverse=True) for n in [e.src, e.dst]})

            # update the list of required gradients to use for states
            for node in backward_nodes:
                if isinstance(node, nodes.AccessNode) and node.data not in given_gradients_all_states:
                    # we want all of the forward AccessNodes that made it to the intersection
                    given_gradients_all_states.add(node.data)

        raise AutoDiffException(f"Parameter {forward_state} is not in the state order")

    def _get_state_required_gradients(self, forward_state: SDFGState):
        """
        Given
        """
        required_gradients_all_states = {n.data for n in self.required_gradients}
        # do the forward bfs iterativly
        for state in self.state_order:
            if state == forward_state:
                return required_gradients_all_states
            forward_nodes: set[nodes.Node] = set()
            state_required_gradients = []
            # get all the access nodes that are neccessary for AD and are present in this state
            for node in state:
                if isinstance(node, nodes.AccessNode) and node.data in required_gradients_all_states:
                    state_required_gradients.append(node)

            forward_nodes = forward_nodes.union(
                {n
                 for e in state.bfs_edges(state_required_gradients) for n in [e.src, e.dst]})

            # update the list of required gradients to use for states
            for node in forward_nodes:
                if isinstance(node, nodes.AccessNode) and node.data not in required_gradients_all_states:
                    # we want all of the forward AccessNodes that made it to the intersection
                    required_gradients_all_states.add(node.data)

        raise AutoDiffException(f"Parameter {forward_state} is not in the state order")

    def _init_grad(self, forward_state: SDFGState, backward_state: SDFGState, data: str):
        desc = self.backward_sdfg.arrays[data]
        # No need to initialize if gradients point to outputs
        if not self.zero_non_transients and not desc.transient:
            return

        # No need to initialize if this is a view
        if isinstance(self.backward_sdfg.arrays[data], dt.ArrayView):
            return

        state = init_grad(data, self.backward_sdfg, backward_state)

        # we only want the init_states to contain the first initial state
        # since this information is used for create edges only
        if backward_state not in self.init_states:
            self.init_states[backward_state] = state

    def _reverse_subgraph(self, forward_state: SDFGState, backward_state: SDFGState,
                          subgraph: dstate.StateSubgraphView):
        """ Reverse a given subgraph. All nodes in the subgraph will be reversed. """
        from dace.libraries.onnx.nodes.onnx_op import ONNXSum  # avoid import loop

        # condiitonal assignement nodes
        conditional_assignement_nodes: List[nodes.Node] = []
        new_backward_state = None

        # a reversed topological sort is a topological sort on the reverse graph
        for node in reversed(list(dutils.dfs_topological_sort(subgraph, subgraph.source_nodes()))):

            try:
                # if this node is a part of the conditional assignement block, we skip it
                if node in conditional_assignement_nodes:
                    continue

                if new_backward_state is not None:
                    backward_state = new_backward_state

                # output names on the forward node
                # (for which the gradient will be connected as an input on the reverse node)
                given_gradients = [
                    edge.src_conn for edge in subgraph.out_edges(node) if _path_src_node_in_subgraph(edge, subgraph)
                ]

                # input names on the forward node that gradients should be generated for
                # note that the edge for the conditional is not included
                required_gradients = [
                    edge.dst_conn for edge in subgraph.in_edges(node) if _path_src_node_in_subgraph(edge, subgraph)
                    and self.sdfg.arrays[edge.data.data].dtype != dace.bool
                ]

                reversed_node, backward_result = self._get_reverse_node(forward_state, backward_state, node,
                                                                        given_gradients, required_gradients)

                self.reverse_map[node] = reversed_node
                self.result_map[node] = backward_result

                # connect the required inputs of the reverse node:
                # the gradients ...
                new_backward_state = self._connect_given_gradients(forward_state=forward_state,
                                                                   backward_state=backward_state,
                                                                   subgraph=subgraph,
                                                                   forward_node=node)

                # in some cases (if the an access node points to a view)
                # there will be no edges added and this will be an isolated node
                if isinstance(reversed_node, nodes.AccessNode) and backward_state.in_degree(
                        reversed_node) == 0 and backward_state.out_degree(reversed_node) == 0:
                    backward_state.remove_node(reversed_node)
                    continue

                # ... and any required input values from the forward pass
                ####################################
                # Determine which forward inputs we need to connect.
                # these are the in_connectors on the reverse node, minus what has already been connected.
                already_connected = {e.dst_conn for e in backward_state.in_edges(reversed_node)}
                required_inputs = set(reversed_node.in_connectors).difference(already_connected)
                required_inputs = {c: c for c in required_inputs}
                self._connect_forward_inputs(forward_state, backward_state, node, reversed_node, required_inputs)

                if isinstance(node, nodes.AccessNode):

                    # this means we are writing out a grad to an array.
                    # initialize the gradient if it hasn't been initialized already (this can also happen in
                    # _connect_given_gradients
                    array_grad_name = self.array_grad_name(node.data)
                    if array_grad_name not in self.backward_sdfg.arrays:
                        # this grad hasn't been written before: initialize it
                        self._add_gradient_data_descriptor(node.data)

                    # we need to make all incoming gradients sum
                    if backward_state.in_degree(reversed_node) > 1:
                        summation_node = ONNXSum(f"sum_{array_grad_name}")

                        grad_desc = self.backward_sdfg.arrays[array_grad_name]
                        cuda = False
                        # connect each incoming edge to the summation node
                        for i, edge in enumerate(backward_state.in_edges(reversed_node)):

                            intermediate_desc = copy.deepcopy(grad_desc)

                            intermediate_desc.transient = True
                            intermediate_name = self.backward_sdfg.add_datadesc(f"{array_grad_name}_contribution_{i}",
                                                                                intermediate_desc,
                                                                                find_new_name=True)
                            access_intermediate = backward_state.add_access(intermediate_name)

                            for mte in backward_state.memlet_tree(edge):
                                # # whatever edges were writing to the reversed node
                                if mte.data.data == array_grad_name:
                                    # will now need to write to the intermediate node
                                    mte.data.data = intermediate_name

                            new_edge = backward_state.add_edge(edge.src, edge.src_conn, access_intermediate, None,
                                                               edge.data)

                            self._set_wcr_sum_if_needed(forward_state, backward_state, new_edge)
                            summation_node.add_in_connector(f"data_0__{i}")
                            backward_state.add_edge(access_intermediate, None, summation_node, f"data_0__{i}",
                                                    self.backward_sdfg.make_array_memlet(intermediate_name))
                            backward_state.remove_edge(edge)

                        # add the final connection to the summation node
                        self._connect_summation_node(forward_state=forward_state,
                                                     backward_state=backward_state,
                                                     summation_node=summation_node,
                                                     forward_node=node,
                                                     reversed_node=reversed_node,
                                                     array_grad_name=array_grad_name)

                        if dtypes.can_access(dtypes.ScheduleType.CPU_Multicore, grad_desc.storage):
                            pass
                        elif dtypes.can_access(dtypes.ScheduleType.GPU_Default, grad_desc.storage):
                            summation_node.schedule = dtypes.ScheduleType.GPU_Default
                        else:
                            raise ValueError(f"Unsupported storage {grad_desc.storage}")
                    elif backward_state.in_degree(reversed_node) == 1:
                        self._set_wcr_sum_if_needed(forward_state, backward_state,
                                                    backward_state.in_edges(reversed_node)[0])

                # if this node is a tasklet with a condition, we add some modification to the backward state
                elif (isinstance(node, nodes.Tasklet)
                      and self._conditional_tasklet(node)) or (isinstance(node, nodes.NestedSDFG)
                                                               and self._conditional_nested_sdfg(forward_state, node)):
                    # extract the conditional assignement block or fail if this is an unexpected structure
                    conditional_block = self._extract_conditional_array_assignement_block(forward_state=forward_state,
                                                                                          tasklet_node=node,
                                                                                          subgraph=subgraph)

                    # add these nodes to be skipped in the future
                    conditional_assignement_nodes.extend(conditional_block)

            except AutoDiffException as e:
                raise AutoDiffException("Failed at node {}: {}".format(node, str(e))) from e

    def _connect_summation_node(self, forward_state: SDFGState, backward_state: SDFGState,
                                forward_node: nodes.AccessNode, summation_node: nodes.CodeBlock,
                                reversed_node: nodes.AccessNode, array_grad_name: str):
        """
        """
        read_write_intersection = self._check_write_read_ranges(forward_state=forward_state, forward_node=forward_node)
        # check if the summation result can be directly assigned to the reversed node
        if not read_write_intersection:
            backward_state.add_edge(summation_node, "sum", reversed_node, None,
                                    self.backward_sdfg.make_array_memlet(array_grad_name))
        else:
            # we need to create an assignement for each peace of data
            # first create an intermediate array
            grad_desc = self.backward_sdfg.arrays[array_grad_name]
            intermediate_desc = copy.deepcopy(grad_desc)

            intermediate_desc.transient = True
            intermediate_name = self.backward_sdfg.add_datadesc(f"__{array_grad_name}_wcr_intermediate__",
                                                                intermediate_desc,
                                                                find_new_name=True)
            access_intermediate = backward_state.add_access(intermediate_name)
            # connect it to the summation node
            backward_state.add_edge(summation_node, "sum", access_intermediate, None,
                                    self.backward_sdfg.make_array_memlet(intermediate_name))

            # replicate the access node and add it to the SDFG
            wcr_replicated_node = copy.deepcopy(reversed_node)
            backward_state.add_node(wcr_replicated_node)

            for memlet_subset, memlet_wcr in read_write_intersection:
                # get the memlet data
                memlet_data = self.backward_sdfg.make_array_memlet(intermediate_name)
                memlet_data.subset = memlet_subset
                memlet_data.dst_subset = memlet_subset
                if memlet_wcr:
                    memlet_data.wcr = memlet_wcr

                    # connect it to the summation node
                    backward_state.add_edge(
                        access_intermediate,
                        None,
                        wcr_replicated_node,
                        None,
                        memlet_data,
                    )
                else:
                    # connect it to the summation node without wcr
                    backward_state.add_edge(
                        access_intermediate,
                        None,
                        reversed_node,
                        None,
                        memlet_data,
                    )

    def _check_write_read_ranges(self, forward_state: SDFGState, forward_node):
        """
        Check if this stencil computation only writes to a subset of its array in a loop
        """
        return None
        read_edge = None
        read_range: List[List[Tuple[int, int, int]]] = []
        write_edge = None
        write_range: List[List[Tuple[int, int, int]]] = []

        # get read/write range of this data in the state
        for node in forward_state.nodes():
            if isinstance(
                    node,
                    nodes.AccessNode) and node.data == forward_node.data and len(forward_state.out_edges(node)) > 0:
                # get the read memlet
                read_edges = forward_state.out_edges(node)
                for read_edge in read_edges:
                    read_memlet = read_edge.data.subset
                    dim_ranges = read_memlet.ndrange()
                    for i, dim in enumerate(dim_ranges):
                        range_start, range_end, step = dim
                        if not read_range:
                            read_range.append([range_start, range_end, step])
                        else:
                            if i >= len(read_range):
                                read_range.append([range_start, range_end, step])
                                continue
                            if range_start < read_range[i][0]:
                                read_range[i][0] = range_start
                            if range_end > read_range[i][1]:
                                read_range[i][1] = range_end

            if isinstance(
                    node,
                    nodes.AccessNode) and node.data == forward_node.data and len(forward_state.in_edges(node)) > 0:

                # get the write memlet
                write_edges = forward_state.in_edges(node)
                for write_edge in write_edges:
                    write_memlet = write_edge.data.subset
                    dim_ranges = write_memlet.ndrange()
                    for i, dim in enumerate(dim_ranges):
                        range_start, range_end, step = dim
                        if not write_range:
                            write_range.append([range_start, range_end, step])
                        else:
                            if i >= len(write_range):
                                write_range.append([range_start, range_end, step])
                                continue
                            if range_start < write_range[i][0]:
                                write_range[i][0] = range_start
                            if range_end > write_range[i][1]:
                                write_range[i][1] = range_end

        if not read_edge or not write_edge:
            # this data is either read or write only
            return None

        ranges_list: List[List[Tuple[Tuple, str]]] = []
        intersection_ranges_list: List[List[Tuple[Tuple, str]]] = []

        # dimensions should match in reads and writes
        assert len(write_range) == len(read_range)
        edges_ranges: List[List[Tuple[subsets.Range, str]]] = []
        intersection_ranges: List[List[Tuple[subsets.Range, str]]] = []
        difference = False
        for dim in range(len(write_range)):
            dim_ranges_list = []
            # get a list of indicies for this assignement
            range_start, range_end, step = write_range[dim]
            write_indicies = []
            for i in range(range_start, range_end + 1, step):
                write_indicies.append(i)

            range_start, range_end, step = read_range[dim]
            read_indicies = []
            for i in range(range_start, range_end + 1, step):
                read_indicies.append(i)

            # get the difference in a sorted list
            sorted_difference = sorted(list(set(read_indicies).difference(set(write_indicies))))
            wcr = "lambda x, y: x + y"
            if len(sorted_difference) != 0:
                difference = True
            else:
                continue

            i_start = sorted_difference[0]
            for index, i in enumerate(sorted_difference):
                if index == len(sorted_difference) - 1:
                    dim_ranges_list.append(((i_start, i, step), wcr))
                    break

                if sorted_difference[index + 1] != sorted_difference[index] + step:
                    # end the list here and add it to the ranges list
                    dim_ranges_list.append(((i_start, i, step), wcr))
                    # set i_start for the next range
                    i_start = sorted_difference[index + 1]

            # get the intersection in a sorted list
            sorted_intersection = sorted(list(set(write_indicies).intersection(set(read_indicies))))
            assert len(sorted_intersection) > 0

            intersection_ranges.append(((sorted_intersection[0], sorted_intersection[-1], step), None))
            ranges_list.append(dim_ranges_list)
            intersection_ranges_list.append(intersection_ranges)

        # add the intersecting edges
        for edge_index in range(len(intersection_ranges_list[0])):
            # build the memlet dimensions
            memlet_subset = []
            for dim in range(len(intersection_ranges_list)):
                memlet_subset.append(intersection_ranges_list[dim][edge_index][0])
                memlet_wcr = intersection_ranges_list[dim][edge_index][1]

            edges_ranges.append((subsets.Range(memlet_subset), memlet_wcr))

        # for each range in the non-overlapping ranges
        for edge_index in range(len(ranges_list[0])):
            # build the memlet dimensions
            memlet_subset = []
            for dim in range(len(ranges_list)):
                memlet_subset.append(ranges_list[dim][edge_index][0])
                memlet_wcr = ranges_list[dim][edge_index][1]

            edges_ranges.append((subsets.Range(memlet_subset), memlet_wcr))
        # if all the write and read ranges match
        if not difference:
            return None
        return edges_ranges

    def _get_backward_state(self, state: SDFGState) -> SDFGState:
        """
        Given a state in the forward SDFG, return the equivelent state in the backward SDFG.
        """
        # make sure this state is in this SDFG
        assert state in self.sdfg.states()
        # make sure we already initiated the reversal for this node
        assert state in self.reversed_states_map
        # get the backward state for this node
        return self.reversed_states_map[state]

    def _set_wcr_sum_if_needed(self, forward_state: SDFGState, backward_state: SDFGState,
                               edge: dgraph.MultiConnectorEdge):
        """ Set the WCR to sum for all edges along the path of edge, if needed.
            This function will also add gradient initialization to zero if necessary.
            The initialization is necessary if there will be a wcr or if the write will be to only a subset of the array.
            :param edge: the root edge to start from
        """
        inverse_array_grad_map = {v: k for k, v in self.array_grad_map.items()}

        add_wcr = False

        # this method assumes that the memlet tree is iterated from the root backwards
        for path_edge in backward_state.memlet_tree(edge):
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
            if is_write_conflicted_with_reason(backward_state, path_edge):
                # if we have a write conflict, we need WCR
                add_wcr = True
            path_edge.data.wcr = old_wcr

            # count the amount of in edges per connector
            connector_in_edges = collections.defaultdict(int)
            for _, _, _, dst_conn, _ in backward_state.in_edges(path_edge.dst):
                connector_in_edges[dst_conn] += 1

            more_than_one_edge_to_connector = any(v > 1 for v in connector_in_edges.values())

            if more_than_one_edge_to_connector:
                add_wcr = True

        # check if any of the inputs from this accessnode will be used to contribute in a wcr way
        if not add_wcr:
            add_wcr = self._input_used_with_a_wcr(forward_state=forward_state, backward_node=edge.dst)

        if add_wcr:
            for tree_edge in backward_state.memlet_tree(edge):
                tree_edge.data.wcr = "lambda x, y: x + y"
            self._init_grad(forward_state=forward_state, backward_state=backward_state, data=edge.data.data)

        # if the initialization has not already been added
        # if not add_wcr:
        #     # check if the write to the gradient contribution is to a subset of of the array
        #     contribution_accessnode: nodes.AccessNode = edge.dst
        #     assert isinstance(contribution_accessnode, nodes.AccessNode)

        #     # get the memlet range
        #     write_memlet = edge.data

        #     # compare the memlet range with the size of the array
        #     comparison_result = self._compare_memlet_accesses_to_array_size(contribution_accessnode.data, write_memlet)
        #     if comparison_result is None or comparison_result > 0:
        #         #TODO this should only be done if this is the first write to the gradient
        #         self._init_grad(forward_state=forward_state, backward_state=backward_state, data=edge.data.data)

    def _input_used_with_a_wcr(self, forward_state: SDFGState, backward_node: nodes.AccessNode):
        """
        Check if this node is used in the forward pass to write to using a wcr
        in this case, a wcr needs to be added to the gradients as well
        """
        forward_node = None
        # get the accessnode of the forward_pass
        for fwd_node, bwd_node in self.reverse_map.items():
            if bwd_node == backward_node:
                forward_node = fwd_node

        # TODO: in the case the forward node is not found
        # is this because of the first/last iteration exception
        if not forward_node:
            return False

        given_gradients_all_states = self._get_state_given_gradients(forward_state=forward_state)
        required_gradients_all_states = self._get_state_required_gradients(forward_state=forward_state)

        state_given_gradients: List[nodes.AccessNode] = []
        state_required_gradients: List[nodes.AccessNode] = []
        # add all of the write occurances of these nodes into the list
        for state_node in forward_state:
            if isinstance(state_node, nodes.AccessNode) and state_node.data in required_gradients_all_states:
                state_required_gradients.append(state_node)

        if forward_node not in state_required_gradients:
            return False

        state_given_gradients: List[nodes.AccessNode] = []
        # add all of the write occurances of these nodes into the list
        for state_node in forward_state:
            if isinstance(state_node, nodes.AccessNode) and state_node.data in given_gradients_all_states:
                state_given_gradients.append(state_node)

        # do reverse bfs starting from this new set of nodes
        backward_nodes = {
            n
            for e in forward_state.bfs_edges(state_given_gradients, reverse=True) for n in [e.src, e.dst]
        }

        forward_nodes = {n for e in forward_state.bfs_edges(forward_node) for n in [e.src, e.dst]}

        intersection = backward_nodes.intersection(forward_nodes)

        for node in intersection:
            if node in state_given_gradients:
                # check the writes to this node
                in_edges = forward_state.in_edges(node)
                for edge in in_edges:
                    edge_path = forward_state.memlet_path(edge)
                    for memlet in edge_path:
                        if memlet.data.wcr is not None:
                            return True

        return False

    def _connect_given_gradients(self, forward_state: SDFGState, backward_state: SDFGState,
                                 subgraph: dstate.StateSubgraphView, forward_node: nodes.Node) -> SDFGState:
        """ 
        Connect the gradients of the outputs of forward_node as inputs to the corresponding reverse node. 
        """
        new_backward_state = None
        for edge in subgraph.out_edges(forward_node):
            if not _path_src_node_in_subgraph(edge, subgraph) or edge.dst not in self.reverse_map:
                # skip connecting edges for which we don't need to generate grads.
                continue

            # skip connecting boolean edges
            if self.sdfg.arrays[edge.data.data].dtype == dace.bool:
                # we also need to remove this connector otherwise it will be dangeling
                backward_node = self.reverse_map[edge.src]
                conn_to_remove = _invert_map_connector(edge.src_conn)
                assert conn_to_remove in backward_node.in_connectors
                assert backward_node.remove_in_connector(conn_to_remove)
                if len(backward_node.in_connectors) == 0:
                    new_backward_state = self._connect_conditional_map_exist(forward_state=forward_state,
                                                                             backward_state=backward_state,
                                                                             backward_map_exit=backward_node,
                                                                             fwd_tasklet=edge.dst)
                continue

            src_node, output_conn, dest_node, input_conn, memlet = edge

            memlet = copy.deepcopy(memlet)

            # remove the WCR since these are now read edges
            memlet.wcr = None

            grad_name = self.array_grad_name(memlet.data)
            if grad_name not in self.backward_sdfg.arrays:
                # this grad hasn't been written before: initialize it
                self._add_gradient_data_descriptor(memlet.data)
            memlet.data = grad_name

            backward_state.add_edge(
                self.reverse_map[dest_node],
                self._lookup_required_grad_name(dest_node, input_conn),
                self.reverse_map[forward_node],
                self._lookup_given_grad_name(forward_node, output_conn),
                memlet,
            )
        return new_backward_state

    def _connect_forward_inputs(self, state: SDFGState, backward_state: SDFGState, forward_node: nodes.Node,
                                backward_node: nodes.Node, required_inputs: Dict[str, str]):
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
        input_edges_to_connect = (edge for edge in state.in_edges(forward_node) if edge.dst_conn in required_inputs)

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
                replicated_edge_src = self._find_backward_entry_node_for_map_entry(forward_state=state,
                                                                                   backward_state=backward_state,
                                                                                   entry_node=edge_src)
                new_in_conn, new_out_conn = _add_through_connector(replicated_edge_src)

                replicated_edge_src_conn = new_out_conn

                # the inverse connector of edge.src_conn should be connected
                # to the new through connector we made
                next_required_inputs = {_invert_map_connector(edge.src_conn): new_in_conn}

            elif isinstance(edge_src, nodes.AccessNode):
                is_base_level = state.scope_dict()[edge_src] is None
                data_name = edge_src.data
                data_desc = copy.deepcopy(edge_src.desc(self.sdfg))
                if self.separate_sdfgs:
                    # need to copy over the descriptor from the forward pass
                    if data_name not in self.backward_sdfg.arrays:
                        self.backward_sdfg.add_datadesc(data_name, data_desc)

                if not is_base_level:
                    # this is a temporary value that needs to be stored
                    # replicate the node now since we need to connect it to the new AccessNode
                    replicate_node = False
                    replicated_edge_src_conn = edge.src_conn

                    # node has not been replicated yet: do it now
                    replicated_edge_src = copy.deepcopy(edge_src)
                    backward_state.add_node(replicated_edge_src)

                    # add the connection from the repliacted node to the tasklet
                    new_edge_data = copy.deepcopy(edge.data)
                    backward_state.add_edge(replicated_edge_src, replicated_edge_src_conn, backward_node,
                                            required_inputs[edge.dst_conn], new_edge_data)
                    connect_replicated_node = False

                    # get the sink edge going into this AccessNode
                    sink_edge = state.in_edges(edge_src)

                    assert len(sink_edge) == 1
                    sink_edge = sink_edge[0]

                    # get the AccessNode that the temporary values are coming from
                    edge_list = state.memlet_path(sink_edge)
                    assert len(edge_list) > 0
                    source_edge = edge_list[0]
                    original_accessnode = source_edge.src
                    assert isinstance(original_accessnode, nodes.AccessNode)

                    # check if this AccessNode has been overwritten
                    overwritten, recomputable = self._check_node_overwrite(state=state, node=edge_src)
                    if overwritten:
                        # we lost values that are necessary for the backward pass
                        if self.strategy == "store_all":
                            # we need to modify the forward pass to store these neccessary values
                            new_store_accessnode, memlets = self._store_data(state=state, edge=source_edge)

                            # duplicate the memlets because this object is stored and can be used multiple times
                            memlets = copy.deepcopy(memlets)

                            self._connect_temporary_to_accessnode(state, new_store_accessnode, replicated_edge_src,
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
                        backward_state.add_node(replicated_original_accessnode)
                        # connect the nodes
                        self._connect_temporary_to_accessnode(state, backward_state, replicated_original_accessnode,
                                                              replicated_edge_src, memlet_list, sink_edge)
                else:
                    # base-case: we have reached a base level AccessNode.
                    # check if this AccessNode has been overwritten
                    overwritten, recomputable = self._check_node_overwrite(state=state, node=edge_src)

                    if overwritten:
                        # we lost values that are necessary for the backward pass
                        if self.strategy == "store_all":

                            # we need to modify the forward pass to store these neccessary values
                            new_store_accessnode, memlets = self._store_data(state=state, edge=edge)

                            # replicate the new store node from the forward state
                            replicated_new_store_accessnode = copy.deepcopy(new_store_accessnode)
                            backward_state.add_node(replicated_new_store_accessnode)

                            # we will traverse the memlets in the reverse order
                            # that they were added from the forward pass
                            last_memlet = memlets.pop()
                            last_memlet = copy.deepcopy(last_memlet)

                            # add the new edge
                            backward_state.add_edge(replicated_new_store_accessnode, None, backward_node,
                                                    required_inputs[edge.dst_conn], last_memlet)

                            # set the boolean to False to avoid adding the connection again
                            connect_replicated_node = False

                            # extract the edge to the new store node to get the memlet path
                            new_edge = backward_state.out_edges(replicated_new_store_accessnode)

                            # sanity check: there should be only one edge on this node
                            assert len(new_edge) == 1
                            new_edge = new_edge[0]

                            # get the memlet path to update the memlets accordingly
                            edge_list = backward_state.memlet_path(new_edge)

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
                                self._recompute_data(state=state, backward_state=backward_state, edge=edge)
                                connector_to_clean = required_inputs[edge.dst_conn]
                                self._clean_after_recomputation(forward_state=state,
                                                                backward_state=backward_state,
                                                                edge=edge,
                                                                connector_to_clean=connector_to_clean)
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

                # always replicate the access node
                replicated_edge_src = copy.deepcopy(edge_src)
                backward_state.add_node(replicated_edge_src)

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

                if isinstance(edge_src, nodes.AccessNode) and isinstance(data_desc, dt.View):
                    if self.separate_sdfgs:
                        # remove the view connector
                        assert replicated_edge_src.remove_in_connector("views")
                    else:
                        # if this is a view, we need to connect it to the AccessNode it is viewing
                        edge_src_in_edge = state.in_edges(edge_src)

                        # a view should only have one incoming edge
                        assert len(edge_src_in_edge) == 1
                        edge_src_in_edge = edge_src_in_edge[0]

                        # replicate the viewed node and its memlet and connect it
                        view_origin = edge_src_in_edge.src
                        replicated_view = copy.deepcopy(view_origin)
                        view_memlet = copy.deepcopy(edge_src_in_edge.data)
                        if self.separate_sdfgs:
                            # if the sdfgs are separate, we need to add the descriptor for this data
                            origin_desc = self.sdfg.arrays[view_origin.data]
                            origin_desc.transient = False
                            backward_state.sdfg.add_datadesc(view_origin.data, origin_desc)
                        backward_state.add_edge(replicated_view, None, replicated_edge_src, "views", view_memlet)

                # add the new edge
                backward_state.add_edge(replicated_edge_src, replicated_edge_src_conn, backward_node,
                                        required_inputs[edge.dst_conn], new_edge_data)

            if next_required_inputs:
                # if there are any required inputs on the new node, we need to
                # recursively call
                self._connect_forward_inputs(state, backward_state, edge.src, replicated_edge_src, next_required_inputs)

    def _connect_temporary_to_accessnode(self, state: SDFGState, backward_state: SDFGState,
                                         source_node: nodes.AccessNode, sink_node: nodes.AccessNode,
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
        edge_list = state.memlet_path(forward_sink_edge)
        assert len(edge_list) > 0

        # we will iterate and connect parent -> child
        child_node = sink_node
        child_node_in_connector = None
        # ietarte through the maps in the path in reverse
        for edge in reversed(edge_list):
            edge_src = edge.src
            if isinstance(edge_src, nodes.MapEntry):
                # get the correponding map exist
                map_exit = self._find_map_exist_for_map_entry(map_entry=edge_src, state=state)

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
                backward_state.add_edge(bwd_map_entry, pranet_node_out_connector, child_node, child_node_in_connector,
                                        memlet_data)

                child_node = bwd_map_entry
                child_node_in_connector = parent_node_in_connector

        # there should be the same number of memlets through the new path
        assert len(memlets) == 0

        # make the final memlet
        memlet_data = Memlet.from_array(source_node.data, self.sdfg.arrays[source_node.data])

        # add the final connection to the source node
        backward_state.add_edge(source_node, None, child_node, child_node_in_connector, memlet_data)

    def _clean_after_recomputation(self, forward_state: SDFGState, backward_state: SDFGState,
                                   edge: dgraph.MultiConnectorEdge, connector_to_clean: str):
        """
        In the case of the recomputation of a base-level AccessNode, 
        we will only know whether this node can be recomputed after adding a path from the tasklet that required the computation
        to the maps serrounding this tasklet.
        If recomputation is applied, this path of memlets is no longer required and needs to be removed.
        :param edge: the edge leading to the first backward map containing the first out edge in the path to clean
        :param connector_to_clean: name of the out connector of the first out edge in the path to clean
        """
        # get the map in the backward pass
        bwd_map = self._find_backward_entry_node_for_map_entry(forward_state=forward_state,
                                                               backward_state=backward_state,
                                                               entry_node=edge.dst)
        assert isinstance(bwd_map, nodes.MapEntry)

        # find the starting edge of the path we want to delete
        out_edges = backward_state.out_edges(bwd_map)
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
            next_map_edges = backward_state.out_edges(starting_edge.dst)
            for e in next_map_edges:
                if isinstance(e.dst, nodes.MapEntry):
                    if e.dst_conn == connector_to_clean:
                        next_edge = e
                        break
                else:
                    if e.src_conn == connector_to_clean.replace("IN", "OUT"):
                        backward_state.remove_edge(e)
                        e.src.remove_out_connector(connector_to_clean.replace("IN", "OUT"))
                        break

            starting_edge.src.remove_out_connector(starting_edge.src_conn)
            starting_edge.dst.remove_in_connector(starting_edge.dst_conn)
            backward_state.remove_edge(starting_edge)

            starting_edge = next_edge

    def _recompute_data(self, state: SDFGState, backward_state: SDFGState, edge: dgraph.MultiConnectorEdge):
        """
        Given an edge leading from a base-level AccessNode to a map in the forward state,
        add an sdfg to recompute the values of this node to the backward state.
        :param edge: the edge connecting the AccessNode to recompute data from to a map node.
        """
        replicate_nodes = {}
        # treat the case where the recomputation can be merged into the gradient maps
        # get the subgraph neccessary to calculate the AccessNode itself
        subgraph: dstate.StateSubgraphView = self._get_computation_subgraph(state=state, node=edge.src)
        # check if this is the case
        mergeable = self._check_if_recomputation_is_mergeable(state, edge, subgraph)
        if mergeable:
            # get the maps from the backward pass to modify
            edge_list = state.memlet_path(edge)
            backward_maps = []
            for e in edge_list:
                if isinstance(e.src, nodes.MapEntry):
                    bwd_map_entry = self._find_backward_entry_node_for_map_entry(forward_state=state,
                                                                                 backward_state=backward_state,
                                                                                 entry_node=e.src)
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
                    in_edges = state.in_edges(fwd_map_entry)
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
                                backward_state.add_node(replicated_edge_src)
                                replicate_nodes[e.src] = replicated_edge_src
                        else:
                            replicated_edge_src = backward_maps[map_index - 1]

                        memlet_data = copy.deepcopy(e.data)
                        # add a new edge between the backward map and the new replicated node
                        backward_state.add_edge(replicated_edge_src, replicated_edge_src_src_con, bwd_map_entry,
                                                replicated_edge_src_dst_con, memlet_data)
                    map_index += 1

            next_level = []
            node = fwd_map_entry
            node_bwd = bwd_map_entry
            # we go level by level through the content of the map nest
            while node:
                # we start with all the edges coming out of the last map
                node_out_edges = state.out_edges(node)
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
                        assert assign_tsaklet in backward_state.nodes()
                        assert "assign" in assign_tsaklet.name

                        # get the edge from the AccessNode coming to the assign tasklet
                        # this will be the edge that is connected to the reversed tasklet
                        assign_tasklet_in_edge = backward_state.in_edges(assign_tsaklet)
                        assert len(assign_tasklet_in_edge) == 1
                        assign_tasklet_in_edge = assign_tasklet_in_edge[0]
                        backward_state.remove_edge(assign_tasklet_in_edge)

                        # remove the tasklet
                        backward_state.remove_node(assign_tsaklet)

                        # add the new edge between the final AccessNode and the reversed tasklet
                        last_accessnode = assign_tasklet_in_edge.src
                        assert isinstance(last_accessnode, nodes.AccessNode)

                        memlet_data = assign_tasklet_in_edge.data
                        assert tasklet_conn in bwd_tasklet.in_connectors
                        backward_state.add_edge(last_accessnode, None, bwd_tasklet, tasklet_conn, memlet_data)
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
                            backward_state.add_node(replicated_edge_dst)
                            replicate_nodes[e.dst] = replicated_edge_dst

                        # add a new edge between the two nodes in the backward state
                        backward_state.add_edge(node_bwd, replicated_edge_src_src_con, replicated_edge_dst,
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

    def _check_if_recomputation_is_mergeable(self, state: SDFGState, edge: dgraph.MultiConnectorEdge,
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
        edge_list = state.memlet_path(edge)
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

    def _get_computation_subgraph(self, state: SDFGState, node: nodes.AccessNode) -> SDFG:
        """
        Given an access node get the subgraph from the forward state that writes to this access node
        """
        # reverse bfs from the accesss node
        backward_nodes = {n for e in state.bfs_edges(node, reverse=True) for n in [e.src, e.dst]}
        forward_nodes = {n for n in state.nodes()}
        # intersection with all the nodes in the forward state
        forward_subgraph = dstate.StateSubgraphView(state, list(forward_nodes.intersection(backward_nodes)))
        return forward_subgraph

    def _store_data_direct_access(self, state: SDFGState,
                                  edge: dgraph.MultiConnectorEdge) -> Tuple[nodes.AccessNode, List[Memlet]]:
        """
        """
        # get the AccessNode we want to save data from
        node: nodes.AccessNode = edge.src

        # make sure this is indeed an AccessNode
        assert isinstance(node, nodes.AccessNode)

        # get all the maps in the path from the accessnode to the desired tasklet
        edge_list = state.memlet_path(edge)

        # there should be at least two edges in the path
        assert len(edge_list) == 1

        # create and add the new access node that contains the saved values
        if "stored_" + node.data not in self.backward_sdfg.arrays:
            new_store_node_name = "stored_" + node.data
        else:
            i = 0
            while True:
                if f"stored_{i}_" + node.data not in self.backward_sdfg.arrays:
                    new_store_node_name = f"stored_{i}_" + node.data
                    break
                i += 1
        # first, get the shape of the new array
        shape_list = []
        # we will also need the starting range of the maps in the path
        start_range = []
        # and the names of the parameters of the maps in the path
        param_list = []

        nb_enclosing_loops = 0
        if self._state_within_loop(state):
            # add an extra dimension to the shapes
            # we need to check all of the loops tha enclose this state and their information
            loop_guards = self._get_loop_guard_enclosing_state(forward_state=state)
            nb_enclosing_loops = len(loop_guards)
            for guard in loop_guards:
                found, begin_edge, after_loop_edge, after_loop_state = self._find_for_loop(guard=guard)
                itervar, (start, end, step), (start_state, end_state) = found

                # add to the shapes and ranges list
                shape_list.insert(0, end + 1)
                start_range.insert(0, start)
                param_list.insert(0, itervar)

        # get the shape of the original array
        shape = self.sdfg.arrays[node.data].shape
        shape_list += shape
        # create a map to copy all of the array
        # add the array descriptor and AccessNode to the forward state
        original_desc = node.desc(state)
        new_store_node = state.add_array(
            name=new_store_node_name,
            shape=shape_list,
            dtype=original_desc.dtype,
            transient=True,
        )

        # the loop accesses will be the same within the state
        # prepare them for all edges
        loop_access = ','.join([f'{param_list[i]}' for i in range(0, nb_enclosing_loops)])

        in_state_access = ','.join([f'0:{s}' for s in shape])

        memlet_data = Memlet(expr=f"{new_store_node.data}[{loop_access},{in_state_access}]")

        # connect the last map exist to the newly created store node
        # add the new edge
        state.add_edge(
            node,  # the last MapExist
            None,  # the last MapExist connector
            new_store_node,  # the new store AccessNode
            None,
            memlet_data)

        # create the memlet to add to the backward pass
        for start, end, stride in edge.data.subset:
            # add the subset to the list
            loop_access += f",{start}:{end+1}" if len(start.free_symbols) == 0 else f",{start}"

        memlet_data = Memlet(expr=f"{new_store_node.data}[{loop_access}]")
        return new_store_node, [memlet_data]

    def _store_data(self, state: SDFGState, edge: dgraph.MultiConnectorEdge) -> Tuple[nodes.AccessNode, List[Memlet]]:
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
        edge_list = state.memlet_path(edge)

        # there should be at least two edges in the path
        if len(edge_list) == 1:
            new_store_node, memlets_stack = self._store_data_direct_access(state=state, edge=edge)
            return new_store_node, memlets_stack

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
        state.add_node(assign_tasklet_node)

        # create the memlet for the assignement
        # this will be the same as the memlet going to the tasklet
        assign_memlet_data = copy.deepcopy(last_edge.data)

        # add the new edge from the last map to the new assign tasklet
        state.add_edge(last_map, last_map_connector, assign_tasklet_node, assign_tasklet_node_in_connector,
                       assign_memlet_data)

        # create and add the new access node that contains the saved values
        # first check if a stored array with this name already exists
        if "stored_" + node.data not in self.backward_sdfg.arrays:
            new_store_node_name = "stored_" + node.data
        else:
            i = 0
            while True:
                if f"stored_{i}_" + node.data not in self.backward_sdfg.arrays:
                    new_store_node_name = f"stored_{i}_" + node.data
                    break
                i += 1

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

        nb_enclosing_loops = 0
        if self._state_within_loop(state):
            # add an extra dimension to the shapes
            # we need to check all of the loops tha enclose this state and their information
            loops = self._get_state_enclosing_loops(forward_state=state)

            for loop in loops:
                # TODO create the find loop function for LoopRegions
                itervar, start, end = loop.loop_variable

                # add to the shapes and ranges list
                # TODO if the loop bounds are symbolic what to do
                shape_list.insert(0, end + 1)
                start_range.insert(0, start)
                param_list.insert(0, itervar)

        # add the array descriptor and AccessNode to the forward state
        original_desc = node.desc(state)
        new_store_node = state.add_array(
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

        # the loop accesses will be the same within the state
        # prepare them for all edges
        loop_access = ','.join([f'{param_list[i]}' for i in range(0, nb_enclosing_loops)])

        # for each map in the path
        for e in reversed(edge_list):
            edge_src = e.src
            if isinstance(edge_src, nodes.MapEntry):
                # get the corresponding map exit
                child_node = self._find_map_exist_for_map_entry(map_entry=edge_src, state=state)
                # get a new connector id
                next_conn = child_node.next_connector()

                # add a new in connector to the mapexit
                child_node_in_connector = "IN_stored_" + node.data + "_" + next_conn
                assert child_node.add_in_connector(child_node_in_connector)

                # add a new out connector to the mapexit
                child_node_out_connector = "OUT_stored_" + node.data + "_" + next_conn
                assert child_node.add_out_connector(child_node_out_connector)
                in_state_access = ','.join([
                    f'{param_list[i]}' if param_list[i] in params_to_add else f'{start_range[i]}:{shape_list[i]}'
                    for i in range(nb_enclosing_loops, len(param_list))
                ])

                memlet_data = Memlet(
                    expr=f"{new_store_node.data}[{loop_access},{in_state_access}]") if loop_access else Memlet(
                        expr=f"{new_store_node.data}[{in_state_access}]")
                # save the memlet for later
                memlets_stack.append(memlet_data)

                # add the edge with the corresponding memlet
                state.add_edge(parent_node, parent_node_connector, child_node, child_node_in_connector, memlet_data)

                # prepare values for the next iteration
                parent_node = child_node
                parent_node_connector = child_node_out_connector

                # remove the parameters seen in the current map
                # since they will become out of scope in the next iteration
                params_to_add = [element for element in params_to_add if element not in edge_src.params]

        # sanity check: since we are out of scope for all maps,
        # only the enclosig loop parameters should be left in the list
        assert len(params_to_add) == nb_enclosing_loops

        in_state_access = ','.join(
            [f'{start_range[i]}:{shape_list[i]}' for i in range(nb_enclosing_loops, len(param_list))])
        # get the memlet data for the connection between the last map exit and the new store AccessNode
        memlet_data = Memlet(expr=f"{new_store_node.data}[{loop_access},{in_state_access}]") if loop_access else Memlet(
            expr=f"{new_store_node.data}[{in_state_access}]")
        memlets_stack.append(memlet_data)

        # connect the last map exist to the newly created store node
        # add the new edge
        state.add_edge(
            parent_node,  # the last MapExist
            parent_node_connector,  # the last MapExist connector
            new_store_node,  # the new store AccessNode
            None,
            memlet_data)

        return new_store_node, memlets_stack

    def _get_state_enclosing_loops(self, forward_state: SDFGState) -> List[LoopRegion]:
        """
        Get all the enclosing loops of this state.
        """
        loop_list = []
        parent_graph = forward_state.parent_graph
        while isinstance(parent_graph, LoopRegion):
            loop_list.append()
            parent_graph = parent_graph.parent_graph
        return loop_list

    def _check_node_overwrite(self, state: SDFGState, node: nodes.AccessNode) -> Tuple[bool, bool]:
        """
        Given an AccessNode from the forward state, check if the data of this node has changed.
        We look at all the AccessNodes with the same data that occur after the 'node' parameter
        if any of them has an incoming edge, return the node has been overwritten.
        
        :param edge: the AccessNode to perform the check for.
        :return: a tuple of wether this node has been overwritten, and if it can be recomputed
        """
        #TODO remove developement code
        return False, False

        overwritten = False
        decided = False
        recomputable = False

        # get the decendant states to look in for an overwrite
        assert state in self.state_order
        index = self.state_order.index(state)
        decendant_states = self.state_order[index:]
        ascendant_states = self.state_order[:index]

        # check if this access node is a view
        if type(node.desc(self.sdfg)) is dt.ArrayView:
            # the view should have one incoming edge from the original access node
            in_edges = state.in_edges(node)

            # sanity checks
            assert len(in_edges) == 1
            assert "views" in node.in_connectors

            # we want to check if the source has been overwritten
            node = in_edges[0].src

        # get all the AccessNodes with the same data
        matches = []
        for d_state in decendant_states:
            matches += [
                nd for nd, _ in d_state.all_nodes_recursive()
                if isinstance(nd, nodes.AccessNode) and nd.data == node.data
            ]

        # there needs to be at least one occurance which is the node passed as a parameter
        assert len(matches) > 0 and node in matches

        # if there is only one occurance of this data, it will not be overwritten later in the graph
        if len(matches) == 1:
            overwritten = False
            decided = True

        # get the index of the parameter node
        index = matches.index(node)

        # if the parameter node is the last occurance in the forward states
        # return False
        if len(matches) - 1 == index:
            overwritten = False
            decided = True

        # if we haven't already confirmed that this node has not been overwritten
        if not decided:
            # iterate through all the successor occurances
            for nd in matches[index + 1:]:
                # check if this node has an incoming edge
                if len(state.in_edges(nd)) > 0:
                    overwritten = True

        if not overwritten:
            # there is no overwrite so far
            # check if this state is within a loop
            loop_state = self._state_within_loop(state)
            if loop_state:
                loop = state.parent_graph
                # check if there is any write to this access node within the loop
                loop_matches = [
                    nd for nd, _ in loop.all_nodes_recursive()
                    if isinstance(nd, nodes.AccessNode) and nd.data == node.data
                ]

                for match in loop_matches:
                    if len(state.in_edges(match)) > 0:
                        overwritten = True

        if overwritten:
            # we only check if the node is recomputable if it has been overwritten
            # iterate through all the predecessor occurances
            # TODO: currently this only checks for a single state
            for nd in matches[:index + 1]:
                # check if this node has an incoming edge
                if len(state.in_edges(nd)) > 0:
                    recomputable = True

            if not recomputable:
                # fall back to the store strategy
                self.strategy = "store_all"

        return overwritten, recomputable

    def _state_has_write_to_data(self, state: SDFGState, node: nodes.AccessNode) -> bool:
        """
        checks whether this state has any write operation to the data in the node parameter
        """
        # get all the AccessNodes with the same data
        matches = [nd for nd in state.nodes() if isinstance(nd, nodes.AccessNode) and nd.data == node.data]
        # if there is only one occurance of this data, it will not be overwritten later in the graph
        if len(matches) == 0:
            return False

        # iterate through all the successor occurances
        for nd in matches:
            # check if this node has an incoming edge
            if len(state.in_edges(nd)) > 0:
                return True

        return False

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

    def _find_backward_entry_node_for_map_entry(self, forward_state: SDFGState, backward_state: SDFGState,
                                                entry_node: nodes.MapEntry) -> nodes.MapExit:
        """Find the entry node in the backward pass corresponding to the exit node opened by
        `entry_node` (where `entry_node` is a node from the forward pass).
        """
        src_candidates = [
            cast(nodes.MapExit, node) for node in backward_state.nodes()
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

    def _get_reverse_node(self, state: SDFGState, backward_state: SDFGState, node, given_gradients,
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
            return getattr(self, "_reverse_" + type(node).__name__)(state, backward_state, node, given_gradients,
                                                                    required_gradients)

        # (2)
        impl = find_backward_implementation(self.sdfg, forward_state=state, node=node)
        if impl is not None:
            backward_node, backward_result = impl.backward(forward_node=node,
                                                           context=BackwardContext(
                                                               forward_state=state,
                                                               forward_sdfg=self.sdfg,
                                                               backward_state=backward_state,
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
        forward_state: SDFGState,
        backward_state: SDFGState,
        node: nodes.NestedSDFG,
        given_gradients: List[str],
        required_gradients: List[str],
    ) -> ReverseNodeReturnType:

        reverse_sdfg = dace.SDFG(node.sdfg.name + "_backward")
        # recursive call
        gen = BackwardPassGenerator(sdfg=node.sdfg,
                                    given_gradients=given_gradients,
                                    required_gradients=required_gradients,
                                    backward_sdfg=reverse_sdfg,
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

                desc = node.sdfg.arrays[name]

                # if the original view node is in the in-connector, no need to connect it, continue
                # if forwarded_name in node.in_connectors:
                #     continue

                # (1)
                new_name = find_str_not_in_set(set(self.sdfg.arrays), name + "_forwarded")
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
                node.sdfg.arrays[name].transient = False
                assert node.add_out_connector(name, force=True)
                write = forward_state.add_write(new_name)
                forward_state.add_edge(node, name, write, None, self.sdfg.make_array_memlet(new_name))

                # (3)
                read = backward_state.add_read(new_name)
                deferred_edges.append(
                    dict(u=read,
                         u_connector=None,
                         v_connector=name,
                         memlet=self.backward_sdfg.make_array_memlet(new_name)))
                inputs.add(name)
            else:
                inputs.add(name)

        outputs = set(backward_result.required_grad_names[name] for name in required_gradients)

        for inp in inputs:
            reverse_sdfg.arrays[inp].transient = False
        for outp in outputs:
            reverse_sdfg.arrays[outp].transient = False

        # actually create the sdfg and return it
        nsdfg = backward_state.add_nested_sdfg(
            reverse_sdfg,
            None,
            inputs=inputs,
            outputs=outputs,
        )

        for edge_args in deferred_edges:
            edge_args["v"] = nsdfg
            backward_state.add_edge(**edge_args)

        return nsdfg, BackwardResult(required_grad_names=backward_result.required_grad_names,
                                     given_grad_names=backward_result.given_grad_names)

    def _reverse_AccessNode(
        self,
        forward_state: SDFGState,
        backward_state: SDFGState,
        node: nodes.AccessNode,
        given_gradients: List[str],
        required_gradients: List[str],
    ) -> ReverseNodeReturnType:
        rev = nodes.AccessNode(self.array_grad_name(node.data))
        backward_state.add_node(rev)
        required_grad_names = {None: None}
        given_grad_names = {None: None}

        if "views" in node.in_connectors:
            required_grad_names = {"views": "views"}
        if "views" in node.out_connectors:
            given_grad_names = {"views": "views"}

        return rev, BackwardResult(required_grad_names=required_grad_names, given_grad_names=given_grad_names)

    def _reverse_MapEntry(
        self,
        forward_state: SDFGState,
        backward_state: SDFGState,
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

        backward_state.add_node(rev)
        return rev, result

    def _reverse_MapExit(
        self,
        forward_state: SDFGState,
        backward_state: SDFGState,
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

        backward_state.add_node(rev)
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
        state: SDFGState,
        backward_state: SDFGState,
        tasklet: nodes.Tasklet,
        given_gradients: List[str],
        required_gradients: List[str],
    ) -> ReverseNodeReturnType:
        if tasklet.language is not dtypes.Language.Python:
            raise AutoDiffException("Expected tasklet with language Python, got language {}".format(tasklet.language))

        # tasklets should have scalar inputs (can be relaxed)
        for _, _, _, _, memlet in state.in_edges(tasklet):
            try:
                _is_int_value(memlet.subset.num_elements(), 1)
            except AutoDiffException as e:
                raise AutoDiffException("Autodiff only supported for tasklets with scalar inputs and outputs") from e

        for _, _, _, _, memlet in state.out_edges(tasklet):
            try:
                _is_int_value(memlet.subset.num_elements(), 1)
            except AutoDiffException as e:
                raise AutoDiffException("Autodiff only supported for tasklets with scalar inputs and outputs") from e

        code_str = tasklet.code.as_string

        # check if this is a conditional tasklet
        if self._conditional_tasklet(tasklet):
            # we want to extract the if and else expressions and pass them to sympy
            if_expression, else_expression, conditional = self._extract_conitional_expressions(tasklet)

            if_code, if_rev_inputs, if_rev_outputs, if_result = self._differentiate_code_symbolically(
                if_expression, state, tasklet, given_gradients, required_gradients)

            if else_expression:
                else_code, else_rev_inputs, else_rev_outputs, else_result = self._differentiate_code_symbolically(
                    else_expression, state, tasklet, given_gradients, required_gradients)
                assert else_rev_inputs == if_rev_inputs
                assert if_rev_outputs == else_rev_outputs
                assert else_result == if_result

            # prepare the tasklet code depending on the conditional type
            # add the same conditional to the if_code
            # first, add indentation
            if_code = if_code.replace("\n", "\n\t")
            if_code = f"if {conditional}:\n{if_code}"

            # add the conditional to the in connectors
            if_rev_inputs.add(conditional)
            joint_code = if_code

            if ":" not in code_str:
                # only an if in the original code
                assert else_expression
                else_code = else_code.replace("\n", "\n\t")
                else_code = f"else:\n{else_code}"
                joint_code = f"{if_code}\n{else_code}"

            # in case there are no out_connectors, we will zero out the assigned-to AccessNode
            if len(if_rev_outputs) == 0:
                if_rev_outputs = {"__zero_out_conn__"}

            rev = nodes.Tasklet("_" + tasklet.label + "_reverse_",
                                inputs=if_rev_inputs,
                                outputs=if_rev_outputs,
                                code=joint_code,
                                debuginfo=tasklet.debuginfo)

            result = if_result
        else:
            code, rev_inputs, rev_outputs, result = self._differentiate_code_symbolically(
                code_str, state, tasklet, given_gradients, required_gradients)
            rev = nodes.Tasklet("_" + tasklet.label + "_reverse_",
                                inputs=rev_inputs,
                                outputs=rev_outputs,
                                code=code,
                                debuginfo=tasklet.debuginfo)
            backward_state.add_node(rev)
        return rev, result


class SympyCleaner(ast.NodeTransformer):
    def visit_Name(self, node):
        if node.id == "pi":
            return ast.copy_location(ast.parse("(3.141592653589)").body[0], node)
        else:
            return self.generic_visit(node)
