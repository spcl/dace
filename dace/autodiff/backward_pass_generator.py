"""
    Automatic Differentiation of an SDFG.
    This module exposes the add_backward_pass method that can be used to add a backward pass to an
    SDFGState.
"""
import ast
import astunparse
import collections
import copy
import logging
import re
from typing import List, Tuple, Set, Dict, Union, cast, Optional, Callable, Sequence
import numpy as np
import sympy as sp

# DaCe imports
import dace
from dace.properties import CodeBlock
import dace.sdfg.nodes as nodes
import dace.transformation.transformation as xf
from dace import dtypes, data as dt
from dace.sdfg import SDFG, SDFGState, graph as dgraph, state as dstate, utils as dace_utils
from dace.sdfg.state import LoopRegion, ControlFlowRegion
from dace.memlet import Memlet
from dace.util import find_str_not_in_set
from dace.libraries.onnx.forward_implementation_abc import ONNXForward
from dace.libraries.onnx.nodes.onnx_op import ONNXOp

# Autodiff imports
from dace.autodiff.base_abc import (BackwardContext, BackwardResult, AutoDiffException, find_backward_implementation)
import dace.autodiff.utils as ad_utils
from dace.autodiff.implementations.dace_nodes import DaceNodeBackwardImplementations

log = logging.getLogger(__name__)


class BackwardPassGenerator:
    """ Class that holds the states for one backward pass creation.

        See autodiff.py, _reverse_NestedSDFG and pytorch.py for examples of usage.
        :param given_gradients: the outputs that gradients must be provided for (i.e. access nodes will be created for
               these)
        :param required_gradients: the inputs to generate gradients for
        :param backward_sdfg: the sdfg the backward pass will be contained in. If it is the same as the forward_sdfg,
                              outputs must be a list containing a single scalar.
        :param array_grad_map: A mapping from array name to the gradient array name. May be passed when certain
                               mappings already exist.
        :param conflicted_gradient_buffers: A list of forward pass value names for which multiple backward passes will
                                            be computed, and thus gradients should be computed with
                                            write-conflict-resolution.
        :param overwrite_strategy: The strategy to use to provide overwritten values from the forward pass to the backward pass. 
                                    Should be either: store_all, recompute_all, or user_defined.
    """

    def __init__(
        self,
        *,
        sdfg: SDFG,
        given_gradients: Sequence[Union[nodes.AccessNode, str]],
        required_gradients: Sequence[Union[nodes.AccessNode, str]],
        backward_sdfg: SDFG,  # This can be the same as sdfg
        array_grad_map: Optional[Dict[str, str]] = None,
        conflicted_gradient_buffers: Optional[Set[str]] = None,
        overwrite_strategy: str = "store_all",
        data_to_recompute: List[str] = None,
    ):

        self.sdfg: SDFG = sdfg
        self.strategy = overwrite_strategy
        self.data_to_recompute = data_to_recompute
        self.backward_sdfg: SDFG = backward_sdfg

        # DaCe nodes backward implementations
        self.dace_node_impl = DaceNodeBackwardImplementations(self)

        given_gradients = [
            n if isinstance(n, nodes.AccessNode) else self._str_to_access(n, "outputs") for n in given_gradients
        ]
        required_gradients = [
            n if isinstance(n, nodes.AccessNode) else self._str_to_access(n, "inputs") for n in required_gradients
        ]
        required_gradients = [n for n in required_gradients if n is not None]

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

        #: Mapping between the map entry of a conditional assignement block and its zeroout AN
        self.conditional_block_entry: Dict[nodes.MapEntry, nodes.AccessNode] = {}

        #: Mapping from forward_node -> BackwardResult for that node
        self.result_map: Dict[nodes.Node, BackwardResult] = {}

        #: Mapping from forward name to gradient name for arrays
        self.array_grad_map: Dict[str, str] = array_grad_map or {}

        #: Mapping from the backward access nodes that will be zeroed out
        # to the transients that contain the values before they are zeroed out
        self.zeroed_out: Dict[nodes.AccessNode, List[nodes.AccessNode]] = {}

        #: The read only arrays of the forward sdfg
        self.read_only_arrays: Set[str] = ad_utils.get_read_only_arrays(self.sdfg)

        #: Mapping from overwritten input name to storing AccessNode
        self.stored_inputs: Dict[str, nodes.AccessNode] = {}

        #: List containing information about all the data to be forwarded to the backward pass
        self._forward_data: List[Tuple[SDFGState, SDFGState, nodes.AccessNode, nodes.Node,
                                       dstate.MultiConnectorEdge]] = []

        # Variable to check if backward has already been applied
        self._applied = False

        # Topological orderning of the states
        self.state_order = ad_utils.get_state_topological_order(self.sdfg)
        self.conflicted_gradient_buffers: Set[str] = conflicted_gradient_buffers or set()

        self.interstate_symbols: Dict[str, str] = {}
        for edge in self.sdfg.all_interstate_edges():
            for assign_symbol, assignement in edge.data.assignments.items():
                self.interstate_symbols[assign_symbol] = assignement

        # Sanity-check: the outputs need to be present in at least one of the states of the sdfg
        for outp in self.given_gradients_data:
            found = self.sdfg.arrays.get(outp) is not None
            if not found:
                raise AutoDiffException(f"Could not find output {outp} in the SDFG array descriptors dictionary")

        # Sanity-check: the inputs need to be present in at least one of the states of the sdfg
        for inp in self.required_gradients_data:
            found = self.sdfg.arrays.get(inp) is not None
            if not found:
                raise AutoDiffException(f"Could not find input {inp} in the SDFG array descriptors dictionary")

        if sdfg is backward_sdfg:
            # We only do reverse mode AD, which requires a single scalar output for now
            if len(given_gradients) != 1:
                raise AutoDiffException("When the forward sdfg is the same as the backward sdfg, outputs must be a"
                                        "single scalar")
            if not ad_utils.is_int_eq_value(sdfg.arrays[given_gradients[0].data].total_size, 1):
                raise AutoDiffException("When the forward sdfg is the same as the backward sdfg, outputs must be a"
                                        "single scalar")
            self.separate_sdfgs = False
        else:
            self.separate_sdfgs = True

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

        # Create state views mapping and expand all the SDFG nodes
        self._create_stateviews_mapping()

        # Reverse each state in the graph
        self._reverse_states()

        # Connect the new reversed states to the other states correctly
        self._connect_reversed_states()

        # Fill the interstate edges with the correct conditions
        self._fill_interstate_edge_conditions()

        # Add interstate assignements for control flow decisions
        self._add_interstate_edge_assignments()

        # Forward required data by the backward pass according to a user defined strategy
        self._forward_data_to_backward_states()

        # In some cases (accessnode -> accessnode), the descriptors for the gradients of the function outputs are not
        # added yet. Add them now.
        for given_grad in sorted(self.given_gradients_data):
            if self.array_grad_name(given_grad) not in self.backward_sdfg.arrays:
                self._add_gradient_data_descriptor(given_grad)

        # Prepare the output
        required_grad_names = {name: self.array_grad_name(name) for name in self.required_gradients_data}
        given_grad_names = {name: self.array_grad_name(name) for name in self.given_gradients_data}

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

                # Check that all edges are float, int, or boolean
                ad_utils.check_edges_type_in_state(state_subgraph_view)

                self._disambiguate_direction_dependent_views(state)

                # Recursively reverse the subgraph
                self._reverse_subgraph(forward_state=state, backward_state=reversed_state, subgraph=state_subgraph_view)

        # We also reverse all the LoopRegions in the graph
        for node in self.sdfg.nodes():
            if not isinstance(node, LoopRegion):
                continue
            self._reverse_loop_region(node)

    def _connect_reversed_states(self):
        """
        Go through all the states of the forward SDFG and connect the equivelent backward state as necessary.
        All the incoming edges of a state in the forward SDFG will result in outgoing edges in the backward SDFG.
        """

        for state in self.state_order:
            # All states should be reversed already
            assert state in self.reversed_states_map
            backward_state = self.reversed_states_map[state]

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
        nodes = dace_utils.dfs_topological_sort(graph, graph.source_nodes())
        for node in nodes:
            # A list of the conditions on all the in edges for this state
            in_edges_conditions: List[str] = []
            if isinstance(node, SDFG) or isinstance(node, LoopRegion):
                # if this is not a reversed loop region
                if not node in self.reversed_loops_map:
                    continue
                self._fill_interstate_edge_conditions_in_scope(node)
            else:

                assert isinstance(node, SDFGState)
                forward_state = node
                parent_graph = forward_state.parent_graph

                # if this is not a reversed state
                if node not in self.reversed_states_map:
                    continue

                # We will iterate through all the incoming edges to the forward state
                edges_list = parent_graph.in_edges(forward_state)

                # If there are none, this is a start state
                # If there is only one incoming edge, no condition necessary
                if len(edges_list) < 2:
                    conditions_map[forward_state] = "1"

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
            conditions_map[node] = condition_for_state

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

            # Add a state to the reversed loop region
            new_start_state = reversed_loop.add_state_before(reversed_loop_start_state,
                                                             is_start_block=True,
                                                             condition=first_state_condition)

            # The condition for this interstate edge should be all iterations expect the fist
            leftover_iterations_condition = f"not {first_state_condition.as_string}"

            # Add a connection between this new start state and the first iteration state
            reversed_loop.add_edge(src=new_start_state,
                                   dst=leftover_loop_state,
                                   data=dace.InterstateEdge(condition=leftover_iterations_condition))

    def _add_interstate_edge_assignments(self):
        """
        We will need to add interstate assignements at the start of the backward SDFG
        This is necessary to make sure the control flow in the backward passs is correctly preserved. 
        """
        # We will add an empty state to the backward pass which will have all the assignements
        assignement_states = SDFGState("_bwd_interstate_assignements_state")

        new_assignements = {}
        # Get all the interstate edges in the forward sdfg
        for edge in self.sdfg.all_interstate_edges():
            if edge.data.assignments:
                # There are assignments to be added to the start of the backward pass
                new_assignements = {**new_assignements, **edge.data.assignments}

                # We need to check if any data needs to be used in these assignement
                # This is important in the case of a NSDFG where data will need to be forwarded
                for lhs, rhs in edge.data.assignments.items():
                    # If any of the sdfg arrays are in the rhs assignement
                    assignement_arrays = [array for array in self.sdfg.arrays.keys() if array in rhs]
                    if assignement_arrays and self.separate_sdfgs:
                        # We need to forward this data to the backward pass
                        for array in assignement_arrays:
                            if array not in self.backward_input_arrays:
                                self.backward_input_arrays[array] = self.sdfg.arrays[array]
                                # Special case if this is a symbol that is doesn't have a descriptor yet
                                if array not in self.backward_sdfg.arrays:
                                    # We add it now
                                    self.backward_sdfg.add_datadesc(array, copy.deepcopy(self.sdfg.arrays[array]))

        if new_assignements:
            # Add the new state to the backward pass
            # First we get the start block of the backward pass
            if self.separate_sdfgs:
                bwd_start_block = self.backward_sdfg.start_block
            else:
                fwd_start_state = self.sdfg.start_block
                if isinstance(fwd_start_state, LoopRegion):
                    bwd_start_block = self.reversed_loops_map[fwd_start_state]
                elif isinstance(fwd_start_state, SDFGState):
                    bwd_start_block = self.reversed_states_map[fwd_start_state]
                else:
                    raise AutoDiffException("Need to add an assignements state but can't find the start block")
            # TODO would this work on a loop region?
            self.backward_sdfg.add_state_before(state=bwd_start_block,
                                                label="_bwd_interstate_assignements_state",
                                                assignments=new_assignements)

    def is_within_map(self, state: SDFGState, node: nodes.AccessNode) -> bool:
        # Get the scope dictionary for the state
        scope_dict = state.scope_dict()

        # Check if the node is within the scope of a map
        scope_entry = scope_dict.get(node, None)
        while scope_entry is not None:
            if isinstance(scope_entry, nodes.MapEntry):
                return True
            scope_entry = scope_dict.get(scope_entry, None)

        return False

    def _zero_out_gradient(self, forward_state: SDFGState, forward_node: nodes.AccessNode, memlet: Memlet):
        """
        Overwritten arrays in the forward pass will need to have their gradients zeroed out for gradient accumelation to work.
        This function will:
            1- copy the current gradient values to a temporary array. These values will still be used in the backward pass one last time.
            2- zero-out the overwritten access in the backward pass.
            3- add the new temporary to a dictionary so that the new read is made to the temporary and not the original gradient array.
        We will also try to avoid doing this as much as possible to optimize performance.
        """
        # Extra checks to only do this if necessary
        # If this access node is not written to in the forward pass except for this one time, we don't need to zero it out
        # An exception is made for required gradients that can be read outside the scope of the SDFG
        clear_out_gradients = forward_node.data in self.required_gradients_data

        # Get the write instances in the forward sdfg to this node that happen in states before the current state
        # These will represent the reads that will happen after this AccessNode
        # This should avoid unnecessary zeroing out of dace generated temporaries
        for state in self.state_order[0:self.state_order.index(forward_state) + 1]:
            # TODO: what if there are multiple views of the same state
            state_view = self.states_view_map[state]
            for node, parent in state_view.all_nodes_recursive():
                if isinstance(node, nodes.AccessNode) and node.data == forward_node.data:
                    if parent.in_degree(node) > 0:
                        # We need to check if the the forward node is inside a map scope or a LoopRegion
                        within_loop, loop = self._state_within_loop(state)
                        within_map = self.is_within_map(state, node)
                        if node != forward_node or (node == forward_node and (within_loop or within_map)):
                            clear_out_gradients = True
                            break

        # We can avoid clearing out the gradients
        if not clear_out_gradients:
            return

        # Get the backward state
        backward_state: SDFGState = self.reversed_states_map[forward_state]

        # Get the backward node
        backward_node: nodes.AccessNode = self.reverse_map[forward_node]

        # Get the original array
        array_desc = self.backward_sdfg.arrays[backward_node.data]

        if dtypes.can_access(dtypes.ScheduleType.CPU_Multicore, array_desc.storage):
            cuda = False
        elif dtypes.can_access(dtypes.ScheduleType.GPU_Default, array_desc.storage):
            cuda = True
        else:
            raise ValueError(f"Unsupported storage {array_desc.storage}")

        # Careful! The order of the ifs here matters since ArrayView is a subclass of Array
        if isinstance(array_desc, (dt.View, dt.ArrayView)):
            # No need to initialize: the viewed array will always be visited
            # (since a view can never be a required grad), and thus the viewed array will be initialized.
            pass
        elif isinstance(array_desc, (dt.Array, dt.Scalar)):
            # Create a new memlet to write to the gradient arrays
            map_exit_memlet = copy.deepcopy(memlet)
            map_exit_memlet.data = backward_node.data

            # Create the tasklet to zero out only the section in the memlet
            # First, Get the range that the zeroout map should iterate over
            # TODO: We are looking at writes in the forward pass,
            # We should take the dst_subset of the memlet
            # Are there cases where dst_subset is None?
            ranges = []
            # TODO: is dst_subset always the right choice?
            for iteration in map_exit_memlet.dst_subset:
                if isinstance(iteration, tuple):
                    # The end of the range is inclusive in the loop
                    # We add 1 to get the upper bound for the map
                    ranges.append((iteration[0], iteration[1] + 1))
                elif isinstance(iteration, sp.Number):
                    # This covers the case of a single element being written
                    ranges.append((int(iteration), int(iteration) + 1))
                else:
                    raise AutoDiffException(f"Unsupported subset type {type(iteration)} in memlet {memlet}")

            # Create the indices dict
            indices = {f"i{i}": f"{start}:{end}" for i, (start, end) in enumerate(ranges)}

            # Create the tasklet memlet from the indices
            tasklet_memlet = dace.Memlet.simple(backward_node.data, ", ".join(indices.keys()))

            # Create the tasklet
            tasklet, map_entry, map_exit = backward_state.add_mapped_tasklet(
                "_clear_" + backward_node.data + "_",
                indices, {},
                f"__out = 0", {
                    "__out": tasklet_memlet,
                },
                schedule=dtypes.ScheduleType.GPU_Device if cuda else dtypes.ScheduleType.Default,
                external_edges=True)

            # Get the edge from the map exit to the backward node
            edge = backward_state.out_edges(map_exit)[0]

            # Get the cleared out AN
            cleared_out_node = edge.dst
            assert isinstance(cleared_out_node, nodes.AccessNode)

            # Create a copy of new memlet that will keep its other subset
            # We want to copy the elements to their same indices in the new tmp array
            # Create a new memlet that copies what memlet is writing to to the tmp
            new_memlet_subset = memlet.subset if memlet.data == forward_node.data else memlet.other_subset
            original_to_tmp_memlet = dace.Memlet(data=backward_node.data,
                                                 subset=new_memlet_subset,
                                                 other_subset=new_memlet_subset)

            # Remove the src_subset of the new memlet and replace the memlet in the edge
            map_exit_memlet.subset = memlet.subset if memlet.data == forward_node.data else memlet.other_subset
            map_exit_memlet.other_subset = None
            edge.data = map_exit_memlet

            # Add an edge from the backward_node to the new map entry
            backward_state.add_edge(backward_node, None, map_entry, None, dace.Memlet())

            # A race will happen unless we make sure the data is being copied being it is zeroed out
            # There is a read from the same array
            # We need to add a transient that reads the content from forward pass before it is zeroed out
            # Create a new array descriptor for the transient
            # TODO: Reuse array descriptors for the same data
            transient_desc = copy.deepcopy(array_desc)
            transient_desc.transient = True

            # Add the new array to the sdfg
            transient_name = self.array_grad_name(forward_node.data) + "_tmp"

            # Check if the array is already in the backward sdfg
            if transient_name not in self.backward_sdfg.arrays:
                self.backward_sdfg.add_datadesc(transient_name, transient_desc)

            # Create an AcessNode for this transient and add it to backward state
            transient_node = backward_state.add_read(transient_name)

            # Add a read from the backward node to the transient
            backward_state.add_edge(backward_node, None, transient_node, None, original_to_tmp_memlet)

            # Add an empty edge from the transient to the map entry
            backward_state.add_edge(transient_node, None, map_entry, None, dace.Memlet())
            if backward_node not in self.zeroed_out:
                self.zeroed_out[backward_node] = [transient_node]
            else:
                self.zeroed_out[backward_node].append(transient_node)
        else:
            raise AutoDiffException("Unsupported data descriptor {}".format(array_desc))

    def _forward_data_to_backward_states(self) -> None:
        """
        Iterate through all the data that needs to be forwarded to the backward pass states.
        """
        # Get the strategy decision for each data that needs to be forwarded to the backward pass
        strategy_choice, recomputation_nsdfgs = self._get_overwrite_resolution_strategy()

        # Make the connection according to the chosen strategy
        for index, (forward_state, backward_state, access_node, node, edge) in enumerate(self._forward_data):
            self._connect_forward_accessnode(forward_state, backward_state, access_node, node, edge,
                                             recomputation_nsdfgs[index], strategy_choice[index])

    def _remove_onnx_attribute_accessnodes(self, nodes_list: List[nodes.Node], state: SDFGState) -> None:
        """
        For some ONNX operators, nodes have attributes as input connectors even if the inputs are actually constant.
        Examples of such attributes are `axis` and `keepdims` in `ReduceSum`. 
        Gradients for these attributes should not be tracked since they represent control flow and not data flow.
        """
        attribute_to_remove = {"axis", "keepdims", "axes", "p", "dilations", "kernel_shape", "strides"}
        for node in nodes_list[:]:  # Iterate over a copy of the list to avoid modification issues
            if isinstance(node, nodes.AccessNode):
                out_edges = state.out_edges(node)
                if out_edges and all(
                        isinstance(edge.dst, ONNXOp) and edge.dst_conn in attribute_to_remove for edge in out_edges):
                    nodes_list.remove(node)

    def _remove_maps_without_input_connectors(self, nodes_list: List[nodes.Node], state: SDFGState) -> None:
        """
        Remove maps that don't have any input connectors from the nodes_list.
        These are maps that won't have an output in the backward pass and thus can be skipped from the reversal process.
        Note that we do not remove the AccessNode that the no-input map writes to
        This is because we might need to zero out the gradient of this node
        If no zeroing out is necessary, the node will be removed in the reverse_subgraph function clean up at the end
        """
        for node in nodes_list[:]:  # Iterate over a copy of the list to avoid modification issues
            if isinstance(node, nodes.MapEntry) and len(node.in_connectors) == 0:
                nodes_list.remove(node)
                # Remove the MapExit and everything in between
                # Get the equivalent map exit for the map entry
                map_exit = state.exit_node(node)
                nodes_list.remove(map_exit)

                # Get all the nodes between the map entry and exit
                for state_node in state.nodes():
                    # Check the scope of the node if it is within the map
                    if state_node in state.scope_dict() and state.scope_dict(
                    )[state_node] == node and state_node in nodes_list:
                        nodes_list.remove(state_node)

    def _find_subgraph_to_differentiate(self) -> None:
        """ 
        Determine which nodes we need to reverse; this forms the subgraph we will differentiate:
        we do a reverse BFS from the target output node. 
        In the case where a state is within a loop, this may result in different subgraphs
        depending on the loop iteration.

        To calculate the gradients for a node x in ``required_gradients``, we need to sum up the gradient
        contributions from every node y where x is used as an input.
        """
        backward_nodes: set[nodes.Node] = set()
        given_gradients_all_states = set(self.given_gradients_data)

        # TODO: this is experimental:
        required_gradients_all_states = {n for n in self.required_gradients_data}
        given_gradients_all_states = given_gradients_all_states | required_gradients_all_states

        # Do the backward BFS iterativly
        for state in reversed(self.state_order):
            state_given_gradients: List[nodes.AccessNode] = []

            for node in state:
                if isinstance(node, nodes.AccessNode) and node.data in given_gradients_all_states:
                    state_given_gradients.append(node)

            backward_nodes = {n for e in state.edge_bfs(state_given_gradients, reverse=True) for n in [e.src, e.dst]}
            nodes_list = list(backward_nodes)

            if state.label == "call_40" and "cavity" in self.sdfg.name:
                # Find access nodes p and b
                p_and_b_nodes = []
                for node in state:
                    if isinstance(node, nodes.AccessNode) and node.data == "p":
                        p_and_b_nodes.append(node)
                    if isinstance(node, nodes.AccessNode) and node.data == "b":
                        p_and_b_nodes.append(node)

                # Do a forward bfs from p and b
                fwd_nodes = {n for e in state.edge_bfs(p_and_b_nodes) for n in [e.src, e.dst]}
                nodes_list = list(backward_nodes.intersection(fwd_nodes))

            if state.label == "BinOp_65" or state.label == "call_58" and "cavity" in self.sdfg.name:
                # Find access nodes p and b
                nodes_care = ["vn", "un", "p", "u", "v", "b"]
                p_and_b_nodes = []
                for node in state:
                    if isinstance(node, nodes.AccessNode) and node.data in nodes_care:
                        p_and_b_nodes.append(node)

                # Do a forward bfs from p and b
                fwd_nodes = {n for e in state.edge_bfs(p_and_b_nodes) for n in [e.src, e.dst]}
                nodes_list = list(backward_nodes.intersection(fwd_nodes))

            # Clean up unwanted elements
            self._remove_maps_without_input_connectors(nodes_list, state)
            self._remove_onnx_attribute_accessnodes(nodes_list, state)

            state_subgraph = dstate.StateSubgraphView(state, nodes_list)

            state_subgraph = self._add_missing_nested_sdfg_connectors_to_view(state=state,
                                                                              state_subgraph=state_subgraph,
                                                                              view_nodes=nodes_list)

            # Add mapping
            self.states_view_map[state] = state_subgraph

            # In the case where this state is within a for loop
            within_loop, _ = self._state_within_loop(state)
            if within_loop:
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
                    for e in state.edge_bfs(state_given_gradients, reverse=True)
                    for n in [e.src, e.dst]
                }

                view_nodes = list(backward_nodes)
                if state.label == "call_40" and "cavity" in self.sdfg.name:
                    # Find access nodes p and b
                    p_and_b_nodes = []
                    for node in state:
                        if isinstance(node, nodes.AccessNode) and node.data == "p":
                            p_and_b_nodes.append(node)
                        if isinstance(node, nodes.AccessNode) and node.data == "b":
                            p_and_b_nodes.append(node)

                    # Do a forward bfs from p and b
                    fwd_nodes = {n for e in state.edge_bfs(p_and_b_nodes) for n in [e.src, e.dst]}
                    view_nodes = list(backward_nodes.intersection(fwd_nodes))

                if state.label == "BinOp_65" or state.label == "call_58" and "cavity" in self.sdfg.name:
                    # Find access nodes p and b
                    nodes_care = ["vn", "un", "p", "u", "v", "b"]
                    p_and_b_nodes = []
                    for node in state:
                        if isinstance(node, nodes.AccessNode) and node.data in nodes_care:
                            p_and_b_nodes.append(node)

                    # Do a forward bfs from p and b
                    fwd_nodes = {n for e in state.edge_bfs(p_and_b_nodes) for n in [e.src, e.dst]}
                    view_nodes = list(backward_nodes.intersection(fwd_nodes))
                self._remove_maps_without_input_connectors(nodes_list, state)

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

        # TODO: remove hack
        if "fdtd_2d" in self.sdfg.name or "jacobi" in self.sdfg.name or "seidel2d" in self.sdfg.name:
            cloned_datadesc.lifetime = dtypes.AllocationLifetime.Persistent
        # TODO: sus
        self.backward_grad_arrays[grad_name] = cloned_datadesc
        self.backward_sdfg.arrays[grad_name] = copy.deepcopy(cloned_datadesc)

    def _state_within_loop(self, forward_state: SDFGState) -> Tuple[bool, LoopRegion]:
        """
        Check if this state will be executed several times within a loop.
        We check if any of the parents of this state is a loop region.
        """
        parent = forward_state.parent_graph
        while parent is not None:
            if isinstance(parent, LoopRegion):
                return True, parent
            parent = parent.parent_graph
        return False, None

    def _get_all_enclosing_loops(self, forward_state: SDFGState) -> List[LoopRegion]:
        """
        Check if this state will be executed several times within a loop.
        We check if any of the parents of this state is a loop region.
        """
        all_loops = []
        parent = forward_state.parent_graph
        while parent is not None:
            if isinstance(parent, LoopRegion):
                all_loops.append(parent)
            parent = parent.parent_graph
        return all_loops

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

        # TODO: need more generalized solution for functions in the loop bounds
        if "floor" not in conditional:
            # There is no function call in the statement, remove parenthesis
            end = end.replace("(", "")
            end = end.replace(")", "")
            end = end.replace(" ", "")
        else:
            if expression_to_remove.startswith("(") and not expression_to_remove.endswith(")") and expression.endswith(
                    ")"):
                # Remove extra parenthesis
                end = end[:-1]

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

                # Remove from the backward SDFG
                self.backward_sdfg.remove_node(bwd_node)

                # Add it to the loop region
                reversed_loop.add_node(bwd_node)

                # Also add loop states if any
                if node in self.reversed_loop_states_map:
                    # Get the backward_node
                    bwd_node = self.reversed_loop_states_map[node]

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
        forward_src = forward_edge.src
        forward_dst = forward_edge.dst

        if isinstance(forward_src, LoopRegion):
            fwd_src_is_loop = True
            assert forward_src in self.reversed_loops_map
        else:
            fwd_src_is_loop = False
            assert forward_src in self.reversed_states_map

        if isinstance(forward_dst, LoopRegion):
            fwd_dst_is_loop = True
            assert forward_dst in self.reversed_loops_map
        else:
            fwd_dst_is_loop = False
            assert forward_dst in self.reversed_states_map

        # Note that the source will become the destination
        backward_dst = self.reversed_states_map[forward_src] if not fwd_src_is_loop else self.reversed_loops_map[
            forward_src]
        backward_src = self.reversed_states_map[forward_dst] if not fwd_dst_is_loop else self.reversed_loops_map[
            forward_dst]

        # Each one of these in edges needs to have an equivelent
        # out edge in the backward part of the SDFG
        bwd_edge = None
        connection_state = backward_dst

        # Find the equivelent edge in the backward SDFG
        for b_edge in connection_state.parent_graph.in_edges(connection_state):
            if b_edge.src == backward_src:
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
        assert forward_state_src in self.reversed_states_map or forward_state_src in self.reversed_loops_map
        assert forward_state_dst in self.reversed_states_map or forward_state_src in self.reversed_loops_map

        # Note that the src will become the destination
        backward_state_dst = self.reversed_states_map[
            forward_state_src] if forward_state_src in self.reversed_states_map else self.reversed_loops_map[
                forward_state_src]
        backward_state_src = self.reversed_states_map[
            forward_state_dst] if forward_state_dst in self.reversed_states_map else self.reversed_loops_map[
                forward_state_dst]

        # Each one of these in edges needs to have an equivelent
        # out edge in the backward part of the SDFG
        bwd_edge = None
        connection_state = backward_state_dst

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
        for state_view in self.states_view_map.values():
            for node, parent_graph in state_view.all_nodes_recursive():
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

    def _connect_conditional_map_exist(self, forward_state: SDFGState, backward_state: SDFGState,
                                       backward_map_exit: nodes.MapExit, fwd_tasklet: nodes.Tasklet):
        """
        This function connects the map exit of a conditional tasklet to a new access node which will zero out the gradient
        # TODO: in the generalization of this a wcr sum should be added in case we are not zeroing out the gradients
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

        # We need to get the map entry that starts the conditional block
        # First get the conditional tasklet
        conditional_block = self._extract_conditional_array_assignement_block(
            forward_state=forward_state, tasklet_node=fwd_tasklet, subgraph=self.states_view_map[forward_state])
        # Get the map entry of the conditional bloc
        map_entries = [n for n in conditional_block if isinstance(n, nodes.MapEntry)]

        if len(map_entries) != 1:
            raise AutoDiffException(
                f"Expected a single MapEntry node in the conditional block, found {len(map_entries)}")
        else:
            map_entry = map_entries[0]

        # Add the new access node to a dictionary in case it needs to be connected
        self.conditional_block_entry[map_entry] = replicated_bwd_target_an

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
                for e in forward_state.edge_bfs(bools_an, reverse=True)
                for n in [e.src, e.dst]
            }

            # if any of the nodes in the block are required for gradient tracking
            nodes_to_keep_tracking: set[nodes.Node] = self._get_gradient_nodes_to_track(
                forward_state=forward_state, block_nodes=conditional_assingement_block_nodes, subgraph=subgraph)
            for node in nodes_to_keep_tracking:
                # we get the reverse bfs of this node and remove it from block nodes to avoid skipping these nodes
                node_subgraph = {n for e in forward_state.edge_bfs(node, reverse=True) for n in [e.src, e.dst]}

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
        gradinet_nodes = [n for n in self.required_gradients_data]
        gradinet_nodes += [n for n in self.given_gradients_data]

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

    def _reverse_subgraph(self, forward_state: SDFGState, backward_state: SDFGState,
                          subgraph: dstate.StateSubgraphView):
        """ Reverse a given subgraph. All nodes in the subgraph will be reversed. """
        from dace.libraries.onnx.nodes import ONNXSum  # avoid import loop

        # Condiitonal assignement nodes
        conditional_assignement_nodes: List[nodes.Node] = []
        new_backward_state = None

        # A reversed topological sort is a topological sort on the reverse graph
        for node in reversed(list(dace_utils.dfs_topological_sort(subgraph, subgraph.source_nodes()))):

            try:
                # If this node is a part of the conditional assignement block, we skip it
                if node in conditional_assignement_nodes:
                    continue

                # Output names on the forward node
                # (for which the gradient will be connected as an input on the reverse node)
                given_gradients = [
                    edge.src_conn for edge in subgraph.out_edges(node)
                    if ad_utils.path_src_node_in_subgraph(edge, subgraph)
                ]

                # Input names on the forward node that gradients should be generated for
                # note that the edge for the conditional is not included
                required_gradients = [
                    edge.dst_conn for edge in subgraph.in_edges(node)
                    if ad_utils.path_src_node_in_subgraph(edge, subgraph)
                    and self.sdfg.arrays[edge.data.data].dtype != dace.bool
                ]

                reversed_node, backward_result = self._get_reverse_node(forward_state, backward_state, node,
                                                                        given_gradients, required_gradients)

                self.reverse_map[node] = reversed_node
                self.result_map[node] = backward_result

                # Connect the required inputs of the reverse node:
                # the gradients ...
                self._connect_given_gradients(forward_state=forward_state,
                                              backward_state=backward_state,
                                              subgraph=subgraph,
                                              forward_node=node)

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

                        # Add a wcr to all the writes to the AccessNode
                        for edge in backward_state.in_edges(reversed_node):
                            # Add wcr to the memlet
                            for tree_edge in backward_state.memlet_tree(edge):
                                tree_edge.data.wcr = "lambda x, y: x + y"

                # If this node is a tasklet with a condition, we add some modification to the backward state
                elif (isinstance(node, nodes.Tasklet)
                      and self._conditional_tasklet(node)) or (isinstance(node, nodes.NestedSDFG)
                                                               and self._conditional_nested_sdfg(forward_state, node)):
                    # extract the conditional assignement block or fail if this is an unexpected structure
                    conditional_block = self._extract_conditional_array_assignement_block(forward_state=forward_state,
                                                                                          tasklet_node=node,
                                                                                          subgraph=subgraph)

                    # add these nodes to be skipped in the future
                    conditional_assignement_nodes.extend(conditional_block)

                # If the node is an AccessNode and it is being overwritten in the forward pass,
                # we need to zero-out the gradients of the overwritten values
                if isinstance(node, nodes.AccessNode):
                    # Check if there is an incoming edge to this node
                    incoming_edges = forward_state.in_edges(node)

                    # If there is an incoming edge, we need to zero-out the gradient
                    for edge in incoming_edges:

                        # Check, if possible, if the written subset is not zero
                        write_size = edge.data.subset.num_elements()

                        # Check if the node doesn't have a wcr
                        # If it does, this is not an overwrite and the gradients should not be cleared
                        has_wcr = edge.data.wcr is not None

                        # Check if the edge is dynamic, this means not all values are overwritten
                        # We will skip zeroeing out the gradient in this case
                        if edge.data.dynamic:
                            Warning("Dynamic memlets are not fully supported in the reverse pass. "
                                    "The gradient of the overwritten values may not be zeroed out.")
                        if not has_wcr and not edge.data.dynamic:
                            # Determine if we need to zero out the gradient
                            zero_out = not (isinstance(write_size, int) and write_size == 0)

                            # We need to zero out the same memlet accesses in the backward pass
                            if zero_out:
                                self._zero_out_gradient(forward_state=forward_state,
                                                        forward_node=node,
                                                        memlet=edge.data)

                # Clean up of isolated nodes
                # We will have an isolated node if it is not connected to any other node in the state view
                # And it has not been cleared out if it is an AccessNode
                # Isolated nodes should only appear from clearing out gradients
                # Check if this is an isolated node and remove it if it is
                if backward_state.out_degree(reversed_node) == 0 and backward_state.in_degree(reversed_node) == 0:
                    if isinstance(node, nodes.AccessNode) and node not in self.zeroed_out:
                        backward_state.remove_node(reversed_node)

            except AutoDiffException as e:
                raise AutoDiffException("Failed at node {}: {}".format(node, str(e))) from e

    def _set_wcr_if_needed(self, backward_state: SDFGState, backward_node: nodes.Node, edge: dstate.MultiConnectorEdge):
        """
        If this Access node represents a gradient that has already been used in other places.
        We want to accumulat the gradients and not overwrite them.
        """

        # Check if the forward node is an AccessNode
        if not isinstance(backward_node, nodes.AccessNode):
            return

        # Otherwise, we add up the gradients, not overwrite them
        for tree_edge in backward_state.memlet_tree(edge):
            tree_edge.data.wcr = "lambda x, y: x + y"

    def _connect_given_gradients(self, forward_state: SDFGState, backward_state: SDFGState,
                                 subgraph: dstate.StateSubgraphView, forward_node: nodes.Node) -> SDFGState:
        """ 
        Connect the gradients of the outputs of forward_node as inputs to the corresponding reverse node. 
        """
        new_backward_state = None
        # First, create the data descriptot if this is an access node and it hasn't been added before
        if isinstance(forward_node, nodes.AccessNode):
            grad_name = self.array_grad_name(forward_node.data)
            if grad_name not in self.backward_sdfg.arrays:
                # This grad hasn't been written before: initialize it
                self._add_gradient_data_descriptor(forward_node.data)

        for edge in subgraph.out_edges(forward_node):
            if not ad_utils.path_src_node_in_subgraph(edge, subgraph) or edge.dst not in self.reverse_map:
                if edge.dst in self.conditional_block_entry:
                    backward_node = self.reverse_map[edge.src]
                    assert isinstance(edge.dst, nodes.MapEntry)
                    conditional_zero_out_an = self.conditional_block_entry[edge.dst]
                    # Add an empty edge to skip the conditional block
                    backward_state.add_edge(conditional_zero_out_an, None, backward_node, None, Memlet())
                # skip connecting edges for which we don't need to generate grads.
                continue

            # Skip connecting boolean edges
            if self.sdfg.arrays[edge.data.data].dtype == dace.bool:
                # we also need to remove this connector otherwise it will be dangeling
                backward_node = self.reverse_map[edge.src]
                if not (isinstance(backward_node, nodes.MapEntry) or isinstance(backward_node, nodes.MapExit)):
                    # If this is not a map entry or exit, the boolean gradients will not be added
                    # No need to remove the connector in this case
                    continue

                conn_to_remove = ad_utils.invert_map_connector(edge.src_conn)
                assert conn_to_remove in backward_node.in_connectors
                assert backward_node.remove_in_connector(conn_to_remove)
                if len(backward_node.in_connectors) == 0:
                    self._connect_conditional_map_exist(forward_state=forward_state,
                                                        backward_state=backward_state,
                                                        backward_map_exit=backward_node,
                                                        fwd_tasklet=edge.dst)
                continue

            src_node, output_conn, dest_node, input_conn, fwd_memlet = edge

            memlet = copy.deepcopy(fwd_memlet)

            # Remove the WCR since these are now read edges
            memlet.wcr = None

            grad_name = self.array_grad_name(memlet.data)
            if grad_name not in self.backward_sdfg.arrays:
                # This grad hasn't been written before: initialize it
                self._add_gradient_data_descriptor(memlet.data)

            # We should not rely on the memlet data because that depends on the subset and other subset attibutes
            # If this is an access node, and the memlet data is not the same as the AN data
            memlet.data = grad_name

            # Check of the values have been zeroed out
            backward_dst_node = self.reverse_map[dest_node]
            if backward_dst_node in self.zeroed_out:
                # The values will be zeroed out in the backward node
                # We use the transient array instead
                copied_zeroed_nodes = self.zeroed_out[backward_dst_node]
                if len(copied_zeroed_nodes) == 1:
                    backward_dst_node = copied_zeroed_nodes[0]
                else:
                    for node in copied_zeroed_nodes:
                        # Get the memlet to this node
                        zero_in_dege = backward_state.in_edges(node)
                        assert len(zero_in_dege) == 1
                        zeroed_memlet = zero_in_dege[0].data
                        if zeroed_memlet.subset == edge.data.subset:
                            backward_dst_node = node
                            break

                memlet.data = backward_dst_node.data

                # We also need to Add an empty edge from the cleared node to where the data will be used
                tmp_clear_node_out_edges = backward_state.out_edges(backward_dst_node)
                for e in tmp_clear_node_out_edges:
                    if e.data.data is None and e.data.subset is None and e.data.other_subset is None:
                        clearing_map_entry = e.dst
                        assert isinstance(clearing_map_entry, nodes.MapEntry)
                        clearing_map_exit = backward_state.exit_node(clearing_map_entry)
                        assert isinstance(clearing_map_exit, nodes.MapExit)
                        # Check that this only has a single output edge and get the destination
                        assert backward_state.out_degree(clearing_map_exit) == 1
                        cleared_out_node = backward_state.out_edges(clearing_map_exit)[0].dst
                backward_node = self.reverse_map[forward_node]
                backward_state.add_edge(cleared_out_node, None, backward_node, None, dace.Memlet())

                # If this is a connection between two access nodes we need to flip the memlet subsets
                if isinstance(forward_node, nodes.AccessNode):
                    # Special case for when the two access nodes are the same
                    if forward_node.data == dest_node.data and fwd_memlet.other_subset is not None:
                        new_memlet = dace.Memlet(data=self.reverse_map[forward_node].data,
                                                 subset=fwd_memlet.other_subset,
                                                 other_subset=fwd_memlet.subset)
                    else:
                        new_memlet = dace.Memlet(data=self.reverse_map[forward_node].data,
                                                 subset=fwd_memlet.subset
                                                 if fwd_memlet.data == forward_node.data else fwd_memlet.other_subset,
                                                 other_subset=fwd_memlet.other_subset
                                                 if fwd_memlet.data == forward_node.data else fwd_memlet.subset)
                    memlet = new_memlet
            if input_conn not in self.result_map[dest_node].required_grad_names:
                continue
            new_edge = backward_state.add_edge(
                backward_dst_node,
                self._lookup_required_grad_name(dest_node, input_conn),
                self.reverse_map[forward_node],
                self._lookup_given_grad_name(forward_node, output_conn),
                memlet,
            )

            # Change the access data in the memlet path if it has been zeroed out
            # Calling the memlet path while reversing will raise an error
            # Because the map has not been completely added for the backward state yet
            # We also don't need to do anything for an AccessNode -> AccessNode connection
            if (not isinstance(forward_node,
                               (nodes.MapExit, nodes.MapEntry))) and not (isinstance(forward_node, nodes.AccessNode)
                                                                          and isinstance(dest_node, nodes.AccessNode)):
                # Check if we can call the memlet path on new_edge safely
                path = backward_state.memlet_path(new_edge)

                # Get the source access node in the path
                source_access_node = list(path)[0].src
                if isinstance(source_access_node, nodes.AccessNode):
                    # Check if this is a zeroed out node
                    in_values = any(source_access_node in values for values in self.zeroed_out.values())
                    if source_access_node.data != memlet.data and in_values:
                        memlet.data = source_access_node.data
            self._set_wcr_if_needed(backward_state=backward_state,
                                    backward_node=self.reverse_map[forward_node],
                                    edge=new_edge)

        return new_backward_state

    def _connect_forward_accessnode_not_overwritten(self,
                                                    forward_state: SDFGState,
                                                    backward_state: SDFGState,
                                                    forward_node: nodes.AccessNode,
                                                    target_node: nodes.Node,
                                                    starting_edge: dgraph.MultiConnectorEdge,
                                                    replicated_node: nodes.AccessNode = None):
        """
        Replicate and connect the forward AccessNode to the requesting node in the backward pass.
        Because the AccessNode has not been overwritten, we just need to create the same connection
        in the backward pass.
        """

        # First, replicate the AccessNode and add it to the backward pass
        # If it has not already been replicated and passed as a parameter
        if replicated_node is None:
            replicated_node = copy.deepcopy(forward_node)
            backward_state.add_node(replicated_node)
            if self.separate_sdfgs:
                # Need to copy over the descriptor from the forward pass
                data_name = replicated_node.data
                data_desc = copy.deepcopy(forward_node.desc(self.sdfg))
                data_desc.transient = False
                if data_name not in self.backward_sdfg.arrays:
                    self.backward_sdfg.add_datadesc(data_name, data_desc)

                # We also need to forward this array
                if data_name not in self.backward_input_arrays:
                    # If the data is needed inside a NestedSDFG
                    # This will make sure the added array is correcyly forwarded
                    # and an in connector to the NestedSDFG is added
                    self.backward_input_arrays[data_name] = data_desc

        # We replicate the excat link between this forward access node and the target node
        # Get all the edges in the path
        all_edges_inbetween = ad_utils.get_all_path_edges(state=forward_state,
                                                          source=forward_node,
                                                          starting_edge=starting_edge)

        # A dictionary to keep track of temporary nodes in the path
        replicated_tmp_nodes = []

        # For each edge in the path
        for edge in all_edges_inbetween:
            src, src_conn, dst, dst_conn, data = edge
            bwd_src, bwd_src_conn, bwd_dst, bwd_dst_conn, bwd_data = src, src_conn, dst, dst_conn, copy.deepcopy(data)

            # If the destination is a map entry,
            if isinstance(dst, nodes.MapEntry):
                # We need to get the corresponding map entry in the backward pass.
                bwd_dst = self._find_backward_entry_node_for_map_entry(backward_state=backward_state, entry_node=dst)
                # Add the dst connector to the map
                assert bwd_dst.add_in_connector(bwd_dst_conn)

            # If the destination is a map entry,
            if isinstance(src, nodes.MapEntry):
                # We need to get the corresponding map entry in the backward pass.
                bwd_src = self._find_backward_entry_node_for_map_entry(backward_state=backward_state, entry_node=src)
                # Add the src connector to the map
                assert bwd_src.add_out_connector(bwd_src_conn)

            if src is forward_node:
                # If this is the node we replicated
                bwd_src = replicated_node
            elif isinstance(src, nodes.AccessNode):
                # This is a temporary AccessNodes
                # we should have already seen and replicated this
                assert src in replicated_tmp_nodes
                bwd_src = replicated_tmp_nodes[src]

            if dst is target_node:
                # If this is the final connection node
                bwd_dst = self.reverse_map[dst]
            elif isinstance(dst, nodes.AccessNode):
                # This is a temporary AccessNodes
                # we want to replicate and add it to the path
                bwd_dst = copy.deepcopy(dst)
                backward_state.add_node(bwd_dst)
                replicated_tmp_nodes[dst] = bwd_dst

            # Modify the data in the memlet in case the array is replicated outside of the function
            bwd_data.data = replicated_node.data

            # Add the edge to the backward state
            backward_state.add_edge(bwd_src, bwd_src_conn, bwd_dst, bwd_dst_conn, bwd_data)

        # If we just connected a view, we need to remove the view in connector
        data_desc = self.sdfg.arrays[forward_node.data]
        if isinstance(forward_node, nodes.AccessNode) and isinstance(data_desc, dt.View):
            if self.separate_sdfgs:
                # Remove the view connector
                assert replicated_node.remove_in_connector("views")
            else:
                # if this is a view, we need to connect it to the AccessNode it is viewing
                edge_src_in_edge = forward_state.in_edges(forward_node)

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
                backward_state.add_edge(replicated_view, None, replicated_node, "views", view_memlet)

    def _connect_forward_accessnode(self, forward_state: SDFGState, backward_state: SDFGState,
                                    forward_node: nodes.AccessNode, target_node: nodes.Node,
                                    starting_edge: dgraph.MultiConnectorEdge, recomputation_nsdfg, strategy: str):
        """
        We need to forward an array from the forward pass to the backward pass.
        To do this we first check if this array has been overwritten or not.
        If the array has not been overwritten, we just need to replicate it
        in the backward pass and then forward it. 
        If the array has been overwritten, we pick a strategy for this AN:
            - Store strategy: 
                - We modify the forward pass to save the values in a new array
                - Connect this new array to the node in the backward pass
            - Recomputation:
                - Add the recomputation as a NestedSDFG
                - Connect the output of the NestedSDFG to the node in the backward pass
        """

        # First, we check if the node has been overwritten
        overwritten, recomputable = self._check_node_overwrite(forward_state=forward_state, node=forward_node)

        # Boolean indicating wether we should fall back to storing
        fallback = False
        if strategy == "recompute" and recomputable:
            try:
                print(f"Recomputing {forward_node.data}")
                if recomputation_nsdfg is None:
                    recomputation_nsdfg = self._get_recomputation_nsdfg(forward_state, target_an=forward_node)
                self._resolve_overwrite_with_recomputation(recomputation_nsdfg=recomputation_nsdfg,
                                                           forward_state=forward_state,
                                                           backward_state=backward_state,
                                                           target_an=forward_node,
                                                           target_node=target_node,
                                                           starting_edge=starting_edge)
            except:
                # If anything goes bad, print a warning and fall back to storing
                print(f"AutoDiff Warning: failed to recompute {forward_node.data}. Falling back to storing")
                fallback = True

        if strategy == "store" or (strategy == "recompute" and not recomputable) or fallback:
            # We store if:
            #   - This was the specified strategy
            #   - We tried to recompute a program input
            #   - We tried to recompute something that didn't work and we're falling back to storing

            # The data has been overwritten
            if not overwritten:
                # We still have access to this data
                self._connect_forward_accessnode_not_overwritten(forward_state, backward_state, forward_node,
                                                                 target_node, starting_edge)
                return

            self._resolve_overwrite_with_store(forward_state=forward_state,
                                               backward_state=backward_state,
                                               forward_node=forward_node,
                                               target_node=target_node,
                                               starting_edge=starting_edge)

    def _resolve_overwrite_with_recomputation(
        self,
        recomputation_nsdfg: nodes.NestedSDFG,
        forward_state: SDFGState,
        backward_state: SDFGState,
        target_an: nodes.AccessNode,
        target_node: nodes.Node,
        starting_edge: dstate.MultiConnectorEdge,
    ):
        """
        """

        # Add the nsdfg where it is required
        self._connect_recomputation_nsdfg(forward_state=forward_state,
                                          backward_state=backward_state,
                                          nsdfg=recomputation_nsdfg,
                                          target_an=target_an,
                                          target_node=target_node,
                                          starting_edge=starting_edge)

    def _prune_decendants_recomputation_nsdfg(self, forward_state: SDFGState, target_an: nodes.AccessNode,
                                              nsdfg: nodes.NestedSDFG):
        """
        1: From this Nested-SDFG, we remove everything that will be executed after the target access node to be recomputed
        2: Prune the unnecessary computation inside the forward state 
            Note: this is even necessary sometimes since the output could be overwritten in the same state
        """

        # 1
        # Get the states order for the nested_sdfg
        states_order: List[SDFGState] = ad_utils.get_state_topological_order(nsdfg.sdfg)
        state_index = states_order.index(forward_state)
        decendant_states: List[SDFGState] = states_order[state_index:]
        assert decendant_states.pop(0) == forward_state

        # Check if the target state is within a loop
        target_within_loop, target_loop = self._state_within_loop(forward_state)

        # We will save the states that are within the same loop because they require special treatement
        same_loop_states: List[SDFGState] = []
        for state in decendant_states:
            # We want to avoid removing the decendant states that are inside the same loop region
            if target_within_loop:
                decendant_within_loop, decendant_loop = self._state_within_loop(state)
                if decendant_within_loop and decendant_loop == target_loop:
                    # If the state is within the same loop, we don't remove it
                    same_loop_states.add(state)
                    continue

            # Remove the state from the nested_sdfg
            parent = state.parent_graph
            parent.remove_node(state)

        # Cleanup empty LoopRegions if any
        for node in nsdfg.sdfg.all_nodes_recursive():
            if isinstance(node, LoopRegion) and len(node.nodes()) == 0:
                parent = node.parent_graph
                parent.remove_node(node)

        # 2
        # Within the same state
        if target_within_loop:
            # For now we keep all of the computation inside the loop
            # TODO: if there is an overwrite to the same array in the decendnat computation
            # We need to make a special case for the last iteration of the loop where the
            # else branch of this if is executed and a spacial version of the loop is added
            pass
        else:
            # If the target state is not within a loop
            # We remove all the decendant computation from the graph

            # Do a reverse bfs to get all the necessary computation
            backward_nodes = {n for e in forward_state.edge_bfs(target_an, reverse=True) for n in [e.src, e.dst]}

            # Remove everything else
            decendant_nodes = set(forward_state.nodes()) - backward_nodes

            for node in decendant_nodes:
                if node is not target_an:
                    forward_state.remove_node(node)

    def _prune_acendant_recomputation_nsdfg(self, forward_state: SDFGState, target_an: nodes.AccessNode,
                                            nsdfg: nodes.NestedSDFG):
        """
        Removes the unnecesary computation done before the main computation.
        To do this we wil go through the graph in reverse bfs and only keep the computation that is contributing to this array
        The pruning will be done on a state-per-state basis
        """

    def _prune_recomputation_sdfg(self, forward_state: SDFGState, target_an: nodes.AccessNode, nsdfg: nodes.NestedSDFG):
        """
        1: From this Nested-SDFG, we remove everything that will be executed after the target access node to be recomputed
        2: Prune the unnecessary computation inside the forward state 
            Note: this is even necessary sometimes since the output could be overwritten in the same state
        3: From the target access node, we go backward in the graph and see what elements are required to get this array
        """

        # 1 and 2
        self._prune_decendants_recomputation_nsdfg(forward_state=forward_state, target_an=target_an, nsdfg=nsdfg)

        # 3
        self._prune_acendant_recomputation_nsdfg(forward_state=forward_state, target_an=target_an, nsdfg=nsdfg)

    def _rename_desciptors_for_recomputation_nsdfg(self, nsdfg: nodes.NestedSDFG):
        """
        """
        # Get all the nodes to renmae in the NestedSDFG
        to_rename = []
        for inp in nsdfg.in_connectors:
            for node, parent in nsdfg.sdfg.all_nodes_recursive():
                if isinstance(node, nodes.AccessNode) and node.data == inp and parent.in_degree(node) > 0:
                    # This is an input that will be written to in the SDFG we need to rename it
                    to_rename.append(inp)
                    break

        if len(to_rename) > 0:
            # Add a new state to copy the data at the start of the SDFG
            initi_state = nsdfg.sdfg.add_state_before(nsdfg.sdfg.start_state, label=f"init_{nsdfg.label}")

        # Rename the descriptors in the nested SDFG in addition to the in connector
        for name in to_rename:
            # Create a new array
            new_name = f"recomputation_{name}"

            # Change the accessnodes in the NestedSDFG
            for node, parent in nsdfg.sdfg.all_nodes_recursive():
                if isinstance(node, nodes.AccessNode) and node.data == name:
                    node.data = new_name

            # Change the memlets in the SDFG
            for edge, parent in nsdfg.sdfg.all_edges_recursive():
                # Skip interstate edges
                if isinstance(edge.data, dace.InterstateEdge):
                    continue

                if edge.data.data == name:
                    edge.data.data = new_name

            # Add the desciptor
            old_desc = nsdfg.sdfg.arrays[name]
            new_desc = copy.deepcopy(old_desc)

            # Check if this is the output of the recomputation block
            if name not in nsdfg.out_connectors:
                new_desc.transient = True
            else:
                new_desc.transient = False

            nsdfg.sdfg.add_datadesc(name=new_name, datadesc=new_desc)

            # Add a copy operation between the input node and the new descriptor
            input_node = nodes.AccessNode(name)
            new_node = nodes.AccessNode(new_name)
            initi_state.add_node(input_node)
            initi_state.add_node(new_node)

            # Add memory copy edge
            initi_state.add_edge(input_node, None, new_node, None, self.sdfg.make_array_memlet(name))

            # Change the output if necessary
            if name in nsdfg.out_connectors:
                nsdfg.remove_out_connector(name)
                nsdfg.add_out_connector(new_name)

    def _get_recomputation_nsdfg(self, forward_state: SDFGState, target_an: nodes.AccessNode) -> nodes.NestedSDFG:
        """
        Given an AccessNode for data that needs to be forwarded from the forward pass to the backward pass,
        Return a nested SDFG that recomputes this data from input data.
        """
        # TODO: Get the correct recomputation nsdfg for loops
        nsdfg_label = "recomputation_nsdfg_" + target_an.data

        # Initially, we will replicate the whole SDFG into a Nested-SDFG and connect it
        nsdfg = nodes.NestedSDFG(label=nsdfg_label,
                                 sdfg=copy.deepcopy(self.original_forward_sdfg),
                                 inputs=self.sdfg.arg_names,
                                 outputs=[target_an.data])

        # We need to make sure the output inside the NestedSDFG is not a transient (anymore)
        nsdfg.sdfg.arrays[target_an.data].transient = False

        # Find the same target node and state in the nsdfg
        nsdfg_forward_state: SDFGState = None
        nb_occurances = 0
        for state in nsdfg.sdfg.states():
            if state.label == forward_state.label:
                nsdfg_forward_state = state
                nb_occurances += 1

        # Sanity check
        assert nb_occurances == 1
        assert nsdfg_forward_state

        # Find the target AccessNode within the state
        nsdfg_target_node: nodes.AccessNode = None
        nb_occurances = 0
        for node in nsdfg_forward_state.nodes():
            if isinstance(node, nodes.AccessNode) and node.data == target_an.data and nsdfg_forward_state.node_id(
                    node) == forward_state.node_id(target_an):
                nsdfg_target_node = node
                nb_occurances += 1

        # Sanity check
        assert nb_occurances == 1
        assert nsdfg_target_node

        self._prune_recomputation_sdfg(nsdfg=nsdfg, forward_state=nsdfg_forward_state, target_an=nsdfg_target_node)

        # Change descriptors if the inputs are written to
        self._rename_desciptors_for_recomputation_nsdfg(nsdfg=nsdfg)

        return nsdfg

    def _connect_recomputation_nsdfg(self, forward_state: SDFGState, backward_state: SDFGState,
                                     target_an: nodes.AccessNode, target_node: nodes.Node, nsdfg: nodes.NestedSDFG,
                                     starting_edge: dstate.MultiConnectorEdge):
        """
        
        """
        initialization_state = None
        # Connect all the SDFG inputs to the nested SDFG
        # First, add the nested sdfg
        for input in nsdfg.in_connectors.keys():
            # For each argument
            input_name = input if "recomputation_" not in input else input[14:]

            # Get the first instance of this AN in the SDFG
            first_instance = None
            for node, parent in self.sdfg.all_nodes_recursive():
                if isinstance(node, nodes.AccessNode) and node.data == input:
                    first_instance = node
                    first_node_state = parent
                    break

            assert first_instance

            new_an = nodes.AccessNode(input_name)
            backward_state.add_node(new_an)

            # Create a memlet passing all the data to the nested-SDFG
            memlet = self.sdfg.make_array_memlet(input_name)

            # Add the connection to the nested SDFG
            backward_state.add_edge(new_an, None, nsdfg, input, memlet)

        # Write the data to a new access node in the backward state
        # Add a new AccessNode and array to the forward pass
        # First, check if a recomputated array with this name already exists
        if "recomputed_" + target_an.data not in self.backward_sdfg.arrays:
            new_recomp_node_name = "recomputed_" + target_an.data
        else:
            i = 0
            while True:
                if f"recomputed_{i}_" + target_an.data not in self.backward_sdfg.arrays:
                    new_recomp_node_name = f"recomputed_{i}_" + target_an.data
                    break
                i += 1

        # Get the new array shape
        # This will be the shape of the current array
        shape: List[int] = list(self.sdfg.arrays[target_an.data].shape)

        # Add the array descriptor and AccessNode to the forward state
        original_desc = target_an.desc(forward_state)
        new_recomp_node = backward_state.add_array(
            name=new_recomp_node_name,
            shape=shape,
            dtype=original_desc.dtype,
            transient=True,
        )
        new_recomp_node.setzero = True

        # Create a memlet passing all the data to the nested-SDFG
        memlet = self.sdfg.make_array_memlet(new_recomp_node.data)

        nsdfg_out_conn = list(nsdfg.out_connectors.keys())
        assert len(nsdfg_out_conn) == 1
        nsdfg_out_conn = nsdfg_out_conn[0]

        # Connect the output of the NestedSDFG
        backward_state.add_edge(nsdfg, nsdfg_out_conn, new_recomp_node, None, memlet)

        # Connect the new AccessNode to the required computation
        self._connect_forward_accessnode_not_overwritten(forward_state=forward_state,
                                                         backward_state=backward_state,
                                                         forward_node=target_an,
                                                         target_node=target_node,
                                                         starting_edge=starting_edge,
                                                         replicated_node=new_recomp_node)

    def _resolve_overwrite_with_store(self, forward_state: SDFGState, backward_state: SDFGState,
                                      forward_node: nodes.AccessNode, target_node: nodes.Node,
                                      starting_edge: dstate.MultiConnectorEdge):
        """
        Given the AccessNode pointing to the data required by the backward pass,
        We will save the values of this array in a new array and forward it to the backward pass.
        """

        # Modify the forward pass to save the data in a new array
        new_stored_array, memlets = self._store_data(forward_state=forward_state,
                                                     backward_state=backward_state,
                                                     forward_an=forward_node,
                                                     target_node=target_node,
                                                     edge=starting_edge)

        # Check if this data needs to be forwarded through a SDFGs
        if self.separate_sdfgs or ad_utils.within_nested_sdfg(forward_state):
            # We need to make sure the new array is forwarded to the backward SDFG
            if new_stored_array.data not in self.backward_input_arrays:
                # If the data is needed inside a NestedSDFG
                # This will make sure the added array is correctly forwarded
                # and an in connector to the NestedSDFG is added
                data_desc = new_stored_array.desc(forward_state)
                self.backward_input_arrays[new_stored_array.data] = data_desc

        # Connect the new array to the target node
        self._connect_stored_data_to_target(forward_state=forward_state,
                                            backward_state=backward_state,
                                            source_node=new_stored_array,
                                            forward_node=forward_node,
                                            starting_edge=starting_edge,
                                            memlets=memlets,
                                            target_node=target_node)

    def _get_overwrite_resolution_strategy(self) -> Tuple[List[bool], List[nodes.NestedSDFG]]:
        """
        Choose a strategy for resolving overwritten data that we need to forward to the backward passs.
        If the user wants a specific strategy, we use it.
        Otherwise, we evaluate what strategy is best for this specific node.
        """
        strategy_choice: List[bool] = []
        recomputation_nsdfgs: List[nodes.NestedSDFG] = []

        # As preprocessing step,
        # We will store all of the global program inputs,
        # if they are required for the backward pass
        # NOTE: This can be relaxed since if an input is not overwritten
        # if can be recomputed
        to_remove = []
        for i, (forward_state, backward_state, access_node, node, edge) in enumerate(self._forward_data):
            if access_node.data not in self.sdfg.arg_names:
                continue

            # Store the input
            self._connect_forward_accessnode(forward_state, backward_state, access_node, node, edge, None, "store")

            # Remove this element from the list of the data to forward
            to_remove.append(i)

        # Remove elements from the list of data to be forwarded
        self._forward_data = [item for idx, item in enumerate(self._forward_data) if idx not in to_remove]

        if self.strategy == "store_all":
            strategy_choice = ["store"] * len(self._forward_data)

            # A recomputation block is not necessary
            recomputation_nsdfgs = [None] * len(self._forward_data)
        elif self.strategy == "recompute_all":
            strategy_choice = ["recompute"] * len(self._forward_data)

            # We will delay getting the recomputation block for now
            recomputation_nsdfgs = [None] * len(self._forward_data)
        elif self.strategy == "user_defined":
            if self.data_to_recompute is None:
                raise AutoDiffException("The overwrite resolution strategy is User Defined "
                                        "but no recomputation list has been provided."
                                        "Please set the data_to_recompute parameter.")

            for forward_state, backward_state, access_node, node, edge in self._forward_data:

                if access_node.data in self.data_to_recompute:
                    try:
                        nsdfg = self._get_recomputation_nsdfg(forward_state, access_node)
                        choice = "recompute"
                    except:
                        print(f"WARNING! couldn't get the recomputation nested SDFG for {access_node.label}")
                        nsdfg = None
                        choice = "store"
                    recomputation_nsdfgs.append(nsdfg)
                    strategy_choice.append(choice)
                else:
                    # We store everything else
                    recomputation_nsdfgs.append(None)
                    strategy_choice.append("store")
        else:
            raise AutoDiffException("Please specify a valid overwrite resolution strategy."
                                    "Expected either store_all, recompute_all, or user_defined"
                                    f"but got {self.strategy}")
        return strategy_choice, recomputation_nsdfgs

    def _get_accessnode_to_forward(self, forward_state: SDFGState, forward_node: nodes.AccessNode):
        """
        Check if this AccessNode is at the base level of the state. If yes, this is the node we want to connect
        Otherwise, in the case the AN is encolsed by maps, we walk up the maps until we find the source AN.
        """
        scope_dict = forward_state.scope_dict()[forward_node]
        is_base_level = scope_dict is None
        if is_base_level:
            return forward_node
        else:
            # The node is within a map nest
            # It should have an in edge leading to the original AN
            in_edges = forward_state.in_edges(forward_node)
            assert len(in_edges) == 1

            # Get the memlet path and the original AN
            memlet_path = forward_state.memlet_path(in_edges[0])
            original_an = memlet_path[0]
            assert isinstance(original_an, nodes.AccessNode)

            # This should be a base level AN
            assert forward_state.scope_dict()[original_an] is None
            return original_an

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
            # Boolean to decide if the source of this edge needs to be replicated
            replicate_node = False

            # Boolean to decide if the connection to the replicated node is required
            # This is set to False if the connection has already been established
            connect_replicated_node = True
            edge_src = edge.src
            next_required_inputs: Dict[Optional[str], Optional[str]]
            replicated_edge_src: nodes.Node
            replicated_edge_src_conn: str

            if isinstance(edge_src, nodes.MapEntry):
                # In the map case, we need to connect the AN at the start of this memlet path
                memlet_path = state.memlet_path(edge)

                # Get the AccessNode at the start of this path
                starting_edge = memlet_path[0]
                starting_an = starting_edge.src
                assert isinstance(starting_an, nodes.AccessNode)

                # Save the information about the data to be forwarded
                # to call the function to connect this required AccessNode
                # after the reversal
                self._forward_data.append((state, backward_state, starting_an, forward_node, edge))
                # self._connect_forward_accessnode(state, backward_state, starting_an, forward_node, edge)

                # No further recusrive calls are required
                # in this branch; next_required_inputs stays empty
                next_required_inputs = {}

                # Everything will be done in the connect forward accessnode function
                replicate_node = False
                connect_replicated_node = False

            elif isinstance(edge_src, nodes.AccessNode):
                # Get the AccessNode to connect
                an_to_connect = self._get_accessnode_to_forward(state, edge_src)

                # Save the information about the data to be forwarded
                # to call the function to connect this required AccessNode
                # after the reversal
                self._forward_data.append((state, backward_state, an_to_connect, forward_node, edge))
                # self._connect_forward_accessnode(state, backward_state, an_to_connect, forward_node, edge)

                # No further recusrive calls are required
                # in this branch; next_required_inputs stays empty
                next_required_inputs = {}

                # Everything will be done in the connect forward accessnode function
                replicate_node = False
                connect_replicated_node = False

            elif isinstance(edge_src, nodes.Tasklet):

                replicate_node = True
                # In the tasklet case, we need to connect all inputs
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
                        # Remove the view connector
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

                # Add the new edge
                backward_state.add_edge(replicated_edge_src, replicated_edge_src_conn, backward_node,
                                        required_inputs[edge.dst_conn], new_edge_data)

            if next_required_inputs:
                # If there are any required inputs on the new node, we need to
                # recursively call
                self._connect_forward_inputs(state, backward_state, edge.src, replicated_edge_src, next_required_inputs)

    def _connect_stored_data_to_target(self, forward_state: SDFGState, backward_state: SDFGState,
                                       source_node: nodes.AccessNode, forward_node: nodes.AccessNode,
                                       target_node: nodes.Node, memlets: List[Memlet],
                                       starting_edge: dgraph.MultiConnectorEdge):
        """
        Connect the source node to the sink target node (both in the backawrd state) through a set of maps using the parameter memelets.
        We use the forward_sink_edge to track which maps to make this connection through.
        :param source_node: the source node of the new memlet path
        :param sink_node: the sink node of the new memlet path
        :param memlets: the set of memlets to use for the edges in the path
        :param forward_sink_edge: the sink edge connecting the original nodes in the forward state
        """
        # First, if the stored data is not already in the sdfg descriptors, add it
        # This is the case for NestedSDFGs
        if source_node.data not in backward_state.sdfg.arrays:
            # Get the data descriptor from the original sdfg
            data_desc = copy.deepcopy(self.sdfg.arrays[source_node.data])
            data_desc.transient = False  # The stored data will be forwarded
            backward_state.sdfg.add_datadesc(source_node.data, data_desc)

        # Get the memlet path from the forward state
        all_edges = ad_utils.get_all_path_edges(forward_state, forward_node, starting_edge)
        assert len(all_edges) > 0

        # We will iterate and connect parent -> child
        reversed_child_node = self.reverse_map[target_node]
        child_node = reversed_child_node
        child_node_in_connector = all_edges[-1].dst_conn

        # Itetarte through the maps in the path in reverse
        for edge in reversed(all_edges):
            edge_src = edge.src
            if isinstance(edge_src, nodes.MapEntry):
                # Get the correponding map exist
                map_exit = self._find_map_exist_for_map_entry(map_entry=edge_src, state=forward_state)

                # Use the lookup table to get the map entry in the backward state corresponding to this map exist in the forward state
                # Sanity check: this map entry should already exist in the backward state
                assert map_exit in self.reverse_map
                bwd_map_entry = self.reverse_map[map_exit]

                # Get a new connector id
                next_conn = bwd_map_entry.next_connector()

                # Add a new in connector to the mapexit
                parent_node_in_connector = "IN_stored_" + source_node.data + "_" + next_conn
                assert bwd_map_entry.add_in_connector(parent_node_in_connector)

                # Add a new out connector to the mapexit
                pranet_node_out_connector = "OUT_stored_" + source_node.data + "_" + next_conn
                assert bwd_map_entry.add_out_connector(pranet_node_out_connector)

                memlet_data = copy.deepcopy(memlets.pop(0))

                # Add the edge with the corresponding memlet
                backward_state.add_edge(bwd_map_entry, pranet_node_out_connector, child_node, child_node_in_connector,
                                        memlet_data)

                child_node = bwd_map_entry
                child_node_in_connector = parent_node_in_connector

            if isinstance(edge_src, nodes.AccessNode):
                # The connection from the stored data will be made here
                assert edge_src == forward_node
                memlet_data = copy.deepcopy(memlets.pop(0))

                # Replicate the source stored node
                replicated_source_node = copy.deepcopy(source_node)
                backward_state.add_node(replicated_source_node)

                # Change the memlet data to read from the stored data and not the original data
                memlet_data.data = replicated_source_node.data

                # Add the final connection to the source node
                backward_state.add_edge(replicated_source_node, None, child_node, child_node_in_connector, memlet_data)

                # If this connection was made to a NestedSDFG and the forward node was a view,
                # We need to change the strides in the data descriptor this points to
                # Since the stored data is not a view
                # For example, if the stride of A is 5 (because it points to a column in  a 2d array),
                # The stored data will only contain the row and the stride for it should be one
                # This is only a problem if the view points to a NestedSDFG input,
                # that expects a descriptor with the original view stride
                if isinstance(child_node, nodes.NestedSDFG) and isinstance(forward_node.desc(self.sdfg),
                                                                           (dt.View, dt.ArrayView)):
                    # Get the strides of the stored data
                    stored_data_desc = self.sdfg.arrays[source_node.data]
                    stored_strides = stored_data_desc.strides

                    # Get the NestedSDFG input descriptor
                    input_desc = child_node.sdfg.arrays[child_node_in_connector]

                    # Set the strides to be the last elements of the stored strides
                    # We take the last elements since we might add loop indices to the shape
                    # Sanity check the strides for this desc should be less than or equal to the stored strides
                    assert len(input_desc.strides) <= len(stored_strides)
                    input_desc.strides = stored_strides[-len(input_desc.shape):]

        # There should be the same number of memlets through the new path
        assert len(memlets) == 0

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
        bwd_map = self._find_backward_entry_node_for_map_entry(backward_state=backward_state, entry_node=edge.dst)
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
                    bwd_map_entry = self._find_backward_entry_node_for_map_entry(backward_state=backward_state,
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
        backward_nodes = {n for e in state.edge_bfs(node, reverse=True) for n in [e.src, e.dst]}
        forward_nodes = {n for n in state.nodes()}
        # intersection with all the nodes in the forward state
        forward_subgraph = dstate.StateSubgraphView(state, list(forward_nodes.intersection(backward_nodes)))
        return forward_subgraph

    def _get_assign_tasklet(self,
                            forward_state: SDFGState,
                            node: nodes.AccessNode,
                            stored_node: nodes.AccessNode,
                            last_edge: dgraph.MultiConnectorEdge,
                            loop_iterators: str,
                            cuda: bool = False):
        """
        """
        # Create the assign tasklet
        assign_tasklet_node_in_connector = "in_stored_" + node.data
        assign_tasklet_node_out_connector = "out_stored_" + node.data
        assign_tasklet_node = nodes.Tasklet(
            label=f"__store_{node.data}_assign_",
            inputs={assign_tasklet_node_in_connector},
            outputs={assign_tasklet_node_out_connector},
            code=f"{assign_tasklet_node_out_connector} = {assign_tasklet_node_in_connector}",
        )

        # Add it to the state
        forward_state.add_node(assign_tasklet_node)

        # Connect it to the last map entry node
        # Create the memlet for the assignement
        # This will be the same as the memlet going to the tasklet
        assign_memlet_data = copy.deepcopy(last_edge.data)
        assign_block = assign_tasklet_node
        assign_block_in_connector = assign_tasklet_node_in_connector
        return_node = assign_tasklet_node
        return_connector = assign_tasklet_node_out_connector
        param_dict = {}
        memlet_access_iterators = []

        # We check the incoming memlet volumn
        if assign_memlet_data.volume != 1:
            # We need to add a map to iterate through the missing dimensions
            # For this we will create an assign block containing a map

            # First, Get the missing dimensions
            # Iterate through the subset
            for element in last_edge.data.subset:
                if str(element[0]) in last_edge.data.free_symbols:
                    # This is a symbol we will keep in the store memlet
                    memlet_access_iterators.append(str(element[0]))
                else:
                    # This is a range tuple we need to add an iterator for
                    # Create a random new free symbol
                    free_symbol = "si"
                    i = 0
                    while free_symbol in memlet_access_iterators or free_symbol in forward_state.free_symbols:
                        free_symbol = f"{free_symbol}_{i}"
                        i += 1
                    memlet_access_iterators.append(free_symbol)
                    param_dict.update({free_symbol: element})

            # Create the map and add it to the SDFG
            map = nodes.Map("flatten_assignement_map",
                            params=list(param_dict.keys()),
                            ndrange=list(param_dict.values()),
                            schedule=dtypes.ScheduleType.GPU_Device if cuda else dtypes.ScheduleType.Default)
            map_entry = nodes.MapEntry(map)
            map_exit = nodes.MapExit(map)
            forward_state.add_nodes_from([map_entry, map_exit])

            # Add the necessary connectors
            assert map_entry.add_in_connector(f"IN_store_block")
            assert map_entry.add_out_connector(f"OUT_store_block")
            assert map_exit.add_in_connector(f"IN_store_block")
            assert map_exit.add_out_connector(f"OUT_store_block")

            # Create the memlet from the map entry to the assign tasklet
            in_state_access = ','.join(memlet_access_iterators)
            memlet_data = Memlet(expr=f"{last_edge.data.data}[{in_state_access}]")

            # Create the edge between the map entry and assign tasklet
            forward_state.add_edge(map_entry, "OUT_store_block", assign_tasklet_node, assign_tasklet_node_in_connector,
                                   memlet_data)

            # Create the memlet from the assign tasklet to the map exist
            memlet_data = Memlet(
                expr=f"{stored_node.data}[{loop_iterators},{in_state_access}]") if loop_iterators else Memlet(
                    expr=f"{stored_node.data}[{in_state_access}]")

            # Create the edge between the map entry and assign tasklet
            forward_state.add_edge(assign_tasklet_node, assign_tasklet_node_out_connector, map_exit, "IN_store_block",
                                   memlet_data)

            # Make sure this block is connected correctly
            assign_block = map_entry
            assign_block_in_connector = "IN_store_block"
            return_node = map_exit
            return_connector = "OUT_store_block"

        # Get the last map
        last_map = last_edge.src
        last_map_connector = last_edge.src_conn

        # Add the new edge from the last map entrance to the new assign block
        forward_state.add_edge(last_map, last_map_connector, assign_block, assign_block_in_connector,
                               assign_memlet_data)
        return return_node, return_connector

    def _get_symbol_upper_bound_from_loop(self, s: sp.Symbol, loops: List[LoopRegion]) -> int:
        """
        Given a symbol and a list of loops, get the upper bound of the symbol from the loops.
        Raises an error if the symbol is not a loop index or the upper bound cannot be extracted correctly.
        """
        # Get the symbol to match
        if isinstance(s, (sp.Symbol, sp.Expr)):
            # We don't want to match global SDFG symbols
            loop_indices = {symb for symb in s.free_symbols if str(symb) not in self.sdfg.free_symbols}
            if len(loop_indices) != 1:
                raise AutoDiffException(f"Symbol dimension {s} couldn't be parsed correctly during storing")
            loop_index = str(list(loop_indices)[0])
        elif isinstance(s, str):
            # Extract the free symbols in the string besides the constants and operators and remove white space
            variable_regex = r'\b[a-zA-Z_][a-zA-Z0-9_]*\b'
            loop_indices = re.findall(variable_regex, s)

            # If there are multiple symbols in the string
            if len(loop_indices) != 1 or loop_indices[0] not in self.sdfg.symbols:
                raise AutoDiffException(f"Symbol dimension {s} couldn't be parsed correctly during storing")
            loop_index = loop_indices[0]
        else:
            raise AutoDiffException(f"Symbol dimesnion {s} is not a string and not a sympy symbol")

        # If the loop bound can be directly extracted from the interstate edges
        if loop_index in self.interstate_symbols:
            loop_size = self.interstate_symbols[loop_index]
        else:
            # Get the loop range for this symbol
            loop_size = None
            for l in loops:
                # Convert the sympy symbol to string to check if it macthes the loop variable
                if loop_index in l.loop_variable:
                    # Get the max loop range
                    start, end = self._extract_loop_region_info(l)

                    # If this is the case of exmaple of a loop like 6 - i
                    # TODO: How can we do this better?
                    matched = f"-{loop_index}" in str(s) or f"- {loop_index}" in str(s)
                    smallest, largest = ad_utils.get_loop_end(start, end, l)
                    if not matched:
                        loop_size = largest
                    else:
                        loop_size = smallest

        if loop_size is None:
            raise AutoDiffException(
                f"Can't figure out how to save the data inside: {l.label} because of its symbol shape {s}")

        # We will call this function recusrively until loop size is numeric or it is a global SDFG symbol
        if ad_utils.shape_has_symbols_to_replace(self.sdfg, loop_size):
            loop_size, _ = self._get_symbol_upper_bound_from_loop(loop_size, loops)
        return loop_size, loop_index

    def _store_data(self, forward_state: SDFGState, backward_state: SDFGState, forward_an: nodes.AccessNode,
                    target_node: nodes.Node, edge: dgraph.MultiConnectorEdge) -> Tuple[nodes.AccessNode, List[Memlet]]:
        """
        Given an edge leading an AccessNode or a map to the target node in the forward state,
        add a path from the connector for this AccessNode to store its values for all iterations.
        This can increase the dimension of the array. i.e. the size of the stored array is
        greater or equal to the size of the original array.
        
        :param edge: the edge connecting the AccessNode to save data from to a map node.
        :return: the new AccessNode which contains the stored data, 
                 a list of memlets connecting an assign tasklet to this new AccessNode.
        """

        # Get the connector and edge to save
        if isinstance(edge.src, nodes.AccessNode) and edge.src is not forward_an:

            # Get the incoming edge to this AccessNode
            in_edges = forward_state.in_edges(edge.src)

            # There should only be one incoming edge
            assert len(in_edges) == 1

            # Get the memlet path for the edge incoming to this AccessNode
            memlet_path = forward_state.memlet_path(in_edges[0])

            # The start of this path should be the forward AccessNode
            assert forward_an is memlet_path[0].src

            # The last edge in the memlet path has the connector we want to save
            edge = memlet_path[-1]

        # Add a new AccessNode and array to the forward pass
        # First, check if a stored array with this name already exists
        if "stored_" + forward_an.data not in self.backward_sdfg.arrays:
            new_store_node_name = "stored_" + forward_an.data
        else:
            i = 0
            while True:
                if f"stored_{i}_" + forward_an.data not in self.backward_sdfg.arrays:
                    new_store_node_name = f"stored_{i}_" + forward_an.data
                    break
                i += 1

        # Get the new array shape
        # This will be the shape of the current array
        shape: List[int] = list(self.sdfg.arrays[forward_an.data].shape)

        # If the shape is an expression:
        if any((isinstance(s, dace.symbol) or isinstance(s, sp.Expr)) and not (str(s) in self.sdfg.free_symbols)
               for s in shape):
            # Otherwise, replace all the loop dependant allocations with the max length of the loop
            # For example, an array of size [i+1] in a range(2, 10) loop will be stored in a [10, 10] array (1)
            # Additionally, an array of size [32-i] in the same loop will be stored in a [10, 30]  (2)
            loops = self._get_all_enclosing_loops(forward_state)

            if len(loops) > 0:
                # Loop over the shape dimensions
                for i, s in enumerate(shape):
                    if ad_utils.shape_has_symbols_to_replace(self.sdfg, s):
                        loop_size, loop_index = self._get_symbol_upper_bound_from_loop(s, loops)
                        # Replace the symbol with the loop size and evaluate the expression
                        # Check if loop size can be converted to an integer
                        if isinstance(loop_size, int) or (isinstance(loop_size, str) and ad_utils.is_int(loop_size)):
                            shape[i] = s.subs(sp.Symbol(loop_index), loop_size)
                        else:
                            shape[i] = s.subs(sp.Symbol(loop_index), dace.symbol(loop_size))

        # Plus the size of any enclosing loops
        encolsed, _ = self._state_within_loop(forward_state=forward_state)
        nb_enclosing_loops = 0
        loop_param_list = []
        if encolsed:
            # Get all incolsing loops
            all_encolsing_loops = self._get_all_enclosing_loops(forward_state=forward_state)
            nb_enclosing_loops = len(all_encolsing_loops)
            # Get the size of each loop and add it to the list
            for loop in all_encolsing_loops:
                # Get the end of the loop
                start, end = self._extract_loop_region_info(loop)

                # Check if the loop is increasing or decreasing
                # First, try to convert the strings to ints if possible
                # Note that we look for the start or end of the loop
                # And not the size of the loop.
                # This is because we access using the loop indices
                # Using the loop sizes instead would require shifting accesses
                # TODO: Do we need to treat the case where 1-i is in the shape here too?
                _, new_dim = ad_utils.get_loop_end(start, end, loop)

                # First we check if the new dimension contains symbols
                # These will need to be replaced with scalars for correct allocation
                # The sdfg symbols are allowed to be in the shape
                if ad_utils.shape_has_symbols_to_replace(self.sdfg, new_dim):
                    # Take the expression to sympy for easier processing
                    if isinstance(new_dim, str):
                        new_dim = sp.Symbol(new_dim)

                    # Try to replace the symbols with the loop size
                    # TODO: this can be extended to a loop over the symbols in new_dim
                    loop_size, loop_index = self._get_symbol_upper_bound_from_loop(new_dim, all_encolsing_loops)
                    if isinstance(loop_size, int) or (isinstance(loop_size, str) and ad_utils.is_int(loop_size)):
                        new_dim = new_dim.subs(sp.Symbol(loop_index), loop_size)
                    else:
                        new_dim = new_dim.subs(sp.Symbol(loop_index), dace.symbol(loop_size))
                shape.insert(0, new_dim)
                loop_param_list.insert(0, loop.loop_variable)

        # Add the array descriptor and AccessNode to the forward state
        original_desc = forward_an.desc(forward_state)

        # We make a special case for a memlet of the type A[i, j] in an i, j loop
        # In this case we only need an array of the same size as the forward node
        if encolsed and edge.data.data == forward_an.data and len(edge.data.subset) == nb_enclosing_loops:
            # Check that the memlet subset matches perfectly the order of loop nest
            # Make sure the subset elements are (i,i,1) and (j,j,1)
            # Then check if this matches the loop indices
            if all(
                    str(subset[0]) == loop_param_list[i] and subset[0] == subset[1] and subset[2] == 1
                    for i, subset in enumerate(edge.data.subset)):
                # We only use the loop accesses
                # Both should work since shape[:nb_enclosing_loops] == shape[nb_enclosing_loops:]
                shape = shape[nb_enclosing_loops:]

                # We want to build the memlet as if this was not in a a loop
                nb_enclosing_loops = 0

        new_store_node = forward_state.add_array(
            name=new_store_node_name,
            shape=shape,
            dtype=original_desc.dtype,
            transient=True,
        )
        new_store_node.setzero = True

        # Connect the edge source and connector to the new access node
        # We will save the memlets we create and return them
        # This is useful to make the connections for the backward state
        memlets_stack = []

        # The loop accesses will be the same within the state
        # Prepare them for all edges
        loop_access = ','.join([f'{loop_param_list[i]}' for i in range(nb_enclosing_loops)])

        # In the other cases, we need to route the storing through maps
        all_edges = ad_utils.get_all_path_edges(forward_state, forward_an, edge)

        # Get the map nest memlet informtation
        start_range, param_list, shape_list, param_dict = ad_utils.get_map_nest_information(all_edges)

        # The parameters to add for the current memlet in the loop
        # At first we will use all of the parameters that are used in the memlet
        # param_dict = {key: val for key, val in param_dict.items() if key in edge.data.free_symbols}
        new_param_dict = {}

        # Iterate through the subset
        for index, element in enumerate(edge.data.subset):
            if str(element[0]) in edge.data.free_symbols and str(element[0]) in param_dict.keys():
                # Add the range from the param_dict
                new_param_dict.update({str(element[0]): param_dict[str(element[0])]})
            else:
                # Add the range from the param_dict
                new_param_dict.update({index: element})

        params_to_add = new_param_dict
        # First, we need to add an assign tasklet
        assign_tasklet_node, assign_tasklet_node_out_connector = self._get_assign_tasklet(forward_state=forward_state,
                                                                                          node=forward_an,
                                                                                          stored_node=new_store_node,
                                                                                          last_edge=edge,
                                                                                          loop_iterators=loop_access)

        # Start iterating
        previous_node = assign_tasklet_node
        previous_node_out_connector = assign_tasklet_node_out_connector
        map_exist = None
        for edge in reversed(all_edges):
            if isinstance(edge.src, nodes.MapEntry):
                # Get the corresponding map exit
                map_exist = self._find_map_exist_for_map_entry(map_entry=edge.src, state=forward_state)

                # Add the Connectors to the map
                map_exit_in_connector = f"IN_stored_{new_store_node.label}"
                map_exit_out_connector = f"OUT_stored_{new_store_node.label}"
                assert map_exist.add_in_connector(map_exit_in_connector)
                assert map_exist.add_out_connector(map_exit_out_connector)

                # Prepare the memlet data for this edge
                access_list = []
                for key, val in new_param_dict.items():
                    if isinstance(key, str):
                        if key in params_to_add.keys():
                            access_list.append(key)
                        else:
                            start = val[0]
                            end = val[1]
                            access_list.append(f'{start}:{end}')
                    elif isinstance(key, int):
                        start = val[0]
                        end = val[1] + 1
                        access_list.append(f'{start}:{end}')
                    else:
                        raise AutoDiffException("Found unexepected type in memlet parameters dictionary")

                in_state_access = ','.join(access_list)

                memlet_data = Memlet(
                    expr=f"{new_store_node.data}[{loop_access},{in_state_access}]") if loop_access else Memlet(
                        expr=f"{new_store_node.data}[{in_state_access}]")

                # Save the memlet for later
                memlets_stack.append(memlet_data)

                # Connect the previous node to this map exist
                forward_state.add_edge(previous_node, previous_node_out_connector, map_exist, map_exit_in_connector,
                                       memlet_data)

                previous_node = map_exist
                previous_node_out_connector = map_exit_out_connector

                # Remove the parameters seen in the current map
                # Since they will become out of scope in the next iteration
                params_to_add = {}
                for key, val in new_param_dict.items():
                    if isinstance(key, str):
                        if key not in edge.src.params:
                            start = val[0]
                            end = val[1]
                            params_to_add.update({key: (start, end)})
                    elif isinstance(key, int):
                        params_to_add.update({key: val})
                    else:
                        raise AutoDiffException("Found unexepected type in memlet parameters dictionary")

            else:
                # Prepare the memlet data for this edge
                access_list = []
                for key, val in new_param_dict.items():
                    if isinstance(key, str):
                        start = val[0]
                        end = val[1]
                        access_list.append(f'{start}:{end}')
                    elif isinstance(key, int):
                        start = val[0]
                        end = val[1] + 1
                        access_list.append(f'{start}:{end}')
                    else:
                        raise AutoDiffException("Found unexepected type in memlet parameters dictionary")

                in_state_access = ','.join(access_list)

                # Get the memlet data for the connection between the last map exit and the new store AccessNode
                memlet_data = Memlet(
                    expr=f"{new_store_node.data}[{loop_access},{in_state_access}]") if loop_access else Memlet(
                        expr=f"{new_store_node.data}[{in_state_access}]")

                memlets_stack.append(memlet_data)

                # This should be the last connection
                forward_state.add_edge(previous_node, previous_node_out_connector, new_store_node, None, memlet_data)
                break

        # We need to add an empty memlet from the new store AccessNode to make sure the data is stored before it is
        # potentially altered
        # First, we check if this can be avoided
        # We do a BFS exploration to see if the data we are trying to store is overwritten within the same execution state
        bfs_nodes = list(forward_state.bfs_nodes(source=forward_an))
        if any(
                isinstance(n, nodes.AccessNode) and n.data == forward_an.data and n is not forward_an
                for n in bfs_nodes):
            to_connect = []
            for out_edge in forward_state.out_edges(forward_an):
                # Get the destination of the edge
                dst = out_edge.dst
                if not isinstance(dst, nodes.MapEntry) and dst is not assign_tasklet_node:
                    # This will not be necessary for maps since the storing is added to the same map
                    # We also don't connect the newly created assign tasklet to avoid creating a cycle
                    if dst not in to_connect:
                        # We only need to make a single connection to the new stored data
                        to_connect.append(dst)

            for node in to_connect:
                # Connect the new store AccessNode to assure the store happens first
                # If there isn't already a connnection between these two nodes
                if not any(e.dst == node for e in forward_state.out_edges(new_store_node)):
                    forward_state.add_edge(new_store_node, None, node, None, Memlet())

            # Another case for making sure data is stored before it is altered is when the map we save from writes itself to the data we want to save
            # In this case this would depend on the codegen order of the tasklets within the map and is thus not safe
            # Detect if this is the case
            if map_exist:
                # Check if this map exit writes to the data we want to save
                if any(
                        isinstance(e.dst, nodes.AccessNode) and e.dst.data == forward_an.data
                        for e in forward_state.out_edges(map_exist)):
                    # Get the map entry of this map exit
                    tasklet_in_edges = forward_state.in_edges(assign_tasklet_node)
                    assert len(tasklet_in_edges) == 1
                    tasklet_in_edge = tasklet_in_edges[0]

                    # Safety check
                    if not isinstance(tasklet_in_edge.src, nodes.MapEntry):
                        raise AutoDiffException(
                            "The map exit writes to the data we want to save, but the storing strcuture is not what we expect"
                        )

                    # Get all the edges coming out of this specific in connector
                    collusion_edges = [
                        e for e in forward_state.out_edges(tasklet_in_edge.src)
                        if e.src_conn == tasklet_in_edge.src_conn and e.dst != assign_tasklet_node
                    ]

                    # We need to add an empty memlet from the new store tasklet to everything else that reads from that connector
                    for out_edge in collusion_edges:
                        forward_state.add_edge(assign_tasklet_node, None, out_edge.dst, None, Memlet())

        return new_store_node, memlets_stack

    def _check_node_overwrite(self, forward_state: SDFGState, node: nodes.AccessNode) -> Tuple[bool, bool]:
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

        # Get the decendant and accendant states to look in for an overwrite
        assert forward_state in self.state_order
        index = self.state_order.index(forward_state)
        decendant_states = self.state_order[index:]

        # Check if this access node is a view
        if type(node.desc(self.sdfg)) is dt.ArrayView:
            # The view should have one incoming edge from the original access node
            in_edges = forward_state.in_edges(node)

            # Sanity checks
            assert len(in_edges) == 1
            assert "views" in node.in_connectors

            # We want to check if the source has been overwritten
            node = in_edges[0].src

        # Get all the AccessNodes with the same data
        matches = []
        for d_state in decendant_states:
            matches += [(nd, parent) for nd, parent in d_state.all_nodes_recursive()
                        if isinstance(nd, nodes.AccessNode) and nd.data == node.data]

        # There needs to be at least one occurance which is the node passed as a parameter
        assert len(matches) > 0 and (node, forward_state) in matches

        # If there is only one occurance of this data, it will not be overwritten later in the graph
        if len(matches) == 1:
            overwritten = False
            decided = True

        # Get the index of the parameter node
        index = matches.index((node, forward_state))

        # If the parameter node is the last occurance in the decendant states,
        # it will not be overwritten
        if len(matches) - 1 == index:
            overwritten = False
            decided = True

        # If we haven't already confirmed that this node has not been overwritten
        if not decided:
            # Iterate through all the successor occurances
            for nd, parent in matches[index + 1:]:
                # Check if this node has an incoming edge
                if len(parent.in_edges(nd)) > 0:
                    overwritten = True

        if not overwritten:
            # There is no overwrite so far
            # Check if this state is within a loop
            is_in_loop, loop = self._state_within_loop(forward_state)
            if is_in_loop:

                # Check if there is any write to this access node within the loop
                loop_matches = [(nd, parent) for nd, parent in loop.all_nodes_recursive()
                                if isinstance(nd, nodes.AccessNode) and nd.data == node.data]
                for match, match_parent in loop_matches:
                    # Check if this node has an incoming edge
                    if len(match_parent.in_edges(match)) > 0:
                        overwritten = True

                if overwritten and len(matches) == 1:
                    # Check if the overwrite is from constant arrays
                    # This means that the same value will be assigned at each iteration of the loop
                    # And no storing is necessary
                    match, match_parent = loop_matches[0]
                    all_read_only = True
                    for edge in match_parent.edge_bfs(match, reverse=True):
                        if edge.data.subset is not None and len(edge.data.subset.free_symbols) != 0:
                            all_read_only = False
                            break
                        if isinstance(edge.src, nodes.AccessNode):
                            # The memlet needs to be constant
                            if edge.src.data not in self.read_only_arrays:
                                all_read_only = False
                                break
                            # Check if the data is read only
                    if all_read_only:
                        overwritten = False

        # Iterate through all the predecessor occurances
        for nd, parent in matches[:index + 1]:
            # Check if this node has an incoming edge
            if len(parent.in_edges(nd)) > 0:
                recomputable = True
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

    def _find_backward_entry_node_for_map_entry(self, backward_state: SDFGState,
                                                entry_node: nodes.MapEntry) -> nodes.MapExit:
        """
        Find the entry node in the backward pass corresponding to the exit node opened by
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
                          required_gradients) -> Tuple[nodes.Node, BackwardResult]:
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

        # (1)
        if hasattr(self.dace_node_impl, "_reverse_" + type(node).__name__):
            return getattr(self.dace_node_impl, "_reverse_" + type(node).__name__)(state, backward_state, node,
                                                                                   given_gradients, required_gradients)

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
            return backward_node, backward_result

        raise AutoDiffException("Unable to differentiate node type {}. Either add a pure forward implementation "
                                "or a backward implementation to progress.".format(type(node)))
