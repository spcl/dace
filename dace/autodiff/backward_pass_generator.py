# Copyright 2019-2025 ETH Zurich and the DaCe authors. All rights reserved.
import copy
from typing import List, Tuple, Set, Dict, Union, Optional, Sequence
import sympy as sp

# DaCe imports
import dace
from dace.properties import CodeBlock
import dace.sdfg.nodes as nodes
import dace.transformation.transformation as xf
from dace import dtypes, data as dt
from dace.sdfg import SDFG, SDFGState, state as dstate, utils as dace_utils
from dace.sdfg.state import LoopRegion
from dace.memlet import Memlet

try:
    from dace.libraries.onnx.forward_implementation_abc import ONNXForward
    from dace.libraries.onnx.nodes.onnx_op import ONNXOp
    ONNX_AVAILABLE = True
except ImportError:
    ONNXForward = None
    ONNXOp = None
    ONNX_AVAILABLE = False

# Autodiff imports
from dace.autodiff.base_abc import (BackwardContext, BackwardResult, AutoDiffException, find_backward_implementation,
                                    ExpansionTemplate)
import dace.autodiff.utils as ad_utils
from dace.autodiff.implementations.dace_nodes import DaceNodeBackwardImplementations
from dace.autodiff.data_forwarding.manager import DataForwardingManager


class BackwardPassGenerator:
    """Generator for automatic differentiation backward passes on DaCe SDFGs.

    This class orchestrates the creation of backward passes for automatic differentiation
    using reverse-mode AD. It handles gradient computation, data forwarding between
    forward and backward passes, and complex control flow structures.

    :param sdfg: The forward SDFG to differentiate.
    :param given_gradients: Output arrays for which gradients are provided (seed gradients).
    :param required_gradients: Input arrays for which gradients should be computed.
    :param backward_sdfg: SDFG to contain the backward pass. Can be same as forward SDFG.
    :param array_grad_map: Optional mapping from array names to gradient array names.
    :param conflicted_gradient_buffers: Arrays with potential write conflicts requiring special handling.
    :param data_forwarding_strategy: Strategy for forwarding data ('store_all', 'recompute_all', 'user_defined').
    :param data_to_recompute: Arrays to recompute instead of storing (when strategy='user_defined').
    :raises AutoDiffException: If the backward pass generation fails.

    Example::

        gen = BackwardPassGenerator(
            sdfg=forward_sdfg,
            given_gradients=['loss'],
            required_gradients=['weights', 'input']
        )
        gen.backward()
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
        data_forwarding_strategy: str = "store_all",
        data_to_recompute: Optional[List[str]] = None,
    ):

        self.sdfg: SDFG = sdfg
        self.data_to_recompute = data_to_recompute
        self.backward_sdfg: SDFG = backward_sdfg

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

        #: Arrays descriptors for the gradients
        self.backward_grad_arrays: Dict[str, dt.Array] = {}

        #: Arrays descriptors for inputs that are required from the forward pass
        self.backward_input_arrays: Dict[str, dt.Array] = {}

        #: Mapping from forward node -> backward node, and forward map -> backward map
        self.reverse_map: Dict[nodes.Node, Union[nodes.Node, nodes.Map]] = {}

        #: Mapping from forward state -> backward state
        self.reversed_states_map: Dict[SDFGState, SDFGState] = {}

        #: Mapping from forward LoopRegion -> backward LoopRegion
        self.reversed_loops_map: Dict[LoopRegion, LoopRegion] = {}

        #: Mapping from forward state -> backward state for loop states
        self.reversed_loop_states_map: Dict[nodes.Node, nodes.Node] = {}

        #: Mapping between states and their subgraph views for AD processing
        self.states_view_map: Dict[SDFGState, dstate.StateSubgraphView] = {}

        #: Mapping between loop states and their subgraph views for AD processing
        self.loop_states_view_map: Dict[SDFGState, dstate.StateSubgraphView] = {}

        #: Mapping between the map entry of a conditional assignment block and its zero-out AN
        self.conditional_block_entry: Dict[nodes.MapEntry, nodes.AccessNode] = {}

        #: Mapping from forward_node -> BackwardResult for that node
        self.result_map: Dict[nodes.Node, BackwardResult] = {}

        #: Mapping from forward name to gradient name for arrays
        self.array_grad_map: Dict[str, str] = array_grad_map or {}

        #: Mapping from the backward access nodes that will be zeroed out
        # to the transients that contain the values before they are zeroed out
        self.zeroed_out: Dict[nodes.AccessNode, List[nodes.AccessNode]] = {}

        #: The read-only arrays of the forward SDFG. Used in data forwarding decisions
        self.read_only_arrays: Set[str] = ad_utils.get_read_only_arrays(self.sdfg)

        #: Mapping from overwritten input name to storing AccessNode
        self.stored_inputs: Dict[str, nodes.AccessNode] = {}

        # Variable to check if backward has already been applied
        self._applied = False

        self.data_forwarding_strategy = data_forwarding_strategy

        # Topological ordering of the states
        self.state_order = ad_utils.get_state_topological_order(self.sdfg)
        self.conflicted_gradient_buffers: Set[str] = conflicted_gradient_buffers or set()

        self.interstate_symbols: Dict[str, str] = {}
        for edge in self.sdfg.all_interstate_edges():
            for assign_symbol, assignment in edge.data.assignments.items():
                self.interstate_symbols[assign_symbol] = assignment

        # Validate parameters and setup SDFG configuration
        self._validate_gradients()
        self._setup_sdfg_configuration(sdfg, backward_sdfg, given_gradients)

        # DaCe nodes backward implementations
        self.dace_node_impl = DaceNodeBackwardImplementations(self)

        #: List containing information about all the data to be forwarded to the backward pass
        self.data_to_forward: List[Tuple[SDFGState, SDFGState, nodes.AccessNode, nodes.Node,
                                         dstate.MultiConnectorEdge]] = []

        # Data forwarding manager
        self.data_forwarding_manager = DataForwardingManager(self)

    def _validate_gradients(self) -> None:
        """Validate that gradient arrays exist in the SDFG.

        Raises:
            AutoDiffException: If gradient arrays are not found in SDFG arrays.
        """
        # Check outputs (given gradients)
        for outp in self.given_gradients_data:
            if outp not in self.sdfg.arrays:
                raise AutoDiffException(f"Could not find output '{outp}' in SDFG array descriptors")

        # Check inputs (required gradients)
        for inp in self.required_gradients_data:
            if inp not in self.sdfg.arrays:
                raise AutoDiffException(f"Could not find input '{inp}' in SDFG array descriptors")

    def _setup_sdfg_configuration(self, sdfg: SDFG, backward_sdfg: SDFG,
                                  given_gradients: List[nodes.AccessNode]) -> None:
        """Setup SDFG configuration for separate or combined forward/backward passes.

        :param sdfg: Forward SDFG.
        :param backward_sdfg: Backward SDFG.
        :param given_gradients: List of gradient output nodes.
        :raises AutoDiffException: If configuration is invalid for combined SDFG mode.
        """
        if sdfg is backward_sdfg:
            # Combined mode requires single scalar output
            if len(given_gradients) != 1:
                raise AutoDiffException("When forward and backward SDFGs are the same, exactly one output is required, "
                                        f"got {len(given_gradients)}")

            output_array = sdfg.arrays[given_gradients[0].data]
            if not ad_utils.is_int_eq_value(output_array.total_size, 1):
                raise AutoDiffException("When forward and backward SDFGs are the same, output must be a single scalar")

            self.separate_sdfgs = False
        else:
            self.separate_sdfgs = True

    def create_child_generator(self, **kwargs) -> 'BackwardPassGenerator':
        """Create a child generator for nested SDFG differentiation.

        This factory method creates a new BackwardPassGenerator instance for differentiating
        nested SDFGs, propagating relevant configuration from the parent generator.

        :param kwargs: Parameters to pass to the child generator constructor.
                       Required: sdfg, given_gradients, required_gradients, backward_sdfg.
        :return: A new BackwardPassGenerator instance configured for the nested SDFG.
        """
        defaults = {
            'data_forwarding_strategy': self.data_forwarding_strategy,
            'data_to_recompute': self.data_to_recompute,
        }
        defaults.update(kwargs)
        return BackwardPassGenerator(**defaults)

    def backward(self) -> Tuple[BackwardResult, Dict[str, dt.Array], Dict[str, dt.Array]]:
        """Generate the backward pass in backward_sdfg."""
        return self.reverse_sdfg()

    def reverse_sdfg(self) -> Tuple[BackwardResult, Dict[str, dt.Array], Dict[str, dt.Array]]:
        """Generate the backward pass by reversing all SDFG states.

        Processes all states in the SDFG and creates their backward counterparts,
        connecting them with appropriate control flow for gradient computation.

        :return: A tuple containing:

            * ``BackwardResult`` - Contains gradient mappings and metadata.
            * ``Dict[str, dt.Array]`` - Gradient array descriptors (backward pass outputs).
            * ``Dict[str, dt.Array]`` - Forward pass arrays required by backward pass.

        :raises AutoDiffException: If backward pass was already applied to this generator.
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

        # Add interstate assignments for control flow decisions
        self._add_interstate_edge_assignments()

        # Forward required data by the backward pass according to a user defined strategy
        self.data_forwarding_manager.forward_data_to_backward_pass()

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

    def _create_stateviews_mapping(self) -> None:
        """Map each state in the SDFG to views that indicate what to differentiate."""
        self._find_subgraph_to_differentiate()
        # Expand until there is nothing left to expand
        while self._expand_nodes():
            # Nodes have been expanded again on the expanded graph; recalculate the forward graph
            self._find_subgraph_to_differentiate()

    def _reverse_states(self) -> None:
        """Go through all states of the forward SDFG, reverse them and add them to the backward SDFG."""
        # For reversal we want to iterate through the states in reverse topological order
        for state in reversed(self.state_order):
            # Get all the views of this state
            if state not in self.states_view_map:
                raise AutoDiffException(f"State {state} not found in states view map")
            state_subgraph_views = [self.states_view_map[state]]

            # In case this is a state loop
            state_subgraph_loop_view = []
            if state in self.loop_states_view_map:
                loop_view = self.loop_states_view_map[state]
                state_subgraph_loop_view.append(loop_view)

            for state_subgraph_view in state_subgraph_views:

                # Make sure this state has not already been reversed
                if state in self.reversed_states_map:
                    raise AutoDiffException(f"State {state} has already been reversed")

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

                # Recursively reverse the subgraph
                self._reverse_subgraph(forward_state=state, backward_state=reversed_state, subgraph=state_subgraph_view)

        # We also reverse all the LoopRegions in the graph
        for node in self.sdfg.nodes():
            if not isinstance(node, LoopRegion):
                continue
            self._reverse_loop_region(node)

    def _connect_reversed_states(self) -> None:
        """Connect backward states corresponding to forward SDFG states.

        All incoming edges of a forward state become outgoing edges in the backward SDFG.
        """

        for state in self.state_order:
            # All states should be reversed already
            if state not in self.reversed_states_map:
                raise AutoDiffException(f"State {state} not found in reversed states map")
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

    def _fill_interstate_edge_conditions_in_scope(self, graph: Union[SDFG, LoopRegion]) -> None:
        """
        Get all the nodes within this graph in topological order,
        Connect the states and call the function recursively on the nested scopes.
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

                if not isinstance(node, SDFGState):
                    raise AutoDiffException(f"Expected SDFGState, got {type(node)}")
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
                    # Handle "1" (unconditional) to avoid creating expressions like "1 and condition"
                    if src_state_condition == "1" and current_edge_condition == "1":
                        new_bwd_edge_condition = "1"
                    elif src_state_condition == "1":
                        new_bwd_edge_condition = current_edge_condition
                    elif current_edge_condition == "1":
                        new_bwd_edge_condition = src_state_condition
                    else:
                        new_bwd_edge_condition = f"({src_state_condition}) and ({current_edge_condition})"

                    bwd_edge = self._get_backward_state_edge(edge)

                    # Add the condition to the edge
                    bwd_edge.data.condition = CodeBlock(new_bwd_edge_condition)

                    # If there is a special case for the first iteration of the backward state
                    if forward_state in self.loop_states_view_map:

                        # Get the corresponding edge between the loop states
                        bwd_loop_edge = self._get_backward_loop_state_edge(edge)

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

    def _fill_interstate_edge_conditions(self) -> None:
        """
        Go through all of the states in the forward graph and fill the necessary conditions in the backward states.
        Each edge in the backward SDFG will be the logical AND between the equivalent edge in the forward SDFG and
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

    def _add_interstate_edge_assignments(self) -> None:
        """
        We will need to add interstate assignments at the start of the backward SDFG
        This is necessary to make sure the control flow in the backward pass is correctly preserved.
        """
        # We will add an empty state to the backward pass which will have all the assignments

        new_assignments = {}
        # Get all the interstate edges in the forward sdfg
        for edge in self.sdfg.all_interstate_edges():
            if edge.data.assignments:
                # There are assignments to be added to the start of the backward pass
                new_assignments = {**new_assignments, **edge.data.assignments}

                # We need to check if any data needs to be used in these assignment
                # This is important in the case of a NSDFG where data will need to be forwarded
                for _, rhs in edge.data.assignments.items():
                    # If any of the sdfg arrays are in the rhs assignment
                    assignment_arrays = [array for array in self.sdfg.arrays.keys() if array in rhs]
                    if assignment_arrays and self.separate_sdfgs:
                        # We need to forward this data to the backward pass
                        for array in assignment_arrays:
                            if array not in self.backward_input_arrays:
                                self.backward_input_arrays[array] = self.sdfg.arrays[array]
                                # Special case if this is a symbol that is doesn't have a descriptor yet
                                if array not in self.backward_sdfg.arrays:
                                    # We add it now
                                    self.backward_sdfg.add_datadesc(array, copy.deepcopy(self.sdfg.arrays[array]))

        if new_assignments:
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
                    raise AutoDiffException("Need to add an assignments state but can't find the start block")
            # TODO would this work on a loop region?
            self.backward_sdfg.add_state_before(state=bwd_start_block,
                                                label="_bwd_interstate_assignments_state",
                                                assignments=new_assignments)

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

    def _zero_out_gradient(self, forward_state: SDFGState, forward_node: nodes.AccessNode, memlet: Memlet) -> None:
        """
        Zero out gradients for overwritten arrays in the forward pass.

        Overwritten arrays need their gradients zeroed for gradient accumulation
        to work correctly. This method:

        1. Copies current gradient values to a temporary array (for one last use
           in the backward pass)
        2. Zeros out the overwritten access in the backward pass
        3. Updates the read mapping to use the temporary instead of the original

        The operation is skipped when possible to optimize performance.

        :param forward_state: The state in the forward pass containing the write.
        :param forward_node: The access node being overwritten.
        :param memlet: The memlet describing the write operation.
        """
        # Extra checks to only do this if necessary
        # If this access node is not written to in the forward pass except for this one time, we don't need to zero it out
        # An exception is made for required gradients that can be read outside the scope of the SDFG
        clear_out_gradients = forward_node.data in self.required_gradients_data

        # Get the write instances in the forward sdfg to this node that happen in states before the current state
        # These will represent the reads that will happen after this AccessNode
        # This should avoid unnecessary zeroing out of dace generated temporaries
        for state in self.state_order[0:self.state_order.index(forward_state) + 1]:
            state_view = self.states_view_map[state]
            for node, parent in state_view.all_nodes_recursive():
                if isinstance(node, nodes.AccessNode) and node.data == forward_node.data:
                    if parent.in_degree(node) > 0:
                        # We need to check if the the forward node is inside a map scope or a LoopRegion
                        within_loop, _ = ad_utils.state_within_loop(state)
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
        elif dtypes.can_access(dtypes.ScheduleType.GPU_Device, array_desc.storage):
            cuda = True
        else:
            raise ValueError(f"Unsupported storage {array_desc.storage}")

        # Careful! The order of the ifs here matters since ArrayView is a subclass of Array
        if isinstance(array_desc, dt.View):
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
            _, map_entry, map_exit = backward_state.add_mapped_tasklet(
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
            if not isinstance(cleared_out_node, nodes.AccessNode):
                raise AutoDiffException(f"Expected AccessNode as cleared out node, got {type(cleared_out_node)}")

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

    def _remove_onnx_attribute_accessnodes(self, nodes_list: List[nodes.Node], state: SDFGState) -> None:
        """Remove ONNX attribute AccessNodes that don't need gradient tracking.

        For some ONNX operators, nodes have attributes as input connectors even if the inputs are actually constant.
        Examples of such attributes are `axis` and `keepdims` in `ReduceSum`.
        Gradients for these attributes should not be tracked since they represent control flow and not data flow.
        """
        attribute_to_remove = {"axis", "keepdims", "axes", "p", "dilations", "kernel_shape", "strides"}
        for node in nodes_list[:]:  # Iterate over a copy of the list to avoid modification issues
            if isinstance(node, nodes.AccessNode):
                out_edges = state.out_edges(node)
                if out_edges and all(
                        ONNX_AVAILABLE and isinstance(edge.dst, ONNXOp) and edge.dst_conn in attribute_to_remove
                        for edge in out_edges):
                    nodes_list.remove(node)

    def _remove_maps_without_input_connectors(self, nodes_list: List[nodes.Node], state: SDFGState) -> None:
        """Remove maps that don't have any input connectors from the nodes_list.

        These are maps that won't have an output in the backward pass and thus can be skipped from the reversal process.
        Note that we do not remove the AccessNode that the no-input map writes to.
        This is because we might need to zero out the gradient of this node.
        If no zeroing out is necessary, the node will be removed in the reverse_subgraph function cleanup at the end.
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
        """Determine which nodes we need to reverse; this forms the subgraph we will differentiate.

        We do a reverse BFS from the target output node.
        In the case where a state is within a loop, this may result in different subgraphs
        depending on the loop iteration.

        To calculate the gradients for a node x in ``required_gradients``, we need to sum up the gradient
        contributions from every node y where x is used as an input.
        """
        backward_nodes: set[nodes.Node] = set()
        given_gradients_all_states = set(self.given_gradients_data)

        required_gradients_all_states = {n for n in self.required_gradients_data}
        given_gradients_all_states = given_gradients_all_states | required_gradients_all_states

        # Do the backward BFS iteratively
        for state in reversed(self.state_order):
            state_given_gradients: List[nodes.AccessNode] = []

            for node in state:
                if isinstance(node, nodes.AccessNode) and node.data in given_gradients_all_states:
                    state_given_gradients.append(node)

            backward_nodes = {n for e in state.edge_bfs(state_given_gradients, reverse=True) for n in [e.src, e.dst]}
            nodes_list = list(backward_nodes)

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
            within_loop, _ = ad_utils.state_within_loop(state)
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
        """Return the gradient name of a name from the forward pass."""
        if forward_name not in self.array_grad_map:
            self.array_grad_map[forward_name] = \
                self.backward_sdfg._find_new_name("gradient_" + forward_name)

        return self.array_grad_map[forward_name]

    def _add_gradient_data_descriptor(self, data_name: str) -> dt.Array:
        """Add the data descriptor for the gradient for `data_name`.

        :param data_name: The name of the forward descriptor.
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

        # Store references
        self.backward_grad_arrays[grad_name] = cloned_datadesc
        self.backward_sdfg.arrays[grad_name] = cloned_datadesc
        return cloned_datadesc

    def _reverse_loop_conditional(self, loop: LoopRegion) -> str:
        """Given a loop region as a parameter, create the conditional for the reversed version of this loop."""

        # Get the loop iterator
        it = loop.loop_variable

        # Get the loop start
        start, _ = ad_utils.extract_loop_region_info(loop)

        # Get the stride sign
        stride_sign = ad_utils.get_stride_sign(loop)

        # Reverse the conditional to end at the start of the original loop
        # This will be incremented or decremented depending on the stride
        if stride_sign > 0:
            reversed_condition = f"{it} > {start}-1"
        else:
            reversed_condition = f"{it} < {start}+1"

        return reversed_condition

    def _reverse_loop_initial_statement(self, loop: LoopRegion) -> str:
        """Given a loop region as a parameter, create the initialization statement for the reversed version of this loop."""
        # Get the loop iterator
        it = loop.loop_variable

        stride_sign = ad_utils.get_stride_sign(loop)

        # Get the loop end
        _, end = ad_utils.extract_loop_region_info(loop)

        # Reverse the initialization to start from the end of the forward loop
        # This will be incremented or decremented depending on the stride
        if stride_sign > 0:
            init_expr = f"{it} = {end}-1"
        else:
            init_expr = f"{it} = {end}+1"

        return init_expr

    def _reverse_loop_update_statement(self, loop: LoopRegion) -> str:
        """Given a loop region as a parameter, create the update statement for the reversed version of this loop."""

        # Get the original update statement
        fwd_update = loop.update_statement.as_string

        stride_sign = ad_utils.get_stride_sign(loop)

        # If the stride is positive
        if stride_sign > 0:
            update_statement = fwd_update.replace("+", "-")
        else:
            # If the stride is negative
            update_statement = fwd_update.replace("-", "+")

        return update_statement

    def _match_loop_region(self, fwd_loop: LoopRegion) -> LoopRegion:
        """Create the backward LoopRegion and fill it with the reversal of the forward LoopRegion."""

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
        """Given a LoopRegion as a parameter, reverse it, add the loop states that belong in this region."""

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
                if node in self.reversed_loops_map:
                    raise AutoDiffException(f"Loop {node} has already been reversed")
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
        """Add missing NestedSDFG connectors to the view for correctness.

        There is a special case for NestedSDFGs that we need to fix
        in the case where a NestedSDFG has an inout connector,
        but we only care about one of those connectors for the sake of AD.
        We need to add the missing connector for correctness.
        TODO: This is only a problem if the said connector is written to
              inside the NestedSDFG.
        """
        # In the case where a NestedSDFG has an inout connector,
        # but we only care about one of those connectors for the sake of AD
        # we need to add the missing connector for correctness
        # TODO: this is only a problem if the said connector is written to
        #       inside the NestedSDFG
        # Iterate over the nested SDFGs in the view
        for g in state_subgraph.nodes():
            if isinstance(g, nodes.NestedSDFG):

                inout_connectors = set(g.in_connectors).intersection(set(g.out_connectors))
                # If there are any inout connectors
                if len(inout_connectors) > 0:
                    out_connectors = {edge.src_conn: edge for edge in state.out_edges(g)}
                    in_connectors = {edge.dst_conn: edge for edge in state.in_edges(g)}
                    view_out_connectors = {edge.src_conn: edge for edge in state_subgraph.out_edges(g)}
                    view_in_connectors = {edge.dst_conn: edge for edge in state_subgraph.in_edges(g)}
                    for con in inout_connectors:
                        # Check if it is missing in the out or in connectors of the view
                        if con in view_out_connectors and con not in view_in_connectors:
                            # Get the equivalent in node and connector
                            edge = in_connectors[con]
                            if not isinstance(edge.src, nodes.AccessNode):
                                raise AutoDiffException(f"Expected AccessNode as source, got {type(edge.src)}")
                            view_nodes.append(edge.src)
                        if con not in view_out_connectors and con in view_in_connectors:
                            # Add the corresponding edge to the view
                            edge = out_connectors[con]
                            if not isinstance(edge.dst, nodes.AccessNode):
                                raise AutoDiffException(f"Expected AccessNode as destination, got {type(edge.dst)}")
                            view_nodes.append(edge.dst)

        return dstate.StateSubgraphView(state, view_nodes)

    def _compare_memlet_accesses_to_array_size(self, data_name: str, memlet: Memlet) -> int:
        """Compare the memlet range with the size of the array to see if the array is being overwritten."""
        total_size = self.backward_sdfg.arrays[data_name].total_size
        try:
            if total_size > memlet.num_accesses:
                return 1
            elif memlet.num_accesses == total_size:
                return 0

            # Something is wrong here raise an exception
            raise AutoDiffException(f"Memlet {memlet} has more accesses than the size of the data {data_name}")

        # If the comparison can not be made, return None
        except TypeError:
            return None

    def _get_reversed_parent_graph(self, forward_node: nodes.Node):
        """Given a node in the SDFG, get the reversed parent of this node."""
        fwd_parent_graph = forward_node.parent_graph

        if fwd_parent_graph == self.sdfg:
            parent_graph = self.backward_sdfg
        elif isinstance(fwd_parent_graph, SDFGState):
            parent_graph = self.reversed_states_map[fwd_parent_graph]
        elif isinstance(fwd_parent_graph, LoopRegion):
            parent_graph = self.reversed_loops_map[fwd_parent_graph]

        return parent_graph

    def _get_backward_loop_state_edge(self, forward_edge: dace.InterstateEdge) -> dace.InterstateEdge:
        """Given an edge from the forward pass, return the equivalent edge in the backward pass."""
        # Get the source and destination states
        forward_src = forward_edge.src
        forward_dst = forward_edge.dst

        if isinstance(forward_src, LoopRegion):
            fwd_src_is_loop = True
            if forward_src not in self.reversed_loops_map:
                raise AutoDiffException(f"Forward loop {forward_src} not found in reversed loops map")
        else:
            fwd_src_is_loop = False
            if forward_src not in self.reversed_states_map:
                raise AutoDiffException(f"Forward state {forward_src} not found in reversed states map")

        if isinstance(forward_dst, LoopRegion):
            fwd_dst_is_loop = True
            if forward_dst not in self.reversed_loops_map:
                raise AutoDiffException(f"Forward loop {forward_dst} not found in reversed loops map")
        else:
            fwd_dst_is_loop = False
            if forward_dst not in self.reversed_states_map:
                raise AutoDiffException(f"Forward state {forward_dst} not found in reversed states map")

        # Note that the source will become the destination
        backward_dst = self.reversed_states_map[forward_src] if not fwd_src_is_loop else self.reversed_loops_map[
            forward_src]
        backward_src = self.reversed_states_map[forward_dst] if not fwd_dst_is_loop else self.reversed_loops_map[
            forward_dst]

        # Each one of these in edges needs to have an equivalent
        # out edge in the backward part of the SDFG
        bwd_edge = None
        connection_state = backward_dst

        # Find the equivalent edge in the backward SDFG
        for b_edge in connection_state.parent_graph.in_edges(connection_state):
            if b_edge.src == backward_src:
                bwd_edge = b_edge
                break

        if not bwd_edge:
            raise AutoDiffException(f"Can't find the equivalent edge of {forward_edge} in the backward pass")

        return bwd_edge

    def _get_backward_state_edge(self, forward_edge: dace.InterstateEdge) -> dace.InterstateEdge:
        """Given an edge from the forward pass, return the equivalent edge in the backward pass."""
        # Get the source and destination states
        forward_state_src = forward_edge.src
        forward_state_dst = forward_edge.dst

        # Get the equivalent states in the backward pass
        if (forward_state_src not in self.reversed_states_map and forward_state_src not in self.reversed_loops_map):
            raise AutoDiffException(f"Forward state source {forward_state_src} not found in reversed maps")
        if (forward_state_dst not in self.reversed_states_map and forward_state_src not in self.reversed_loops_map):
            raise AutoDiffException(f"Forward state destination {forward_state_dst} not found in reversed maps")

        # Note that the src will become the destination
        backward_state_dst = self.reversed_states_map[
            forward_state_src] if forward_state_src in self.reversed_states_map else self.reversed_loops_map[
                forward_state_src]
        backward_state_src = self.reversed_states_map[
            forward_state_dst] if forward_state_dst in self.reversed_states_map else self.reversed_loops_map[
                forward_state_dst]

        # Each one of these in edges needs to have an equivalent
        # out edge in the backward part of the SDFG
        bwd_edge = None
        connection_state = backward_state_dst

        # Find the equivalent edge in the backward SDFG
        for b_edge in connection_state.parent_graph.in_edges(connection_state):
            if b_edge.src == backward_state_src:
                bwd_edge = b_edge
                break

        if not bwd_edge:
            raise AutoDiffException(f"Can't find the equivalent edge of {forward_edge} in the backward pass")

        return bwd_edge

    def _str_to_access(self, data: str, source: str) -> nodes.AccessNode:
        """Given a string containing the name of the accessed array, return the AccessNode in the state.

        Given a string containing the name of the accessed array, return the AccessNode in the state
        that points to this array.
        If there are multiple AccessNodes, the behavior will depend on whether we want
        an output or input AccessNode.
        Input: We will return the first occurrence of this node in the state and make sure there are
            only outgoing edges from this node.
        Output: We will return the last occurrence of this node in the state
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
            # There are multiple occurrences of the same AccessNode
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
        """Expand all library nodes in the sdfg to pure implementations.

        Returns whether something was expanded.
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
                if ONNX_AVAILABLE and isinstance(node, ONNXOp):
                    impls = ONNXForward.registered_implementations(node.schema.name)

                    # Order the implementations so that implementations containing "pure" are tried first
                    impls = [i for name, i in impls if "pure" in name] + [i for name, i in impls if "pure" not in name]
                    for impl in impls:
                        if impl.forward_can_be_applied(node, parent_graph, self.sdfg):
                            # Configure the module-level expansion class
                            ExpansionTemplate.environments = impl.environments if hasattr(impl, "environments") else []
                            ExpansionTemplate._impl = impl
                            ExpansionTemplate._match_node = xf.PatternNode(type(node))
                            ExpansionTemplate.apply_to(parent_graph.parent, verify=False, _match_node=node)
                            expanded_something = True
                            break

                # This could later on be changed to check if the expansion is differentiable and if not, move
                # on to the next expansion. For now we will just apply the first one that matches, prioritizing ones that
                # have "pure" in the name
                if isinstance(node, nodes.LibraryNode) and not (ONNX_AVAILABLE and isinstance(node, ONNXOp)):
                    # Try to select an expansion
                    if hasattr(node, "implementations"):
                        implementations = node.implementations

                        pure_candidates = [name for name, _ in sorted(implementations.items()) if "pure" in name]
                        if len(pure_candidates) > 0:
                            expansion = pure_candidates[0]
                        else:
                            expansion = node.implementation
                    else:
                        expansion = node.implementation

                    node.implementation = expansion
                    node.expand(parent_graph)
                    expanded_something = True

        return expanded_something

    def _get_node_state(self, node: nodes.Node) -> SDFGState:
        """Return the SDFG state that contains this node."""
        matches = []
        for state in self.sdfg.states():
            if node in state.nodes():
                matches.append(state)

        if len(matches) != 1:
            raise AutoDiffException(f"Expected exactly one match, got {len(matches)}")
        return matches[0]

    def _connect_conditional_map_exist(self, forward_state: SDFGState, backward_state: SDFGState,
                                       backward_map_exit: nodes.MapExit, fwd_tasklet: nodes.Tasklet):
        """Connect the map exit of a conditional tasklet to a new access node which will zero out the gradient.
        """

        if len(backward_map_exit.in_connectors) != 0:
            raise AutoDiffException(
                f"Expected no input connectors on backward map exit, got {len(backward_map_exit.in_connectors)}")

        # Add the in and out connectors for the zero-out operation
        backward_map_exit.add_in_connector("IN_zero_out")
        backward_map_exit.add_out_connector("OUT_zero_out")

        # Get the memlet data for the edge from the tasklet to the map exist
        tasklet_out_edge = forward_state.out_edges(fwd_tasklet)
        if len(tasklet_out_edge) != 1:
            raise AutoDiffException(f"Expected exactly one tasklet output edge, got {len(tasklet_out_edge)}")
        tasklet_out_edge = tasklet_out_edge[0]
        tasklet_memlet_path = forward_state.memlet_path(tasklet_out_edge)
        if len(tasklet_memlet_path) != 2:
            raise AutoDiffException(f"Expected tasklet memlet path of length 2, got {len(tasklet_memlet_path)}")

        # Copy the memlet and change the data name
        memlet_data = copy.deepcopy(tasklet_memlet_path[0].data)
        memlet_data.data = self.array_grad_map[memlet_data.data]

        # Get the reversed tasklet
        bwd_tasklet = self.reverse_map[fwd_tasklet]

        # Connect this map exist to the tasklet
        backward_state.add_edge(bwd_tasklet, "__zero_out_conn__", backward_map_exit, "IN_zero_out", memlet_data)

        # Replicate the target accedd node and connect it
        fwd_target_an: nodes.AccessNode = tasklet_memlet_path[-1].dst
        if not isinstance(fwd_target_an, nodes.AccessNode):
            raise AutoDiffException(f"Expected AccessNode for forward target, got {type(fwd_target_an)}")
        if fwd_target_an not in self.reverse_map:
            raise AutoDiffException(f"Forward target AccessNode {fwd_target_an} not found in reverse map")
        bwd_target_an = self.reverse_map[fwd_target_an]

        replicated_bwd_target_an = copy.deepcopy(bwd_target_an)
        backward_state.add_node(replicated_bwd_target_an)

        an_memlet_data: nodes.AccessNode = copy.deepcopy(tasklet_memlet_path[1].data)
        an_memlet_data.data = self.array_grad_map[an_memlet_data.data]
        backward_state.add_edge(backward_map_exit, "OUT_zero_out", replicated_bwd_target_an, None, an_memlet_data)

        # We need to get the map entry that starts the conditional block
        # First get the conditional tasklet
        conditional_block = self._extract_conditional_array_assignment_block(
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
        """Check if this tasklet contains a conditional.

        This only happens in conditional array assignments and requires special treatment in reversing the graph.
        """
        # sanity check
        if not isinstance(tasklet_node, nodes.Tasklet):
            raise AutoDiffException(f"Expected Tasklet node, got {type(tasklet_node)}")

        # get the code string and check if there is an if
        # TODO: How to more accurately check this?
        return "if" in tasklet_node.code.as_string

    def _conditional_nested_sdfg(self, forward_state: SDFGState, node: nodes.NestedSDFG):
        """Check if this NestedSDFG contains a conditional.

        This only happens in conditional array assignments and requires special treatment in reversing the graph.
        """
        # sanity check
        if not isinstance(node, nodes.NestedSDFG):
            raise AutoDiffException(f"Expected NestedSDFG node, got {type(node)}")

        # get the incoming edges to the sdfg
        in_edges = forward_state.in_edges(node)

        # check if any of the incoming edges are boolean edges
        for edge in in_edges:
            if self.sdfg.arrays[edge.data.data].dtype == dace.bool:
                return True

        # get the code string and check if there is an if
        return False

    def _extract_conditional_array_assignment_block(self, forward_state: SDFGState, tasklet_node: nodes.Node,
                                                    subgraph: dstate.SubgraphView):
        """Extract a conditional array assignment block.

        Given a conditional tasklet, check if this is a conditional array assignment of the type
        A[A>=0 and A<=5] = cst. At the moment the function only supports constant assignments.
        """
        try:

            if not isinstance(tasklet_node, nodes.Tasklet):
                raise AutoDiffException(f"Expected Tasklet node, got {type(tasklet_node)}")
            # This applies to both Tasklet and NestedSDFG nodes
            # get the AccessNode containing the boolean values for this assignment
            tasklet_in_edges = forward_state.in_edges(tasklet_node)
            tasklet_boolean_edge = None
            single_boolean_edge_found = False
            for edge in tasklet_in_edges:
                edge_type = self.sdfg.arrays[edge.data.data].dtype
                if edge_type == dace.bool:
                    # sanity check
                    if single_boolean_edge_found:
                        # we expect there to be a single AccessNode where the booleans come from
                        raise AutoDiffException(
                            "Multiple boolean edges found for conditional assignment. Expected only one.")
                    tasklet_boolean_edge = edge
                    single_boolean_edge_found = True

            if tasklet_boolean_edge is None:
                raise AutoDiffException("Expected to find a boolean edge for conditional assignment")
            tasklet_in_memlet_path = forward_state.memlet_path(tasklet_boolean_edge)
            # the first element in the path is the boolean AN
            bools_an = tasklet_in_memlet_path[0].src
            if not isinstance(bools_an, nodes.AccessNode):
                raise AutoDiffException(f"Expected AccessNode for boolean values, got {type(bools_an)}")

            # save all the nodes in the path to the assignment block list
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
        """Get gradient nodes that need tracking in conditional assignments.

        When extracting the block for a conditional assignment, we need to make sure we keep tracking
        the required gradient AccessNodes.
        This function checks all the required access nodes that are in the conditional block.
        At the moment this is just the target access node.
        """
        nodes_to_track: List[nodes.AccessNode] = []
        gradient_nodes = [n for n in self.required_gradients_data]
        gradient_nodes += [n for n in self.given_gradients_data]

        # get the subgraph difference
        difference = set(subgraph.nodes()).difference(set(block_nodes))

        # go through all the access nodes in the conditional block
        for node in block_nodes:
            if not isinstance(node, nodes.AccessNode):
                continue

            # we always want to track the gradient nodes
            if node.data in gradient_nodes:
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
                          subgraph: dstate.StateSubgraphView) -> None:
        """Reverse a given subgraph by reversing all nodes within it.

        :param forward_state: The forward state containing the subgraph.
        :param backward_state: The backward state to add reversed nodes to.
        :param subgraph: The subgraph view containing nodes to reverse.
        """

        # Conditional assignment nodes
        conditional_assignment_nodes: List[nodes.Node] = []

        # A reversed topological sort is a topological sort on the reverse graph
        for node in reversed(list(dace_utils.dfs_topological_sort(subgraph, subgraph.source_nodes()))):

            try:
                # If this node is a part of the conditional assignment block, we skip it
                if node in conditional_assignment_nodes:
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
                    # extract the conditional assignment block or fail if this is an unexpected structure
                    conditional_block = self._extract_conditional_array_assignment_block(forward_state=forward_state,
                                                                                         tasklet_node=node,
                                                                                         subgraph=subgraph)

                    # add these nodes to be skipped in the future
                    conditional_assignment_nodes.extend(conditional_block)

                # If the node is an AccessNode and it is being overwritten in the forward pass,
                # we need to zero-out the gradients of the overwritten values
                if isinstance(node, nodes.AccessNode):
                    # Check if there is an incoming edge to this node
                    incoming_edges = forward_state.in_edges(node)

                    # If there is an incoming edge, we need to zero-out the gradient
                    for edge in incoming_edges:

                        # Check, if possible, if the written subset is not zero
                        write_size = edge.data.subset.num_elements()

                        # Check if the node doesn't have a WCR
                        # If it does, this is not an overwrite and the gradients should not be cleared
                        has_wcr = edge.data.wcr is not None

                        # Check if the edge is dynamic, this means not all values are overwritten
                        # We will skip zeroing out the gradient in this case
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

                # Cleanup of isolated nodes
                # We will have an isolated node if it is not connected to any other node in the state view
                # And it has not been cleared out if it is an AccessNode
                # Isolated nodes should only appear from clearing out gradients
                # Check if this is an isolated node and remove it if it is
                if backward_state.out_degree(reversed_node) == 0 and backward_state.in_degree(reversed_node) == 0:
                    if isinstance(node, nodes.AccessNode) and node not in self.zeroed_out:
                        backward_state.remove_node(reversed_node)

            except AutoDiffException as e:
                raise AutoDiffException("Failed at node {}: {}".format(node, str(e))) from e

    def _set_wcr_if_needed(self, backward_state: SDFGState, backward_node: nodes.Node,
                           edge: dstate.MultiConnectorEdge) -> None:
        """Set write-conflict resolution (WCR) for gradient accumulation if needed.

        If this AccessNode represents a gradient that has already been used elsewhere,
        we want to accumulate the gradients rather than overwrite them.

        :param backward_state: The backward state containing the edge.
        :param backward_node: The backward node (should be AccessNode for gradients).
        :param edge: The edge that may need WCR for gradient accumulation.
        """

        # Check if the forward node is an AccessNode
        if not isinstance(backward_node, nodes.AccessNode):
            return

        # Otherwise, we add up the gradients, not overwrite them
        for tree_edge in backward_state.memlet_tree(edge):
            tree_edge.data.wcr = "lambda x, y: x + y"

    def _connect_given_gradients(self, forward_state: SDFGState, backward_state: SDFGState,
                                 subgraph: dstate.StateSubgraphView, forward_node: nodes.Node) -> Optional[SDFGState]:
        """Connect output gradients of forward_node as inputs to the corresponding reverse node.

        :param forward_state: The forward state containing the node.
        :param backward_state: The backward state to add connections to.
        :param subgraph: The subgraph view for the current operation.
        :param forward_node: The forward node whose output gradients to connect.
        :return: The backward state (possibly modified) or None.
        """
        new_backward_state = None
        # First, create the data descriptor if this is an access node and it hasn't been added before
        if isinstance(forward_node, nodes.AccessNode):
            grad_name = self.array_grad_name(forward_node.data)
            if grad_name not in self.backward_sdfg.arrays:
                # This grad hasn't been written before: initialize it
                self._add_gradient_data_descriptor(forward_node.data)

        for edge in subgraph.out_edges(forward_node):
            if not ad_utils.path_src_node_in_subgraph(edge, subgraph) or edge.dst not in self.reverse_map:
                if edge.dst in self.conditional_block_entry:
                    backward_node = self.reverse_map[edge.src]
                    if not isinstance(edge.dst, nodes.MapEntry):
                        raise AutoDiffException(f"Expected MapEntry in conditional block, got {type(edge.dst)}")
                    conditional_zero_out_an = self.conditional_block_entry[edge.dst]
                    # Add an empty edge to skip the conditional block
                    backward_state.add_edge(conditional_zero_out_an, None, backward_node, None, Memlet())
                # skip connecting edges for which we don't need to generate grads.
                continue

            # Skip connecting boolean edges
            if self.sdfg.arrays[edge.data.data].dtype == dace.bool:
                # we also need to remove this connector otherwise it will be dangling
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

            _, output_conn, dest_node, input_conn, fwd_memlet = edge

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
                                backward_node: nodes.Node, required_inputs: Dict[str, str]) -> None:
        """Connect the reversed node to all required non-gradient inputs.

        This function handles non-trivial routing scenarios:
        1. When reading from an AccessNode in forward pass, route through maps in backward pass
        2. Save connector values to arrays when backward pass needs to read them

        Currently supports two strategies: store-all and recompute-all.

        :param state: Forward state containing the forward node.
        :param backward_state: Backward state containing the backward node.
        :param forward_node: The forward pass node.
        :param backward_node: The backward pass node (not necessarily a reversed node).
        :param required_inputs: Maps forward pass connector names to backward pass connector names.
        :raises AutoDiffException: If required connectors don't exist on forward node.
        """

        if set(required_inputs).difference(forward_node.in_connectors):
            missing_connectors = \
                set(required_inputs).difference(forward_node.in_connectors)
            raise AutoDiffException(f"Cannot connect connectors {missing_connectors} to {backward_node} "
                                    f"because they don't exist on the corresponding forward node {forward_node}")

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
                if not isinstance(starting_an, nodes.AccessNode):
                    raise AutoDiffException(f"Expected AccessNode at start of memlet path, got {type(starting_an)}")

                # Save the information about the data to be forwarded
                # to call the function to connect this required AccessNode
                # after the reversal
                self.data_to_forward.append((state, backward_state, starting_an, forward_node, edge))

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
                self.data_to_forward.append((state, backward_state, an_to_connect, forward_node, edge))

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
                        # If this is a view, we need to connect it to the AccessNode it is viewing
                        edge_src_in_edge = state.in_edges(edge_src)

                        # A view should only have one incoming edge
                        assert len(edge_src_in_edge) == 1
                        edge_src_in_edge = edge_src_in_edge[0]

                        # Replicate the viewed node and its memlet and connect it
                        view_origin = edge_src_in_edge.src
                        replicated_view = copy.deepcopy(view_origin)
                        view_memlet = copy.deepcopy(edge_src_in_edge.data)
                        if self.separate_sdfgs:
                            # If the SDFGs are separate, we need to add the descriptor for this data
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

    def _lookup_required_grad_name(self, node: nodes.Node, connector: str) -> str:
        """Look up the required gradient name for a given node and connector.

        :param node: The forward pass node.
        :param connector: The connector name to look up.
        :return: The required gradient name for the connector.
        :raises AutoDiffException: If the node's backward result is not available.
        """
        if node not in self.result_map:
            raise AutoDiffException(f"Attempted to access required gradient of {node} "
                                    f"before the backward node was created")
        return self.result_map[node].required_grad_names[connector]

    def _lookup_given_grad_name(self, node: nodes.Node, connector: str) -> str:
        """Look up the given gradient name for a given node and connector.

        :param node: The forward pass node.
        :param connector: The connector name to look up.
        :return: The given gradient name for the connector.
        :raises AutoDiffException: If the node's backward result is not available.
        """
        if node not in self.result_map:
            raise AutoDiffException(f"Attempted to access given gradient of {node} "
                                    f"before the backward node was created")
        return self.result_map[node].given_grad_names[connector]

    def _find_backward_entry_node_for_map_entry(self, backward_state: SDFGState,
                                                entry_node: nodes.MapEntry) -> nodes.MapEntry:
        """Find the entry node in the backward pass corresponding to a forward pass entry node.

        :param backward_state: The backward state to search in.
        :param entry_node: The MapEntry node from the forward pass.
        :return: The corresponding MapEntry node in the backward pass.
        :raises AutoDiffException: If exactly one corresponding node is not found.
        """
        src_candidates = [
            node for node in backward_state.nodes()
            if isinstance(node, nodes.MapEntry) and node.map == self.reverse_map[entry_node.map]
        ]
        if len(src_candidates) != 1:
            raise AutoDiffException(f"Expected exactly one backward MapEntry for forward MapEntry {entry_node}, "
                                    f"but found {len(src_candidates)} candidates")

        return src_candidates[0]

    def _get_reverse_node(self, state: SDFGState, backward_state: SDFGState, node: nodes.Node,
                          given_gradients: List[str],
                          required_gradients: List[str]) -> Tuple[nodes.Node, BackwardResult]:
        """Add the reverse node for a node from the forward pass to the backward pass.

        Resolution order:
        1) Check for methods on this class
        2) Check the backward pass repository

        :param state: Forward state containing the node.
        :param backward_state: Backward state to add the reverse node to.
        :param node: Node from the forward pass to reverse.
        :param given_gradients: Output names on the forward node for gradient input connections.
        :param required_gradients: Input names on the forward node that need gradients generated.
        :return: Tuple of (reversed node, BackwardResult with gradient connector names).
        :raises AutoDiffException: If no backward implementation is found for the node type.
        """

        # (1)
        if hasattr(self.dace_node_impl, "_reverse_" + type(node).__name__):
            reverse_method = getattr(self.dace_node_impl, f"_reverse_{type(node).__name__}")
            return reverse_method(state, backward_state, node, given_gradients, required_gradients)

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
            if isinstance(backward_node, nodes.LibraryNode) and hasattr(node, 'schedule'):
                backward_node.schedule = node.schedule
            return backward_node, backward_result

        raise AutoDiffException(f"Unable to differentiate node type {type(node)}. "
                                f"Either add a pure forward implementation or a backward implementation to progress.")
