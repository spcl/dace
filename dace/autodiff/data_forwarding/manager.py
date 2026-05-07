# Copyright 2019-2025 ETH Zurich and the DaCe authors. All rights reserved.
import copy
from typing import List, Tuple, Optional

# DaCe imports
import dace.sdfg.nodes as nodes
from dace import config, data as dt
from dace.sdfg import SDFGState, graph as dgraph

# Autodiff imports
from dace.autodiff.base_abc import AutoDiffException
import dace.autodiff.utils as ad_utils
import dace.autodiff.data_forwarding as data_forwarding


class DataForwardingManager:

    def __init__(self, bwd_generator: 'BackwardPassGenerator'):

        # The user specified strategy for forwarding
        # Whether to forward data through separate SDFGs
        self.bwd_generator: 'BackwardPassGenerator' = bwd_generator

    def forward_data_to_backward_pass(self) -> None:
        """
        Iterate through all the data that needs to be forwarded to the backward pass states.
        """
        # Get the strategy decision for each data that needs to be forwarded to the backward pass
        strategy_choice, recomputation_nsdfgs = self._get_overwrite_resolution_strategy()

        # Make the connection according to the chosen strategy
        for index, (forward_state, backward_state, access_node, node,
                    edge) in enumerate(self.bwd_generator.data_to_forward):
            self._connect_forward_accessnode(forward_state, backward_state, access_node, node, edge,
                                             recomputation_nsdfgs[index], strategy_choice[index])

    def _get_overwrite_resolution_strategy(self) -> Tuple[List[str], List[Optional[nodes.NestedSDFG]]]:
        """
        Choose a strategy for resolving overwritten data that we need to forward to the backward pass.
        If the user wants a specific strategy, we use it.
        Otherwise, we evaluate what strategy is best for this specific node.
        """
        strategy_choice: List[str] = []
        recomputation_nsdfgs: List[Optional[nodes.NestedSDFG]] = []

        # As preprocessing step,
        # We will store all of the global program inputs,
        # if they are required for the backward pass
        # NOTE: This can be relaxed since if an input is not overwritten
        # it can be recomputed
        to_remove = []
        for i, (forward_state, backward_state, access_node, node,
                edge) in enumerate(self.bwd_generator.data_to_forward):
            if access_node.data not in self.bwd_generator.sdfg.arg_names:
                continue

            # Store the input
            self._connect_forward_accessnode(forward_state, backward_state, access_node, node, edge, None, "store")

            # Remove this element from the list of the data to forward
            to_remove.append(i)

        # Remove elements from the list of data to be forwarded (in reverse order to maintain indices)
        for idx in sorted(to_remove, reverse=True):
            del self.bwd_generator.data_to_forward[idx]

        if self.bwd_generator.data_forwarding_strategy == "store_all":
            strategy_choice = ["store"] * len(self.bwd_generator.data_to_forward)

            # A recomputation block is not necessary
            recomputation_nsdfgs = [None] * len(self.bwd_generator.data_to_forward)
        elif self.bwd_generator.data_forwarding_strategy == "recompute_all":
            strategy_choice = ["recompute"] * len(self.bwd_generator.data_to_forward)

            # We will delay getting the recomputation block for now
            recomputation_nsdfgs = [None] * len(self.bwd_generator.data_to_forward)
        elif self.bwd_generator.data_forwarding_strategy == "user_defined":
            if self.bwd_generator.data_to_recompute is None:
                raise AutoDiffException("The overwrite resolution strategy is User Defined "
                                        "but no recomputation list has been provided."
                                        "Please set the data_to_recompute parameter.")

            for forward_state, backward_state, access_node, node, edge in self.bwd_generator.data_to_forward:

                if access_node.data in self.bwd_generator.data_to_recompute:
                    try:
                        nsdfg = data_forwarding.get_recomputation_nsdfg(self.bwd_generator, forward_state, access_node)
                        choice = "recompute"
                    except Exception as e:
                        # If anything goes wrong, print a warning and fall back to storing
                        if config.Config.get_bool('debugprint'):
                            print(
                                f"Warning: Couldn't get the recomputation nested SDFG for {access_node.label} because {e}"
                            )
                        nsdfg = None
                        choice = "store"
                    recomputation_nsdfgs.append(nsdfg)
                    strategy_choice.append(choice)
                else:
                    # We store everything else
                    recomputation_nsdfgs.append(None)
                    strategy_choice.append("store")
        else:
            raise AutoDiffException("Please specify a valid overwrite resolution strategy. "
                                    "Expected either store_all, recompute_all, or user_defined "
                                    f"but got {self.bwd_generator.data_forwarding_strategy}")
        return strategy_choice, recomputation_nsdfgs

    def _connect_forward_accessnode(self, forward_state: SDFGState, backward_state: SDFGState,
                                    forward_node: nodes.AccessNode, target_node: nodes.Node,
                                    starting_edge: dgraph.MultiConnectorEdge,
                                    recomputation_nsdfg: Optional[nodes.NestedSDFG], strategy: str):
        """
        We need to forward an array from the forward pass to the backward pass.
        To do this we first check if this array has been overwritten or not.
        If the array has not been overwritten, we just need to replicate it
        in the backward pass and then forward it.
        If the array has been overwritten, we pick a strategy for this AccessNode:
            - Store strategy:
                - We modify the forward pass to save the values in a new array
                - Connect this new array to the node in the backward pass
            - Recomputation:
                - Add the recomputation as a NestedSDFG
                - Connect the output of the NestedSDFG to the node in the backward pass
        """

        # First, we check if the node has been overwritten
        overwritten, recomputable = self._check_node_overwrite(forward_state=forward_state, node=forward_node)

        # Boolean indicating whether we should fall back to storing
        fallback = False
        if strategy == "recompute" and recomputable:
            try:
                if recomputation_nsdfg is None:
                    recomputation_nsdfg = data_forwarding.get_recomputation_nsdfg(self.bwd_generator,
                                                                                  forward_state,
                                                                                  target_an=forward_node)
                data_forwarding.resolve_overwrite_with_recomputation(recomputation_nsdfg=recomputation_nsdfg,
                                                                     forward_state=forward_state,
                                                                     backward_state=backward_state,
                                                                     target_an=forward_node,
                                                                     target_node=target_node,
                                                                     starting_edge=starting_edge)
            except Exception as e:
                # If anything goes bad, print a warning and fall back to storing
                if config.Config.get_bool('debugprint'):
                    print(f"Warning: Failed to recompute {forward_node.data}: {e}. Falling back to storing")
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

            data_forwarding.resolve_overwrite_with_store(bwd_generator=self.bwd_generator,
                                                         forward_state=forward_state,
                                                         backward_state=backward_state,
                                                         forward_node=forward_node,
                                                         target_node=target_node,
                                                         starting_edge=starting_edge)

    def _check_node_overwrite(self, forward_state: SDFGState, node: nodes.AccessNode) -> Tuple[bool, bool]:
        """
        Given an AccessNode from the forward state, check if the data of this node has changed.
        We look at all the AccessNodes with the same data that occur after the 'node' parameter
        if any of them has an incoming edge, return the node has been overwritten.

        :param node: the AccessNode to perform the check for.
        :return: a tuple of whether this node has been overwritten, and if it can be recomputed
        """
        overwritten = False
        decided = False
        recomputable = False

        # Get the descendant and ascendant states to look in for an overwrite
        if forward_state not in self.bwd_generator.state_order:
            raise AutoDiffException(f"Forward state {forward_state} not found in state order")
        index = self.bwd_generator.state_order.index(forward_state)
        descendant_states = self.bwd_generator.state_order[index:]

        # Check if this access node is a view
        if isinstance(node.desc(self.bwd_generator.sdfg), dt.ArrayView):
            # The view should have one incoming edge from the original access node
            in_edges = forward_state.in_edges(node)

            # Sanity checks
            if len(in_edges) != 1:
                raise AutoDiffException(f"Expected exactly one incoming edge for view node {node}, got {len(in_edges)}")
            if "views" not in node.in_connectors:
                raise AutoDiffException(f"Expected 'views' connector in node {node}, but not found")

            # We want to check if the source has been overwritten
            node = in_edges[0].src

        # Get all the AccessNodes with the same data
        matches = []
        for d_state in descendant_states:
            matches += [(nd, parent) for nd, parent in d_state.all_nodes_recursive()
                        if isinstance(nd, nodes.AccessNode) and nd.data == node.data]

        # There needs to be at least one occurrence which is the node passed as a parameter
        if len(matches) == 0 or (node, forward_state) not in matches:
            raise AutoDiffException(f"Node {node} not found in descendant states")

        # If there is only one occurrence of this data, it will not be overwritten later in the graph
        if len(matches) == 1:
            overwritten = False
            decided = True

        # Get the index of the parameter node
        index = matches.index((node, forward_state))

        # If the parameter node is the last occurrence in the descendant states,
        # it will not be overwritten
        if len(matches) - 1 == index:
            overwritten = False
            decided = True

        # If we haven't already confirmed that this node has not been overwritten
        if not decided:
            # Iterate through all the successor occurrences
            for nd, parent in matches[index + 1:]:
                # Check if this node has an incoming edge
                if len(parent.in_edges(nd)) > 0:
                    overwritten = True

        if not overwritten:
            # There is no overwrite so far
            # Check if this state is within a loop
            is_in_loop, loop = ad_utils.state_within_loop(forward_state)
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
                            if edge.src.data not in self.bwd_generator.read_only_arrays:
                                all_read_only = False
                                break
                            # Check if the data is read only
                    if all_read_only:
                        overwritten = False

        # Iterate through all the predecessor occurrences
        for nd, parent in matches[:index + 1]:
            # Check if this node has an incoming edge
            if len(parent.in_edges(nd)) > 0:
                recomputable = True
        return overwritten, recomputable

    def _connect_forward_accessnode_not_overwritten(self,
                                                    forward_state: SDFGState,
                                                    backward_state: SDFGState,
                                                    forward_node: nodes.AccessNode,
                                                    target_node: nodes.Node,
                                                    starting_edge: dgraph.MultiConnectorEdge,
                                                    replicated_node: Optional[nodes.AccessNode] = None):
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
            if self.bwd_generator.separate_sdfgs:
                # Need to copy over the descriptor from the forward pass
                data_name = replicated_node.data
                data_desc = copy.deepcopy(forward_node.desc(self.bwd_generator.sdfg))
                data_desc.transient = False
                if data_name not in self.bwd_generator.backward_sdfg.arrays:
                    self.bwd_generator.backward_sdfg.add_datadesc(data_name, data_desc)

                # We also need to forward this array
                if data_name not in self.bwd_generator.backward_input_arrays:
                    # If the data is needed inside a NestedSDFG
                    # This will make sure the added array is correctly forwarded
                    # and an in connector to the NestedSDFG is added
                    self.bwd_generator.backward_input_arrays[data_name] = data_desc

        # We replicate the exact link between this forward access node and the target node
        # Get all the edges in the path
        all_edges_inbetween = ad_utils.get_all_path_edges(state=forward_state,
                                                          source=forward_node,
                                                          starting_edge=starting_edge)

        # A dictionary to keep track of temporary nodes in the path
        replicated_tmp_nodes = {}

        # For each edge in the path
        for edge in all_edges_inbetween:
            src, src_conn, dst, dst_conn, data = edge
            bwd_src, bwd_src_conn, bwd_dst, bwd_dst_conn, bwd_data = src, src_conn, dst, dst_conn, copy.deepcopy(data)

            # If the destination is a map entry,
            if isinstance(dst, nodes.MapEntry):
                # We need to get the corresponding map entry in the backward pass.
                bwd_dst = self.bwd_generator._find_backward_entry_node_for_map_entry(backward_state=backward_state,
                                                                                     entry_node=dst)
                # Add the dst connector to the map
                added = bwd_dst.add_in_connector(bwd_dst_conn)
                assert added

            # If the destination is a map entry,
            if isinstance(src, nodes.MapEntry):
                # We need to get the corresponding map entry in the backward pass.
                bwd_src = self.bwd_generator._find_backward_entry_node_for_map_entry(backward_state=backward_state,
                                                                                     entry_node=src)
                # Add the src connector to the map
                added = bwd_src.add_out_connector(bwd_src_conn)
                assert added

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
                bwd_dst = self.bwd_generator.reverse_map[dst]
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
        data_desc = self.bwd_generator.sdfg.arrays[forward_node.data]
        if isinstance(forward_node, nodes.AccessNode) and isinstance(data_desc, dt.View):
            if self.bwd_generator.separate_sdfgs:
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
                if self.bwd_generator.separate_sdfgs:
                    # if the sdfgs are separate, we need to add the descriptor for this data
                    origin_desc = self.bwd_generator.sdfg.arrays[view_origin.data]
                    origin_desc.transient = False
                    backward_state.sdfg.add_datadesc(view_origin.data, origin_desc)
                backward_state.add_edge(replicated_view, None, replicated_node, "views", view_memlet)
