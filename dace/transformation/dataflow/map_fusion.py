# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
"""Implements the serial map fusing transformation."""

import copy
from typing import Any, Dict, List, Optional, Set, Tuple, Union, Iterable

import dace
from dace import data, dtypes, properties, subsets, symbolic, transformation
from dace.sdfg import SDFG, SDFGState, graph, nodes, validation
from dace.transformation import helpers


@properties.make_properties
class MapFusion(transformation.SingleStateTransformation):
    """Fuse two serial maps together.

    The transformation combines two maps into one that are connected through some
    access nodes. Conceptually this transformation removes the exit of the first
    or upper map and the entry of the lower or second map and then rewrites the
    connections appropriately. Depending on the situation the transformation will
    either fully remove or make the intermediate a new output of the second map.

    By default `strict_dataflow` is enabled. In this mode the transformation is
    more conservative. The main difference is, that it will not adjust the
    subsets of the intermediate, i.e. turning an array with shape `(1, 1, 1, 1)`
    into a scalar.
    Furthermore, shared intermediates, see `partition_first_outputs()` will only
    be created if the data is not referred downstream in the dataflow.

    Args:
        only_inner_maps: Only match Maps that are internal, i.e. inside another Map.
        only_toplevel_maps: Only consider Maps that are at the top.
        strict_dataflow: Which dataflow mode should be used, see above.

    Notes:
        - This transformation modifies more nodes than it matches.
        - After the transformation has been applied simplify should be run to remove
            some dead data flow, that was introduced to ensure validity.
        - A `MapFusion` obejct can be initialized and be reused. However,
            after new access nodes are added to any state, it is no longer valid
            to use the object.

    Todo:
        - Consider the case that only shared nodes are created (thus no inspection of
            the graph is needed) and make all shared. Then use the dead dataflow
            elimination transformation to get rid of the ones we no longer need.
        - Increase the applicability.
    """

    # Pattern Nodes
    map_exit_1 = transformation.transformation.PatternNode(nodes.MapExit)
    intermediate_access_node = transformation.transformation.PatternNode(nodes.AccessNode)
    map_entry_2 = transformation.transformation.PatternNode(nodes.MapEntry)


    # Settings
    only_toplevel_maps = properties.Property(
        dtype=bool,
        default=False,
        desc="Only perform fusing if the Maps are in the top level.",
    )
    only_inner_maps = properties.Property(
        dtype=bool,
        default=False,
        desc="Only perform fusing if the Maps are inner Maps, i.e. does not have top level scope.",
    )
    strict_dataflow = properties.Property(
        dtype=bool,
        default=True,
        desc="If `True` then the transformation will ensure a more stricter data flow.",
    )
    # Maps SDFGs to the set of data that can not be removed,
    #  because they transmit data _between states_, such data will be made 'shared'.
    #  This variable acts as a cache, and is managed by 'is_shared_data()'.
    _shared_data: Dict[SDFG, Set[str]]


    def __init__(
        self,
        only_inner_maps: Optional[bool] = None,
        only_toplevel_maps: Optional[bool] = None,
        strict_dataflow: Optional[bool] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        if only_toplevel_maps is not None:
            self.only_toplevel_maps = bool(only_toplevel_maps)
        if only_inner_maps is not None:
            self.only_inner_maps = bool(only_inner_maps)
        if strict_dataflow is not None:
            self.strict_dataflow = bool(strict_dataflow)
        self._shared_data = {}


    @classmethod
    def expressions(cls) -> Any:
        """Get the match expression.

        The transformation matches the exit node of the top Map that is connected to
        an access node that again is connected to the entry node of the second Map.
        An important note is, that the transformation operates not just on the
        matched nodes, but more or less on anything that has an incoming connection
        from the first Map or an outgoing connection to the second Map entry.
        """
        return [dace.sdfg.utils.node_path_graph(cls.map_exit_1, cls.intermediate_access_node, cls.map_entry_2)]


    def can_be_applied(
        self,
        graph: Union[SDFGState, SDFG],
        expr_index: int,
        sdfg: dace.SDFG,
        permissive: bool = False,
    ) -> bool:
        """Tests if the matched Maps can be merged.

        The two Maps are mergeable iff:
        - Checks general requirements, see `can_topologically_be_fused()`.
        - Tests if there are read write dependencies.
        - Tests if the decomposition exists.
        """
        map_entry_1: nodes.MapEntry = graph.entry_node(self.map_exit_1)
        map_exit_1: nodes.MapExit = self.map_exit_1
        map_entry_2: nodes.MapEntry = self.map_entry_2

        # This essentially test the structural properties of the two Maps.
        if not self.can_topologically_be_fused(map_entry_1=map_entry_1, map_entry_2=map_entry_2, graph=graph, sdfg=sdfg):
            return False

        # Test for read-write conflicts
        if self.has_read_write_dependency(
                map_entry_1=map_entry_1,
                map_entry_2=map_entry_2,
                state=graph,
                sdfg=sdfg,
        ):
            return False

        # Two maps can be serially fused if the node decomposition exists and
        #  at least one of the intermediate output sets is not empty. The state
        #  of the pure outputs is irrelevant for serial map fusion.
        output_partition = self.partition_first_outputs(
            state=graph,
            sdfg=sdfg,
            map_exit_1=map_exit_1,
            map_entry_2=map_entry_2,
        )
        if output_partition is None:
            return False
        _, exclusive_outputs, shared_outputs = output_partition
        if not (exclusive_outputs or shared_outputs):
            return False
        return True


    def apply(self, graph: Union[dace.SDFGState, dace.SDFG], sdfg: dace.SDFG) -> None:
        """Performs the serial Map fusing.

        The function first computes the map decomposition and then handles the
        three sets. The pure outputs are handled by `relocate_nodes()` while
        the two intermediate sets are handled by `handle_intermediate_set()`.

        By assumption we do not have to rename anything.

        Args:
            graph: The SDFG state we are operating on.
            sdfg: The SDFG we are operating on.
        """
        # NOTE: `self.map_*` actually stores the ID of the node.
        #  once we start adding and removing nodes it seems that their ID changes.
        #  Thus we have to save them here, this is a known behaviour in DaCe.
        assert isinstance(graph, dace.SDFGState)
        assert isinstance(self.map_exit_1, nodes.MapExit)
        assert isinstance(self.map_entry_2, nodes.MapEntry)

        map_exit_1: nodes.MapExit = self.map_exit_1
        map_entry_2: nodes.MapEntry = self.map_entry_2
        map_exit_2: nodes.MapExit = graph.exit_node(self.map_entry_2)
        map_entry_1: nodes.MapEntry = graph.entry_node(self.map_exit_1)

        # Before we do anything we perform the renaming.
        self.rename_map_parameters(
            first_map=map_exit_1.map,
            second_map=map_entry_2.map,
            second_map_entry=map_entry_2,
            state=graph,
        )

        output_partition = self.partition_first_outputs(
            state=graph,
            sdfg=sdfg,
            map_exit_1=map_exit_1,
            map_entry_2=map_entry_2,
        )
        assert output_partition is not None  # Make MyPy happy.
        pure_outputs, exclusive_outputs, shared_outputs = output_partition

        if len(exclusive_outputs) != 0:
            self.handle_intermediate_set(
                intermediate_outputs=exclusive_outputs,
                state=graph,
                sdfg=sdfg,
                map_exit_1=map_exit_1,
                map_entry_2=map_entry_2,
                map_exit_2=map_exit_2,
                is_exclusive_set=True,
            )
        if len(shared_outputs) != 0:
            self.handle_intermediate_set(
                intermediate_outputs=shared_outputs,
                state=graph,
                sdfg=sdfg,
                map_exit_1=map_exit_1,
                map_entry_2=map_entry_2,
                map_exit_2=map_exit_2,
                is_exclusive_set=False,
            )
        assert pure_outputs == set(graph.out_edges(map_exit_1))
        if len(pure_outputs) != 0:
            self.relocate_nodes(
                from_node=map_exit_1,
                to_node=map_exit_2,
                state=graph,
                sdfg=sdfg,
            )

        # Above we have handled the input of the second map and moved them
        #  to the first map, now we must move the output of the first map
        #  to the second one, as this one is used.
        self.relocate_nodes(
            from_node=map_entry_2,
            to_node=map_entry_1,
            state=graph,
            sdfg=sdfg,
        )

        for node_to_remove in [map_exit_1, map_entry_2]:
            assert graph.degree(node_to_remove) == 0
            graph.remove_node(node_to_remove)

        # Now turn the second output node into the output node of the first Map.
        map_exit_2.map = map_entry_1.map


    def partition_first_outputs(
        self,
        state: SDFGState,
        sdfg: SDFG,
        map_exit_1: nodes.MapExit,
        map_entry_2: nodes.MapEntry,
    ) -> Union[
            Tuple[
                Set[graph.MultiConnectorEdge[dace.Memlet]],
                Set[graph.MultiConnectorEdge[dace.Memlet]],
                Set[graph.MultiConnectorEdge[dace.Memlet]],
            ],
            None,
    ]:
        """Partition the output edges of `map_exit_1` for serial map fusion.

        The output edges of the first map are partitioned into three distinct sets,
        defined as follows:
        - Pure Output Set `\mathbb{P}`:
            These edges exits the first map and does not enter the second map. These
            outputs will be simply be moved to the output of the second map.
        - Exclusive Intermediate Set `\mathbb{E}`:
            Edges in this set leaves the first map exit, enters an access node, from
            where a Memlet then leads immediately to the second map. The memory
            referenced by this access node is not used anywhere else, thus it can
            be removed.
        - Shared Intermediate Set `\mathbb{S}`:
            These edges are very similar to the one in `\mathbb{E}` except that they
            are used somewhere else, thus they can not be removed and have to be
            recreated as output of the second map.

        If strict data flow mode is enabled the function is rather strict if an
        output can be added to either intermediate set and might fail to compute
        the partition, even if it would exist.

        Returns:
            If such a decomposition exists the function will return the three sets
            mentioned above in the same order.
            In case the decomposition does not exist, i.e. the maps can not be fused
            the function returns `None`.

        Args:
            state: The in which the two maps are located.
            sdfg: The full SDFG in whcih we operate.
            map_exit_1: The exit node of the first map.
            map_entry_2: The entry node of the second map.
        """
        # The three outputs set.
        pure_outputs: Set[graph.MultiConnectorEdge[dace.Memlet]] = set()
        exclusive_outputs: Set[graph.MultiConnectorEdge[dace.Memlet]] = set()
        shared_outputs: Set[graph.MultiConnectorEdge[dace.Memlet]] = set()

        # Compute the renaming that for translating the parameter of the _second_
        #  map to the ones used by the first map.
        repl_dict: Dict[str, str] = self.find_parameter_remapping(
            first_map=map_exit_1.map,
            second_map=map_entry_2.map,
        )
        assert repl_dict is not None

        # Set of intermediate nodes that we have already processed.
        processed_inter_nodes: Set[nodes.Node] = set()

        # Now scan all output edges of the first exit and classify them
        for out_edge in state.out_edges(map_exit_1):
            intermediate_node: nodes.Node = out_edge.dst

            # We already processed the node, this should indicate that we should
            #  run simplify again, or we should start implementing this case.
            # TODO(phimuell): Handle this case, already partially handled here.
            if intermediate_node in processed_inter_nodes:
                return None
            processed_inter_nodes.add(intermediate_node)

            # The intermediate can only have one incoming degree. It might be possible
            #  to handle multiple incoming edges, if they all come from the top map.
            #  However, the resulting SDFG might be invalid.
            # NOTE: Allow this to happen (under certain cases) if the only producer
            #   is the top map.
            if state.in_degree(intermediate_node) != 1:
                return None

            # If the second map is not reachable from the intermediate node, then
            #  the output is pure and we can end here.
            if not self.is_node_reachable_from(
                    graph=state,
                    begin=intermediate_node,
                    end=map_entry_2,
            ):
                pure_outputs.add(out_edge)
                continue

            # The following tests are _after_ we have determined if we have a pure
            #  output node, because this allows us to handle more exotic pure node
            #  cases, as handling them is essentially rerouting an edge, whereas
            #  handling intermediate nodes is much more complicated.

            # For us an intermediate node must always be an access node, because
            #  everything else we do not know how to handle. It is important that
            #  we do not test for non transient data here, because they can be
            #  handled has shared intermediates.
            if not isinstance(intermediate_node, nodes.AccessNode):
                return None
            if self.is_view(intermediate_node, sdfg):
                return None

            # Empty Memlets are only allowed if they are in `\mathbb{P}`, which
            #  is also the only place they really make sense (for a map exit).
            #  Thus if we now found an empty Memlet we reject it.
            if out_edge.data.is_empty():
                return None

            # It can happen that multiple edges converges at the `IN_` connector
            #  of the first map exit, but there is only one edge leaving the exit.
            #  It is complicate to handle this, so for now we ignore it.
            # TODO(phimuell): Handle this case properly.
            #   To handle this we need to associate a consumer edge (the outgoing edges
            #   of the second map) with exactly one producer.
            producer_edges: List[graph.MultiConnectorEdge[dace.Memlet]] = list(
                state.in_edges_by_connector(map_exit_1, "IN_" + out_edge.src_conn[4:]))
            if len(producer_edges) > 1:
                return None

            # Now check the constraints we have on the producers.
            #   - The source of the producer can not be a view (we do not handle this)
            #   - The edge shall also not be a reduction edge.
            #   - Defined location to where they write.
            #   - No dynamic Melets.
            #  Furthermore, we will also extract the subsets, i.e. the location they
            #  modify inside the intermediate array.
            #  Since we do not allow for WCR, we do not check if the producer subsets intersects.
            producer_subsets: List[subsets.Subset] = []
            for producer_edge in producer_edges:
                if isinstance(producer_edge.src, nodes.AccessNode) and self.is_view(producer_edge.src, sdfg):
                    return None
                if producer_edge.data.dynamic:
                    return None
                if producer_edge.data.wcr is not None:
                    return None
                if producer_edge.data.dst_subset is None:
                    return None
                producer_subsets.append(producer_edge.data.dst_subset)

            # Check if the producer do not intersect
            if len(producer_subsets) == 1:
                pass
            elif len(producer_subsets) == 2:
                if producer_subsets[0].intersects(producer_subsets[1]):
                    return None
            else:
                for i, psbs1 in enumerate(producer_subsets):
                    for j, psbs2 in enumerate(producer_subsets):
                        if i == j:
                            continue
                        if psbs1.intersects(psbs2):
                            return None

            # Now we determine the consumer of nodes. For this we are using the edges
            #  leaves the second map entry. It is not necessary to find the actual
            #  consumer nodes, as they might depend on symbols of nested Maps.
            #  For the covering test we only need their subsets, but we will perform
            #  some scan and filtering on them.
            found_second_map = False
            consumer_subsets: List[subsets.Subset] = []
            for intermediate_node_out_edge in state.out_edges(intermediate_node):

                # If the second map entry is not immediately reachable from the intermediate
                #  node, then ensure that there is not path that goes to it.
                if intermediate_node_out_edge.dst is not map_entry_2:
                    if self.is_node_reachable_from(graph=state, begin=intermediate_node_out_edge.dst, end=map_entry_2):
                        return None
                    continue

                # Ensure that the second map is found exactly once.
                # TODO(phimuell): Lift this restriction.
                if found_second_map:
                    return None
                found_second_map = True

                # The output of the top map can not define a dynamic map range in the
                #  second map.
                if not intermediate_node_out_edge.dst_conn.startswith("IN_"):
                    return None

                # Now we look at all edges that leave the second map entry, i.e. the
                #  edges that feeds the consumer and define what is read inside the map.
                #  We do not check them, but collect them and inspect them.
                #  NOTE: The subset still uses the old iteration variables.
                for inner_consumer_edge in state.out_edges_by_connector(
                        map_entry_2, "OUT_" + intermediate_node_out_edge.dst_conn[3:]):
                    if inner_consumer_edge.data.src_subset is None:
                        return None
                    if inner_consumer_edge.data.dynamic:
                        # TODO(phimuell): Is this restriction necessary, I am not sure.
                        return None
                    consumer_subsets.append(inner_consumer_edge.data.src_subset)
            assert found_second_map, f"Found '{intermediate_node}' which looked like a pure node, but is not one."
            assert len(consumer_subsets) != 0

            # The consumer still uses the original symbols of the second map, so we must rename them.
            if repl_dict:
                consumer_subsets = copy.deepcopy(consumer_subsets)
                for consumer_subset in consumer_subsets:
                    symbolic.safe_replace(mapping=repl_dict, replace_callback=consumer_subset.replace)

            # Now we are checking if a single iteration of the first (top) map
            #  can satisfy all data requirements of the second (bottom) map.
            #  For this we look if the producer covers the consumer. A consumer must
            #  be covered by exactly one producer.
            for consumer_subset in consumer_subsets:
                nb_coverings = sum(producer_subset.covers(consumer_subset) for producer_subset in producer_subsets)
                if nb_coverings != 1:
                    return None

            # After we have ensured coverage, we have to decide if the intermediate
            #  node can be removed (`\mathbb{E}`) or has to be restored (`\mathbb{S}`).
            #  Note that "removed" here means that it is reconstructed by a new
            #  output of the second map.
            if self.is_shared_data(intermediate_node, sdfg):
                # The intermediate data is used somewhere else, either in this or another state.
                # NOTE: If the intermediate is shared, then we will turn it into a
                #   sink node attached to the combined map exit. Technically this
                #   should be enough, even if the same data appears again in the
                #   dataflow down streams. However, some DaCe transformations,
                #   I am looking at you `auto_optimizer()` do not like that. Thus
                #   if the intermediate is used further down in the same datadflow,
                #   then we consider that the maps can not be fused. But we only
                #   do this in the strict data flow mode.
                if self.strict_dataflow:
                    if self._is_data_accessed_downstream(
                            data=intermediate_node.data,
                            graph=state,
                            begin=intermediate_node,  # is ignored itself.
                    ):
                        return None
                shared_outputs.add(out_edge)
            else:
                # The intermediate can be removed, as it is not used anywhere else.
                exclusive_outputs.add(out_edge)

        assert len(processed_inter_nodes) == sum(len(x) for x in [pure_outputs, exclusive_outputs, shared_outputs])
        return (pure_outputs, exclusive_outputs, shared_outputs)


    def relocate_nodes(
        self,
        from_node: Union[nodes.MapExit, nodes.MapEntry],
        to_node: Union[nodes.MapExit, nodes.MapEntry],
        state: SDFGState,
        sdfg: SDFG,
    ) -> None:
        """Move the connectors and edges from `from_node` to `to_nodes` node.

        This function will only rewire the edges, it does not remove the nodes
        themselves. Furthermore, this function should be called twice per Map,
        once for the entry and then for the exit.
        While it does not remove the node themselves if guarantees that the
        `from_node` has degree zero.
        The function assumes that the parameter renaming was already done.

        Args:
            from_node: Node from which the edges should be removed.
            to_node: Node to which the edges should reconnect.
            state: The state in which the operation happens.
            sdfg: The SDFG that is modified.
        """

        # Now we relocate empty Memlets, from the `from_node` to the `to_node`
        for empty_edge in list(filter(lambda e: e.data.is_empty(), state.out_edges(from_node))):
            helpers.redirect_edge(state, empty_edge, new_src=to_node)
        for empty_edge in list(filter(lambda e: e.data.is_empty(), state.in_edges(from_node))):
            helpers.redirect_edge(state, empty_edge, new_dst=to_node)

        # We now ensure that there is only one empty Memlet from the `to_node` to any other node.
        #  Although it is allowed, we try to prevent it.
        empty_targets: Set[nodes.Node] = set()
        for empty_edge in list(filter(lambda e: e.data.is_empty(), state.all_edges(to_node))):
            if empty_edge.dst in empty_targets:
                state.remove_edge(empty_edge)
            empty_targets.add(empty_edge.dst)

        # We now determine which edges we have to migrate, for this we are looking at
        #  the incoming edges, because this allows us also to detect dynamic map ranges.
        #  TODO(phimuell): If there is already a connection to the node, reuse this.
        for edge_to_move in list(state.in_edges(from_node)):
            assert isinstance(edge_to_move.dst_conn, str)

            if not edge_to_move.dst_conn.startswith("IN_"):
                # Dynamic Map Range
                #  The connector name simply defines a variable name that is used,
                #  inside the Map scope to define a variable. We handle it directly.
                dmr_symbol = edge_to_move.dst_conn

                # TODO(phimuell): Check if the symbol is really unused in the target scope.
                if dmr_symbol in to_node.in_connectors:
                    raise NotImplementedError(f"Tried to move the dynamic map range '{dmr_symbol}' from {from_node}'"
                                              f" to '{to_node}', but the symbol is already known there, but the"
                                              " renaming is not implemented.")
                if not to_node.add_in_connector(dmr_symbol, force=False):
                    raise RuntimeError(  # Might fail because of out connectors.
                        f"Failed to add the dynamic map range symbol '{dmr_symbol}' to '{to_node}'.")
                helpers.redirect_edge(state=state, edge=edge_to_move, new_dst=to_node)
                from_node.remove_in_connector(dmr_symbol)

            else:
                # We have a Passthrough connection, i.e. there exists a matching `OUT_`.
                old_conn = edge_to_move.dst_conn[3:]  # The connection name without prefix
                new_conn = to_node.next_connector(old_conn)

                to_node.add_in_connector("IN_" + new_conn)
                for e in list(state.in_edges_by_connector(from_node, "IN_" + old_conn)):
                    helpers.redirect_edge(state, e, new_dst=to_node, new_dst_conn="IN_" + new_conn)
                to_node.add_out_connector("OUT_" + new_conn)
                for e in list(state.out_edges_by_connector(from_node, "OUT_" + old_conn)):
                    helpers.redirect_edge(state, e, new_src=to_node, new_src_conn="OUT_" + new_conn)
                from_node.remove_in_connector("IN_" + old_conn)
                from_node.remove_out_connector("OUT_" + old_conn)

        # Check if we succeeded.
        if state.out_degree(from_node) != 0:
            raise validation.InvalidSDFGError(
                f"Failed to relocate the outgoing edges from `{from_node}`, there are still `{state.out_edges(from_node)}`",
                sdfg,
                sdfg.node_id(state),
            )
        if state.in_degree(from_node) != 0:
            raise validation.InvalidSDFGError(
                f"Failed to relocate the incoming edges from `{from_node}`, there are still `{state.in_edges(from_node)}`",
                sdfg,
                sdfg.node_id(state),
            )
        assert len(from_node.in_connectors) == 0
        assert len(from_node.out_connectors) == 0


    def handle_intermediate_set(
        self,
        intermediate_outputs: Set[graph.MultiConnectorEdge[dace.Memlet]],
        state: SDFGState,
        sdfg: SDFG,
        map_exit_1: nodes.MapExit,
        map_entry_2: nodes.MapEntry,
        map_exit_2: nodes.MapExit,
        is_exclusive_set: bool,
    ) -> None:
        """This function handles the intermediate sets.

        The function is able to handle both the shared and exclusive intermediate
        output set, see `partition_first_outputs()`. The main difference is that
        in exclusive mode the intermediate nodes will be fully removed from
        the SDFG. While in shared mode the intermediate node will be preserved.
        The function assumes that the parameter renaming was already done.

        Args:
            intermediate_outputs: The set of outputs, that should be processed.
            state: The state in which the map is processed.
            sdfg: The SDFG that should be optimized.
            map_exit_1: The exit of the first/top map.
            map_entry_2: The entry of the second map.
            map_exit_2: The exit of the second map.
            is_exclusive_set: If `True` `intermediate_outputs` is the exclusive set.

        Notes:
            Before the transformation the `state` does not have to be valid and
            after this function has run the state is (most likely) invalid.
        """

        map_params = map_exit_1.map.params.copy()

        # Now we will iterate over all intermediate edges and process them.
        #  If not stated otherwise the comments assume that we run in exclusive mode.
        for out_edge in intermediate_outputs:
            # This is the intermediate node that, that we want to get rid of.
            #  In shared mode we want to recreate it after the second map.
            inter_node: nodes.AccessNode = out_edge.dst
            inter_name = inter_node.data
            inter_desc = inter_node.desc(sdfg)
            inter_shape = inter_desc.shape

            # Now we will determine the shape of the new intermediate. This size of
            #  this temporary is given by the Memlet that goes into the first map exit.
            pre_exit_edges = list(state.in_edges_by_connector(map_exit_1, "IN_" + out_edge.src_conn[4:]))
            if len(pre_exit_edges) != 1:
                raise NotImplementedError()
            pre_exit_edge = pre_exit_edges[0]
            new_inter_shape_raw = symbolic.overapproximate(pre_exit_edge.data.subset.size())

            # Over approximation will leave us with some unneeded size one dimensions.
            #  If they are removed some dace transformations (especially auto optimization)
            #  will have problems.
            if not self.strict_dataflow:
                squeezed_dims: List[int] = []  # These are the dimensions we removed.
                new_inter_shape: List[int] = []  # This is the final shape of the new intermediate.
                for dim, (proposed_dim_size, full_dim_size) in enumerate(zip(new_inter_shape_raw, inter_shape)):
                    if full_dim_size == 1:  # Must be kept!
                        new_inter_shape.append(proposed_dim_size)
                    elif proposed_dim_size == 1:  # This dimension was reduced, so we can remove it.
                        squeezed_dims.append(dim)
                    else:
                        new_inter_shape.append(proposed_dim_size)
            else:
                squeezed_dims = []
                new_inter_shape = list(new_inter_shape_raw)

            # This is the name of the new "intermediate" node that we will create.
            #  It will only have the shape `new_inter_shape` which is basically its
            #  output within one Map iteration.
            #  NOTE: The insertion process might generate a new name.
            new_inter_name: str = f"__s{sdfg.node_id(state)}_n{state.node_id(out_edge.src)}{out_edge.src_conn}_n{state.node_id(out_edge.dst)}{out_edge.dst_conn}"

            # Now generate the intermediate data container.
            if len(new_inter_shape) == 0:
                assert pre_exit_edge.data.subset.num_elements() == 1
                is_scalar = True
                new_inter_name, new_inter_desc = sdfg.add_scalar(
                    new_inter_name,
                    dtype=inter_desc.dtype,
                    transient=True,
                    find_new_name=True,
                )

            else:
                assert (pre_exit_edge.data.subset.num_elements() > 1) or all(x == 1 for x in new_inter_shape)
                is_scalar = False
                new_inter_name, new_inter_desc = sdfg.add_transient(
                    new_inter_name,
                    shape=new_inter_shape,
                    dtype=inter_desc.dtype,
                    find_new_name=True,
                )
            new_inter_node: nodes.AccessNode = state.add_access(new_inter_name)

            # Get the subset that defined into which part of the old intermediate
            #  the old output edge wrote to. We need that to adjust the producer
            #  Memlets, since they now write into the new (smaller) intermediate.
            assert pre_exit_edge.data.data == inter_name
            assert pre_exit_edge.data.dst_subset is not None
            producer_offset = self.compute_offset_subset(
                    original_subset=pre_exit_edge.data.dst_subset,
                    intermediate_desc=inter_desc,
                    map_params=map_params,
                    producer_offset=None,
            )

            # Memlets have a lot of additional informations, such as dynamic.
            #  To ensure that we get all of them, we will now copy them and modify
            #  the one that was originally there. We also hope that propagate will
            #  set the rest for us correctly.
            new_pre_exit_memlet = copy.deepcopy(pre_exit_edge.data)
            new_pre_exit_memlet.data = new_inter_name
            new_pre_exit_memlet.dst_subset = subsets.Range.from_array(new_inter_desc)

            # New we will reroute the output Memlet, thus it will no longer pass
            #  through the Map exit but through the newly created intermediate.
            #  NOTE: We will delete the previous edge later.
            new_pre_exit_edge = state.add_edge(
                pre_exit_edge.src,
                pre_exit_edge.src_conn,
                new_inter_node,
                None,
                new_pre_exit_memlet,
            )

            # We now handle the MemletTree defined by this edge.
            #  The newly created edge, only handled the last collection step.
            for producer_tree in state.memlet_tree(new_pre_exit_edge).traverse_children(include_self=False):
                producer_edge = producer_tree.edge

                # Associate the (already existing) Memlet with the new data.
                # TODO(phimuell): Improve the code below to remove the check.
                assert producer_edge.data.data == inter_name
                producer_edge.data.data = new_inter_name

                if is_scalar:
                    producer_edge.data.dst_subset = "0"
                elif producer_edge.data.dst_subset is not None:
                    # Since we now write into a smaller memory patch, we must
                    #  compensate for that. We do this by substracting where the write
                    #  originally had begun.
                    producer_edge.data.dst_subset.offset(producer_offset, negative=True)
                    producer_edge.data.dst_subset.pop(squeezed_dims)

            # Now after we have handled the input of the new intermediate node,
            #  we must handle its output. For this we have to "inject" the newly
            #  created intermediate into the second map. We do this by finding
            #  the input connectors on the map entry, such that we know where we
            #  have to reroute inside the Map.
            # NOTE: Assumes that map (if connected is the direct neighbour).
            conn_names: Set[str] = set()
            for inter_node_out_edge in state.out_edges(inter_node):
                if inter_node_out_edge.dst == map_entry_2:
                    assert inter_node_out_edge.dst_conn.startswith("IN_")
                    conn_names.add(inter_node_out_edge.dst_conn)
                else:
                    # If we found another target than the second map entry from the
                    #  intermediate node it means that the node _must_ survive,
                    #  i.e. we are not in exclusive mode.
                    assert not is_exclusive_set

            # Now we will reroute the connections inside the second map, i.e.
            #  instead of consuming the old intermediate node, they will now
            #  consume the new intermediate node.
            for in_conn_name in conn_names:
                out_conn_name = "OUT_" + in_conn_name[3:]

                for inner_edge in state.out_edges_by_connector(map_entry_2, out_conn_name):
                    assert inner_edge.data.data == inter_name  # DIRECTION!!

                    # As for the producer side, we now read from a smaller array,
                    #  So we must offset them, we use the original edge for this.
                    assert inner_edge.data.src_subset is not None
                    consumer_offset = self.compute_offset_subset(
                            original_subset=inner_edge.data.src_subset,
                            intermediate_desc=inter_desc,
                            map_params=map_params,
                            producer_offset=producer_offset,
                    )

                    # Now we create a new connection that instead reads from the new
                    #  intermediate, instead of the old one. For this we use the
                    #  old Memlet as template. However it is not fully initialized.
                    new_inner_memlet = copy.deepcopy(inner_edge.data)
                    new_inner_memlet.data = new_inter_name

                    # Now we replace the edge from the SDFG.
                    state.remove_edge(inner_edge)
                    new_inner_edge = state.add_edge(
                        new_inter_node,
                        None,
                        inner_edge.dst,
                        inner_edge.dst_conn,
                        new_inner_memlet,
                    )

                    # Now modifying the Memlet, we do it after the insertion to make
                    #  sure that the Memlet was properly initialized.
                    if is_scalar:
                        new_inner_memlet.subset = "0"
                    elif new_inner_memlet.src_subset is not None:
                        new_inner_memlet.src_subset.offset(consumer_offset, negative=True)
                        new_inner_memlet.src_subset.pop(squeezed_dims)

                    # Now we have to make sure that all consumers are properly updated.
                    for consumer_tree in state.memlet_tree(new_inner_edge).traverse_children(include_self=False):
                        assert consumer_tree.edge.data.data == inter_name

                        consumer_edge = consumer_tree.edge
                        consumer_edge.data.data = new_inter_name
                        if is_scalar:
                            consumer_edge.data.src_subset = "0"
                        elif consumer_edge.data.src_subset is not None:
                            consumer_edge.data.src_subset.offset(consumer_offset, negative=True)
                            consumer_edge.data.src_subset.pop(squeezed_dims)

                # The edge that leaves the second map entry was already deleted. We now delete
                #  the edges that connected the intermediate node with the second map entry.
                for edge in list(state.in_edges_by_connector(map_entry_2, in_conn_name)):
                    assert edge.src == inter_node
                    state.remove_edge(edge)
                map_entry_2.remove_in_connector(in_conn_name)
                map_entry_2.remove_out_connector(out_conn_name)

            if is_exclusive_set:
                # In exclusive mode the old intermediate node is no longer needed.
                #  This will also remove `out_edge` from the SDFG.
                assert state.degree(inter_node) == 1
                state.remove_edge_and_connectors(out_edge)
                state.remove_node(inter_node)

                state.remove_edge(pre_exit_edge)
                map_exit_1.remove_in_connector(pre_exit_edge.dst_conn)
                map_exit_1.remove_out_connector(out_edge.src_conn)
                del sdfg.arrays[inter_name]

            else:
                # This is the shared mode, so we have to recreate the intermediate
                #  node, but this time it is at the exit of the second map.
                state.remove_edge(pre_exit_edge)
                map_exit_1.remove_in_connector(pre_exit_edge.dst_conn)

                # This is the Memlet that goes from the map internal intermediate
                #  temporary node to the Map output. This will essentially restore
                #  or preserve the output for the intermediate node. It is important
                #  that we use the data that `preExitEdge` was used.
                final_pre_exit_memlet = copy.deepcopy(pre_exit_edge.data)
                assert pre_exit_edge.data.data == inter_name
                final_pre_exit_memlet.other_subset = subsets.Range.from_array(new_inter_desc)

                new_pre_exit_conn = map_exit_2.next_connector()
                state.add_edge(
                    new_inter_node,
                    None,
                    map_exit_2,
                    "IN_" + new_pre_exit_conn,
                    final_pre_exit_memlet,
                )
                state.add_edge(
                    map_exit_2,
                    "OUT_" + new_pre_exit_conn,
                    inter_node,
                    out_edge.dst_conn,
                    copy.deepcopy(out_edge.data),
                )
                map_exit_2.add_in_connector("IN_" + new_pre_exit_conn)
                map_exit_2.add_out_connector("OUT_" + new_pre_exit_conn)

                map_exit_1.remove_out_connector(out_edge.src_conn)
                state.remove_edge(out_edge)


    def compute_offset_subset(
            self,
            original_subset: subsets.Range,
            intermediate_desc: data.Data,
            map_params: List[str],
            producer_offset: Union[subsets.Range, None],
    ) -> subsets.Range:
        """Computes the memlet to correct read and writes of the intermediate.

        This is the value that must be substracted from the memlets to adjust, i.e
        (`memlet_to_adjust(correction, negative=True)`). If `producer_offset` is
        `None` then the function computes the correction that should be applied to
        the producer memlets, i.e. the memlets of the tree converging at
        `intermediate_node`. If `producer_offset` is given, it should be the output
        of the previous call to this function, with `producer_offset=None`. In this
        case the function computes the correction for the consumer side, i.e. the
        memlet tree that originates at `intermediate_desc`.

        Args:
            original_subset: The original subset that was used to write into the
                intermediate, must be renamed to the final map parameter.
            intermediate_desc: The original intermediate data descriptor.
            map_params: The parameter of the final map.
            producer_offset: The correction that was applied to the producer side.
        """
        assert not isinstance(intermediate_desc, data.View)
        final_offset: subsets.Range = None
        if isinstance(intermediate_desc, data.Scalar):
            # If the intermediate was a scalar, then it will remain a scalar.
            #  Thus there is no correction that we must apply.
            return subsets.Range.from_string("0")

        elif isinstance(intermediate_desc, data.Array):
            basic_offsets = original_subset.min_element()
            offset_list = []
            for d in range(original_subset.dims()):
                d_range = subsets.Range([original_subset[d]])
                if d_range.free_symbols.intersection(map_params):
                    offset_list.append(d_range[0])
                else:
                    offset_list.append((basic_offsets[d], basic_offsets[d], 1))
            final_offset = subsets.Range(offset_list)

        else:
            raise TypeError(f"Does not know how to compute the subset offset for '{type(intermediate_desc).__name__}'.")

        if producer_offset is not None:
            # Here we are correcting some parts that over approximate (which partially
            #  does under approximate) might screw up. Consider two maps, the first
            #  map only writes the subset `[:, 2:6]`, thus the new intermediate will
            #  have shape `(1, 4)`. Now also imagine that the second map only reads
            #  the elements `[:, 3]`. From this we see that we can only correct the
            #  consumer side if we also take the producer side into consideration!
            #  See also the `transformations/mapfusion_test.py::test_offset_correction_*`
            #  tests for more.
            final_offset.offset(
                    final_offset.offset_new(
                        producer_offset,
                        negative=True,
                    ),
                    negative=True,
            )
        return final_offset


    def can_topologically_be_fused(
        self,
        map_entry_1: nodes.MapEntry,
        map_entry_2: nodes.MapEntry,
        graph: Union[dace.SDFGState, dace.SDFG],
        sdfg: dace.SDFG,
        permissive: bool = False,
    ) -> bool:
        """Performs basic checks if the maps can be fused.

        This function only checks constrains that are common between serial and
        parallel map fusion process, which includes:
        - The scope of the maps.
        - The scheduling of the maps.
        - The map parameters.

        Args:
            map_entry_1: The entry of the first (in serial case the top) map.
            map_exit_2: The entry of the second (in serial case the bottom) map.
            graph: The SDFGState in which the maps are located.
            sdfg: The SDFG itself.
            permissive: Currently unused.
        """
        if self.only_inner_maps and self.only_toplevel_maps:
            raise ValueError("You specified both `only_inner_maps` and `only_toplevel_maps`.")

        # Ensure that both have the same schedule
        if map_entry_1.map.schedule != map_entry_2.map.schedule:
            return False

        # Fusing is only possible if the two entries are in the same scope.
        scope = graph.scope_dict()
        if scope[map_entry_1] != scope[map_entry_2]:
            return False
        elif self.only_inner_maps:
            if scope[map_entry_1] is None:
                return False
        elif self.only_toplevel_maps:
            if scope[map_entry_1] is not None:
                return False

        # We will now check if there exists a remapping that of the map parameter
        if self.find_parameter_remapping(first_map=map_entry_1.map, second_map=map_entry_2.map) is None:
            return False

        return True


    def has_read_write_dependency(
        self,
        map_entry_1: nodes.MapEntry,
        map_entry_2: nodes.MapEntry,
        state: SDFGState,
        sdfg: SDFG,
    ) -> bool:
        """Test if there is a read write dependency between the two maps to be fused.

        The function checks two different things.
        - The function will make sure that there is no read write dependency between
            the input and output of the fused maps. For that it will inspect the
            respective subsets.
        - The second part partially checks the intermediate nodes, it mostly ensures
            that there are not views and that they are not used as inputs or outputs
            at the same time. However, the function will not check for read write
            conflicts in this set, this is done in the partition function.

        Returns:
            `True` if there is a conflict between the maps that can not be handled.
            If there is no conflict or if the conflict can be handled `False`
            is returned.

        Args:
            map_entry_1: The entry node of the first map.
            map_entry_2: The entry node of the second map.
            state: The state on which we operate.
            sdfg: The SDFG on which we operate.
        """
        map_exit_1: nodes.MapExit = state.exit_node(map_entry_1)
        map_exit_2: nodes.MapExit = state.exit_node(map_entry_2)

        # Get the read and write sets of the different maps, note that Views
        #  are not resolved yet.
        access_sets: List[Dict[str, nodes.AccessNode]] = []
        for scope_node in [map_entry_1, map_exit_1, map_entry_2, map_exit_2]:
            access_set: Set[nodes.AccessNode] = self.get_access_set(scope_node, state)
            access_sets.append({node.data: node for node in access_set})
            # If two different access nodes of the same scoping node refers to the
            #  same data, then we consider this as a dependency we can not handle.
            #  It is only a problem for the intermediate nodes and might be possible
            #  to handle, but doing so is hard, so we just forbid it.
            if len(access_set) != len(access_sets[-1]):
                return True
        read_map_1, write_map_1, read_map_2, write_map_2 = access_sets

        # It might be possible that there are views, so we have to resolve them.
        #  We also already get the name of the data container.
        #  Note that `len(real_read_map_1) <= len(read_map_1)` holds because of Views.
        resolved_sets: List[Set[str]] = []
        for unresolved_set in [read_map_1, write_map_1, read_map_2, write_map_2]:
            resolved_sets.append({
                self.track_view(node, state, sdfg).data if self.is_view(node, sdfg) else node.data
                for node in unresolved_set.values()
            })
            # If the resolved and unresolved names do not have the same length.
            #  Then different views point to the same location, which we forbid
            if len(unresolved_set) != len(resolved_sets[-1]):
                return False
        real_read_map_1, real_write_map_1, real_read_map_2, real_write_map_2 = resolved_sets

        # We do not allow that the first and second map each write to the same data.
        if not real_write_map_1.isdisjoint(real_write_map_2):
            return True

        # These are the names (unresolved) and the access nodes of the data that is used
        #  to transmit information between the maps. The partition function ensures that
        #  these nodes are directly connected to the two maps.
        exchange_names: Set[str] = set(write_map_1.keys()).intersection(read_map_2.keys())
        exchange_nodes: Set[nodes.AccessNode] = set(write_map_1.values()).intersection(read_map_2.values())

        # If the number are different then a data is accessed through multiple nodes.
        if len(exchange_names) != len(exchange_nodes):
            return True
        assert all(exchange_node.data in exchange_names for exchange_node in exchange_nodes)

        # For simplicity we assume that the nodes used for exchange are not views.
        if any(self.is_view(exchange_node, sdfg) for exchange_node in exchange_nodes):
            return True

        # This is the names of the node that are used as input of the first map and
        #  as output of the second map. We have to ensure that there is no data
        #  dependency between these nodes.
        # NOTE: This set is not required to be empty. It might look as this would
        #   create a data race, but it is save. The reason is because all data has
        #   to pass through the intermediate we create, this will separate the reads
        #   from the writes.
        fused_inout_data_names: Set[str] = set(read_map_1.keys()).intersection(write_map_2.keys())

        # If a data container is used as input and output then it can not be a view (simplicity)
        if any(self.is_view(read_map_1[name], sdfg) for name in fused_inout_data_names):
            return True

        # A data container can not be used as output (of the second as well as the
        #  combined map) and as intermediate. If we would allow that the map would
        #  have two output nodes one the original one and the second is the created
        #  node that is created because the intermediate is shared.
        # TODO(phimuell): Handle this case.
        if not fused_inout_data_names.isdisjoint(exchange_names):
            return True

        # If there is no intersection between the input and output data, then we can
        #  we have nothing to check.
        if len(fused_inout_data_names) == 0:
            return False

        # Get the replacement dict for changing the map variables from the subsets of
        #  the second map.
        repl_dict = self.find_parameter_remapping(map_entry_1.map, map_exit_2.map)

        # Now we inspect if there is a read write dependency, between data that is
        #  used as input and output of the fused map. There is no problem is they
        #  are pointwise, i.e. in each iteration the same locations are accessed.
        #  Essentially they all boil down to `a += 1`.
        for inout_data_name in fused_inout_data_names:
            all_subsets: List[subsets.Subset] = []
            # The subsets that define reading are given by the first map's entry node
            all_subsets.extend(
                self.find_subsets(
                    node=read_map_1[inout_data_name],
                    scope_node=map_entry_1,
                    state=state,
                    sdfg=sdfg,
                    repl_dict=None,
                ))
            #  While the subsets defining writing are given by the second map's exit
            #  node, there we also have to apply renaming.
            all_subsets.extend(
                self.find_subsets(
                    node=write_map_2[inout_data_name],
                    scope_node=map_exit_2,
                    state=state,
                    sdfg=sdfg,
                    repl_dict=repl_dict,
                ))
            # Now we can test if these subsets are point wise
            if not self.test_if_subsets_are_point_wise(all_subsets):
                return True

        # No read write dependency was found.
        return False


    def test_if_subsets_are_point_wise(self, subsets_to_check: List[subsets.Subset]) -> bool:
        """Point wise means that they are all the same.

        If a series of subsets are point wise it means that all Memlets, access
        the same data. This is an important property because the whole map fusion
        is build upon this.
        If the subsets originates from different maps, then they must have been
        renamed.

        Args:
            subsets_to_check: The list of subsets that should be checked.
        """
        assert len(subsets_to_check) > 1

        # We will check everything against the master subset.
        master_subset = subsets_to_check[0]
        for ssidx in range(1, len(subsets_to_check)):
            subset = subsets_to_check[ssidx]
            if isinstance(subset, subsets.Indices):
                subset = subsets.Range.from_indices(subset)
                # Do we also need the reverse? See below why.
                if any(r != (0, 0, 1) for r in subset.offset_new(master_subset, negative=True)):
                    return False
            else:
                # The original code used `Range.offset` here, but that one had trouble
                #  for `r1 = 'j, 0:10'` and `r2 = 'j, 0`. The solution would be to test
                #  symmetrically, i.e. `r1 - r2` and `r2 - r1`. However, if we would
                #  have `r2_1 = 'j, 0:10'` it consider it as failing, which is not
                #  what we want. Thus we will use symmetric cover.
                if not master_subset.covers(subset):
                    return False
                if not subset.covers(master_subset):
                    return False

        # All subsets are equal to the master subset, thus they are equal to each other.
        #  This means that the data accesses, described by this transformation is
        #  point wise
        return True


    def is_shared_data(
        self,
        data: nodes.AccessNode,
        sdfg: dace.SDFG,
    ) -> bool:
        """Tests if `data` is shared data, an can not be removed.

        Interstate data is used to transmit data, this includes:
        - The data is referred in multiple states.
        - The data is referred to multiple times in the same state, either the state
            has multiple access nodes for that data or an access node has an
            out degree larger than one.
        - The data is read inside interstate edges.

        This definition is stricter than the one employed by `SDFG.shared_transients()`,
        as it also includes usage within a state.

        Args:
            transient: The transient that should be checked.
            sdfg: The SDFG containing the array.

        Note:
            The function computes the this set once for every SDFG and then caches it.
            There is no mechanism to detect if the cache must be evicted. However,
            as long as no additional data is added to the SDFG, there is no problem.
        """
        if sdfg not in self._shared_data:
            self._compute_shared_data_in(sdfg)
        return data.data in self._shared_data[sdfg]


    def _compute_shared_data_in(
        self,
        sdfg: dace.SDFG,
    ) -> None:
        """Updates the internal set of shared data/interstate data of `self` for `sdfg`.

        See the documentation for `self.is_shared_data()` for a description.

        Args:
            sdfg: The SDFG for which the set of shared data should be computed.
        """
        # Shared data of this SDFG.
        shared_data: Set[str] = set()

        # All global data can not be removed, so it must always be shared.
        for data_name, data_desc in sdfg.arrays.items():
            if not data_desc.transient:
                shared_data.add(data_name)
            elif isinstance(data_desc, dace.data.Scalar):
                shared_data.add(data_name)

        # We go through all states and classify the nodes/data:
        #   - Data is referred to in different states.
        #   - The access node is a view (both have to survive).
        #   - Transient sink or source node.
        #   - The access node has output degree larger than 1 (input degrees larger
        #       than one, will always be partitioned as shared anyway).
        prevously_seen_data: Set[str] = set()
        for state in sdfg.nodes():
            for access_node in state.data_nodes():

                if access_node.data in shared_data:
                    # The data was already classified to be shared data
                    pass

                elif access_node.data in prevously_seen_data:
                    # We have seen this data before, either in this state or in
                    #  a previous one, but we did not classifies it as shared back then
                    shared_data.add(access_node.data)

                if state.in_degree(access_node) == 0:
                    # (Transient) sink nodes are used in other states, or simplify
                    #  will get rid of them.
                    shared_data.add(access_node.data)

                elif state.out_degree(access_node) != 1:  # state.out_degree() == 0 or state.out_degree() > 1
                    # The access node is either a source node (it is shared in another
                    #  state) or the node has a degree larger than one, so it is used
                    #  in this state somewhere else.
                    shared_data.add(access_node.data)

                elif self.is_view(node=access_node, sdfg=sdfg):
                    # To ensure that the write to the view happens, both have to be shared.
                    viewed_data: str = self.track_view(view=access_node, state=state, sdfg=sdfg).data
                    shared_data.update([access_node.data, viewed_data])
                    prevously_seen_data.update([access_node.data, viewed_data])

                else:
                    # The node was not classified as shared data, so we record that
                    #  we saw it. Note that a node that was immediately classified
                    #  as shared node will never be added to this set, but a data
                    #  that was found twice will be inside this list.
                    prevously_seen_data.add(access_node.data)

        # Now we collect all symbols that are read in interstate edges.
        #  Because, they might refer to data inside states and must be kept alive.
        interstate_read_symbols: Set[str] = set()
        for edge in sdfg.edges():
            interstate_read_symbols.update(edge.data.read_symbols())
        data_read_in_interstate_edges = interstate_read_symbols.intersection(prevously_seen_data)

        # Compute the final set of shared data and update the internal cache.
        shared_data.update(data_read_in_interstate_edges)
        self._shared_data[sdfg] = shared_data


    def find_parameter_remapping(self, first_map: nodes.Map, second_map: nodes.Map) -> Union[Dict[str, str], None]:
        """Computes the parameter remapping for the parameters of the _second_ map.

        The returned `dict` maps the parameters of the second map (keys) to parameter
        names of the first map (values). Because of how the replace function works
        the `dict` describes how to replace the parameters of the second map
        with parameters of the first map.
        Parameters that already have the correct name and compatible range, are not
        included in the return value, thus the keys and values are always different.
        If no renaming at is _needed_, i.e. all parameter have the same name and range,
        then the function returns an empty `dict`.
        If no remapping exists, then the function will return `None`.

        Args:
            first_map:  The first map (these parameters will be replaced).
            second_map: The second map, these parameters acts as source.
        """

        # The parameter names
        first_params: List[str] = first_map.params
        second_params: List[str] = second_map.params

        if len(first_params) != len(second_params):
            return None

        # The ranges, however, we apply some post processing to them.
        simp = lambda e: symbolic.simplify_ext(symbolic.simplify(e))
        first_rngs: Dict[str, Tuple[Any, Any, Any]] = {
            param: tuple(simp(r) for r in rng)
            for param, rng in zip(first_params, first_map.range)
        }
        second_rngs: Dict[str, Tuple[Any, Any, Any]] = {
            param: tuple(simp(r) for r in rng)
            for param, rng in zip(second_params, second_map.range)
        }

        # Parameters of the second map that have not yet been matched to a parameter
        #  of the first map and vice versa.
        unmapped_second_params: Set[str] = set(second_params)
        unused_first_params: Set[str] = set(first_params)

        # This is the result (`second_param -> first_param`), note that if no renaming
        #  is needed then the parameter is not present in the mapping.
        final_mapping: Dict[str, str] = {}

        # First we identify the parameters that already have the correct name.
        for param in set(first_params).intersection(second_params):
            first_rng = first_rngs[param]
            second_rng = second_rngs[param]

            if first_rng == second_rng:
                # They have the same name and the same range, this is already a match.
                #  Because the names are already the same, we do not have to enter them
                #  in the `final_mapping`
                unmapped_second_params.discard(param)
                unused_first_params.discard(param)

        # Check if no remapping is needed.
        if len(unmapped_second_params) == 0:
            return {}

        # Now we go through all the parameters that we have not mapped yet.
        #  All of them will result in a remapping.
        for unmapped_second_param in unmapped_second_params:
            second_rng = second_rngs[unmapped_second_param]
            assert unmapped_second_param not in final_mapping

            # Now look in all not yet used parameters of the first map which to use.
            for candidate_param in unused_first_params:
                candidate_rng = first_rngs[candidate_param]
                if candidate_rng == second_rng:
                    final_mapping[unmapped_second_param] = candidate_param
                    unused_first_params.discard(candidate_param)
                    break
            else:
                # We did not find a candidate, so the remapping does not exist
                return None

        assert len(unused_first_params) == 0
        assert len(final_mapping) == len(unmapped_second_params)
        return final_mapping


    def rename_map_parameters(
        self,
        first_map: nodes.Map,
        second_map: nodes.Map,
        second_map_entry: nodes.MapEntry,
        state: SDFGState,
    ) -> None:
        """Replaces the map parameters of the second map with names from the first.

        The replacement is done in a safe way, thus `{'i': 'j', 'j': 'i'}` is
        handled correct. The function assumes that a proper replacement exists.
        The replacement is computed by calling `self.find_parameter_remapping()`.

        Args:
            first_map:  The first map (these are the final parameter).
            second_map: The second map, this map will be replaced.
            second_map_entry: The entry node of the second map.
            state: The SDFGState on which we operate.
        """
        # Compute the replacement dict.
        repl_dict: Dict[str, str] = self.find_parameter_remapping(first_map=first_map, second_map=second_map)

        if repl_dict is None:
            raise RuntimeError("The replacement does not exist")
        if len(repl_dict) == 0:
            return

        second_map_scope = state.scope_subgraph(entry_node=second_map_entry)
        # Why is this thing is symbolic and not in replace?
        symbolic.safe_replace(
            mapping=repl_dict,
            replace_callback=second_map_scope.replace_dict,
        )

        # For some odd reason the replace function does not modify the range and
        #  parameter of the map, so we will do it the hard way.
        second_map.params = copy.deepcopy(first_map.params)
        second_map.range = copy.deepcopy(first_map.range)


    def is_node_reachable_from(
        self,
        graph: Union[dace.SDFG, dace.SDFGState],
        begin: nodes.Node,
        end: nodes.Node,
    ) -> bool:
        """Test if the node `end` can be reached from `begin`.

        Essentially the function starts a DFS at `begin`. If an edge is found that lead
        to `end` the function returns `True`. If the node is never found `False` is
        returned.

        Args:
            graph: The graph to operate on.
            begin: The start of the DFS.
            end: The node that should be located.
        """

        def next_nodes(node: nodes.Node) -> Iterable[nodes.Node]:
            return (edge.dst for edge in graph.out_edges(node))

        to_visit: List[nodes.Node] = [begin]
        seen: Set[nodes.Node] = set()

        while len(to_visit) > 0:
            node: nodes.Node = to_visit.pop()
            if node == end:
                return True
            elif node not in seen:
                to_visit.extend(next_nodes(node))
            seen.add(node)

        # We never found `end`
        return False


    def _is_data_accessed_downstream(
        self,
        data: str,
        graph: dace.SDFGState,
        begin: nodes.Node,
    ) -> bool:
        """Tests if there is an AccessNode for `data` downstream of `begin`.

        Essentially, this function starts a DFS at `begin` and checks every
        AccessNode that is reachable from it. If it finds such a node it will
        check if it refers to `data` and if so, it will return `True`.
        If no such node is found it will return `False`.
        Note that the node `begin` will be ignored.

        Args:
            data: The name of the data to look for.
            graph: The graph to explore.
            begin: The node to start exploration; The node itself is ignored.
        """
        def next_nodes(node: nodes.Node) -> Iterable[nodes.Node]:
            return (edge.dst for edge in graph.out_edges(node))

        # Dataflow graph is acyclic, so we do not need to keep a list of
        #  what we have visited.
        to_visit: List[nodes.Node] = list(next_nodes(begin))
        while len(to_visit) > 0:
            node = to_visit.pop()
            if isinstance(node, nodes.AccessNode) and node.data == data:
                return True
            to_visit.extend(next_nodes(node))

        return False


    def get_access_set(
        self,
        scope_node: Union[nodes.MapEntry, nodes.MapExit],
        state: SDFGState,
    ) -> Set[nodes.AccessNode]:
        """Computes the access set of a "scope node".

        If `scope_node` is a `MapEntry` it will operate on the set of incoming edges
        and if it is an `MapExit` on the set of outgoing edges. The function will
        then determine all access nodes that have a connection through these edges
        to the scope nodes (edges that does not lead to access nodes are ignored).
        The function returns a set that contains all access nodes that were found.
        It is important that this set will also contain views.

        Args:
            scope_node: The scope node that should be evaluated.
            state: The state in which we operate.
        """
        if isinstance(scope_node, nodes.MapEntry):
            get_edges = lambda node: state.in_edges(node)
            other_node = lambda e: e.src
        else:
            get_edges = lambda node: state.out_edges(node)
            other_node = lambda e: e.dst
        access_set: Set[nodes.AccessNode] = {
            node
            for node in map(other_node, get_edges(scope_node)) if isinstance(node, nodes.AccessNode)
        }

        return access_set


    def find_subsets(
        self,
        node: nodes.AccessNode,
        scope_node: Union[nodes.MapExit, nodes.MapEntry],
        state: SDFGState,
        sdfg: SDFG,
        repl_dict: Optional[Dict[str, str]],
    ) -> List[subsets.Subset]:
        """Finds all subsets that access `node` within `scope_node`.

        The function will not start a search for all consumer/producers.
        Instead it will locate the edges which is immediately inside the
        map scope.

        Args:
            node: The access node that should be examined.
            scope_node: We are only interested in data that flows through this node.
            state: The state in which we operate.
            sdfg: The SDFG object.
        """

        # Is the node used for reading or for writing.
        #  This influences how we have to proceed.
        if isinstance(scope_node, nodes.MapEntry):
            outer_edges_to_inspect = [e for e in state.in_edges(scope_node) if e.src == node]
            get_subset = lambda e: e.data.src_subset
            get_inner_edges = lambda e: state.out_edges_by_connector(scope_node, "OUT_" + e.dst_conn[3:])
        else:
            outer_edges_to_inspect = [e for e in state.out_edges(scope_node) if e.dst == node]
            get_subset = lambda e: e.data.dst_subset
            get_inner_edges = lambda e: state.in_edges_by_connector(scope_node, "IN_" + e.src_conn[4:])

        found_subsets: List[subsets.Subset] = []
        for edge in outer_edges_to_inspect:
            found_subsets.extend(get_subset(e) for e in get_inner_edges(edge))
        assert len(found_subsets) > 0, "Could not find any subsets."
        assert not any(subset is None for subset in found_subsets)

        found_subsets = copy.deepcopy(found_subsets)
        if repl_dict:
            for subset in found_subsets:
                # Replace happens in place
                symbolic.safe_replace(repl_dict, subset.replace)

        return found_subsets


    def is_view(
        self,
        node: nodes.AccessNode,
        sdfg: SDFG,
    ) -> bool:
        """Tests if `node` points to a view or not."""
        node_desc: data.Data = node.desc(sdfg)
        return isinstance(node_desc, data.View)


    def track_view(
        self,
        view: nodes.AccessNode,
        state: SDFGState,
        sdfg: SDFG,
    ) -> nodes.AccessNode:
        """Find the original data of a View.

        Given the View `view`, the function will trace the view back to the original
        access node. For convenience, if `view` is not a `View` the argument will be
        returned.

        Args:
            view: The view that should be traced.
            state: The state in which we operate.
            sdfg: The SDFG on which we operate.
        """

        # Test if it is a view at all, if not return the passed node as source.
        if not self.is_view(view, sdfg):
            return view

        # First determine if the view is used for reading or writing.
        curr_edge = dace.sdfg.utils.get_view_edge(state, view)
        if curr_edge is None:
            raise RuntimeError(f"Failed to determine the direction of the view '{view}'.")
        if curr_edge.dst_conn == "views":
            # The view is used for reading.
            next_node = lambda curr_edge: curr_edge.src
        elif curr_edge.src_conn == "views":
            # The view is used for writing.
            next_node = lambda curr_edge: curr_edge.dst
        else:
            raise RuntimeError(f"Failed to determine the direction of the view '{view}' | {curr_edge}.")

        # Now trace the view back.
        org_view = view
        view = next_node(curr_edge)
        while self.is_view(view, sdfg):
            curr_edge = dace.sdfg.utils.get_view_edge(state, view)
            if curr_edge is None:
                raise RuntimeError(f"View tracing of '{org_view}' failed at note '{view}'.")
            view = next_node(curr_edge)
        return view
