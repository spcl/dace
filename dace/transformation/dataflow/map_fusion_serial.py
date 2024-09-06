# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.

"""Implements the serial map fusing transformation."""

import copy
from typing import Any, Dict, List, Set, Tuple, Union, Optional

import dace
from dace import data, dtypes, properties, subsets, symbolic, transformation
from dace.sdfg import SDFG, SDFGState, graph, nodes

from dace.transformation.dataflow import map_fusion_helper as mfh


@properties.make_properties
class MapFusionSerial(mfh.MapFusionHelper):
    """Fuse two serial maps together.

    The transformation combines two maps into one that are connected through some
    access nodes. Conceptually this transformation removes the exit of the first
    or upper map and the entry of the lower or second map and then rewrites the
    connections appropriately. Depending on the situation the transformation will
    either fully remove or make the intermediate a new output of the second map.

    By default, the transformation does not use the strict data flow mode, see
    `MapFusionHelper` for more, however, it might be useful in come cases to enable
    it.

    Args:
        only_inner_maps: Only match Maps that are internal, i.e. inside another Map.
        only_toplevel_maps: Only consider Maps that are at the top.
        strict_dataflow: If `True`, the transformation ensures a more
            stricter version of the data flow.

    Notes:
        - This transformation modifies more nodes than it matches.
        - After the transformation has been applied simplify should be run to remove
            some dead data flow, that was introduced to ensure validity.
        - A `MapFusionSerial` obejct can be initialized and be reused. However,
            after new access nodes are added to any state, it is no longer valid
            to use the object.

    Todo:
        - Consider the case that only shared nodes are created (thus no inspection of
            the graph is needed) and make all shared. Then use the dead dataflow
            elimination transformation to get rid of the ones we no longer need.
        - Increase the applicability.
    """

    map_exit_1 = transformation.transformation.PatternNode(nodes.MapExit)
    intermediate_access_node = transformation.transformation.PatternNode(nodes.AccessNode)
    map_entry_2 = transformation.transformation.PatternNode(nodes.MapEntry)

    def __init__(
        self,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)


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
        - Checks general requirements, see `MapFusionHelper.can_be_fused()`.
        - Tests if the decomposition exists.
        - Tests if there are read write dependencies.
        """
        map_entry_1: nodes.MapEntry = graph.entry_node(self.map_exit_1)
        map_exit_1: nodes.MapExit = self.map_exit_1
        map_entry_2: nodes.MapEntry = self.map_entry_2

        # This essentially test the structural properties of the two Maps.
        if not self.can_be_fused(
            map_entry_1=map_entry_1, map_entry_2=map_entry_2, graph=graph, sdfg=sdfg
        ):
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
            access_sets.append({
                node.data: node
                for node in access_set
            })
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
                self.track_view(node).data if self.is_view(node, sdfg) else node.data
                for node in unresolved_set.values()
            })
            # If the resolved and unresolved names do not have the same length.
            #  Then different views point to the same location, which we forbid
            if len(unresolved_set) != len(resolved_sets[-1]):
                return None
        real_read_map_1, real_write_map_1, real_read_map_2, real_write_map_2 = resolved_sets

        # We do not allow that the first and second map each write to the same data.
        if not real_write_map_1.isdisjoint(real_write_map_2):
            return True

        # If there is no overlap in what is (totally) read and written, there will be no conflict.
        #  This must come before the check of disjoint write.
        if (real_read_map_1 | real_read_map_2).isdisjoint(real_write_map_1 | real_write_map_2):
            return False

        # These are the names (unresolved) and the access nodes of the data that is used
        #  to transmit information between the maps. The partition function ensures that
        #  these nodes are directly connected to the two maps.
        exchange_names: Set[str] = set(write_map_1.keys()).intersection(read_map_2.keys())
        exchange_nodes: Set[node.AccessNode] = set(write_map_1.values()).intersection(read_map_2.values())

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
        fused_inout_data_names: Set[str] = set(read_map_1.keys()).intersection(write_map_2.keys())

        # If a data container is used as input and output then it can not be a view (simplicity)
        if any(self.is_view(read_map_1[name], sdfg) for name in fused_inout_data_names):
            return True

        # A data container can be used as input and output. But we do not allow that
        #  it is also used as intermediate or exchange data. This is an important check.
        if not fused_inout_data_names.isdisjoint(exchange_names):
            return True

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
                )
            )
            #  While the subsets defining writing are given by the second map's exit
            #  node, there we also have to apply renaming.
            all_subsets.extend(
                self.find_subsets(
                    node=write_map_2[inout_data_name],
                    scope_node=map_exit_2,
                    state=state,
                    sdfg=sdfg,
                    repl_dict=repl_dict,
                )
            )
            # Now we can test if these subsets are point wise
            if not self.test_if_subsets_are_point_wise(all_subsets):
                return True

        # No read write dependency was found.
        return False


    def test_if_subsets_are_point_wise(
            self,
            subsets_to_check: List[subsets.Subset]
    ) -> bool:
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
                if any(r != (0, 0, 1) for r in test in subset.offset_new(master_subset,negative=True)):
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

        # These are the iteration parameters of the two maps.
        #  They are not yet modified, that they match each other.
        map_params_1: Sequence[str] = map_exit_1.map.params
        map_params_2: Sequence[str] = map_entry_2.map.params

        # Compute the renaming that for translating the parameter of the _second_
        #  map to the ones used by the first map.
        repl_dict: Dict[str, str] = self.find_parameter_remapping(
                first_map=map_exit_1.map,
                second_map=map_entry_2.map,
        )
        assert repl_dict is not None

        # Set of intermediate nodes that we have already processed.
        processed_inter_nodes: Set[nodes.Node] = set()

        # These are the data that is written to multiple times in _this_ state.
        #  If a data is written to multiple time in a state, it could be
        #  classified as shared. However, it might happen that the node has zero
        #  degree. This is not a problem as the maps also induced a before-after
        #  relationship. But some DaCe transformations do not catch this.
        #  Thus we will never modify such intermediate nodes and fail instead.
        if self.strict_dataflow:
            multi_write_data: Set[str] = self._compute_multi_write_data(state, sdfg)
        else:
            multi_write_data = set()

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
            intermediate_desc: data.Data = intermediate_node.desc(sdfg)
            if self.is_view(intermediate_node, sdfg):
                return None

            # Checks if the intermediate node refers to data that is accessed by
            #  _other_ access nodes in _this_ state. If this is the case then never
            #  touch this intermediate node.
            #  TODO(phimuell): Technically it would be enough to turn the node into
            #   a shared output node, because this will still fulfil the dependencies.
            #   However, some DaCe transformation can not handle this properly, so we
            #   are _forced_ to reject this node.
            if intermediate_node.data in multi_write_data:
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
            producer_edges: List[graph.MultiConnectorEdge[dace.Memlet]] = list(state.in_edges_by_connector(map_exit_1, "IN_" + out_edge.src_conn[4:]))
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
                for inner_consumer_edge in state.out_edges_by_connector(map_entry_2, "OUT_" + intermediate_node_out_edge.dst_conn[3:]):
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
                shared_outputs.add(out_edge)
            else:
                # The intermediate can be removed, as it is not used anywhere else.
                exclusive_outputs.add(out_edge)

        assert len(processed_inter_nodes) == sum(len(x) for x in [pure_outputs, exclusive_outputs, shared_outputs])
        return (pure_outputs, exclusive_outputs, shared_outputs)


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
            pre_exit_edges = list(
                state.in_edges_by_connector(map_exit_1, "IN_" + out_edge.src_conn[4:])
            )
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
                for dim, (proposed_dim_size, full_dim_size) in enumerate(
                    zip(new_inter_shape_raw, inter_shape)
                ):
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
                    storage=dtypes.StorageType.Register,
                    find_new_name=True,
                )

            else:
                assert (pre_exit_edge.data.subset.num_elements() > 1) or all(
                    x == 1 for x in new_inter_shape
                )
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
            old_pre_exit_edge_subset = pre_exit_edge.data.dst_subset

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
                    producer_edge.data.dst_subset.offset(old_pre_exit_edge_subset, negative=True)
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
                    inner_edge_correction_offset = copy.deepcopy(inner_edge.data.src_subset)

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
                        new_inner_memlet.src_subset.offset(inner_edge_correction_offset, negative=True)
                        new_inner_memlet.src_subset.pop(squeezed_dims)

                    # Now we have to make sure that all consumers are properly updated.
                    for consumer_tree in state.memlet_tree(new_inner_edge).traverse_children(include_self=False):
                        assert consumer_tree.edge.data.data == inter_name

                        consumer_edge = consumer_tree.edge
                        consumer_edge.data.data = new_inter_name
                        if is_scalar:
                            consumer_edge.data.src_subset = "0"
                        elif consumer_edge.data.src_subset is not None:
                            consumer_edge.data.src_subset.offset(inner_edge_correction_offset, negative=True)
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
