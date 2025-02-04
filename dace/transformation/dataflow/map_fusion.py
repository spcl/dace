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
    """Implements the MapFusion transformation.

    From a high level perspective it will remove the MapExit node of the first and the MapEntry node of
    the second Map. It will then rewire and modify the Memlets such that the data flow bypasses the
    intermediate node. For this a new intermediate node will be created, which is much smaller because
    it has no longer to store the whole output of the first map, but only the data that is produced by
    a single iteration of the first map. The transformation will then remove the old intermediate.
    Thus by merging the two Maps together the transformation will reduce the memory footprint. It is
    important that it is not always possible to fully remove the intermediate node. For example the
    data might be used somewhere else. In this case the intermediate will become an output of the Map.

    An example would be the following:
    ```python
    for i in range(N):
        T[i] = foo(A[i])
    for j in range(N):
        B[j] = bar(T[i])
    ```
    which would be translated into:
    ```python
    for i in range(N):
        temp: scalar = foo(A[i])
        B[i] = bar(temp)
    ```

    The checks that two Maps can be fused are quite involved, however, they essentially check:
    * If the two Maps cover the same iteration space, essentially have the same start, stop and
        iteration , see `find_parameter_remapping()`.
    * Furthermore, they verify if the new fused Map did not introduce read write conflict,
        essentially it tests if the data is pointwise, i.e. what is read is also written,
        see `has_read_write_dependency()`.
    * Then it will examine the intermediate data. This will essentially test if the data that
        is needed by a single iteration of the second Map is produced by a single iteration of
        the first Map, see `partition_first_outputs()`.

    By default `strict_dataflow` is enabled. In this mode the transformation is more conservative.
    The main difference is, that it will not adjust the subsets of the intermediate, i.e. turning
    an array with shape `(1, 1, 1, 1)` into a scalar. Furthermore, shared intermediates, see
    `partition_first_outputs()` will only be created if the data is not referred downstream in
    the dataflow.

    In order to determine if an intermediate can be removed or has to be kept, it is in general
    necessary to scan the whole SDFG, which is the default behaviour. There are two ways to
    speed this up. The first way is to set `assume_always_shared` to `True`. In this case the
    transformation will not perform the scan, but assume that the data is shared, i.e. used
    somewhere else. This might lead to dead data flow.
    The second way is to use the transformation inside a pipeline, which includes the
    `FindSingleUseData` analysis pass. If the result of this pass is present then the
    transformation will use it instead to determine if a intermediate can be removed.
    Note that `assume_always_shared` takes precedence.
    For this pattern the `FullMapFusion` pass is provided, that combines the analysis
    pass and `MapFusion`

    :param only_inner_maps: Only match Maps that are internal, i.e. inside another Map.
    :param only_toplevel_maps: Only consider Maps that are at the top.
    :param strict_dataflow: Which dataflow mode should be used, see above.
    :param assume_always_shared: Assume that all intermediates are shared.

    :note: This transformation modifies more nodes than it matches.
    :note: If `assume_always_shared` is `True` then the transformation will assume that
            all intermediates are shared. This avoids the problems mentioned above with
            the cache at the expense of the creation of dead dataflow.
    """

    # Pattern Nodes
    first_map_exit = transformation.transformation.PatternNode(nodes.MapExit)
    array = transformation.transformation.PatternNode(nodes.AccessNode)
    second_map_entry = transformation.transformation.PatternNode(nodes.MapEntry)


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
    assume_always_shared = properties.Property(
        dtype=bool,
        default=False,
        desc="If `True` then all intermediates will be classified as shared.",
    )


    def __init__(
        self,
        only_inner_maps: Optional[bool] = None,
        only_toplevel_maps: Optional[bool] = None,
        strict_dataflow: Optional[bool] = None,
        assume_always_shared: Optional[bool] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        if only_toplevel_maps is not None:
            self.only_toplevel_maps = only_toplevel_maps
        if only_inner_maps is not None:
            self.only_inner_maps = only_inner_maps
        if strict_dataflow is not None:
            self.strict_dataflow = strict_dataflow
        if assume_always_shared is not None:
            self.assume_always_shared = assume_always_shared

        # See comment in `is_shared_data()` for more information.
        self._single_use_data: Optional[Dict[dace.SDFG, Set[str]]] = None


    @classmethod
    def expressions(cls) -> Any:
        """Get the match expression.

        The transformation matches the exit node of the top Map that is connected to
        an access node that again is connected to the entry node of the second Map.
        An important note is, that the transformation operates not just on the
        matched nodes, but more or less on anything that has an incoming connection
        from the first Map or an outgoing connection to the second Map entry.
        """
        return [dace.sdfg.utils.node_path_graph(cls.first_map_exit, cls.array, cls.second_map_entry)]


    def can_be_applied(
        self,
        graph: Union[SDFGState, SDFG],
        expr_index: int,
        sdfg: dace.SDFG,
        permissive: bool = False,
    ) -> bool:
        """Tests if the matched Maps can be merged.

        The two Maps are mergeable iff:
        * Checks general requirements, see `can_topologically_be_fused()`.
        * Tests if there are read write dependencies.
        * Tests if the decomposition exists.
        """
        # To ensures that the `{src,dst}_subset` are properly set, run initialization.
        #  See [issue 1708](https://github.com/spcl/dace/issues/1703)
        for edge in graph.edges():
            edge.data.try_initialize(sdfg, graph, edge)

        first_map_entry: nodes.MapEntry = graph.entry_node(self.first_map_exit)
        first_map_exit: nodes.MapExit = self.first_map_exit
        second_map_entry: nodes.MapEntry = self.second_map_entry

        # Check the structural properties of the Maps. The function will return
        #  the `dict` that describes how the parameters must be renamed (for caching)
        #  or `None` if the maps can not be structurally fused.
        param_repl = self.can_topologically_be_fused(
                first_map_entry=first_map_entry,
                second_map_entry=second_map_entry,
                graph=graph,
                sdfg=sdfg
        )
        if param_repl is None:
            return False

        # Tests if there are read write dependencies that are caused by the bodies
        #  of the Maps, such as referring to the same data. Note that this tests are
        #  different from the ones performed by `has_read_write_dependency()`, which
        #  only checks the data dependencies that go through the scope nodes.
        if self.has_inner_read_write_dependency(
                first_map_entry=first_map_entry,
                second_map_entry=second_map_entry,
                state=graph,
                sdfg=sdfg,
        ):
            return False

        # Tests for read write conflicts of the two maps, this is only checking
        #  the data that goes through the scope nodes. `has_inner_read_write_dependency()`
        #  if used to check if there are internal dependencies.
        if self.has_read_write_dependency(
                first_map_entry=first_map_entry,
                second_map_entry=second_map_entry,
                param_repl=param_repl,
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
            first_map_exit=first_map_exit,
            second_map_entry=second_map_entry,
            param_repl=param_repl,
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

        :param graph: The SDFG state we are operating on.
        :param sdfg: The SDFG we are operating on.
        """
        # To ensures that the `{src,dst}_subset` are properly set, run initialization.
        #  See [issue 1708](https://github.com/spcl/dace/issues/1703)
        for edge in graph.edges():
            edge.data.try_initialize(sdfg, graph, edge)

        first_map_exit: nodes.MapExit = self.first_map_exit
        second_map_entry: nodes.MapEntry = self.second_map_entry
        second_map_exit: nodes.MapExit = graph.exit_node(self.second_map_entry)
        first_map_entry: nodes.MapEntry = graph.entry_node(self.first_map_exit)

        # Before we do anything we perform the renaming.
        self.rename_map_parameters(
            first_map=first_map_exit.map,
            second_map=second_map_entry.map,
            second_map_entry=second_map_entry,
            state=graph,
        )

        # Now compute the partition. Because we have already renamed the parameters
        #  of the second Map, there is no need to perform any renaming, thus we can
        #  pass an empty `dict`.
        output_partition = self.partition_first_outputs(
            state=graph,
            sdfg=sdfg,
            first_map_exit=first_map_exit,
            second_map_entry=second_map_entry,
            param_repl=dict(),
        )
        assert output_partition is not None  # Make MyPy happy.
        pure_outputs, exclusive_outputs, shared_outputs = output_partition

        # Now perform the actual rewiring, we handle each partition separately.
        if len(exclusive_outputs) != 0:
            self.handle_intermediate_set(
                intermediate_outputs=exclusive_outputs,
                state=graph,
                sdfg=sdfg,
                first_map_exit=first_map_exit,
                second_map_entry=second_map_entry,
                second_map_exit=second_map_exit,
                is_exclusive_set=True,
            )
        if len(shared_outputs) != 0:
            self.handle_intermediate_set(
                intermediate_outputs=shared_outputs,
                state=graph,
                sdfg=sdfg,
                first_map_exit=first_map_exit,
                second_map_entry=second_map_entry,
                second_map_exit=second_map_exit,
                is_exclusive_set=False,
            )
        assert pure_outputs == set(graph.out_edges(first_map_exit))
        if len(pure_outputs) != 0:
            self.relocate_nodes(
                from_node=first_map_exit,
                to_node=second_map_exit,
                state=graph,
                sdfg=sdfg,
            )

        # Now move the input of the second map, that has no connection to the first
        #  map, to the first map. This is needed because we will later delete the
        #  exit of the first map (which we have essentially handled above). Now
        #  we must handle the input of the second map (that has no connection to the
        #  first map) to the input of the first map.
        self.relocate_nodes(
            from_node=second_map_entry,
            to_node=first_map_entry,
            state=graph,
            sdfg=sdfg,
        )

        for node_to_remove in [first_map_exit, second_map_entry]:
            assert graph.degree(node_to_remove) == 0
            graph.remove_node(node_to_remove)

        # Now turn the second output node into the output node of the first Map.
        second_map_exit.map = first_map_entry.map


    def partition_first_outputs(
        self,
        state: SDFGState,
        sdfg: SDFG,
        first_map_exit: nodes.MapExit,
        second_map_entry: nodes.MapEntry,
        param_repl: Dict[str, str],
    ) -> Union[
            Tuple[
                Set[graph.MultiConnectorEdge[dace.Memlet]],
                Set[graph.MultiConnectorEdge[dace.Memlet]],
                Set[graph.MultiConnectorEdge[dace.Memlet]],
            ],
            None,
    ]:
        """Partition the output edges of `first_map_exit` for serial map fusion.

        The output edges of the first map are partitioned into three distinct sets,
        defined as follows:
        * Pure Output Set `\mathbb{P}`:
            These edges exits the first map and does not enter the second map. These
            outputs will be simply be moved to the output of the second map.
        * Exclusive Intermediate Set `\mathbb{E}`:
            Edges in this set leaves the first map exit, enters an access node, from
            where a Memlet then leads immediately to the second map. The memory
            referenced by this access node is not used anywhere else, thus it can
            be removed.
        * Shared Intermediate Set `\mathbb{S}`:
            These edges are very similar to the one in `\mathbb{E}` except that they
            are used somewhere else, thus they can not be removed and have to be
            recreated as output of the second map.

        If strict data flow mode is enabled the function is rather strict if an
        output can be added to either intermediate set and might fail to compute
        the partition, even if it would exist.

        :return: If such a decomposition exists the function will return the three sets
            mentioned above in the same order. In case the decomposition does not exist,
            i.e. the maps can not be fused the function returns `None`.

        :param state: The in which the two maps are located.
        :param sdfg: The full SDFG in whcih we operate.
        :param first_map_exit: The exit node of the first map.
        :param second_map_entry: The entry node of the second map.
        :param param_repl: Use this map to rename the parameter of the second Map, such
            that they match the one of the first map.
        """
        # The three outputs set.
        pure_outputs: Set[graph.MultiConnectorEdge[dace.Memlet]] = set()
        exclusive_outputs: Set[graph.MultiConnectorEdge[dace.Memlet]] = set()
        shared_outputs: Set[graph.MultiConnectorEdge[dace.Memlet]] = set()

        # Set of intermediate nodes that we have already processed.
        processed_inter_nodes: Set[nodes.Node] = set()

        # Now scan all output edges of the first exit and classify them
        for out_edge in state.out_edges(first_map_exit):
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
                    end=second_map_entry,
            ):
                pure_outputs.add(out_edge)
                continue

            # The following tests are _after_ we have determined if we have a pure
            #  output node, because this allows us to handle more exotic pure node
            #  cases, as handling them is essentially rerouting an edge, whereas
            #  handling intermediate nodes is much more complicated.

            # Empty Memlets are only allowed if they are in `\mathbb{P}`, which
            #  is also the only place they really make sense (for a map exit).
            #  Thus if we now found an empty Memlet we reject it.
            if out_edge.data.is_empty():
                return None

            # For us an intermediate node must always be an access node, because
            #  everything else we do not know how to handle. It is important that
            #  we do not test for non transient data here, because they can be
            #  handled has shared intermediates.
            if not isinstance(intermediate_node, nodes.AccessNode):
                return None
            intermediate_desc: dace.data.Data = intermediate_node.desc(sdfg)
            if self.is_view(intermediate_desc, sdfg):
                return None

            # It can happen that multiple edges converges at the `IN_` connector
            #  of the first map exit, but there is only one edge leaving the exit.
            #  It is complicate to handle this, so for now we ignore it.
            # TODO(phimuell): Handle this case properly.
            #   To handle this we need to associate a consumer edge (the outgoing edges
            #   of the second map) with exactly one producer.
            producer_edges: List[graph.MultiConnectorEdge[dace.Memlet]] = list(
                state.in_edges_by_connector(first_map_exit, "IN_" + out_edge.src_conn[4:]))
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
                    # TODO(phimuell): Find out if this restriction could be lifted, but it is unlikely.
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
                if intermediate_node_out_edge.dst is not second_map_entry:
                    if self.is_node_reachable_from(graph=state, begin=intermediate_node_out_edge.dst, end=second_map_entry):
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
                # NOTE1: The subset still uses the old iteration variables.
                # NOTE2: In case of consumer Memlet we explicitly allow dynamic Memlets.
                #   This is different compared to the producer Memlet. The reason is
                #   because in a consumer the data is conditionally read, so the data
                #   has to exists anyway.
                for inner_consumer_edge in state.out_edges_by_connector(
                        second_map_entry, "OUT_" + intermediate_node_out_edge.dst_conn[3:]):
                    if inner_consumer_edge.data.src_subset is None:
                        return None
                    consumer_subsets.append(inner_consumer_edge.data.src_subset)
            assert found_second_map, f"Found '{intermediate_node}' which looked like a pure node, but is not one."
            assert len(consumer_subsets) != 0

            # The consumer still uses the original symbols of the second map, so we must rename them.
            if param_repl:
                consumer_subsets = copy.deepcopy(consumer_subsets)
                for consumer_subset in consumer_subsets:
                    symbolic.safe_replace(mapping=param_repl, replace_callback=consumer_subset.replace)

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
            if self.is_shared_data(data=intermediate_node, state=state, sdfg=sdfg):
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

        :param from_node: Node from which the edges should be removed.
        :param to_node: Node to which the edges should reconnect.
        :param state: The state in which the operation happens.
        :param sdfg: The SDFG that is modified.
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
        first_map_exit: nodes.MapExit,
        second_map_entry: nodes.MapEntry,
        second_map_exit: nodes.MapExit,
        is_exclusive_set: bool,
    ) -> None:
        """This function handles the intermediate sets.

        The function is able to handle both the shared and exclusive intermediate
        output set, see `partition_first_outputs()`. The main difference is that
        in exclusive mode the intermediate nodes will be fully removed from
        the SDFG. While in shared mode the intermediate node will be preserved.
        The function assumes that the parameter renaming was already done.

        :param intermediate_outputs: The set of outputs, that should be processed.
        :param state: The state in which the map is processed.
        :param sdfg: The SDFG that should be optimized.
        :param first_map_exit: The exit of the first/top map.
        :param second_map_entry: The entry of the second map.
        :param second_map_exit: The exit of the second map.
        :param is_exclusive_set: If `True` `intermediate_outputs` is the exclusive set.

        :note: Before the transformation the `state` does not have to be valid and
            after this function has run the state is (most likely) invalid.
        """

        map_params = first_map_exit.map.params.copy()

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
            pre_exit_edges = list(state.in_edges_by_connector(first_map_exit, "IN_" + out_edge.src_conn[4:]))
            if len(pre_exit_edges) != 1:
                raise NotImplementedError()
            pre_exit_edge = pre_exit_edges[0]

            (new_inter_shape_raw, new_inter_shape, squeezed_dims) = self.compute_reduced_intermediate(
                    producer_subset=pre_exit_edge.data.dst_subset,
                    inter_desc=inter_desc,
            )

            # This is the name of the new "intermediate" node that we will create.
            #  It will only have the shape `new_inter_shape` which is basically its
            #  output within one Map iteration.
            #  NOTE: The insertion process might generate a new name.
            new_inter_name: str = f"__s{self.state_id}_n{state.node_id(out_edge.src)}{out_edge.src_conn}_n{state.node_id(out_edge.dst)}{out_edge.dst_conn}"

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
            producer_offset = self.compute_offset_subset(
                    original_subset=pre_exit_edge.data.dst_subset,
                    intermediate_desc=inter_desc,
                    map_params=map_params,
                    producer_offset=None,
            )

            # Memlets have a lot of additional informations, to ensure that we get
            #  all of them, we have to do it this way. The main reason for this is
            #  to handle the case were the "Memlet reverse direction", i.e. `data`
            #  refers to the other end of the connection than before.
            assert pre_exit_edge.data.dst_subset is not None
            new_pre_exit_memlet_src_subset = copy.deepcopy(pre_exit_edge.data.src_subset)
            new_pre_exit_memlet_dst_subset = subsets.Range.from_array(new_inter_desc)

            new_pre_exit_memlet = copy.deepcopy(pre_exit_edge.data)
            new_pre_exit_memlet.data = new_inter_name

            new_pre_exit_edge = state.add_edge(
                pre_exit_edge.src,
                pre_exit_edge.src_conn,
                new_inter_node,
                None,
                new_pre_exit_memlet,
            )

            # We can update `{src, dst}_subset` only after we have inserted the
            #  edge, this is because the direction of the Memlet might change.
            new_pre_exit_edge.data.src_subset = new_pre_exit_memlet_src_subset
            new_pre_exit_edge.data.dst_subset = new_pre_exit_memlet_dst_subset

            # We now handle the MemletTree defined by this edge.
            #  The newly created edge, only handled the last collection step.
            for producer_tree in state.memlet_tree(new_pre_exit_edge).traverse_children(include_self=False):
                producer_edge = producer_tree.edge

                # In order to preserve the intrinsic direction of Memlets we only have to change
                #  the `.data` attribute of the producer Memlet if it refers to the old intermediate.
                #  If it refers to something different we keep it. Note that this case can only
                #  occur if the producer is an AccessNode.
                if producer_edge.data.data == inter_name:
                    producer_edge.data.data = new_inter_name

                # Regardless of the intrinsic direction of the Memlet, the subset we care about
                #  is always `dst_subset`.
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
                if inter_node_out_edge.dst == second_map_entry:
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

                for inner_edge in state.out_edges_by_connector(second_map_entry, out_conn_name):
                    # As for the producer side, we now read from a smaller array,
                    #  So we must offset them, we use the original edge for this.
                    assert inner_edge.data.src_subset is not None
                    consumer_offset = self.compute_offset_subset(
                            original_subset=inner_edge.data.src_subset,
                            intermediate_desc=inter_desc,
                            map_params=map_params,
                            producer_offset=producer_offset,
                    )

                    # Now create the memlet for the new consumer. To make sure that we get all attributes
                    #  of the Memlet we make a deep copy of it. There is a tricky part here, we have to
                    #  access `src_subset` however, this is only correctly set once it is put inside the
                    #  SDFG. Furthermore, we have to make sure that the Memlet does not change its direction.
                    #  i.e. that the association of `subset` and `other_subset` does not change. For this
                    #  reason we only modify `.data` attribute of the Memlet if its name refers to the old
                    #  intermediate. Furthermore, to play it safe, we only access the subset, `src_subset`
                    #  after we have inserted it to the SDFG.
                    new_inner_memlet = copy.deepcopy(inner_edge.data)
                    if inner_edge.data.data == inter_name:
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
                        # TODO(phimuell): Figuring out if `src_subset` is None is an error.
                        new_inner_memlet.src_subset.offset(consumer_offset, negative=True)
                        new_inner_memlet.src_subset.pop(squeezed_dims)

                    # Now we have to make sure that all consumers are properly updated.
                    for consumer_tree in state.memlet_tree(new_inner_edge).traverse_children(include_self=False):
                        consumer_edge = consumer_tree.edge

                        # We only modify the data if the Memlet refers to the old intermediate data.
                        #  We can not do this unconditionally, because it might change the intrinsic
                        #  direction of a Memlet and then `src_subset` would at the next `try_initialize`
                        #  be wrong. Note that this case only occurs if the destination is an AccessNode.
                        if consumer_edge.data.data == inter_name:
                            consumer_edge.data.data = new_inter_name

                        # Now we have to adapt the subsets.
                        if is_scalar:
                            consumer_edge.data.src_subset = "0"
                        elif consumer_edge.data.src_subset is not None:
                            # TODO(phimuell): Figuring out if `src_subset` is None is an error.
                            consumer_edge.data.src_subset.offset(consumer_offset, negative=True)
                            consumer_edge.data.src_subset.pop(squeezed_dims)

                # The edge that leaves the second map entry was already deleted. We now delete
                #  the edges that connected the intermediate node with the second map entry.
                for edge in list(state.in_edges_by_connector(second_map_entry, in_conn_name)):
                    assert edge.src == inter_node
                    state.remove_edge(edge)
                second_map_entry.remove_in_connector(in_conn_name)
                second_map_entry.remove_out_connector(out_conn_name)

            if is_exclusive_set:
                # In exclusive mode the old intermediate node is no longer needed.
                #  This will also remove `out_edge` from the SDFG.
                assert state.degree(inter_node) == 1
                state.remove_edge_and_connectors(out_edge)
                state.remove_node(inter_node)

                state.remove_edge(pre_exit_edge)
                first_map_exit.remove_in_connector(pre_exit_edge.dst_conn)
                first_map_exit.remove_out_connector(out_edge.src_conn)
                del sdfg.arrays[inter_name]

            else:
                # TODO(phimuell): Lift this restriction
                assert pre_exit_edge.data.data == inter_name

                # This is the shared mode, so we have to recreate the intermediate
                #  node, but this time it is at the exit of the second map.
                state.remove_edge(pre_exit_edge)
                first_map_exit.remove_in_connector(pre_exit_edge.dst_conn)

                # This is the Memlet that goes from the map internal intermediate
                #  temporary node to the Map output. This will essentially restore
                #  or preserve the output for the intermediate node. It is important
                #  that we use the data that `preExitEdge` was used.
                final_pre_exit_memlet = copy.deepcopy(pre_exit_edge.data)
                final_pre_exit_memlet.other_subset = subsets.Range.from_array(new_inter_desc)

                new_pre_exit_conn = second_map_exit.next_connector()
                state.add_edge(
                    new_inter_node,
                    None,
                    second_map_exit,
                    "IN_" + new_pre_exit_conn,
                    final_pre_exit_memlet,
                )
                state.add_edge(
                    second_map_exit,
                    "OUT_" + new_pre_exit_conn,
                    inter_node,
                    out_edge.dst_conn,
                    copy.deepcopy(out_edge.data),
                )
                second_map_exit.add_in_connector("IN_" + new_pre_exit_conn)
                second_map_exit.add_out_connector("OUT_" + new_pre_exit_conn)

                first_map_exit.remove_out_connector(out_edge.src_conn)
                state.remove_edge(out_edge)


    def compute_reduced_intermediate(
            self,
            producer_subset: subsets.Range,
            inter_desc: dace.data.Data,
    ) -> Tuple[Tuple[int, ...], Tuple[int, ...], List[int]]:
        """Compute the size of the new (reduced) intermediate.

        `MapFusion` does not only fuses map, but, depending on the situation, also
        eliminates intermediate arrays between the two maps. To transmit data between
        the two maps a new, but much smaller intermediate is needed.

        :return: The function returns a tuple with three values with the following meaning:
            * The raw shape of the reduced intermediate.
            * The cleared shape of the reduced intermediate, essentially the raw shape
                with all shape 1 dimensions removed.
            * Which dimensions of the raw shape have been removed to get the cleared shape.

        :param producer_subset: The subset that was used to write into the intermediate.
        :param inter_desc: The data descriptor for the intermediate.
        """
        assert producer_subset is not None

        # Over approximation will leave us with some unneeded size one dimensions.
        #  If they are removed some dace transformations (especially auto optimization)
        #  will have problems.
        new_inter_shape_raw = symbolic.overapproximate(producer_subset.size())
        inter_shape = inter_desc.shape
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

        return (tuple(new_inter_shape_raw), tuple(new_inter_shape), squeezed_dims)


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

        :param original_subset: The original subset that was used to write into the
            intermediate, must be renamed to the final map parameter.
        :param intermediate_desc: The original intermediate data descriptor.
        :param map_params: The parameter of the final map.
        :param producer_offset: The correction that was applied to the producer side.
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
        first_map_entry: nodes.MapEntry,
        second_map_entry: nodes.MapEntry,
        graph: Union[dace.SDFGState, dace.SDFG],
        sdfg: dace.SDFG,
        permissive: bool = False,
    ) -> Optional[Dict[str, str]]:
        """Performs basic checks if the maps can be fused.

        This function only checks constrains that are common between serial and
        parallel map fusion process, which includes:
        * The scope of the maps.
        * The scheduling of the maps.
        * The map parameters.

        :return: If the maps can not be topologically fused the function returns `None`.
            If they can be fused the function returns `dict` that describes parameter
            replacement, see `find_parameter_remapping()` for more.

        :param first_map_entry: The entry of the first (in serial case the top) map.
        :param second_map_exit: The entry of the second (in serial case the bottom) map.
        :param graph: The SDFGState in which the maps are located.
        :param sdfg: The SDFG itself.
        :param permissive: Currently unused.
        """
        if self.only_inner_maps and self.only_toplevel_maps:
            raise ValueError("Only one of `only_inner_maps` and `only_toplevel_maps` is allowed per MapFusion instance.")

        # Ensure that both have the same schedule
        if first_map_entry.map.schedule != second_map_entry.map.schedule:
            return None

        # Fusing is only possible if the two entries are in the same scope.
        scope = graph.scope_dict()
        if scope[first_map_entry] != scope[second_map_entry]:
            return None
        elif self.only_inner_maps:
            if scope[first_map_entry] is None:
                return None
        elif self.only_toplevel_maps:
            if scope[first_map_entry] is not None:
                return None

        # We will now check if we can rename the Map parameter of the second Map such that they
        #  match the one of the first Map.
        param_repl = self.find_parameter_remapping(first_map=first_map_entry.map, second_map=second_map_entry.map)
        return param_repl


    def has_inner_read_write_dependency(
        self,
        first_map_entry: nodes.MapEntry,
        second_map_entry: nodes.MapEntry,
        state: SDFGState,
        sdfg: SDFG,
    ) -> bool:
        """This function tests if there are dependency inside the Maps.

        The function will scan and anaysize the body of the two Maps and look for
        inconsistencies. To detect them the function will scan the body of the maps
        and examine the all AccessNodes and apply the following rules:
        * If an AccessNode refers to a View, it is ignored. Because the source is
            either on the outside, in which case `has_read_write_dependency()`
            takes care of it, or the data source is inside the Map body itself.
        * An inconsistency is detected, if in each bodies there exists an AccessNode
            that refer to the same data.
        * An inconsistency is detected, if there exists an AccessNode that refers
            to non transient data. This is an implementation detail and could be
            lifted.

        Note that some of the restrictions of this function could be relaxed by
        performing more analysis.

        :return: The function returns `True` if an inconsistency has been found.

        :param first_map_entry: The entry node of the first map.
        :param second_map_entry: The entry node of the second map.
        :param state: The state on which we operate.
        :param sdfg: The SDFG on which we operate.
        """
        first_map_body = state.scope_subgraph(first_map_entry, False, False)
        second_map_body = state.scope_subgraph(second_map_entry, False, False)

        # Find the data that is internally referenced. Because of the first rule above,
        #  we filter all views above.
        first_map_body_data, second_map_body_data = [
            {
                dnode.data
                for dnode in map_body.nodes()
                if isinstance(dnode, nodes.AccessNode) and not self.is_view(dnode, sdfg)
            }
            for map_body in [first_map_body, second_map_body]
        ]

        # If there is data that is referenced in both, then we consider this as an error
        #  this is the second rule above.
        if not first_map_body_data.isdisjoint(second_map_body_data):
            return True

        # We consider it as a problem if any map refers to non-transient data.
        #  This is an implementation detail and could be dropped if we do further
        #  analysis.
        if any(
            not sdfg.arrays[data].transient
            for data in first_map_body_data.union(second_map_body_data)
        ):
            return True

        return False


    def has_read_write_dependency(
        self,
        first_map_entry: nodes.MapEntry,
        second_map_entry: nodes.MapEntry,
        param_repl: Dict[str, str],
        state: SDFGState,
        sdfg: SDFG,
    ) -> bool:
        """Test if there is a read write dependency between the two maps to be fused.

        The function checks three different things.
        * The function will make sure that there is no read write dependency between
            the input and output of the fused maps. For that it will inspect the
            respective subsets of the inputs of the MapEntry of the first and the
            outputs of the MapExit node of the second map.
        * The second part partially checks the intermediate nodes, it mostly ensures
            that there are not views and that they are not used as output of the
            combined map. Note that it is allowed that an intermediate node is also
            an input to the first map.
        * In case an intermediate node, is also used as input node of the first map,
            it is forbidden that the data is used as output of the second map, the
            function will do additional checks. This is needed as the partition function
            only checks the data consumption of the second map can be satisfied by the
            data production of the first map, it ignores any potential reads made by
            the first map's MapEntry.

        :return: `True` if there is a conflict between the maps that can not be handled.
            If there is no conflict or if the conflict can be handled `False` is returned.

        :param first_map_entry: The entry node of the first map.
        :param second_map_entry: The entry node of the second map.
        :param param_repl: Dict that describes how to rename the parameters of the second Map.
        :param state: The state on which we operate.
        :param sdfg: The SDFG on which we operate.
        """
        first_map_exit: nodes.MapExit = state.exit_node(first_map_entry)
        second_map_exit: nodes.MapExit = state.exit_node(second_map_entry)

        # Get the read and write sets of the different maps, note that Views
        #  are not resolved yet.
        access_sets: List[Dict[str, nodes.AccessNode]] = []
        for scope_node in [first_map_entry, first_map_exit, second_map_entry, second_map_exit]:
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
        #  This essentially ensures that an intermediate can not be used as output of
        #  the second map at the same time. It is actually stronger as it does not
        #  take their role into account.
        if not real_write_map_1.isdisjoint(real_write_map_2):
            return True

        # These are the names (unresolved) and the access nodes of the data that is used
        #  to transmit information between the maps. The partition function ensures that
        #  these nodes are directly connected to the two maps.
        exchange_names: Set[str] = set(write_map_1.keys()).intersection(read_map_2.keys())
        exchange_nodes: Set[nodes.AccessNode] = set(write_map_1.values()).intersection(read_map_2.values())

        # If the number are different then a data is accessed through different
        #  AccessNodes. We could analyse this, but we will consider this as a data race.
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

        # While it is forbidden that a data container, used as intermediate, is also
        #  used as output of the second map. It is allowed that the data container
        #  is used as intermediate and as input of the first map. The partition only
        #  checks that the data dependencies are mean, i.e. what is read by the second
        #  map is also computed (written to the intermediate) it does not take into
        #  account the first map's read to the data container.
        #  To make an example: The partition function will make sure that if the
        #  second map reads index `i` from the intermediate that the first map writes
        #  to that index. But it will not care if the first map reads (through its
        #  MapEntry) index `i + 1`. In order to be valid me must ensure that the first
        #  map's reads and writes to the intermediate are pointwise.
        #  Note that we only have to make this check if it is also an intermediate node.
        #  Because if it is not read by the second map it is not a problem as the node
        #  will end up as an pure output node anyway.
        read_write_map_1 = set(read_map_1.keys()).intersection(write_map_1.keys())
        datas_to_inspect = read_write_map_1.intersection(exchange_names)
        for data_to_inspect in datas_to_inspect:

            # Now get all subsets of the data container that the first map reads
            #  from or writes to and check if they are pointwise.
            all_subsets: List[subsets.Subset] = []
            all_subsets.extend(
                self.find_subsets(
                    node=read_map_1[data_to_inspect],
                    scope_node=first_map_entry,
                    state=state,
                    sdfg=sdfg,
                    param_repl=None,
                ))
            all_subsets.extend(
                self.find_subsets(
                    node=write_map_1[data_to_inspect],
                    scope_node=first_map_exit,
                    state=state,
                    sdfg=sdfg,
                    param_repl=None,
                ))
            if not self.test_if_subsets_are_point_wise(all_subsets):
                return True

        # If there is no intersection between the input and output data, then we can
        #  we have nothing to check.
        if len(fused_inout_data_names) == 0:
            return False

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
                    scope_node=first_map_entry,
                    state=state,
                    sdfg=sdfg,
                    param_repl=None,
                ))
            #  While the subsets defining writing are given by the second map's exit
            #  node, there we also have to apply renaming.
            all_subsets.extend(
                self.find_subsets(
                    node=write_map_2[inout_data_name],
                    scope_node=second_map_exit,
                    state=state,
                    sdfg=sdfg,
                    param_repl=param_repl,
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

        :param subsets_to_check: The list of subsets that should be checked.
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
        state: dace.SDFGState,
        sdfg: dace.SDFG,
    ) -> bool:
        """Tests if `data` is shared data, i.e. it can not be removed from the SDFG.

        Depending on the situation, the function will not perform a scan of the whole SDFG:
        1) If `assume_always_shared` was set to `True`, the function will return `True` unconditionally.
        2) If `data` is non transient then the function will return `True`, as non transient data
            must be reconstructed always.
        3) If the AccessNode `data` has more than one outgoing edge or more than one incoming edge
            it is classified as shared.
        2) If `FindSingleUseData` is in the pipeline it will be used and no scan will be performed.
        3) The function will perform a scan.

        :param data: The transient that should be checked.
        :param state: The state in which the fusion is performed.
        :param sdfg: The SDFG in which we want to perform the fusing.

        """
        # `assume_always_shared` takes precedence.
        if self.assume_always_shared:
            return True

        # If `data` is non transient then return `True` as the intermediate can not be removed.
        if not data.desc(sdfg).transient:
            return True

        # This means the data is consumed by multiple Maps, through the same AccessNode, in this state
        #  Note currently multiple incoming edges are not handled, but in the spirit of this function
        #  we consider such AccessNodes as shared, because we can not remove the intermediate.
        if state.out_degree(data) > 1:
            return True
        if state.in_degree(data) > 1:
            return True

        # NOTE: Actually, if this transformation is run through the `FullMapFusion` pass, it should
        #  read the results from `FindSingelUseData`, that was computed because it is a dependent
        #  pass through the `self._pipeline_results` which is set by the `SingleStateTransformation`.
        #  However, this member is only set during when `apply()` is called, but not during
        #  `can_be_applied()`, see [issue#1911](https://github.com/spcl/dace/issues/1911).
        #  Because, the whole goal of this separation of scanning and fusion was to make the
        #  transformation stateless, the member `_single_use_data` was introduced. If it is set
        #  then we use it otherwise we use the scanner.
        #  This value is set for example by the `FullMapFusion` pass.
        # TODO(phimuell): Change this once the issue is resolved.
        if self._single_use_data is not None:
            assert sdfg in self._single_use_data, f"`_single_use_data` was set, but does not contain information about the SDFG '{sdfg.name}'."
            single_use_data: Set[str] = self._single_use_data[sdfg]
            return data.data not in single_use_data

        # We have to perform the full scan of the SDFG.
        return self._scan_sdfg_if_data_is_shared(data=data, state=state, sdfg=sdfg)


    def _scan_sdfg_if_data_is_shared(
        self,
        data: nodes.AccessNode,
        state: dace.SDFGState,
        sdfg: dace.SDFG,
    ) -> None:
        """Scans `sdfg` to determine if `data` is shared.

        Essentially, this function determine, if the intermediate AccessNode `data` is
        can be removed or if it has to be restored as output of the Map.
        A data descriptor is classified as shared if any of the following is true:
        - `data` is non transient data.
        - `data` has at most one incoming and/or outgoing edge.
        - There are other AccessNodes beside `data` that refer to the same data.
        - The data is accessed on an interstate edge.

        This function should not be called directly. Instead it is called indirectly
        by `is_shared_data()` if there is no short cut.

        :param data: The AccessNode that should checked if it is shared.
        :param sdfg: The SDFG for which the set of shared data should be computed.
        """
        if not data.desc(sdfg).transient:
            return True

        # See description in `is_shared_data()` for more.
        if state.out_degree(data) > 1:
            return True
        if state.in_degree(data) > 1:
            return True

        data_name: str = data.data
        for state in sdfg.states():
            for dnode in state.data_nodes():
                if dnode is data:
                    # We have found the `data` AccessNode, which we must ignore.
                    continue
                if dnode.data == data_name:
                    # We found a different AccessNode that refers to the same data
                    #  as `data`. Thus `data` is shared.
                    return True

        # Test if the data is referenced in the interstate edges.
        for edge in sdfg.edges():
            if data_name in edge.data.free_symbols:
                # The data is used in the inter state edges. So it is shared.
                return True

        # Test if the data is referenced inside a control flow, such as a conditional
        #  block or loop condition.
        for cfr in sdfg.all_control_flow_regions():
            if data_name in cfr.used_symbols(all_symbols=True, with_contents=False):
                return True

        # The `data` is not used anywhere else, thus `data` is not shared.
        return False


    def find_parameter_remapping(self, first_map: nodes.Map, second_map: nodes.Map) -> Optional[Dict[str, str]]:
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

        :param first_map: The first map (these parameters will be replaced).
        :param second_map: The second map, these parameters acts as source.
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

        :param first_map:  The first map (these are the final parameter).
        :param second_map: The second map, this map will be replaced.
        :param second_map_entry: The entry node of the second map.
        :param state: The SDFGState on which we operate.
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

        :param graph: The graph to operate on.
        :param begin: The start of the DFS.
        :param end: The node that should be located.
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

        :param data: The name of the data to look for.
        :param graph: The graph to explore.
        :param begin: The node to start exploration; The node itself is ignored.
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

        :param scope_node: The scope node that should be evaluated.
        :param state: The state in which we operate.
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
        param_repl: Optional[Dict[str, str]],
    ) -> List[subsets.Subset]:
        """Finds all subsets that access `node` within `scope_node`.

        The function will not start a search for all consumer/producers.
        Instead it will locate the edges which is immediately inside the
        map scope.

        :param node: The access node that should be examined.
        :param scope_node: We are only interested in data that flows through this node.
        :param state: The state in which we operate.
        :param sdfg: The SDFG object.
        :param param_repl: `dict` that describes the parameter renaming that should be
            performed. Can be `None` to skip the processing.
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
        if param_repl:
            for subset in found_subsets:
                # Replace happens in place
                symbolic.safe_replace(param_repl, subset.replace)

        return found_subsets


    def is_view(
        self,
        node: Union[nodes.AccessNode, data.Data],
        sdfg: SDFG,
    ) -> bool:
        """Tests if `node` points to a view or not."""
        node_desc: data.Data =  node if isinstance(node, data.Data) else node.desc(sdfg)
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

        :param view: The view that should be traced.
        :param state: The state in which we operate.
        :param sdfg: The SDFG on which we operate.
        """

        # Test if it is a view at all, if not return the passed node as source.
        if not self.is_view(view, sdfg):
            return view

        # This is the node that defines the view.
        defining_node = dace.sdfg.utils.get_last_view_node(state, view)
        assert isinstance(defining_node, nodes.AccessNode)
        assert not self.is_view(defining_node, sdfg)
        return defining_node
