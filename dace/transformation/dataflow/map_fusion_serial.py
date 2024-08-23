# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.

"""Implements the serial map fusing transformation."""

import copy
from typing import Any, Dict, List, Set, Union

import dace
from dace import dtypes, properties, subsets, symbolic, transformation
from dace.sdfg import SDFG, SDFGState, graph, nodes

from dace.transformation.dataflow import map_fusion_helper


@properties.make_properties
class SerialMapFusion(map_fusion_helper.MapFusionHelper):
    """Specialized replacement for the map fusion transformation that is provided by DaCe.

    As its name is indicating this transformation is only able to handle Maps that
    are in sequence. Compared to the native DaCe transformation, this one is able
    to handle more complex cases of connection between the maps. In that sense, it
    is much more similar to DaCe's `SubgraphFusion` transformation.

    Things that are improved, compared to the native DaCe implementation:
    - Nested Maps.
    - Temporary arrays and the correct propagation of their Memlets.
    - Top Maps that have multiple outputs.

    Conceptually this transformation removes the exit of the first or upper map
    and the entry of the lower or second map and then rewrites the connections
    appropriately.

    This transformation assumes that an SDFG obeys the structure that is outlined
    [here](https://hackmd.io/klvzLnzMR6GZBWtRU8HbDg#Requirements-on-SDFG). For that
    reason it is not true replacement of the native DaCe transformation.

    Args:
        only_inner_maps: Only match Maps that are internal, i.e. inside another Map.
        only_toplevel_maps: Only consider Maps that are at the top.

    Notes:
        - This transformation modifies more nodes than it matches!
    """

    map_exit1 = transformation.transformation.PatternNode(nodes.MapExit)
    access_node = transformation.transformation.PatternNode(nodes.AccessNode)
    map_entry2 = transformation.transformation.PatternNode(nodes.MapEntry)

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
        return [dace.sdfg.utils.node_path_graph(cls.map_exit1, cls.access_node, cls.map_entry2)]

    def can_be_applied(
        self,
        graph: Union[SDFGState, SDFG],
        expr_index: int,
        sdfg: dace.SDFG,
        permissive: bool = False,
    ) -> bool:
        """Tests if the matched Maps can be merged.

        The two Maps are mergeable iff:
        - The `can_be_fused()` of the base succeed, which checks some basic constraints.
        - The decomposition exists and at least one of the intermediate sets
            is not empty.
        """
        assert isinstance(self.map_exit1, nodes.MapExit)
        assert isinstance(self.map_entry2, nodes.MapEntry)
        map_entry_1: nodes.MapEntry = graph.entry_node(self.map_exit1)
        map_entry_2: nodes.MapEntry = self.map_entry2

        # This essentially test the structural properties of the two Maps.
        if not self.can_be_fused(
            map_entry_1=map_entry_1, map_entry_2=map_entry_2, graph=graph, sdfg=sdfg
        ):
            return False

        # Two maps can be serially fused if the node decomposition exists and
        #  at least one of the intermediate output sets is not empty. The state
        #  of the pure outputs is irrelevant for serial map fusion.
        output_partition = self.partition_first_outputs(
            state=graph,
            sdfg=sdfg,
            map_exit_1=self.map_exit1,
            map_entry_2=self.map_entry2,
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
        assert isinstance(self.map_exit1, nodes.MapExit)
        assert isinstance(self.map_entry2, nodes.MapEntry)
        assert self.map_parameter_compatible(self.map_exit1.map, self.map_entry2.map, graph, sdfg)

        map_exit_1: nodes.MapExit = self.map_exit1
        map_entry_2: nodes.MapEntry = self.map_entry2
        map_exit_2: nodes.MapExit = graph.exit_node(self.map_entry2)
        map_entry_1: nodes.MapEntry = graph.entry_node(self.map_exit1)

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

    @staticmethod
    def handle_intermediate_set(
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

        Todo:
            Rewrite using `MemletTree`.
        """

        # Essentially this function removes the AccessNode between the two maps.
        #  However, we still need some temporary memory that we can use, which is
        #  just much smaller, i.e. a scalar. But all Memlets inside the second map
        #  assumes that the intermediate memory has the bigger shape.
        #  To fix that we will create this replacement dict that will replace all
        #  occurrences of the iteration variables of the second map with zero.
        #  Note that this is still not enough as the dimensionality might be different.
        memlet_repl: Dict[str, int] = {str(param): 0 for param in map_entry_2.map.params}

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
            #  That are known to cause some troubles, so we will now remove them.
            squeezed_dims: List[int] = []  # These are the dimensions we removed.
            new_inter_shape: List[int] = []  # This is the final shape of the new intermediate.
            for dim, (proposed_dim_size, full_dim_size) in enumerate(
                zip(new_inter_shape_raw, inter_shape)
            ):
                # Order of checks is important!
                if full_dim_size == 1:  # Must be kept!
                    new_inter_shape.append(proposed_dim_size)
                elif proposed_dim_size == 1:  # This dimension was reduced, so we can remove it.
                    squeezed_dims.append(dim)
                else:
                    new_inter_shape.append(proposed_dim_size)

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

            # New we will reroute the output Memlet, thus it will no longer pass
            #  through the Map exit but through the newly created intermediate.
            #  we will delete the previous edge later.
            pre_exit_memlet: dace.Memlet = pre_exit_edge.data
            new_pre_exit_memlet = copy.deepcopy(pre_exit_memlet)

            # We might operate on a different array, but the check below, ensures
            #  that we do not change the direction of the Memlet.
            assert pre_exit_memlet.data == inter_name
            new_pre_exit_memlet.data = new_inter_name

            # Now we have to modify the subset of the Memlet.
            #  Before the subset of the Memlet was dependent on the Map variables,
            #  however, this is no longer the case, as we removed them. This change
            #  has to be reflected in the Memlet.
            #  NOTE: Assert above ensures that the below is correct.
            new_pre_exit_memlet.replace(memlet_repl)
            if is_scalar:
                new_pre_exit_memlet.subset = "0"
                new_pre_exit_memlet.other_subset = None
            else:
                new_pre_exit_memlet.subset.pop(squeezed_dims)

            # Now we create the new edge between the producer and the new output
            #  (the new intermediate node). We will remove the old edge further down.
            new_pre_exit_edge = state.add_edge(
                pre_exit_edge.src,
                pre_exit_edge.src_conn,
                new_inter_node,
                None,
                new_pre_exit_memlet,
            )

            # We just have handled the last Memlet, but we must actually handle the
            #  whole producer side, i.e. the scope of the top Map.
            for producer_tree in state.memlet_tree(new_pre_exit_edge).traverse_children():
                producer_edge = producer_tree.edge

                # Ensure the correctness of the rerouting below.
                # TODO(phimuell): Improve the code below to remove the check.
                assert producer_edge.data.data == inter_name

                # Will not change the direction, because of test above!
                producer_edge.data.data = new_inter_name
                producer_edge.data.replace(memlet_repl)
                if is_scalar:
                    producer_edge.data.dst_subset = "0"
                elif producer_edge.data.dst_subset is not None:
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

                    # The create the first Memlet to transmit information, within
                    #  the second map, we do this again by copying and modifying
                    #  the original Memlet.
                    # NOTE: Test above is important to ensure the direction of the
                    #       Memlet and the correctness of the code below.
                    new_inner_memlet = copy.deepcopy(inner_edge.data)
                    new_inner_memlet.replace(memlet_repl)
                    new_inner_memlet.data = new_inter_name  # Because of the assert above, this will not change the direction.

                    # Now remove the old edge, that started the second map entry.
                    #  Also add the new edge that started at the new intermediate.
                    state.remove_edge(inner_edge)
                    new_inner_edge = state.add_edge(
                        new_inter_node,
                        None,
                        inner_edge.dst,
                        inner_edge.dst_conn,
                        new_inner_memlet,
                    )

                    # Now we do subset modification to ensure that nothing failed.
                    if is_scalar:
                        new_inner_memlet.src_subset = "0"
                    elif new_inner_memlet.src_subset is not None:
                        new_inner_memlet.src_subset.pop(squeezed_dims)

                    # Now clean the Memlets of that tree to use the new intermediate node.
                    for consumer_tree in state.memlet_tree(new_inner_edge).traverse_children():
                        consumer_edge = consumer_tree.edge
                        assert consumer_edge.data.data == inter_name
                        consumer_edge.data.data = new_inter_name
                        if is_scalar:
                            consumer_edge.data.src_subset = "0"
                        elif consumer_edge.data.subset is not None:
                            consumer_edge.data.subset.pop(squeezed_dims)

                # The edge that leaves the second map entry was already deleted.
                #  We will now delete the edges that brought the data.
                for edge in list(state.in_edges_by_connector(map_entry_2, in_conn_name)):
                    assert edge.src == inter_node
                    state.remove_edge(edge)
                map_entry_2.remove_in_connector(in_conn_name)
                map_entry_2.remove_out_connector(out_conn_name)

            if is_exclusive_set:
                # In exclusive mode the old intermediate node is no longer needed.
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
                new_exit_memlet = copy.deepcopy(pre_exit_edge.data)
                assert new_exit_memlet.data == inter_name
                new_exit_memlet.subset = pre_exit_edge.data.dst_subset
                new_exit_memlet.other_subset = (
                    "0" if is_scalar else subsets.Range.from_array(inter_desc)
                )

                new_pre_exit_conn = map_exit_2.next_connector()
                state.add_edge(
                    new_inter_node,
                    None,
                    map_exit_2,
                    "IN_" + new_pre_exit_conn,
                    new_exit_memlet,
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
