# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.

"""Implements Helper functionaliyies for map fusion"""

import functools
import itertools
import re
import copy
from typing import Any, Dict, Iterable, List, Optional, Set, Sequence, Tuple, Union, overload

import dace
from dace import data, properties, subsets, transformation, symbolic
from dace.sdfg import SDFG, SDFGState, graph, nodes, validation, replace
from dace.transformation import helpers


@properties.make_properties
class MapFusionHelper(transformation.SingleStateTransformation):
    """Contains common part of the fusion for parallel and serial Map fusion.

    The transformation assumes that the SDFG obeys the principals outlined [here](https://hackmd.io/klvzLnzMR6GZBWtRU8HbDg#Requirements-on-SDFG).
    The main advantage of this structure is, that it is rather easy to determine
    if a transient is used anywhere else. This check, performed by
    `is_shared_data()`. It is further speeded up by cashing some computation,
    thus such an object should not be used after interstate optimizations were applied
    to the SDFG.

    Args:
        only_inner_maps: Only match Maps that are internal, i.e. inside another Map.
        only_toplevel_maps: Only consider Maps that are at the top.
    """

    only_toplevel_maps = properties.Property(
        dtype=bool,
        default=False,
        allow_none=False,
        desc="Only perform fusing if the Maps are in the top level.",
    )
    only_inner_maps = properties.Property(
        dtype=bool,
        default=False,
        allow_none=False,
        desc="Only perform fusing if the Maps are inner Maps, i.e. does not have top level scope.",
    )
    shared_data = properties.DictProperty(
        key_type=SDFG,
        value_type=set, #[str]
        default=None,
        allow_none=True,
        desc="Maps SDFGs to the set of data that can not be removed. "
        "The variable acts as a cache, and is managed by 'is_shared_data()'.",
    )

    def __init__(
        self,
        only_inner_maps: Optional[bool] = None,
        only_toplevel_maps: Optional[bool] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        if only_toplevel_maps is not None:
            self.only_toplevel_maps = bool(only_toplevel_maps)
        if only_inner_maps is not None:
            self.only_inner_maps = bool(only_inner_maps)
        self.shared_data = {}


    @classmethod
    def expressions(cls) -> bool:
        raise RuntimeError("The `_MapFusionHelper` is not a transformation on its own.")


    def can_be_fused(
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

        However, for performance reasons, the function does not check if the node
        decomposition exists.

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
            # TODO(phimuell): Figuring out why this is here.
            elif is_nested_sdfg(sdfg):
                return False

        # We will now check if there exists a remapping that of the map parameter
        if self.find_parameter_remapping(first_map=map_entry_1.map, second_map=map_entry_2.map) is None:
            return False

        return True


    @staticmethod
    def relocate_nodes(
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
        for edge_to_move in list(state.in_edges(from_node)):
            assert isinstance(edge_to_move.dst_conn, str)

            if not edge_to_move.dst_conn.startswith("IN_"):
                # Dynamic Map Range
                #  The connector name simply defines a variable name that is used,
                #  inside the Map scope to define a variable. We handle it directly.
                dmr_symbol = edge_to_move.dst_conn

                # TODO(phimuell): Check if the symbol is really unused in the target scope.
                if dmr_symbol in to_node.in_connectors:
                    raise NotImplementedError(
                        f"Tried to move the dynamic map range '{dmr_symbol}' from {from_node}'"
                        f" to '{to_node}', but the symbol is already known there, but the"
                        " renaming is not implemented."
                    )
                if not to_node.add_in_connector(dmr_symbol, force=False):
                    raise RuntimeError(  # Might fail because of out connectors.
                        f"Failed to add the dynamic map range symbol '{dmr_symbol}' to '{to_node}'."
                    )
                helpers.redirect_edge(state=state, edge=edge_to_move, new_dst=to_node)
                from_node.remove_in_connector(dmr_symbol)

                # There is no other edge that we have to consider, so we just end here
                continue

            # We have a Passthrough connection, i.e. there exists a matching `OUT_`.
            old_conn = edge_to_move.dst_conn[3:]  # The connection name without prefix
            new_conn = to_node.next_connector(old_conn)

            to_node.add_in_connector("IN_" + new_conn)
            for e in list(state.in_edges_by_connector(from_node, "IN_" + old_conn)):
                helpers.redirect_edge(state, e, new_dst=to_node, new_dst_conn="IN_" + new_conn)
            to_node.add_out_connector("OUT_" + new_conn)
            for e in list(state.out_edges_by_connector(from_node, "OUT_" + old_conn)):
                helpers.redirect_edge(
                    state, e, new_src=to_node, new_src_conn="OUT_" + new_conn
                )
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


    @staticmethod
    def find_parameter_remapping(
        first_map: nodes.Map,
        second_map: nodes.Map
    ) -> Union[Dict[str, str], None]:
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
        second_rngs: Dict[Tuple[Any, Any, Any]] = {
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
        assert len(final_mapping) == len(first_params)
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


    def is_shared_data(
        self,
        data: nodes.AccessNode,
        sdfg: dace.SDFG,
    ) -> bool:
        """Tests if `data` is interstate data, an can not be removed.

        Interstate data is used to transmit data between multiple state or
        by extension within the state, and thus can not be removed by the
        serial map fusion.

        The function determine this properties, according to the following rules:
        - The access node must be in the top scope.
        - The underlying data is global.
        - The `data` descriptor is used multiple times with the same state.
        - `data` has an out or in degree of zero.
        - The underlying data is referred to in another state.

        The function computes this information and then caches it for later use.

        Args:
            transient: The transient that should be checked.
            sdfg: The SDFG containing the array.

        Note:
            - This function does not inspect the interstate edges, instead the
                set of data that is accessed in interstate edges is approximated
                with the set of sink nodes.
            - This function works best if the SDFG uses SSA style.
        """
        if sdfg not in self.shared_data:
            self._compute_shared_data(sdfg)
        return data.data in self.shared_data[sdfg]


    def _compute_shared_data(
        self,
        sdfg: dace.SDFG,
    ) -> None:
        """This function computes the set of shared data for SDFG `sdfg`.

        See the documentation for `self.is_shared_data()` for a description.

        Args:
            sdfg: The SDFG for which the set of shared data should be computed.
        """
        # Shared data of this SDFG.
        shared_data: Set[str] = set()

        # Add all global data.
        for data_name, data_desc in sdfg.arrays.items():
            if not data_desc.transient:
                shared_data.add(data_name)

        # We go through all states and classify the nodes, according to the rules.
        prevously_seen_data: Set[str] = set()
        for state in sdfg.nodes():
            scope_dict = state.scope_dict()
            for access_node in state.data_nodes():
                if scope_dict[access_node] is not None:
                    # We are only interested in global data.
                    pass
                elif access_node.data in shared_data:
                    # The data was already determined to be shared data
                    pass
                elif access_node.data in prevously_seen_data:
                    # We have seen this data before, either in this state or in
                    #  a previous one, but we did not classifies it as shared,
                    #  let's do this now. Note that we do not remove the data
                    #  also from `previously_seen_data`.
                    shared_data.add(access_node.data)
                elif state.out_degree(access_node) == 0:
                    # Sink and source nodes also have to be kept.
                    shared_data.add(access_node.data)
                elif state.in_degree(access_node) == 0:
                    shared_data.add(access_node.data)
                else:
                    # The node was not classified as shared data, so we record that
                    #  we saw it. Note that a node that was immediately classified
                    #  as shared node will never be added to this set, but a data
                    #  that was found twice will be inside this list.
                    prevously_seen_data.add(access_node.data)

        # Update the internal cache
        self.shared_data[sdfg] = shared_data


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

        # Set of intermediate nodes that we have already processed.
        processed_inter_nodes: Set[nodes.Node] = set()

        # Now scan all output edges of the first exit and classify them
        for out_edge in state.out_edges(map_exit_1):
            intermediate_node: nodes.Node = out_edge.dst

            # We already processed the node, this should indicate that we should
            #  run simplify again, or we should start implementing this case.
            if intermediate_node in processed_inter_nodes:
                return None
            processed_inter_nodes.add(intermediate_node)

            # Now let's look at all nodes that are downstream of the intermediate node.
            #  This, among other things, will tell us, how we have to handle this node.
            downstream_nodes = all_nodes_between(
                graph=state,
                begin=intermediate_node,
                end=map_entry_2,
            )

            # If `downstream_nodes` is `None` this means that `map_entry_2` was never
            #  reached, thus `intermediate_node` does not enter the second map and
            #  the node is a pure output node.
            if downstream_nodes is None:
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
            if isinstance(intermediate_desc, data.View):
                return None

            # Empty Memlets are only allowed if they are in `\mathbb{P}`, which
            #  is also the only place they really make sense (for a map exit).
            #  Thus if we now found an empty Memlet we reject it.
            if out_edge.data.is_empty():
                return None

            # The intermediate now can only have a single source. It might be possible
            #  to extend this to many inputs as long as they come from the top map.
            # NOTE: The output degree is checked implicitly further down, the
            #   general rule is, that multiple outputs are only allowed if only
            #   one enters the second Map, the other output must go to different
            #   consumers, in which case the node is a shared intermediate.
            if state.in_degree(intermediate_node) != 1:
                return None

            # It can happen that multiple edges converges at the `IN_` connector
            #  of the first map exit, but there is only one edge leaving the exit.
            #  It is complicate to handle this, so for now we ignore it.
            # TODO(phimuell): Handle this case properly.
            #   The main reason why we forbid this is because it becomes a bit tricky
            #   to figuring out the size of the intermediate.
            inner_collector_edges = list(state.in_edges_by_connector(intermediate_node, "IN_" + out_edge.src_conn[3:]))
            if len(inner_collector_edges) > 1:
                return None

            # We now look for all producers inside the top map. We need this information
            #  later to determine if the point wise output of the top map is enough
            #  to serve as (point wise) input for the second map.
            # In addition we will check if there is a WCR edge is present, which we
            #  can not handle.
            producers = find_upstream_producers(state, out_edge)

            # More than one producer is not supported.
            if len(producers) != 1:
                return None
            producer_node, producer_edge = next(iter(producers))

            # If the producer is a view, then we give up. It is possible to handle,
            #  but also very complicated.
            if isinstance(producer_node, nodes.AccessNode) and isinstance(producer_node.desc(sdfg), data.View):
                return None

            # We do not allow that the edge has WCR.
            if producer_edge.data.wcr is not None:
                return None

            if len(downstream_nodes) == 0:
                # TODO(phimuell): Refactor this if such that the loop is only there once.

                # There is nothing between intermediate node and the entry of the
                #  second map, thus the edge belongs either in `\mathbb{S}` or
                #  `\mathbb{E}`.

                # If the intermediate access node as more than one outgoing edge
                #  it means (because of `downstream_nodes`) that it has multiple
                #  connections to the second map. We do not allow this.
                # TODO(phimuell): Handle this case.
                if state.out_degree(intermediate_node) != 1:
                    return None

                # We can fuse the maps only if the producer, i.e. the top map,
                #  represented by `producer_edge`, is gives us enough information.
                producer_subset: subsets.Range = producer_edge.data.dst_subset
                consumers = find_downstream_consumers(state=state, begin=intermediate_node)

                for consumer_node, consumer_edge in consumers:
                    if consumer_node is map_entry_2:  # Dynamic map range.
                        return None
                    if isinstance(consumer_node, nodes.LibraryNode):
                        # TODO(phimuell): Allow some library nodes.
                        return None

                    # Tests if consumer consume less or equal the amount we generate.
                    #  If the source of the consumer is not set, we can not check what it reads.
                    if consumer_edge.data.src_subset is None:
                        return None
                    #  For that we have to perform a replace operation of the consumer .
                    mod_consumer_subset = copy.deepcopy(consumer_edge.data.src_subset)
                    symbolic.safe_replace(mapping=repl_dict, replace_callback=mod_consumer_subset.replace)

                    if not producer_subset.covers(mod_consumer_subset):
                        return None

                # Note that "remove" has a special meaning here, regardless of the
                #  output of the check function, from within the second map we remove
                #  the intermediate, it has more the meaning of "do we need to
                #  reconstruct it after the second map again?"
                if self.is_shared_data(intermediate_node, sdfg):
                    shared_outputs.add(out_edge)
                else:
                    exclusive_outputs.add(out_edge)
                continue

            else:
                # There is not only a single connection from the intermediate node to
                #  the second map, but the intermediate has more connections, thus
                #  the node might belong to the shared output. Of the many different
                #  possibilities, we only consider a single case:
                #  - The intermediate has a single connection to the second map, that
                #       fulfills the restriction outlined above.
                #  - All other connections have no connection to the second map.

                # We can fuse the maps only if the producer, i.e. the top map,
                #  represented by `producer_edge`, is gives us enough information.
                producer_subset: subsets.Range = producer_edge.data.dst_subset
                consumers = find_downstream_consumers(state=state, begin=intermediate_node)

                found_second_entry = False
                for edge in state.out_edges(intermediate_node):
                    if edge.dst is map_entry_2:
                        if found_second_entry:  # The second map was found again.
                            return None
                        found_second_entry = True
                        consumers = find_downstream_consumers(state=state, begin=edge)
                        for consumer_node, consumer_edge in consumers:
                            if consumer_node is map_entry_2:  # Dynamic map range.
                                return None
                            if isinstance(consumer_node, nodes.LibraryNode):
                                # TODO(phimuell): Allow some library nodes.
                                return None

                            # Tests if consumer consume less or equal the amount we generate.
                            #  If the source of the consumer is not set, we can not check what it reads.
                            if consumer_edge.data.src_subset is None:
                                return None
                            #  For that we have to perform a replace operation of the consumer .
                            mod_consumer_subset = copy.deepcopy(consumer_edge.data.src_subset)
                            symbolic.safe_replace(mapping=repl_dict, replace_callback=mod_consumer_subset.replace)
                            if not producer_subset.covers(mod_consumer_subset):
                                return None
                    else:
                        # Ensure that there is no path that leads to the second map.
                        after_intermdiate_node = all_nodes_between(graph=state, begin=edge.dst, end=map_entry_2)
                        if after_intermdiate_node is not None:
                            return None
                # If we are here, then we know that the node is a shared output
                shared_outputs.add(out_edge)
                continue

        assert exclusive_outputs or shared_outputs or pure_outputs
        assert len(processed_inter_nodes) == sum(len(x) for x in [pure_outputs, exclusive_outputs, shared_outputs])
        return (pure_outputs, exclusive_outputs, shared_outputs)


def is_nested_sdfg(
    sdfg: Union[dace.SDFG, dace.SDFGState, nodes.NestedSDFG],
) -> bool:
    """Tests if `sdfg` is a NestedSDFG."""
    if isinstance(sdfg, dace.SDFGState):
        sdfg = sdfg.parent
    if isinstance(sdfg, nodes.NestedSDFG):
        return True
    elif isinstance(sdfg, dace.SDFG):
        if sdfg.parent_nsdfg_node is not None:
            return True
        return False
    else:
        raise TypeError(f"Does not know how to handle '{type(sdfg).__name__}'.")


def all_nodes_between(
    graph: Union[dace.SDFG, dace.SDFGState],
    begin: nodes.Node,
    end: nodes.Node,
    reverse: bool = False,
) -> Union[Set[nodes.Node], None]:
    """Find all nodes that are reachable from `begin` but bound by `end`.

    Essentially the function starts a DFS at `begin`. If an edge is found that lead
    to `end`, this edge is ignored. It will thus found any node that is reachable
    from `begin` by a path that does not involve `end`. The returned set will
    never contain `end` nor `begin`. In case `end` is never found the function
    will return `None`.

    If `reverse` is set to `True` the function will start exploring at `end` and
    follows the outgoing edges, i.e. the meaning of `end` and `begin` are swapped.

    Args:
        graph: The graph to operate on.
        begin: The start of the DFS.
        end: The terminator node of the DFS.
        reverse: Perform a backward DFS.

    Notes:
        - The returned set will also contain the nodes of path that starts at
            `begin` and ends at a node that is not `end`.
    """

    def next_nodes(node: nodes.Node) -> Iterable[nodes.Node]:
        if reverse:
            return (edge.src for edge in graph.in_edges(node))
        return (edge.dst for edge in graph.out_edges(node))

    if reverse:
        begin, end = end, begin

    to_visit: List[nodes.Node] = [begin]
    seen: Set[nodes.Node] = set()
    found_end: bool = False

    while len(to_visit) > 0:
        n: nodes.Node = to_visit.pop()
        if n == end:
            found_end = True
            continue
        elif n in seen:
            continue
        seen.add(n)
        to_visit.extend(next_nodes(n))

    if not found_end:
        return None

    seen.discard(begin)
    return seen


def is_parallel(
    graph: Union[dace.SDFG, dace.SDFGState],
    node1: nodes.Node,
    node2: nodes.Node,
) -> bool:
    """Tests if `node1` and `node2` are parallel.

    The nodes are parallel if `node2` can not be reached from `node1` and vice versa.

    Args:
        graph:      The graph to traverse.
        node1:      The first node to check.
        node2:      The second node to check.
    """

    # The `all_nodes_between()` function traverse the graph and returns `None` if
    #  `end` was not found. We have to call it twice, because we do not know
    #  which node is upstream if they are not parallel.
    if all_nodes_between(graph=graph, begin=node1, end=node2) is not None:
        return False
    elif all_nodes_between(graph=graph, begin=node2, end=node1) is not None:
        return False
    return True


def find_downstream_consumers(
    state: dace.SDFGState,
    begin: Union[nodes.Node, graph.MultiConnectorEdge[dace.Memlet]],
    only_tasklets: bool = False,
    reverse: bool = False,
) -> Set[Tuple[nodes.Node, graph.MultiConnectorEdge[dace.Memlet]]]:
    """Find all downstream connectors of `begin`.

    A consumer, in for this function, is any node that is neither an entry nor
    an exit node. The function returns a set of pairs, the first element is the
    node that acts as consumer and the second is the edge that leads to it.
    By setting `only_tasklets` the nodes the function finds are only Tasklets.

    To find this set the function starts a search at `begin`, however, it is also
    possible to pass an edge as `begin`.
    If `reverse` is `True` the function essentially finds the producers that are
    upstream.

    Args:
        state: The state in which to look for the consumers.
        begin: The initial node that from which the search starts.
        only_tasklets: Return only Tasklets.
        reverse: Follow the reverse direction.
    """
    if isinstance(begin, graph.MultiConnectorEdge):
        to_visit: List[graph.MultiConnectorEdge[dace.Memlet]] = [begin]
    elif reverse:
        to_visit = list(state.in_edges(begin))
    else:
        to_visit = list(state.out_edges(begin))
    seen: Set[graph.MultiConnectorEdge[dace.Memlet]] = set()
    found: Set[Tuple[nodes.Node, graph.MultiConnectorEdge[dace.Memlet]]] = set()

    while len(to_visit) != 0:
        curr_edge: graph.MultiConnectorEdge[dace.Memlet] = to_visit.pop()
        next_node: nodes.Node = curr_edge.src if reverse else curr_edge.dst

        if curr_edge in seen:
            continue
        seen.add(curr_edge)

        if isinstance(next_node, (nodes.MapEntry, nodes.MapExit)):
            if reverse:
                target_conn = curr_edge.src_conn[4:]
                new_edges = state.in_edges_by_connector(curr_edge.src, "IN_" + target_conn)
            else:
                # In forward mode a Map entry could also mean the definition of a
                #  dynamic map range.
                if (not curr_edge.dst_conn.startswith("IN_")) and isinstance(
                    next_node, nodes.MapEntry
                ):
                    # This edge defines a dynamic map range, which is a consumer
                    if not only_tasklets:
                        found.add((next_node, curr_edge))
                    continue
                target_conn = curr_edge.dst_conn[3:]
                new_edges = state.out_edges_by_connector(curr_edge.dst, "OUT_" + target_conn)
            to_visit.extend(new_edges)
            del new_edges
        else:
            if only_tasklets and (not isinstance(next_node, nodes.Tasklet)):
                continue
            found.add((next_node, curr_edge))

    return found


def find_upstream_producers(
    state: dace.SDFGState,
    begin: Union[nodes.Node, graph.MultiConnectorEdge[dace.Memlet]],
    only_tasklets: bool = False,
) -> Set[Tuple[nodes.Node, graph.MultiConnectorEdge[dace.Memlet]]]:
    """Same as `find_downstream_consumers()` but with `reverse` set to `True`."""
    return find_downstream_consumers(
        state=state,
        begin=begin,
        only_tasklets=only_tasklets,
        reverse=True,
    )


def get_access_set(
        scope_node: Union[nodes.MapEntry, nodes.MapExit],
        state: SDFGState,
) -> Set[nodes.AccessNode]:
    """Computes the access set of a "scope node".

    If `scope_node` is a `MapEntry` node it will operate on the set of incoming
    edges and if it is an `MapExit` node on the set of outgoing edges. The
    function will then determine all access nodes that have a connection through
    these edges to the scope nodes (edges that does not lead to access nodes are
    ignored).
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
    return {
            node
            for node in map(other_node, get_edges(scope_node))
            if isinstance(node, nodes.AccessNode)
    }


def is_view(
        node: nodes.AccessNode,
        sdfg: SDFG,
) -> bool:
    """Tests if `node` points to a view or not."""
    node_desc: data.Data = node.desc(sdfg)
    return isinstance(node_desc, data.View)


def track_view(
        view: nodes.AccessNode,
        state: SDFGState,
        sdfg: SDFG,
) -> nodes.AccessNode:
    """Find the original data of a View.

    Given the View `view`, the function will trace the view back to the
    original access node.
    For convenience, if `view` is not a `View` but a normal data descriptor,
    then the function will return the argument unmodified.

    Args:
        view: The view that should be traced.
        state: The state in which we operate.
        sdfg: The SDFG on which we operate.
    """

    # Test if it is a view at all, if not return the passed node as source.
    if not is_view(view, sdfg):
        return view

    # First determine if the view is used for reading or writing.
    curr_edge = dace.sdfg.utils.get_view_edge(state, view)
    if curr_edge is None:
        raise RuntimeError(f"Failed to determine the direction of the view '{view}'.")
    if curr_edge.dst_conn == "view":
        # The view is used for reading.
        next_node = lambda curr_edge: curr_edge.src
    elif curr_edge.src_conn == "view":
        # The view is used for writing.
        next_node = lambda curr_edge: curr_edge.dst
    else:
        raise RuntimeError("Failed to determine the direction of the view '{view}'.")

    # Now trace the view back.
    org_view = view
    view = next_node(curr_edge)
    while is_view(view, sdfg):
        curr_edge = dace.sdfg.utils.get_view_edge(state, view)
        if curr_edge is None:
            raise RuntimeError(f"View tracing of '{org_view}' failed at note '{view}'.")
        view = next_node(curr_edge)
    return view
