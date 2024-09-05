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
    """Common parts of the parallel and serial map fusion transformation.

    Args:
        only_inner_maps: Only match Maps that are internal, i.e. inside another Map.
        only_toplevel_maps: Only consider Maps that are at the top.
        strict_dataflow: If `True`, the default, the transformation ensures a more
            stricter version of the data flow.

    Note:
        If `strict_dataflow` mode is enabled then the transformation will not remove
        _direct_ data flow dependency from the graph. Furthermore, the transformation
        will not remove size 1 dimensions of intermediate it crates.
        This is a compatibility mode, that will limit the applicability of the
        transformation, but might help transformations that does not fully analyse
        the graph.
    """

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
    shared_data = properties.DictProperty(
        key_type=SDFG,
        value_type=set, #[str]
        default=None,
        allow_none=True,
        optional=True, # Do not serialize.
        optional_condition=lambda _: False,
        desc="Maps SDFGs to the set of data that can not be removed,"
        " because they transmit data _between states_, such data will be made 'shared'."
        " This variable acts as a cache, and is managed by 'is_shared_data()'.",
    )
    strict_dataflow = properties.Property(
        dtype=bool,
        default=True,
        desc="If `True` then the transformation will ensure a more stricter data flow.",
    )


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

        # We will now check if there exists a remapping that of the map parameter
        if self.find_parameter_remapping(first_map=map_entry_1.map, second_map=map_entry_2.map) is None:
            return False

        return True


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


    def find_parameter_remapping(
        self,
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

        Interstate data is used to transmit data between multiple state or by
        extension within the state. Thus it must be classified as a shared output.
        This function will go through the SDFG to and collect the names of all data
        container that should be classified as shared.

        Args:
            transient: The transient that should be checked.
            sdfg: The SDFG containing the array.

        Note:
            The function computes the this set once for every SDFG and then caches it.
        """
        if sdfg not in self.shared_data:
            self._compute_shared_data(sdfg)
        return data.data in self.shared_data[sdfg]


    def _compute_shared_data(
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
        interstate_read_symbols: Set[str] = set()
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

                elif state.out_degree(access_node) != 1: # state.out_degree() == 0 or state.out_degree() > 1
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

        # Now we are collecting all symbols that interstate edges read from.
        for edge in sdfg.edges():
            interstate_read_symbols.update(edge.data.read_symbols())

        # We also have to keep everything the edges referrers to and is an array.
        shared_data.update(interstate_read_symbols.intersection(prevously_seen_data))

        # Update the internal cache
        self.shared_data[sdfg] = shared_data


    def _compute_multi_write_data(
        self,
        state: SDFGState,
        sdfg: SDFG,
    ) -> Set[str]:
        """Computes data inside a _single_ state, that is written multiple times.

        Essentially this function computes the set of data that does not follow
        the single static assignment idiom. The function also resolves views.
        If an access node that refers to a view, the function will add not only
        the view itself, but also the data it refers to.

        Args:
            state: The state that should be examined.
            sdfg: The SDFG object.

        Note:
            This information is used by the partition function (in case strict data
            flow mode is enabled), if it is legal to turn a intermediate node into
            shared output or if the partition does not exists at all. The current
            implementation is rather simple as it only checks if a data is written
            to multiple times in the same state. A more refined (but still simple)
            implementation would take the location of the access node into
            consideration.
        """
        data_written_to: Set[str] = set()
        multi_write_data: Set[str] = set()

        for access_node in state.data_nodes():
            if state.in_degree(access_node) == 0:
                continue
            if self.is_view(access_node, sdfg):
                # This is an over approximation.
                multi_write_data.update([access_node.data, track_view(access_node, state, sdfg).data])
            elif access_node.data in data_written_to:
                multi_write_data.add(access_node.data)
            data_written_to.add(access_node.data)
        return multi_write_data


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
        assert repl_dict is not None

        # Set of intermediate nodes that we have already processed.
        processed_inter_nodes: Set[nodes.Node] = set()

        # These are the data that is written to multiple times in _this_ state.
        #  If a data is written to multiple time in a state, it could be
        #  classified as shared. However, it might happen that the node has zero
        #  degree. This is not a problem as the maps also induced a before-after
        #  relationship. But some DaCe transformations do not catch this.
        #  Thus we will never modify such intermediate nodes.
        if self.strict_dataflow:
            multi_write_data: Set[str] = self._compute_multi_write_data(state, sdfg)
        else:
            multi_write_data = set()

        # Now scan all output edges of the first exit and classify them
        for out_edge in state.out_edges(map_exit_1):
            intermediate_node: nodes.Node = out_edge.dst

            # We already processed the node, this should indicate that we should
            #  run simplify again, or we should start implementing this case.
            # TODO(phimuell): Handle this case, it is partially implemented.
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

            # Now let's look at all nodes that are downstream of the intermediate node.
            #  This, among other things, will tell us, how we have to handle this node.
            #  NOTE: The traversal will stop at the second map.
            downstream_nodes = self.all_nodes_between(
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

            # Checks if the intermediate node refers to data that is accessed by
            #  _other_ access nodes in _this_ state. If this is the case then never
            #  touch this intermediate node.
            #  TODO(phimuell): Technically it would be enough to turn the node into
            #   a shared output node, because this will still fulfil the dependencies.
            #   However, some DaCe transformation can not handle this properly, so we
            #   are _forced_ to reject this node.
            if intermediate_node.data in multi_write_data:
                return None

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

            # Now we determine the consumer of nodes. For this we are using the edges
            #  leaves the second map entry. It is not necessary to find the actual
            #  consumer nodes, as they might depend on symbols of nested Maps.
            #  For the covering test we only need their subsets, but we will perform
            #  some scan and filtering on them.
            found_second_map = False
            consumer_subsets: List[subsets.Subset] = []
            for intermediate_node_out_edge in state.out_edges(intermediate_node):

                # Ensure that there is no multihop connection to the second map entry.
                if intermediate_node_out_edge.dst is not map_entry_2:
                    if self.all_nodes_between(
                        graph=state,
                        begin=intermediate_node_out_edge.dst,
                        end=map_entry_2,
                    ) is None:
                        continue
                    return None

                # Ensure that the second map is found exactly once.
                if found_second_map:
                    # TODO(phimuell): Lift this restriction.
                    return None
                found_second_map = True

                # Now we look at all edges that leave the second map entry, as they
                #  define what is read inside the map.
                #  NOTE: The subset still uses the old iteration variables.
                assert intermediate_node_out_edge.dst_conn.startswith("IN_")
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
            if len(downstream_nodes) != 0:
                # The intermediate node is connected to more node inside _this_ state.
                shared_outputs.add(out_edge)
            elif self.is_shared_data(intermediate_node, sdfg):
                # The intermediate data is used somewhere else, either in this or another state.
                shared_outputs.add(out_edge)
            else:
                # The intermediate can be removed, as it is not used anywhere else.
                exclusive_outputs.add(out_edge)

        assert exclusive_outputs or shared_outputs or pure_outputs
        assert len(processed_inter_nodes) == sum(len(x) for x in [pure_outputs, exclusive_outputs, shared_outputs])
        return (pure_outputs, exclusive_outputs, shared_outputs)


    def all_nodes_between(
        self,
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

        if reverse:
            def next_nodes(node: nodes.Node) -> Iterable[nodes.Node]:
                return (edge.src for edge in graph.in_edges(node))
            begin, end = end, begin
        else:
            def next_nodes(node: nodes.Node) -> Iterable[nodes.Node]:
                return (edge.dst for edge in graph.out_edges(node))

        to_visit: List[dace_nodes.Node] = [begin]
        seen: Set[dace_nodes.Node] = set()

        while len(to_visit) > 0:
            node: dace_nodes.Node = to_visit.pop()
            if node != end and node not in seen:
                to_visit.extend(next_nodes(node))
            seen.add(node)

        # If `end` was not found we have to return `None` to indicate this.
        #  `begin` and `end` are not included in the output set.
        if end not in seen:
            return None
        return seen - {begin, end}


    def get_access_set(
            self,
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
        access_set: Set[nodes.AccessNode] = {
                node
                for node in map(other_node, get_edges(scope_node))
                if isinstance(node, nodes.AccessNode)
        }
        # As far as I know in a valid SDFG this should not happen.
        assert len(access_set) == len({node.data for node in access_set})
        return access_set


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
        if not self.is_view(view, sdfg):
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
            raise RuntimeError(f"Failed to determine the direction of the view '{view}'.")

        # Now trace the view back.
        org_view = view
        view = next_node(curr_edge)
        while is_view(view, sdfg):
            curr_edge = dace.sdfg.utils.get_view_edge(state, view)
            if curr_edge is None:
                raise RuntimeError(f"View tracing of '{org_view}' failed at note '{view}'.")
            view = next_node(curr_edge)
        return view
