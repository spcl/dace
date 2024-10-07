# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
"""Implements Helper functionaliyies for map fusion"""

import copy
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple, Union

import dace
from dace import data, properties, subsets, symbolic, transformation
from dace.sdfg import SDFG, SDFGState, nodes, validation
from dace.transformation import helpers


@properties.make_properties
class MapFusionHelper(transformation.SingleStateTransformation):
    """Common parts of the parallel and serial map fusion transformation.

    Args:
        only_inner_maps: Only match Maps that are internal, i.e. inside another Map.
        only_toplevel_maps: Only consider Maps that are at the top.
        strict_dataflow: If `True`, the transformation ensures a more
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
    strict_dataflow = properties.Property(
        dtype=bool,
        default=False,
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
    def expressions(cls) -> bool:
        raise RuntimeError("The `MapFusionHelper` is not a transformation on its own.")

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

    def is_shared_data(
        self,
        data: nodes.AccessNode,
        sdfg: dace.SDFG,
    ) -> bool:
        """Tests if `data` is interstate data, an can not be removed.

        Interstate data is used to transmit data between multiple state or by
        extension within the state. Thus it must be classified as a shared output.
        This function will go through the SDFG to and collect the names of all data
        container that should be classified as shared. Note that this is an over
        approximation as it does not take the location into account, i.e. "is no longer
        used".

        Args:
            transient: The transient that should be checked.
            sdfg: The SDFG containing the array.

        Note:
            The function computes the this set once for every SDFG and then caches it.
            There is no mechanism to detect if the cache must be evicted. However,
            as long as no additional data is added, there is no problem.
        """
        if sdfg not in self._shared_data:
            self._compute_shared_data(sdfg)
        return data.data in self._shared_data[sdfg]

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

        # Now we are collecting all symbols that interstate edges read from.
        for edge in sdfg.edges():
            interstate_read_symbols.update(edge.data.read_symbols())

        # We also have to keep everything the edges referrers to and is an array.
        shared_data.update(interstate_read_symbols.intersection(prevously_seen_data))

        # Update the internal cache
        self._shared_data[sdfg] = shared_data

    def _compute_multi_write_data(
        self,
        state: SDFGState,
        sdfg: SDFG,
    ) -> Set[str]:
        """Computes data inside a _single_ state, that is written multiple times.

        Essentially this function computes the set of data that does not follow
        the single static assignment idiom. The function also resolves views.
        If an access node, refers to a view, not only the view itself, but also
        the data it refers to is added to the set.

        Args:
            state: The state that should be examined.
            sdfg: The SDFG object.

        Note:
            This information is used by the partition function (in case strict data
            flow mode is enabled), in strict data flow mode only. The current
            implementation is rather simple as it only checks if a data is written
            to multiple times in the same state.
        """
        data_written_to: Set[str] = set()
        multi_write_data: Set[str] = set()

        for access_node in state.data_nodes():
            if state.in_degree(access_node) == 0:
                continue
            if access_node.data in data_written_to:
                multi_write_data.add(access_node.data)
            elif self.is_view(access_node, sdfg):
                # This is an over approximation.
                multi_write_data.update([access_node.data, self.track_view(access_node, state, sdfg).data])
            data_written_to.add(access_node.data)
        return multi_write_data

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
