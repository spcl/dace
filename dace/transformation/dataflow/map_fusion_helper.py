# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.

"""Implements Helper functionaliyies for map fusion"""

import functools
import itertools
import re
from typing import Any, Dict, Iterable, List, Literal, Optional, Set, Sequence, Tuple, Union, overload

import dace
from dace import data, properties, subsets, transformation
from dace.sdfg import SDFG, SDFGState, graph, nodes, validation
from dace.transformation import helpers
from dace.transformation.dataflow import map_fusion_helper

@properties.make_properties
class MapFusionHelper(transformation.SingleStateTransformation):
    """Contains common part of the fusion for parallel and serial Map fusion.

    The transformation assumes that the SDFG obeys the principals outlined [here](https://hackmd.io/klvzLnzMR6GZBWtRU8HbDg#Requirements-on-SDFG).
    The main advantage of this structure is, that it is rather easy to determine
    if a transient is used anywhere else. This check, performed by
    `is_interstate_transient()`. It is further speeded up by cashing some computation,
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
    shared_transients = properties.DictProperty(
        key_type=SDFG,
        value_type=set, #[str]
        default=None,
        allow_none=True,
        desc="Maps SDFGs to the set of array transients that can not be removed. "
        "The variable acts as a cache, and is managed by 'is_interstate_transient()'.",
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
        self.shared_transients = {}

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
            elif map_fusion_helper.is_nested_sdfg(sdfg):
                return False

        # We will now check if there exists a "remapping" that we can use.
        if not self.map_parameter_compatible(
            map_1=map_entry_1.map, map_2=map_entry_2.map, state=graph, sdfg=sdfg
        ):
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
    def map_parameter_compatible(
        map_1: nodes.Map,
        map_2: nodes.Map,
        state: Union[SDFGState, SDFG],
        sdfg: SDFG,
    ) -> bool:
        """Checks if the parameters of `map_1` are compatible with `map_2`.

        The check follows the following rules:
        - The names of the map variables must be the same, i.e. no renaming
            is performed.
        - The ranges must be the same.
        """
        range_1: subsets.Range = map_1.range
        params_1: Sequence[str] = map_1.params
        range_2: subsets.Range = map_2.range
        params_2: Sequence[str] = map_2.params

        # The maps are only fuseable if we have an exact match in the parameter names
        #  this is because we do not do any renaming. This is in accordance with the
        #  rules.
        if set(params_1) != set(params_2):
            return False

        # Maps the name of a parameter to the dimension index
        param_dim_map_1: Dict[str, int] = {pname: i for i, pname in enumerate(params_1)}
        param_dim_map_2: Dict[str, int] = {pname: i for i, pname in enumerate(params_2)}

        # To fuse the two maps the ranges must have the same ranges
        for pname in params_1:
            idx_1 = param_dim_map_1[pname]
            idx_2 = param_dim_map_2[pname]
            # TODO(phimuell): do we need to call simplify?
            if range_1[idx_1] != range_2[idx_2]:
                return False

        return True

    def is_interstate_transient(
        self,
        transient: Union[str, nodes.AccessNode],
        sdfg: dace.SDFG,
        state: dace.SDFGState,
    ) -> bool:
        """Tests if `transient` is an interstate transient, an can not be removed.

        Essentially this function checks if a transient might be needed in a
        different state in the SDFG, because it transmit information from
        one state to the other.
        If only the name of the data container is passed the function will
        first look for an corresponding access node.

        The set of these "interstate transients" is computed once per SDFG.
        The result is then cached internally for later reuse.

        Args:
            transient: The transient that should be checked.
            sdfg: The SDFG containing the array.
            state: If given the state the node is located in.

        Note:
            This function build upon the structure of the SDFG that is outlined
            in the HackMD document.
        """

        # According to [rule 6](https://hackmd.io/klvzLnzMR6GZBWtRU8HbDg#Requirements-on-SDFG)
        #  the set of such transients is partially given by all source access nodes.
        #  Because of rule 3 we also include all scalars in this set, as an over
        #  approximation. Furthermore, because simplify might violate rule 3,
        #  we also include the sink nodes.

        # See if we have already computed the set
        if sdfg in self.shared_transients:
            shared_sdfg_transients: Set[str] = self.shared_transients[sdfg]
        else:
            # SDFG is not known so we have to compute the set.
            shared_sdfg_transients = set()
            for state_to_scan in sdfg.all_states():
                # TODO(phimuell): Use `all_nodes_recursive()` once it is available.
                shared_sdfg_transients.update(
                    [
                        node.data
                        for node in itertools.chain(
                            state_to_scan.source_nodes(), state_to_scan.sink_nodes()
                        )
                        if isinstance(node, nodes.AccessNode)
                        and sdfg.arrays[node.data].transient
                    ]
                )
            self.shared_transients[sdfg] = shared_sdfg_transients

        if isinstance(transient, str):
            name = transient
            matching_access_nodes = [node for node in state.data_nodes() if node.data == name]
            # Rule 8: There is only one access node per state for data.
            assert len(matching_access_nodes) == 1
            transient = matching_access_nodes[0]
        else:
            assert isinstance(transient, nodes.AccessNode)
            name = transient.data

        desc: data.Data = sdfg.arrays[name]
        if not desc.transient:
            return True
        if isinstance(desc, data.Scalar):
            return True  # Scalars can not be removed by fusion anyway.

        # Rule 8: If degree larger than one then it is used within the state.
        if state.out_degree(transient) > 1:
            return True

        # Now we check if it is used in a different state.
        return name in shared_sdfg_transients

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

        # Set of intermediate nodes that we have already processed.
        processed_inter_nodes: Set[nodes.Node] = set()

        # Now scan all output edges of the first exit and classify them
        for out_edge in state.out_edges(map_exit_1):
            intermediate_node: nodes.Node = out_edge.dst

            # We already processed the node, this should indicate that we should
            #  run simplify again, or we should start implementing this case.
            if intermediate_node in processed_inter_nodes:
                print(f"399")
                return None
            processed_inter_nodes.add(intermediate_node)

            # Now let's look at all nodes that are downstream of the intermediate node.
            #  This, among other things, will tell us, how we have to handle this node.
            downstream_nodes = map_fusion_helper.all_nodes_between(
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
                print(f"428")
                return None
            intermediate_desc: data.Data = intermediate_node.desc(sdfg)
            if isinstance(intermediate_desc, data.View):
                print(f"432")
                return None

            # Empty Memlets are only allowed if they are in `\mathbb{P}`, which
            #  is also the only place they really make sense (for a map exit).
            #  Thus if we now found an empty Memlet we reject it.
            if out_edge.data.is_empty():
                print(f"out_endge empty.")
                return None

            # The intermediate now can only have a single source. It might be possible
            #  to extend this to many inputs as long as they come from the top map.
            # NOTE: The output degree is checked implicitly further down, the
            #   general rule is, that multiple outputs are only allowed if only
            #   one enters the second Map, the other output must go to different
            #   consumers, in which case the node is a shared intermediate.
            if state.in_degree(intermediate_node) != 1:
                print(f"449")
                return None

            # It can happen that multiple edges converges at the `IN_` connector
            #  of the first map exit, but there is only one edge leaving the exit.
            #  It is complicate to handle this, so for now we ignore it.
            # TODO(phimuell): Handle this case properly.
            #   The main reason why we forbid this is because it becomes a bit tricky
            #   to figuring out the size of the intermediate.
            inner_collector_edges = list(
                state.in_edges_by_connector(intermediate_node, "IN_" + out_edge.src_conn[3:])
            )
            if len(inner_collector_edges) > 1:
                print(f"469")
                return None

            # An important assumption we made for fusion is that the data is "point
            #  wise interchangeable/compatible", for a more involved definition see
            #  `is_pointwise_subset()`. We will now check this for the "producer side"
            #  (the consumer side is handled later). There is an important point here,
            #  in case the new intermediate is only a scalar, then this is completely
            #  safe. Due to the fact how a Map is defined in SDFG. If the new
            #  intermediate is not a scalar, such as `A[i, j, :]` in `Map[i=..., j=...]`
            #  then it is a bit of a gamble and to be fully sure we would need to look
            #  at the consumer subset, however, these should be edge cases.
            # TODO(phimuell): Use the `param_association` to evaluate which dimensions
            #   are actually used and store this here, below use this to check if the
            #   same dimensions are accessed by the consumer.
            for inner_collector_edge in inner_collector_edges:
                if not is_pointwise_subset(inner_collector_edge.data.dst_subset, map_params_1):
                    print(f"479")
                    return None

            # Another restriction we impose is that we do not allow WCR.
            for _, produce_edge in map_fusion_helper.find_upstream_producers(state, out_edge):
                if produce_edge.data.wcr is not None:
                    print(f"485")
                    return None

            if len(downstream_nodes) == 0:
                # There is nothing between intermediate node and the entry of the
                #  second map, thus the edge belongs either in `\mathbb{S}` or
                #  `\mathbb{E}`.

                # If the intermediate access node as more than one outgoing edge
                #  it means (because of `downstream_nodes`) that it has multiple
                #  connections to the second map. We do not allow this.
                # TODO(phimuell): Handle this case.
                if state.out_degree(intermediate_node) != 1:
                    print(f"489")
                    return None

                # We now look at the consumers, as above we assume that the consumption.
                #  is point wise, however, we allow multiple consumer. As written
                #  above is safe if the new intermediate is a scalar, in case of an
                #  array it is pretty safe (see todo above).
                # Furthermore, we disallow certain type of consumer.
                consumers = map_fusion_helper.find_downstream_consumers(state=state, begin=intermediate_node)
                for consumer_node, feed_edge in consumers:
                    if not is_pointwise_subset(feed_edge.data.src_subset, map_params_2):
                        print(f"399// {feed_edge.data.src_subset} | {map_params_2}")
                        return None
                    if consumer_node is map_entry_2:  # Dynamic map range.
                        print(f"399_")
                        return None
                    if isinstance(consumer_node, nodes.LibraryNode):
                        # TODO(phimuell): Allow some library nodes.
                        print(f"399__")
                        return None

                # Note that "remove" has a special meaning here, regardless of the
                #  output of the check function, from within the second map we remove
                #  the intermediate, it has more the meaning of "do we need to
                #  reconstruct it after the second map again?"
                if self.is_interstate_transient(intermediate_node, sdfg, state):
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
                found_second_entry = False
                intermediate_size = functools.reduce(lambda a, b: a * b, intermediate_desc.shape)
                for edge in state.out_edges(intermediate_node):
                    if edge.dst is map_entry_2:
                        if found_second_entry:  # The second map was found again.
                            return None
                        found_second_entry = True
                        consumers = map_fusion_helper.find_downstream_consumers(state=state, begin=edge)
                        for consumer_node, feed_edge in consumers:
                            if feed_edge.data.num_elements() == intermediate_size:
                                return None
                            if consumer_node is map_entry_2:  # Dynamic map range
                                return None
                            if isinstance(consumer_node, nodes.LibraryNode):
                                # TODO(phimuell): Allow some library nodes.
                                return None
                    else:
                        # Ensure that there is no path that leads to the second map.
                        after_intermdiate_node = map_fusion_helper.all_nodes_between(
                            graph=state, begin=edge.dst, end=map_entry_2
                        )
                        if after_intermdiate_node is not None:
                            return None
                # If we are here, then we know that the node is a shared output
                shared_outputs.add(out_edge)
                continue

        assert exclusive_outputs or shared_outputs or pure_outputs
        assert len(processed_inter_nodes) == sum(
            len(x) for x in [pure_outputs, exclusive_outputs, shared_outputs]
        )
        return (pure_outputs, exclusive_outputs, shared_outputs)


@overload
def is_pointwise_subset(
    subset: subsets.Range,
    map_params: List[str],
    param_association: Literal[False],
) -> bool:
    ...


@overload
def is_pointwise_subset(
        subset: subsets.Range,
        map_params: List[str],
        param_association: Literal[True],
) -> Optional[List[int]]:
    ...


def is_pointwise_subset(
        subset: subsets.Range,
        map_params: List[str],
        param_association: bool = False,
) -> bool:
    """Tests if `subset` is "point wise" with respect to map parameters `map_params`.

    Essentially a subset is point wise, with respect to map parameters, if it access
    the data in a `A[i, j]` manner. An example for a not point wise access would be
    `A[i + 1, j]`. However, there are some special cases:
    - All map parameters must be used, For example the expression `A[i, :]`, inside
        the map `Map[i=0:N, j=0:M]` is not point wise, because `j` is not used.
    - On the other hand if `A` is a 3D array then expressions such as `A[i, :, j]`
        or `A[i, 3, j]` would be point wise. Although they are not a scalar.
    - Furthermore, all parameters must appear exactly once, i.e. accesses such as
        `A[i, i]`, even inside `Map[i=0:N]` is not point wise.

    It is important to realize that point wise is a very powerful property, since
    it essentially releases us from the check of the order of the parameter.
    However, there are some cases were it might fail.

    If the `param_association` argument is set to `True` the function will return the
    parameter association, This is a list of integer, that indicates which parameter
    was found in which dimension of the subset.
    If the subset is point wise the function will return `None`.

    Args:
        subset:     The subset to inspect.
        map_params: The list of parameters to inspect.
        param_association: Return the parameter association.
    """
    map_patterns = [re.compile(f"\\b{str(map_param)}\\b") for map_param in map_params]
    subset_sizes = subset.size_exact()
    unused_params = set(map_params)
    parameter_to_dim_map: Dict[str, int] = dict()

    # Now go through each dimension of the subset and inspect them.
    for dim in range(subset.dims()):
        if(subset_sizes[dim] == 1):
            # Only a single element is consumed, thus we must test if the access
            #  is done through a yet unused map parameter only.
            ss_idx = str(subset[dim][0])
            for map_param, map_pattern in zip(map_params, map_patterns):
                if(ss_idx == map_param):
                    # The map parameter is used alone without any additions.
                    if(map_param not in unused_params):
                        # The map parameter was already used, so we have something
                        #  like `A[i, i]`. Thus it is not point wise!
                        return None if param_association else False

                    # The parameter is used alone, so this is point wise.
                    unused_params.discard(map_param)
                    parameter_to_dim_map[map_param] = dim
                    break

                elif(map_pattern.match(ss_idx)):
                    # The parameter matches partially, e.g. `A[i + 1]`, and is not point wise
                    return None if param_association else False

            # If we here then `ss_idx` did not depend in any way on the map parameters.
            #  This is the case if it is a literal or an other symbol, but we know that
            #  it is constant (because of how symbols work). If it is really point wise
            #  depends on if all symbols are consumed.

        elif(subset_sizes[dim] == 0):
            # This is a strange case that we ignore but it does not violate point wise.
            pass

        else:
            # There are multiple elements that are consumed. An example would be
            #  expressions such as `A[i, :, j]` again for a 2D Map. For now we allow
            #  them, but it is a bit dangerous to do this because it only works if
            #  the other map also processed that that with that expression.
            #  This is a fair assumption.
            for ss_element in map(str, subset[dim]):
                if any(map_pattern.match(ss) for ss in ss_element):
                    return None if param_association else False

    # Not all parameters were used, so it is not point wise
    if(len(unused_params) != 0):
        return None if param_association else False

    if(param_association):
        return [parameter_to_dim_map[map_param] for map_param in map_params]
    return True


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




