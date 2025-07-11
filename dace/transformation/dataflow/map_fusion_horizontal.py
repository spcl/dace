# Copyright 2019-2025 ETH Zurich and the DaCe authors. All rights reserved.
from typing import Any, Dict, Optional, Set, Union

import dace
from dace import properties, transformation
from dace.sdfg import SDFG, nodes
from dace.transformation.dataflow import map_fusion_helper as mfhelper


@properties.make_properties
class MapFusionHorizontal(transformation.SingleStateTransformation):
    """Implements the horizontal Map fusion transformation.

    This transformation allows to merge any two Maps as long as the following conditions hold:
    - The Maps are parallel, i.e. they are in concurrent subgraph or neither of the Maps is
        reachable from the other nodes.
    - Their Map ranges are compatible, essentially they iterate over the same space.
    - They are in the same scope.

    This is different from VerticalMapFusion that is restricted to the case that the two Maps
    have a linear/serial dependency between each other. Furthermore, this transformation will
    never reduce the memory footprint as no intermediate is ever removed.

    An example would be the following:
    ```python
    for i in range(N):
        T[i] = foo(A[i])
    for j in range(N):
        S[j] = bar(B[j], A[j + 1])
    ```
    which would be translated into:
    ```python
    for i in range(N):
        T[i] = foo(A[i])
        S[j] = bar(B[j], A[j + 1])
    ```

    :param only_inner_maps: Only match Maps that are internal, i.e. inside another Map.
    :param only_toplevel_maps: Only consider Maps that are at the top.
    :param only_if_common_ancestor: Only apply if the both Map have a direct common ancestor.
        Note that this does not mean that both Maps read from the same data, let's say `A`, as
        in the example above, they have to read from the same AccessNode.
    :param consolidate_edges_only_if_not_extending: If `True`, the default is `False`,
        the transformation will only consolidate edges if this does not lead to an
        extension of the subset.
    :param never_consolidate_edges: If `False`, the default, the function will never
        try to consolidate the edges. Thus Maps might have multiple connectors that
        goes to the same AccessNode.

    :note: This transformation modifies more nodes than it matches.
    :note: Since the Maps that should be fused can be everywhere all possible combinations,
        i.e. all pair of Maps, must be checked. It is thus advised to first run `VerticalMapFusion`
        or other transformations that reduces the numbers of Maps.
    """

    first_parallel_map_entry = transformation.transformation.PatternNode(nodes.MapEntry)
    second_parallel_map_entry = transformation.transformation.PatternNode(nodes.MapEntry)

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
    only_if_common_ancestor = properties.Property(
        dtype=bool,
        default=False,
        desc="If `True` restrict parallel Map fusion to maps that have a direct common ancestor.",
    )

    never_consolidate_edges = properties.Property(
        dtype=bool,
        default=False,
        desc="If `True`, always create a new connector, instead of reusing one that referring to the same data.",
    )
    consolidate_edges_only_if_not_extending = properties.Property(
        dtype=bool,
        default=False,
        desc="Only consolidate if this does not lead to an extension of the subset.",
    )

    def __init__(
        self,
        only_inner_maps: Optional[bool] = None,
        only_toplevel_maps: Optional[bool] = None,
        only_if_common_ancestor: Optional[bool] = None,
        consolidate_edges_only_if_not_extending: Optional[bool] = None,
        never_consolidate_edges: Optional[bool] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        if only_toplevel_maps is not None:
            self.only_toplevel_maps = only_toplevel_maps
        if only_inner_maps is not None:
            self.only_inner_maps = only_inner_maps
        if only_if_common_ancestor is not None:
            self.only_if_common_ancestor = only_if_common_ancestor
        if never_consolidate_edges is not None:
            self.never_consolidate_edges = never_consolidate_edges
        if consolidate_edges_only_if_not_extending is not None:
            self.consolidate_edges_only_if_not_extending = consolidate_edges_only_if_not_extending

    @classmethod
    def expressions(cls) -> Any:
        map_fusion_parallel_match = dace.sdfg.graph.OrderedMultiDiConnectorGraph()
        map_fusion_parallel_match.add_nodes_from([cls.first_parallel_map_entry, cls.second_parallel_map_entry])
        return [map_fusion_parallel_match]

    def can_be_applied(
        self,
        graph: Union[dace.SDFGState, SDFG],
        sdfg: dace.SDFG,
    ) -> bool:
        # NOTE: The after this point it is not legal to access the matched nodes
        first_map_entry: nodes.MapEntry = self.first_parallel_map_entry
        second_map_entry: nodes.MapEntry = self.second_parallel_map_entry
        assert isinstance(first_map_entry, nodes.MapEntry)
        assert isinstance(second_map_entry, nodes.MapEntry)

        # Since we matched any two Maps in the state, we have to ensure that they
        #  are in the same scope (e.g. same state, or same parent Map), otherwise it could be that one is inside one Map
        #  while the other is inside another one.
        scope = graph.scope_dict()
        if scope[first_map_entry] != scope[second_map_entry]:
            return False

        # Test if they have they share a node as direct ancestor.
        if self.only_if_common_ancestor:
            first_ancestors: Set[nodes.Node] = {e1.src for e1 in graph.in_edges(first_map_entry)}
            if not any(e2.src in first_ancestors for e2 in graph.in_edges(second_map_entry)):
                return False

        # We will now check if the two maps are parallel.
        if not mfhelper.is_parallel(graph=graph, node1=first_map_entry, node2=second_map_entry):
            return False

        # Check the structural properties of the Maps. The function will return
        #  the `dict` that describes how the parameters must be renamed (for caching)
        #  or `None` if the maps can not be structurally fused.
        param_repl = mfhelper.can_topologically_be_fused(
            first_map_entry=first_map_entry,
            second_map_entry=second_map_entry,
            graph=graph,
            sdfg=sdfg,
            only_inner_maps=self.only_inner_maps,
            only_toplevel_maps=self.only_toplevel_maps,
        )
        if param_repl is None:
            return False

        return True

    def apply(
        self,
        graph: Union[dace.SDFGState, dace.SDFG],
        sdfg: dace.SDFG,
    ) -> None:
        # NOTE: The after this point it is not legal to access the matched nodes
        first_map_entry: nodes.MapEntry = self.first_parallel_map_entry
        second_map_entry: nodes.MapEntry = self.second_parallel_map_entry
        assert isinstance(first_map_entry, nodes.MapEntry)
        assert isinstance(second_map_entry, nodes.MapEntry)

        first_map_exit: nodes.MapExit = graph.exit_node(first_map_entry)
        second_map_exit: nodes.MapExit = graph.exit_node(second_map_entry)

        # To ensures that the `{src,dst}_subset` are properly set, run initialization.
        #  See [issue 1708](https://github.com/spcl/dace/issues/1703)
        #  Because we do not need to look at them, we were able to skip them in the
        #  `can_be_applied()` function, but now we have to do it.
        for edge in graph.edges():
            edge.data.try_initialize(sdfg, graph, edge)

        # We have to get the scope_dict before we start mutating the graph.
        scope_dict: Dict = graph.scope_dict().copy()

        # Before we do anything we perform the renaming, i.e. we will rename the
        #  parameters of the second Map such that they match the one of the first Map.
        mfhelper.rename_map_parameters(
            first_map=first_map_entry.map,
            second_map=second_map_entry.map,
            second_map_entry=second_map_entry,
            state=graph,
        )

        # Now we relocate all connectors from the second to the first Map and remove
        #  the respective node of the second Map.
        for to_node, from_node in [
            (first_map_entry, second_map_entry),
            (first_map_exit, second_map_exit),
        ]:
            mfhelper.relocate_nodes(
                from_node=from_node,
                to_node=to_node,
                state=graph,
                sdfg=sdfg,
                scope_dict=scope_dict,
                never_consolidate_edges=self.never_consolidate_edges,
                consolidate_edges_only_if_not_extending=self.consolidate_edges_only_if_not_extending,
            )
            # The relocate function does not remove the node, so we must do it.
            graph.remove_node(from_node)

        # If we have "consolidated" edges, i.e. reused existing edges, then the set
        #  of that might have expanded, thus we have to propagate them. However,
        #  in case we never consolidated, i.e. all edges were preserved, then we
        #  can skip that step.
        if not self.never_consolidate_edges:
            mfhelper.propagate_memlets_map_scope(sdfg, graph, first_map_entry)
