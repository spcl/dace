# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.

"""Implements the parallel map fusing transformation."""

from typing import Any, Optional, Set, Union

import dace
from dace import properties, transformation
from dace.sdfg import SDFG, SDFGState, graph, nodes

from dace.transformation.dataflow import map_fusion_helper

@properties.make_properties
class ParallelMapFusion(map_fusion_helper.MapFusionHelper):
    """The `ParallelMapFusion` transformation allows to merge two parallel maps.

    While the `SerialMapFusion` transformation fuses maps that are sequentially
    connected by an intermediate node, this transformation is able to fuse any
    two maps that are not sequential and in the same scope.

    Args:
        only_if_common_ancestor: Only perform fusion if both Maps share at least one
            node as direct ancestor. This will increase the locality of the merge.
        only_inner_maps: Only match Maps that are internal, i.e. inside another Map.
        only_toplevel_maps: Only consider Maps that are at the top.

    Note:
        This transformation only matches the entry nodes of the Map, but will also
        modify the exit nodes of the Maps.
    """

    map_entry1 = transformation.transformation.PatternNode(nodes.MapEntry)
    map_entry2 = transformation.transformation.PatternNode(nodes.MapEntry)

    only_if_common_ancestor = properties.Property(
        dtype=bool,
        default=False,
        allow_none=False,
        desc="Only perform fusing if the Maps share a node as parent.",
    )


    def __init__(
        self,
        only_if_common_ancestor: Optional[bool] = None,
        **kwargs: Any,
    ) -> None:
        if only_if_common_ancestor is not None:
            self.only_if_common_ancestor = only_if_common_ancestor
        super().__init__(**kwargs)


    @classmethod
    def expressions(cls) -> Any:
        # This just matches _any_ two Maps inside a state.
        state = graph.OrderedMultiDiConnectorGraph()
        state.add_nodes_from([cls.map_entry1, cls.map_entry2])
        return [state]


    def can_be_applied(
        self,
        graph: Union[SDFGState, SDFG],
        expr_index: int,
        sdfg: dace.SDFG,
        permissive: bool = False,
    ) -> bool:
        map_entry_1: nodes.MapEntry = self.map_entry1
        map_entry_2: nodes.MapEntry = self.map_entry2

        # Check the structural properties of the maps, this will also ensure that
        #  the two maps are in the same scope and the parameters can be renamed
        if not self.can_be_fused(
            map_entry_1=map_entry_1,
            map_entry_2=map_entry_2,
            graph=graph,
            sdfg=sdfg,
            permissive=permissive,
        ):
            return False

        # Since the match expression matches any twp Maps, we have to ensure that
        #  the maps are parallel. The `can_be_fused()` function already verified
        #  if they are in the same scope.
        if not self._is_parallel(graph=graph, node1=map_entry_1, node2=map_entry_2):
            return False

        # Test if they have they share a node as direct ancestor.
        if self.only_if_common_ancestor:
            # This assumes that there is only one access node per data container in the state.
            ancestors_1: Set[nodes.Node] = {e1.src for e1 in graph.in_edges(map_entry_1)}
            if not any(e2.src in ancestors_1 for e2 in graph.in_edges(map_entry_2)):
                return False

        return True


    def _is_parallel(
        self,
        graph: SDFGState,
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

        # In order to be parallel they must be in the same scope.
        scope = graph.scope_dict()
        if scope[node1] != scope[node2]:
            return False

        # The `all_nodes_between()` function traverse the graph and returns `None` if
        #  `end` was not found. We have to call it twice, because we do not know
        #  which node is upstream if they are not parallel.
        if self.all_nodes_between(graph=graph, begin=node1, end=node2) is not None:
            return False
        elif self.all_nodes_between(graph=graph, begin=node2, end=node1) is not None:
            return False
        return True


    def apply(self, graph: Union[SDFGState, SDFG], sdfg: SDFG) -> None:
        """Performs the Map fusing.

        Essentially, the function relocate all edges from the scope nodes (`MapEntry`
        and `MapExit`) of the second map to the scope nodes of the first map.
        """

        map_entry_1: nodes.MapEntry = self.map_entry1
        map_exit_1: nodes.MapExit = graph.exit_node(map_entry_1)
        map_entry_2: nodes.MapEntry = self.map_entry2
        map_exit_2: nodes.MapExit = graph.exit_node(map_entry_2)

        # Before we do anything we perform the renaming.
        self.rename_map_parameters(
                first_map=map_entry_1.map,
                second_map=map_entry_2.map,
                second_map_entry=map_entry_2,
                state=graph,
        )

        for to_node, from_node in zip((map_entry_1, map_exit_1), (map_entry_2, map_exit_2)):
            self.relocate_nodes(
                from_node=from_node,
                to_node=to_node,
                state=graph,
                sdfg=sdfg,
            )
            # The relocate function does not remove the node, so we must do it.
            graph.remove_node(from_node)
