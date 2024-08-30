# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.

"""Implements the parallel map fusing transformation."""

from typing import Any, Optional, Set, Union

import dace
from dace import properties, transformation
from dace.sdfg import SDFG, SDFGState, graph, nodes

from dace.transformation.dataflow import map_fusion_helper

@properties.make_properties
class ParallelMapFusion(map_fusion_helper.MapFusionHelper):
    """The `ParallelMapFusion` transformation allows to merge two parallel maps together.

    The `SerialMapFusion` transformation is only able to handle maps that are sequential,
    however, this transformation is able to fuse _any_ maps that are not sequential
    and are in the same scope.

    Args:
        only_if_common_ancestor: Only perform fusion if both Maps share at least one
            node as direct ancestor. This will increase the locality of the merge.
        only_inner_maps: Only match Maps that are internal, i.e. inside another Map.
        only_toplevel_maps: Only consider Maps that are at the top.

    Note:
        This transformation only matches the entry nodes of the Map, but will also
        modify the exit nodes of the Map.
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
        """The transformation is applicable."""
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
        if not map_fusion_helper.is_parallel(graph=graph, node1=map_entry_1, node2=map_entry_2):
            return False

        # Test if they have they share a node as direct ancestor.
        if self.only_if_common_ancestor:
            # This assumes that there is only one access node per data container in the state.
            ancestors_1: Set[nodes.Node] = {e1.src for e1 in graph.in_edges(map_entry_1)}
            if not any(e2.src in ancestors_1 for e2 in graph.in_edges(map_entry_2)):
                return False

        return True


    def apply(self, graph: Union[SDFGState, SDFG], sdfg: SDFG) -> None:
        """Performs the Map fusing.

        Essentially, the function relocate all edges from the nodes forming the second
        Map to the corresponding nodes of the first Map. Afterwards the nodes of the
        second Map are removed.
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
