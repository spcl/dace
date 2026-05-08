# Copyright 2019-2022 ETH Zurich and the DaCe authors. All rights reserved.

from typing import Any, Dict, Optional

from dace.sdfg import nodes


def input_descriptor_name(node: nodes.Node, state: Any, connector: str) -> Optional[str]:
    edges = list(state.in_edges_by_connector(node, connector))
    if not edges:
        return None
    if len(edges) > 1:
        raise ValueError(f"Expected at most one input edge for MPI connector {connector}, got {len(edges)}")

    edge = edges[0]
    if edge.data is not None and edge.data.data is not None:
        return edge.data.data
    if isinstance(edge.src, nodes.AccessNode):
        return edge.src.data
    return None


def expanded_input_connectors(node: nodes.Node, state: Any) -> Dict[str, Any]:
    connectors = dict(node.in_connectors)
    for edge in state.in_edges(node):
        if edge.dst_conn is not None:
            connectors.setdefault(edge.dst_conn, None)
    return connectors


class MPINode(nodes.LibraryNode):
    """
    Abstract class representing an MPI library node.
    """

    @property
    def has_side_effects(self) -> bool:
        return True
