# Copyright 2019-2022 ETH Zurich and the DaCe authors. All rights reserved.

from typing import Any, Dict, Optional

from dace import dtypes
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


def resolve_comm(node: nodes.Node, state: Any) -> str:
    """Resolve the MPI communicator an MPI library node should use from its
    optional communicator-carrying input connectors.  The connector NAME is the
    semantic discriminator:

      * ``_comm`` -- a raw ``opaque(MPI_Comm)`` value (e.g. produced by a
        ``Comm_f2c`` node from a Fortran integer handle); used DIRECTLY, no
        cartesian topology.
      * ``_grid`` -- a process grid (``ProcessGrid`` / ``FortranProcessGrid``)
        whose cartesian sub-communicator is used.

    ``_comm`` takes priority when both are present.  Falls back to
    ``MPI_COMM_WORLD`` when neither is connected.  The returned string is the
    connector name (or ``"MPI_COMM_WORLD"``); the C tasklet emits it verbatim
    into the ``MPI_*`` call and DaCe codegen substitutes the connector's value.
    """
    if input_descriptor_name(node, state, '_comm'):
        return "_comm"
    if input_descriptor_name(node, state, '_grid'):
        return "_grid"
    return "MPI_COMM_WORLD"


def validate_integer_descriptor(desc: Any, name: str) -> None:
    if desc is None:
        raise ValueError(f"{name} connector is missing")
    if desc.dtype.base_type not in dtypes.INTEGER_TYPES:
        raise ValueError(f"{name} must be an integer")


class MPINode(nodes.LibraryNode):
    """
    Abstract class representing an MPI library node.
    """

    @property
    def has_side_effects(self) -> bool:
        return True
