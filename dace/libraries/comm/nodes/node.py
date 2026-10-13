# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.

from typing import Any

from dace import data, dtypes
from dace.libraries.comm import utils
from dace.sdfg import SDFG, SDFGState, nodes


def input_descriptor_name(node: nodes.Node, state: Any, connector: str) -> str | None:
    edges = list(state.in_edges_by_connector(node, connector))
    if not edges:
        return None
    if len(edges) > 1:
        raise ValueError(f"Expected at most one input edge for communication connector {connector}, got {len(edges)}")

    edge = edges[0]
    if edge.data is not None and edge.data.data is not None:
        return edge.data.data
    if isinstance(edge.src, nodes.AccessNode):
        return edge.src.data
    return None


def expanded_input_connectors(node: nodes.Node, state: Any) -> dict[str, Any]:
    connectors = dict(node.in_connectors)
    for edge in state.in_edges(node):
        if edge.dst_conn is not None:
            connectors.setdefault(edge.dst_conn, None)
    return connectors


def validate_integer_descriptor(desc: Any, name: str) -> None:
    if desc is None:
        raise ValueError(f"{name} connector is missing")
    if desc.dtype.base_type not in dtypes.INTEGER_TYPES:
        raise ValueError(f"{name} must be an integer")


def mpi_communicator(node: nodes.Node, state: SDFGState) -> str:
    """The MPI communicator a node communicates over: its process grid (``_grid``), else the world."""
    return "_grid" if input_descriptor_name(node, state, "_grid") else "MPI_COMM_WORLD"


def nccl_communicator(node: nodes.Node, sdfg: SDFG, state: SDFGState) -> str:
    """The C++ expression of the NCCL communicator a node communicates over.

    The communicator of a process grid is created in the initialization code of ``sdfg``, which runs after
    that of the SDFG defining the grid. Creating it on first use instead would deadlock point-to-point
    operations, since creation is collective over the grid.
    """
    grid = input_descriptor_name(node, state, "_grid")
    if grid is None:
        return "__state->nccl->Get(MPI_COMM_WORLD)"
    create = f"__state->nccl->Create(__state->{grid});\n"
    if "frame" not in sdfg.init_code or create not in sdfg.init_code["frame"].as_string:
        sdfg.append_init_code(create)
    return "__state->nccl->Get(_grid)"


def validate_nccl_buffers(node: nodes.Node, *descs: data.Data) -> None:
    """NCCL communicates device memory only, on the stream of the expanded tasklet."""
    for desc in descs:
        if desc.storage != dtypes.StorageType.GPU_Global:
            raise ValueError(
                f'The NCCL expansion of {type(node).__name__} "{node.label}" requires buffers in GPU global '
                f"memory, but got one in {desc.storage}. Use the MPI expansion for host memory."
            )
        if desc.dtype.veclen > 1:
            raise NotImplementedError("NCCL expansions do not support vector types")


def scaled(count: str, factor: int) -> str:
    """``count`` elements of a DaCe type in units of an NCCL type with ``factor`` per element."""
    return count if factor == 1 else f"({count}) * {factor}"


def nccl_point_to_point(
    call: str, buffer: str, peer: str, count: str, dtype: dtypes.typeclass, ddt: dict | None
) -> str:
    """An ``ncclSend`` or ``ncclRecv`` of ``count`` elements of ``buffer``, or of the strided rows that
    ``ddt`` (see :func:`dace.libraries.comm.utils.create_vector_ddt`) describes. NCCL has no derived
    datatypes, so a strided buffer moves row by row; NCCL matches the rows in order within a group."""
    nccl_type, factor = utils.nccl_transfer_type(dtype)
    if ddt is None:
        return (
            f"DACE_NCCL_CHECK({call}({buffer}, {scaled(count, factor)}, {nccl_type}, {peer}, __comm, "
            "__dace_current_stream));\n"
        )
    return f"""
        for (long long __row = 0; __row < {ddt["count"]}; ++__row) {{
            DACE_NCCL_CHECK({call}({buffer} + __row * ({ddt["stride"]}), {scaled(ddt["blocklen"], factor)}, {nccl_type},
                                   {peer}, __comm, __dace_current_stream));
        }}
        """


def nccl_tasklet(
    node: nodes.LibraryNode, state: SDFGState, comm: str, body: str, blocking: bool = True
) -> nodes.Tasklet:
    """The tasklet of an NCCL expansion. ``body`` enqueues the operation on ``__comm``, inside the open NCCL
    group of the program; a blocking operation then launches the group, a nonblocking one leaves it to
    the ``Wait`` that completes it."""
    code = f"""
        ncclComm_t __comm = {comm};
        __state->nccl->Begin();
        {body}
        """
    if blocking:
        code += "__state->nccl->Launch();\n"
    return nodes.Tasklet(
        node.name,
        expanded_input_connectors(node, state),
        node.out_connectors,
        code,
        language=dtypes.Language.CPP,
        side_effects=True,
    )


class CommNode(nodes.LibraryNode):
    """
    Abstract class representing a communication library node.

    Every node has an ``MPI`` implementation. Nodes that also have an ``NCCL`` implementation (which
    becomes RCCL on the HIP backend) leave ``default_implementation`` unset, so the
    ``library.comm.default_implementation`` configuration entry selects their backend.
    """

    # Communication reads host control values (ranks, tags, requests) next to buffers in any memory
    bridges_storage = True

    def has_side_effects(self, sdfg) -> bool:
        return True
