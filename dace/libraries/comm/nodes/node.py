# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.

from typing import Any

import sympy

from dace import data, dtypes, properties, subsets
from dace.libraries.comm import environments, utils
from dace.memlet import Memlet
from dace.sdfg import SDFG, SDFGState, nodes
from dace.transformation.transformation import ExpandTransformation


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


def external_communicator(node: nodes.Node, state: SDFGState) -> bool:
    """Whether ``node`` communicates over a communicator given as an argument (its ``_comm`` connector)."""
    return input_descriptor_name(node, state, "_comm") is not None


def mpi_communicator(node: nodes.Node, state: SDFGState) -> str:
    """The C++ expression of the MPI communicator a node communicates over: the handle given to its ``_comm``
    connector (an integer holding an ``MPI_Comm``, e.g., ``mpi4py.MPI._handleof(comm)``), else its process grid
    (``_grid``), else the world."""
    if external_communicator(node, state):
        return "((MPI_Comm)(_comm))"
    return "_grid" if input_descriptor_name(node, state, "_grid") else "MPI_COMM_WORLD"


def nccl_communicator(node: nodes.Node, sdfg: SDFG, state: SDFGState) -> str:
    """The C++ expression of the NCCL communicator a node communicates over: the handle given to its ``_comm``
    connector (an integer holding an ``ncclComm_t``, see :func:`dace.libraries.comm.communicators.nccl_init_rank`),
    else the communicator of its process grid (``_grid``), else that of ``MPI_COMM_WORLD``.

    The communicators of the world and of process grids are bootstrapped over MPI in the initialization code of the
    program; that of a grid in the initialization code of ``sdfg``, which runs after that of the SDFG defining the
    grid. Creating them on first use instead would deadlock point-to-point operations, since creation is collective.
    An SDFG whose NCCL nodes all have a ``_comm`` connector needs no MPI.
    """
    if external_communicator(node, state):
        return "((ncclComm_t)(_comm))"
    grid = input_descriptor_name(node, state, "_grid")
    if grid is None:
        _append_init_code_once(sdfg.root_sdfg, "__state->nccl->Create(MPI_COMM_WORLD);\n")
        return "__state->nccl->Get(MPI_COMM_WORLD)"
    _append_init_code_once(sdfg, f"__state->nccl->Create(__state->{grid});\n")
    return "__state->nccl->Get(_grid)"


def _append_init_code_once(sdfg: SDFG, code: str) -> None:
    if "frame" not in sdfg.init_code or code not in sdfg.init_code["frame"].as_string:
        sdfg.append_init_code(code)


def mpi_average(op: str, buffer: str, count: str, comm: str, root: str | None = None) -> str:
    """The code that turns the sum an MPI reduction wrote to ``buffer`` into the average, if ``op`` is the average
    (see :func:`dace.libraries.comm.utils.is_average`): divides the ``count`` elements by the number of processes of
    ``comm``, on ``root`` only if given."""
    if not utils.is_average(op):
        return ""
    condition = "true"
    if root is not None:
        condition = f"__avg_rank == {root}"
    return f"""
        {{
            int __avg_size, __avg_rank;
            MPI_Comm_size({comm}, &__avg_size);
            MPI_Comm_rank({comm}, &__avg_rank);
            if ({condition}) {{
                for (long long __i = 0; __i < ({count}); ++__i) {buffer}[__i] /= __avg_size;
            }}
        }}
        """


def row_major_contiguous(subset: subsets.Range, desc: data.Data) -> bool:
    """Whether ``subset`` of ``desc`` is one contiguous run of memory in row-major (C) element order, the order in
    which a buffer is communicated. A packed permutation (e.g., a transposed view) is contiguous memory, but not in
    this order."""
    if any((step != 1) == True for _, _, step in subset):  # noqa: E712 (symbolic comparison)
        return False
    expected = 1
    for size, stride in reversed(list(zip(subset.size(), desc.strides))):
        if (size == 1) == True:  # noqa: E712 (symbolic comparison)
            continue
        if sympy.simplify(sympy.sympify(stride) - sympy.sympify(expected)) != 0:
            return False
        expected = expected * size
    return True


def pack_noncontiguous_buffers(
    node: nodes.LibraryNode,
    sdfg: SDFG,
    state: SDFGState,
    inputs: tuple[str, ...] = ("_inbuffer",),
    outputs: tuple[str, ...] = ("_outbuffer",),
) -> None:
    """
    Routes the buffers of ``node`` that are not one contiguous run of memory (e.g., a transposed view, which
    simplification may hand to the node in place of a contiguous copy) through contiguous transients: copies
    into them before the node and out of them after it. The expansions communicate ``count`` elements from a
    pointer, so they need contiguous buffers. A buffer that is both an input and an output (in place) gets one
    transient.

    :param inputs: The input connectors of buffers.
    :param outputs: The output connectors of buffers.
    """
    packed: dict[tuple[str, str], str] = {}

    def transient(memlet: Memlet) -> str:
        key = (memlet.data, str(memlet.subset))
        if key not in packed:
            desc = sdfg.arrays[memlet.data]
            packed[key], _ = sdfg.add_array(
                f"__{node.label}_{memlet.data}_packed",
                memlet.subset.size(),
                desc.dtype,
                storage=desc.storage,
                transient=True,
                find_new_name=True,
            )
        return packed[key]

    for edge in list(state.in_edges(node)):
        if edge.dst_conn not in inputs or _contiguous_memlet(sdfg, edge.data):
            continue
        name = transient(edge.data)
        full = subsets.Range.from_array(sdfg.arrays[name])
        access = state.add_access(name)
        state.add_edge(
            edge.src,
            edge.src_conn,
            access,
            None,
            Memlet(data=edge.data.data, subset=edge.data.subset, other_subset=full),
        )
        state.remove_edge(edge)
        state.add_edge(access, None, node, edge.dst_conn, Memlet.from_array(name, sdfg.arrays[name]))
    for edge in list(state.out_edges(node)):
        if edge.src_conn not in outputs or _contiguous_memlet(sdfg, edge.data):
            continue
        name = transient(edge.data)
        full = subsets.Range.from_array(sdfg.arrays[name])
        access = state.add_access(name)
        state.add_edge(node, edge.src_conn, access, None, Memlet.from_array(name, sdfg.arrays[name]))
        state.add_edge(
            access,
            None,
            edge.dst,
            edge.dst_conn,
            Memlet(data=name, subset=full, other_subset=edge.data.subset),
        )
        state.remove_edge(edge)


def _contiguous_memlet(sdfg: SDFG, memlet: Memlet) -> bool:
    desc = sdfg.arrays[memlet.data]
    return isinstance(desc, data.Scalar) or row_major_contiguous(memlet.subset, desc)


def mpi_collective(node: nodes.LibraryNode, name: str, args: str) -> str:
    """
    The call of the MPI collective ``name`` (e.g., ``Allreduce``) with the arguments ``args``. For a nonblocking
    node, the call of its nonblocking variant (``MPI_Iallreduce``), which starts the operation and sets the
    ``_request`` output (see :func:`mpi_request_connector`).
    """
    if isinstance(node, NonblockingCommNode) and node.nonblocking:
        return f"MPI_I{name[0].lower()}{name[1:]}({args}, _request);\n"
    return f"MPI_{name}({args});\n"


def mpi_request_connector(tasklet: nodes.Tasklet) -> nodes.Tasklet:
    """Types the ``_request`` output of an expanded tasklet, if it has one, as a pointer to an ``MPI_Request``."""
    tasklet.out_connectors = {
        c: (dtypes.pointer(dtypes.opaque("MPI_Request")) if c == "_request" else t)
        for c, t in tasklet.out_connectors.items()
    }
    return tasklet


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


def nccl_request(node: nodes.LibraryNode, sdfg: SDFG, state: SDFGState, tasklet: nodes.Tasklet) -> str:
    """The code that sets the ``_request`` output of a nonblocking NCCL operation, which has no request, and the
    type of the connector. An ``MPI_Request`` container is set to ``MPI_REQUEST_NULL`` (which needs MPI); any other
    (e.g., an integer token that orders the operation before its ``Wait`` without MPI) to zero. Without a
    ``_request`` output, there is nothing to set.

    A request in host memory makes the host wait for the GPU stream of the operation after it, as for any host
    output of GPU work; ordering the operation through its GPU buffers instead (ordering connectors) does not."""
    request = next((e for e in state.out_edges(node) if e.src_conn == "_request"), None)
    if request is None:
        return ""
    dtype = sdfg.arrays[request.data.data].dtype
    if isinstance(dtype, dtypes.opaque) and dtype.ctype == "MPI_Request":
        tasklet.out_connectors = {
            c: (dtypes.pointer(dtype) if c == "_request" else t) for c, t in tasklet.out_connectors.items()
        }
        return "*_request = MPI_REQUEST_NULL;\n"
    tasklet.out_connectors = {c: (dtype if c == "_request" else t) for c, t in tasklet.out_connectors.items()}
    return "_request = 0;\n"


def nccl_tasklet(
    node: nodes.LibraryNode,
    state: SDFGState,
    comm: str,
    body: str,
    blocking: bool = True,
    sdfg: SDFG | None = None,
) -> nodes.Tasklet:
    """The tasklet of an NCCL expansion. ``body`` enqueues the operation on ``__comm``, inside the open NCCL
    group of the program; a blocking operation then launches the group, a nonblocking one leaves it to
    the ``Wait`` that completes it.

    A collective is launched right away, nonblocking or not: NCCL runs it asynchronously on the stream of its node,
    and code generation orders its readers after it on the device. The ``_request`` output of a nonblocking one, if
    connected, is a token (see :func:`nccl_request`), which needs ``sdfg``."""
    code = f"""
        ncclComm_t __comm = {comm};
        __state->nccl->Begin();
        {body}
        """
    if blocking:
        code += "__state->nccl->Launch();\n"
    tasklet = nodes.Tasklet(
        node.name,
        expanded_input_connectors(node, state),
        node.out_connectors,
        code,
        language=dtypes.Language.CPP,
        side_effects=True,
    )
    if isinstance(node, NonblockingCommNode) and node.nonblocking:
        if sdfg is None:
            raise ValueError("The tasklet of a nonblocking NCCL collective needs its SDFG")
        tasklet.code = properties.CodeBlock(
            tasklet.code.as_string + nccl_request(node, sdfg, state, tasklet), dtypes.Language.CPP
        )
    return tasklet


class NcclExpansion(ExpandTransformation):
    """Base class of the NCCL expansions. An expansion that communicates over a communicator it bootstraps over MPI
    (that of the world or of a process grid, see :func:`nccl_communicator`) also uses the MPI environment; one whose
    node has a ``_comm`` connector does not."""

    environments = [environments.nccl.NCCL]

    @staticmethod
    def postprocessing(sdfg, state, expansion):
        if "_comm" not in expansion.in_connectors:
            expansion.environments = set(expansion.environments) | {environments.mpi.MPI.full_class_path()}


class CommNode(nodes.LibraryNode):
    """
    Abstract class representing a communication library node.

    Every node has an ``MPI`` implementation. Nodes that also have an ``NCCL`` implementation (which
    becomes RCCL on the HIP backend) leave ``default_implementation`` unset, so the
    ``library.comm.default_implementation`` configuration entry selects their backend.

    A node communicates over the communicator given to its optional ``_comm`` connector, an integer scalar
    holding the handle of an ``MPI_Comm`` (MPI) or ``ncclComm_t`` (NCCL) that matches its implementation. Without
    one, it communicates over its process grid (``_grid``) or the world.
    """

    # Communication reads host control values (ranks, tags, requests) next to buffers in any memory
    bridges_storage = True

    def has_side_effects(self, sdfg) -> bool:
        return True


@properties.make_properties
class NonblockingCommNode(CommNode):
    """
    A collective that can also start its operation and return (``nonblocking``), like the ``MPI_I*`` variants of MPI
    collectives. With MPI, a nonblocking node writes an ``MPI_Request`` to its ``_request`` output, which a ``Wait``
    completes; its buffers must stay untouched until then, and must be contiguous. The average (``avg``) of a
    reduction cannot be nonblocking with MPI: reduce the sum, and divide after the ``Wait``.

    With NCCL (or RCCL), every collective is asynchronous already: it runs on the GPU stream of its node, and code
    generation orders the readers of its output after it on the device (with events across streams), while
    independent work runs on other streams. ``nonblocking`` makes no difference and needs no ``Wait``. A connected
    ``_request`` output is set to a token, but writing it to host memory makes the host wait for the collective.
    """

    nonblocking = properties.Property(
        dtype=bool,
        default=False,
        category="Semantics",
        desc="Start the operation and return; a Wait on the _request output completes it",
    )

    def validate_nonblocking(self, sdfg: SDFG, state: SDFGState) -> None:
        """
        Checks what an MPI expansion of a nonblocking node needs: a connected ``_request`` output, and contiguous
        buffers, since the operation completes at its ``Wait``, after a packed output would have been copied back
        (see :func:`pack_noncontiguous_buffers`) and a packed input freed. (NCCL needs neither: the copies of packed
        buffers are ordered on the stream of the node.)
        """
        if not self.nonblocking:
            return
        if not any(e.src_conn == "_request" for e in state.out_edges(self)):
            raise ValueError(f'The nonblocking {type(self).__name__} "{self.label}" has no _request output')
        for edge in state.in_edges(self) + state.out_edges(self):
            connector = edge.dst_conn if edge.dst is self else edge.src_conn
            if connector in ("_inbuffer", "_outbuffer") and not _contiguous_memlet(sdfg, edge.data):
                raise ValueError(
                    f'The nonblocking {type(self).__name__} "{self.label}" requires contiguous buffers, but '
                    f"{edge.data.data} is not"
                )
