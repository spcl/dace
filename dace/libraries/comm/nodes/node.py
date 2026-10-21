# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.

from collections.abc import Sequence
from typing import Any

from dace import data, dtypes, properties, subsets
from dace.codegen import common
from dace.libraries.comm import environments, utils
from dace.memlet import Memlet
from dace.sdfg import SDFG, SDFGState, nodes
from dace.sdfg import utils as sdutil
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


def integer_input(node: nodes.Node, sdfg: SDFG, state: SDFGState, connector: str) -> data.Data:
    """The descriptor on the input ``connector`` of ``node``, which must be an integer."""
    desc = sdutil.in_desc_with_name(node, state, sdfg, connector)
    validate_integer_descriptor(desc, f'{type(node).__name__} "{node.label}" {connector}')
    return desc


def integer_output(node: nodes.Node, sdfg: SDFG, state: SDFGState, connector: str) -> data.Data:
    """The descriptor on the output ``connector`` of ``node``, which must be an integer."""
    desc = sdutil.out_desc_with_name(node, state, sdfg, connector)
    validate_integer_descriptor(desc, f'{type(node).__name__} "{node.label}" {connector}')
    return desc


def communicator_output(node: nodes.Node, sdfg: SDFG, state: SDFGState, connector: str) -> data.Data:
    """The descriptor on the output ``connector`` of ``node``, which receives a communicator handle: a 64-bit integer,
    as the ``_comm`` connector takes (see :mod:`dace.libraries.comm.communicators`)."""
    desc = sdutil.out_desc_with_name(node, state, sdfg, connector)
    if desc.dtype.base_type not in (dtypes.int64, dtypes.uint64):
        raise ValueError(f'{type(node).__name__} "{node.label}" {connector} must be a 64-bit integer handle')
    return desc


def count_expression(subset: subsets.Subset) -> str:
    """The number of elements of ``subset`` as a C++ expression, parenthesized unless it is one term."""
    from dace.codegen.targets.cpp import sym2cpp  # avoid import loop

    return sym2cpp(subset.num_elements_exact())


def int_array(name: str, values: Sequence, ctype: str = "int", static: bool = False) -> str:
    """A C++ array ``name`` of ``ctype`` assigned the symbolic ``values`` on every run, ``static`` if it must outlive
    the code (e.g., of a nonblocking operation)."""
    from dace.codegen.targets.cpp import sym2cpp  # avoid import loop

    storage = "static " if static else ""
    assignments = "".join(f"{name}[{i}] = ({ctype})({sym2cpp(v)});\n" for i, v in enumerate(values))
    return f"{storage}{ctype} {name}[{len(values)}];\n{assignments}"


#: The connectors of the buffers a communication node sends or receives
BUFFER_CONNECTORS = ("_buffer", "_inbuffer", "_outbuffer")


def comm_tasklet(node: nodes.LibraryNode, state: SDFGState, code: str) -> nodes.Tasklet:
    """The tasklet of a communication expansion that runs the C++ ``code``. Its buffer connectors are pointers, so MPI
    and NCCL take the buffer itself even for a scalar or a single element."""
    inputs = expanded_input_connectors(node, state)
    outputs = dict(node.out_connectors)
    for connector in BUFFER_CONNECTORS:
        if connector in inputs:
            inputs[connector] = dtypes.pointer(sdutil.in_desc_with_name(node, state, state.sdfg, connector).dtype)
        if connector in outputs:
            outputs[connector] = dtypes.pointer(sdutil.out_desc_with_name(node, state, state.sdfg, connector).dtype)
    return nodes.Tasklet(node.name, inputs, outputs, code, language=dtypes.Language.CPP, side_effects=True)


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
        _validate_nccl_handle(node, sdfg, state)
        return "((ncclComm_t)(_comm))"
    grid = input_descriptor_name(node, state, "_grid")
    if grid is None:
        _append_init_code_once(sdfg.root_sdfg, "__state->nccl->Create(MPI_COMM_WORLD);\n")
        return "__state->nccl->Get(MPI_COMM_WORLD)"
    _append_init_code_once(sdfg, f"__state->nccl->Create(__state->{grid});\n")
    return "__state->nccl->Get(_grid)"


def _validate_nccl_handle(node: nodes.Node, sdfg: SDFG, state: SDFGState) -> None:
    """Refuses a ``_comm`` handle that a node of ``sdfg`` writes as an ``MPI_Comm`` (see
    :attr:`CommNode.outputs_mpi_communicator`): an NCCL expansion would reinterpret it as an ``ncclComm_t``."""
    name = input_descriptor_name(node, state, "_comm")
    for block in sdfg.states():
        for access in block.data_nodes():
            if access.data != name:
                continue
            for edge in block.in_edges(access):
                if isinstance(edge.src, CommNode) and edge.src.outputs_mpi_communicator:
                    raise ValueError(
                        f'The NCCL expansion of {type(node).__name__} "{node.label}" takes the communicator {name}, '
                        f'which {type(edge.src).__name__} "{edge.src.label}" writes as an MPI communicator. Use the '
                        "MPI expansion, or a handle from dace.libraries.comm.communicators.nccl_init_rank"
                    )


def _append_init_code_once(sdfg: SDFG, code: str) -> None:
    if "frame" not in sdfg.init_code or code not in sdfg.init_code["frame"].as_string:
        sdfg.append_init_code(code)


def mpi_average(op: str, buffer: str, count: str, comm: str, root: str | None = None) -> str:
    """The code that turns the sum an MPI reduction wrote to ``buffer`` (a pointer expression) into the average, if
    ``op`` is the average (see :func:`dace.libraries.comm.utils.is_average`): divides the ``count`` elements by the
    number of processes of ``comm``, on ``root`` only if given."""
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
                for (long long __i = 0; __i < ({count}); ++__i) ({buffer})[__i] /= __avg_size;
            }}
        }}
        """


def touches_gpu_memory(node: nodes.Node, sdfg: SDFG, state: SDFGState) -> bool:
    """Whether ``node`` is connected to a container in GPU memory, which gives its tasklet a GPU stream."""
    return any(
        e.data.data is not None and sdfg.arrays[e.data.data].storage == dtypes.StorageType.GPU_Global
        for e in state.all_edges(node)
    )


def gpu_buffers_ready(node: nodes.Node, sdfg: SDFG, state: SDFGState) -> str:
    """The code that makes the host wait for the GPU stream of the tasklet of ``node`` before an MPI call on its
    buffers in GPU memory (GPU-aware MPI): the host call is not ordered with the kernels that write or read them."""
    if touches_gpu_memory(node, sdfg, state):
        return f"DACE_GPU_CHECK({common.get_gpu_backend()}StreamSynchronize(__dace_current_stream));\n"
    return ""


def mpi_vector_datatype(ddt: dict | None) -> tuple[str, str, str, str]:
    """The MPI datatype of a point-to-point buffer that ``ddt`` (see :func:`dace.libraries.comm.utils.create_vector_ddt`)
    describes: (code creating it, datatype, count, code freeing it). The datatype is created on every call, since its
    stride and block length may be symbols; freeing it after starting a nonblocking operation is allowed, as the
    operation keeps it until completion. Without ``ddt``, all are empty."""
    if ddt is None:
        return "", "", "", ""
    create = f"""
        MPI_Datatype __vector;
        MPI_Type_vector({ddt["count"]}, {ddt["blocklen"]}, {ddt["stride"]}, {ddt["oldtype"]}, &__vector);
        MPI_Type_commit(&__vector);
        """
    return create, "__vector", "1", "MPI_Type_free(&__vector);\n"


def column_major(sdfg: SDFG, state: SDFGState, access: nodes.Node, name: str) -> bool:
    """Whether the elements of container ``name``, accessed through ``access``, are in column-major (Fortran) order:
    whether the array it views (or itself, if not a view) has packed Fortran strides and not packed C strides."""
    if isinstance(access, nodes.AccessNode) and isinstance(sdfg.arrays[name], data.View):
        viewed = sdutil.get_last_view_node(state, access)
        if viewed is not None:
            name = viewed.data
    desc = sdfg.arrays[name]
    return isinstance(desc, data.Array) and desc.is_packed_fortran_strides() and not desc.is_packed_c_strides()


def in_place(sdfg: SDFG, state: SDFGState, edge, node: nodes.Node) -> bool:
    """Whether the buffer on ``edge`` of ``node`` is communicated in place: one contiguous run of memory (see
    :meth:`dace.subsets.Subset.is_contiguous_subset`) laid out in the element order of the buffer (see
    :func:`column_major`). A transposed view of a row-major array is contiguous, but not in row-major order."""
    desc = sdfg.arrays[edge.data.data]
    if isinstance(desc, data.Scalar):
        return True
    subset = edge.data.subset
    if not subset.is_contiguous_subset(desc):
        return False
    if sum(1 for size in subset.size() if (size == 1) != True) <= 1:  # noqa: E712 (symbolic comparison)
        return True  # One row: the same in any order
    access = edge.src if edge.dst is node else edge.dst
    if column_major(sdfg, state, access, edge.data.data):
        return desc.is_packed_fortran_strides()
    return desc.is_packed_c_strides()


def pack_noncontiguous_buffers(
    node: nodes.LibraryNode,
    sdfg: SDFG,
    state: SDFGState,
    inputs: tuple[str, ...] = ("_inbuffer",),
    outputs: tuple[str, ...] = ("_outbuffer",),
) -> None:
    """
    Routes the buffers of ``node`` that are not communicated in place (see :func:`in_place`) through contiguous
    transients laid out in the element order of the buffer: copies into them before the node and out of them after
    it. The expansions communicate ``count`` elements from a pointer, so they need contiguous buffers. A buffer that
    is both an input and an output (in place) gets one transient.

    :param inputs: The input connectors of buffers.
    :param outputs: The output connectors of buffers.
    """
    packed: dict[tuple[str, str], str] = {}

    def transient(memlet: Memlet, access: nodes.Node) -> str:
        key = (memlet.data, str(memlet.subset))
        if key not in packed:
            desc = sdfg.arrays[memlet.data]
            packed[key], packed_desc = sdfg.add_array(
                f"__{node.label}_{memlet.data}_packed",
                memlet.subset.size(),
                desc.dtype,
                storage=desc.storage,
                transient=True,
                find_new_name=True,
            )
            if column_major(sdfg, state, access, memlet.data):
                packed_desc.strides = packed_desc._get_packed_fortran_strides()
        return packed[key]

    def copy_in(src: nodes.Node, src_conn: str | None, memlet: Memlet, name: str) -> nodes.AccessNode:
        access = state.add_access(name)
        full = subsets.Range.from_array(sdfg.arrays[name])
        state.add_edge(src, src_conn, access, None, Memlet(data=memlet.data, subset=memlet.subset, other_subset=full))
        return access

    for edge in list(state.in_edges(node)):
        if edge.dst_conn not in inputs or in_place(sdfg, state, edge, node):
            continue
        name = transient(edge.data, edge.src)
        access = copy_in(edge.src, edge.src_conn, edge.data, name)
        state.remove_edge(edge)
        state.add_edge(access, None, node, edge.dst_conn, Memlet.from_array(name, sdfg.arrays[name]))
    for edge in list(state.out_edges(node)):
        if edge.src_conn not in outputs or in_place(sdfg, state, edge, node):
            continue
        name = transient(edge.data, edge.dst)
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


def same_buffer(node: nodes.Node, state: SDFGState) -> bool:
    """Whether the input and output buffers of ``node`` are the same memory, communicated in place (``MPI_IN_PLACE``).
    Disjoint subsets of one container are two buffers; different ones that overlap, or may overlap, are refused, as
    MPI forbids aliased buffers."""
    inp = sdutil.in_edge_with_name(node, state, "_inbuffer").data
    out = sdutil.out_edge_with_name(node, state, "_outbuffer").data
    if inp.data != out.data:
        return False
    if inp.subset == out.subset:
        return True
    if subsets.intersects(inp.subset, out.subset) is False:
        return False
    raise ValueError(
        f'{type(node).__name__} "{node.label}" reads {inp.data}[{inp.subset}] and writes the (possibly) overlapping '
        f"{out.data}[{out.subset}]: use the same subset for an in-place operation, or disjoint ones"
    )


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
    tasklet = comm_tasklet(node, state, code)
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

    #: Whether the node outputs the handle of an ``MPI_Comm`` (e.g., ``CommSplit``), which no NCCL expansion may take
    outputs_mpi_communicator = False

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
            if connector not in ("_inbuffer", "_outbuffer"):
                continue
            if not in_place(sdfg, state, edge, self):
                raise ValueError(
                    f'The nonblocking {type(self).__name__} "{self.label}" requires contiguous buffers, but '
                    f"{edge.data.data} is not"
                )
