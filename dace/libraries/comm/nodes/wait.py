# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
import dace.library
import dace.properties
import dace.sdfg.nodes
from dace import dtypes
from dace.libraries.comm import environments
from dace.libraries.comm.nodes.node import CommNode, expanded_input_connectors
from dace.transformation.transformation import ExpandTransformation


@dace.library.expansion
class ExpandWaitMPI(ExpandTransformation):
    environments = [environments.mpi.MPI]

    @staticmethod
    def expansion(node, parent_state, parent_sdfg, n=None, **kwargs):
        req, status = node.validate(parent_sdfg, parent_state)
        code = """
            MPI_Status _s;
            MPI_Wait(_request, &_s);
            """
        code += _status_code(node, parent_state, "_s.MPI_TAG", "_s.MPI_SOURCE")
        tasklet = dace.sdfg.nodes.Tasklet(
            node.name,
            expanded_input_connectors(node, parent_state),
            node.out_connectors,
            code,
            language=dace.dtypes.Language.CPP,
            side_effects=True,
        )
        conn = tasklet.in_connectors
        conn = {c: (dtypes.pointer(dtypes.opaque("MPI_Request")) if c == "_request" else t) for c, t in conn.items()}
        tasklet.in_connectors = conn
        return tasklet


@dace.library.expansion
class ExpandWaitNCCL(ExpandTransformation):
    """Completes every pending NCCL ``Isend``/``Irecv``: launches their group and synchronizes their
    streams. NCCL has no requests, so the ``_request`` connector is optional, and reports no status, so
    the source and tag outputs are set to -1."""

    environments = [environments.nccl.NCCL]

    @staticmethod
    def expansion(node, parent_state, parent_sdfg, n=None, **kwargs):
        code = "__state->nccl->Wait();\n" + _status_code(node, parent_state, "-1", "-1")
        tasklet = dace.sdfg.nodes.Tasklet(
            node.name,
            expanded_input_connectors(node, parent_state),
            node.out_connectors,
            code,
            language=dace.dtypes.Language.CPP,
            side_effects=True,
        )
        return tasklet


def _status_code(node: dace.sdfg.nodes.LibraryNode, state, tag: str, source: str) -> str:
    """Assignments of ``tag`` and ``source`` to the status outputs of a ``Wait`` that are connected."""
    connected = {e.src_conn for e in state.out_edges(node)}
    code = ""
    if "_stat_tag" in connected:
        code += f"_stat_tag = {tag};\n"
    if "_stat_source" in connected:
        code += f"_stat_source = {source};\n"
    return code


@dace.library.node
class Wait(CommNode):
    """
    Completes a nonblocking operation (``Isend``, ``Irecv``) given its request (``_request``), and returns the tag
    and source of a receive (``_stat_tag``, ``_stat_source``; optional).

    Other connectors only order the wait in the dataflow: connect the buffer of the operation through an input and an
    output connector (e.g., ``_in_buffer``, ``_out_buffer``) so that its readers (of a receive) or writers (of a send)
    follow the wait. The expansions ignore them.
    """

    # Global properties
    implementations = {
        "MPI": ExpandWaitMPI,
        "NCCL": ExpandWaitNCCL,
    }
    default_implementation = None

    # Object fields
    n = dace.properties.SymbolicProperty(allow_none=True, default=None, category="Semantics")

    def __init__(self, name, *args, **kwargs):
        super().__init__(name, *args, inputs={"_request"}, outputs={"_stat_tag", "_stat_source"}, **kwargs)

    def validate(self, sdfg, state):
        """
        :return: req, status
        """

        req, status = None, None
        for e in state.in_edges(self):
            if e.dst_conn == "_request":
                req = sdfg.arrays[e.data.data]
        for e in state.out_edges(self):
            if e.src_conn == "_status":
                status = sdfg.arrays[e.data.data]

        return req, status


@dace.library.expansion
class ExpandWaitallMPI(ExpandTransformation):
    environments = [environments.mpi.MPI]

    @staticmethod
    def expansion(node, parent_state, parent_sdfg, n=None, **kwargs):
        count = node.validate(parent_sdfg, parent_state)
        code = f"""
            MPI_Status _s[{count}];
            MPI_Waitall({count}, _request, _s);
            """
        tasklet = dace.sdfg.nodes.Tasklet(
            node.name,
            expanded_input_connectors(node, parent_state),
            node.out_connectors,
            code,
            language=dace.dtypes.Language.CPP,
            side_effects=True,
        )
        conn = tasklet.in_connectors
        conn = {c: (dtypes.pointer(dtypes.opaque("MPI_Request")) if c == "_request" else t) for c, t in conn.items()}
        tasklet.in_connectors = conn
        return tasklet


@dace.library.expansion
class ExpandWaitallNCCL(ExpandTransformation):
    """Completes every pending NCCL ``Isend``/``Irecv``: launches their group and synchronizes their
    streams. NCCL has no requests, so the ``_request`` connector is optional."""

    environments = [environments.nccl.NCCL]

    @staticmethod
    def expansion(node, parent_state, parent_sdfg, n=None, **kwargs):
        tasklet = dace.sdfg.nodes.Tasklet(
            node.name,
            expanded_input_connectors(node, parent_state),
            node.out_connectors,
            "__state->nccl->Wait();",
            language=dace.dtypes.Language.CPP,
            side_effects=True,
        )
        return tasklet


@dace.library.node
class Waitall(CommNode):
    """
    Completes the nonblocking operations of an array of requests (``_request``). Other connectors only order the
    wait in the dataflow, as for :class:`Wait`.
    """

    # Global properties
    implementations = {
        "MPI": ExpandWaitallMPI,
        "NCCL": ExpandWaitallNCCL,
    }
    default_implementation = None

    # Object fields
    n = dace.properties.SymbolicProperty(allow_none=True, default=None, category="Semantics")

    def __init__(self, name, *args, **kwargs):
        super().__init__(name, *args, inputs={"_request"}, outputs={}, **kwargs)

    def validate(self, sdfg, state):
        """
        :return: req, status
        """

        count = None
        for e in state.in_edges(self):
            if e.dst_conn == "_request":
                count = e.data.subset.num_elements()

        if not count:
            raise ValueError("At least 1 request object must be passed to Waitall")

        return count
