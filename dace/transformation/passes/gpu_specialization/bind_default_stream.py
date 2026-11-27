# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Binds every GPU stream of a single-stream SDFG to the default stream, ``nullptr``.

:class:`~dace.transformation.passes.gpu_specialization.gpu_stream_wiring.GPUStreamWiring` reaches each stream
through the ``gpu_streams`` array, which the generated code points at a runtime context. A unit without that
context (a CPF unit without boilerplate) has one stream, the default one, so :class:`BindDefaultStream` removes the
array, its access nodes and every edge and connector it reached. What consumed a stream is left to name the default
one: a kernel launch without a stream edge already uses ``nullptr``, a sync tasklet waits on it, and a tasklet that
names :data:`~dace.libraries.standard.helper.CURRENT_STREAM_NAME` (a copy, a library call) reads the unit's one
constant of that name.
"""

from typing import Any

from dace import SDFG, SDFGState, dtypes
from dace.libraries.standard.helper import CURRENT_STREAM_NAME
from dace.properties import CodeBlock
from dace.sdfg import nodes
from dace.transformation import pass_pipeline as ppl
from dace.transformation import transformation
from dace.transformation.passes.gpu_specialization.gpu_stream_scheduling import stream_sync_code
from dace.transformation.passes.gpu_specialization.helpers.gpu_helpers import get_gpu_stream_array_name

#: The default stream, which needs no context to name.
DEFAULT_STREAM = "nullptr"


def bind_tasklet_streams(tasklet: nodes.Tasklet, streams: list[str]) -> None:
    """Make ``tasklet``'s code name the default stream where it named the connectors ``streams``.

    A sync tasklet, whose code is :func:`stream_sync_code` of its connectors, waits on the default stream once. The
    connector :data:`CURRENT_STREAM_NAME` needs nothing: the unit declares the default stream under that name. Any
    other connector becomes a local holding the default stream.
    """
    code = tasklet.code.as_string
    if code == stream_sync_code(streams):
        tasklet.code = CodeBlock(stream_sync_code([DEFAULT_STREAM]), dtypes.Language.CPP)
        return
    others = [stream for stream in streams if stream != CURRENT_STREAM_NAME]
    if not others:
        return
    if tasklet.language is not dtypes.Language.CPP:
        raise NotImplementedError(f"tasklet {tasklet.label} reads a GPU stream in {tasklet.language.name}")
    declarations = "".join(f"gpuStream_t {stream} = {DEFAULT_STREAM};\n" for stream in others)
    tasklet.code = CodeBlock(declarations + code, dtypes.Language.CPP)


@transformation.explicit_cf_compatible
class BindDefaultStream(ppl.Pass):
    """Remove the ``gpu_streams`` array from every SDFG and bind each stream consumer to the default stream."""

    CATEGORY: str = "GPU"

    def modifies(self) -> ppl.Modifies:
        return ppl.Modifies.AccessNodes | ppl.Modifies.Memlets | ppl.Modifies.Tasklets | ppl.Modifies.Descriptors

    def should_reapply(self, modified: ppl.Modifies) -> bool:
        return False

    def apply_pass(self, sdfg: SDFG, pipeline_results: dict[str, Any]) -> int | None:
        name = get_gpu_stream_array_name()
        removed = 0
        names_current_stream = False
        for nested in sdfg.all_sdfgs_recursive():
            if name not in nested.arrays:
                continue
            for state in nested.states():
                edges = [edge for edge in state.edges() if edge.data.data == name]
                streams: dict[nodes.Tasklet, list[str]] = {}
                for edge in edges:
                    if isinstance(edge.dst, nodes.Tasklet) and edge.dst_conn is not None:
                        if edge.dst.in_connectors[edge.dst_conn] != dtypes.gpuStream_t:
                            raise NotImplementedError(f"tasklet {edge.dst.label} indexes the GPU stream array")
                        streams.setdefault(edge.dst, []).append(edge.dst_conn)
                for tasklet, connectors in streams.items():
                    names_current_stream |= CURRENT_STREAM_NAME in connectors
                    # In connector order, the order the tasklet's code was built in.
                    bind_tasklet_streams(tasklet, [c for c in tasklet.in_connectors if c in connectors])
                for edge in edges:
                    state.remove_edge(edge)
                    if edge.src_conn is not None:
                        edge.src.remove_out_connector(edge.src_conn)
                    if edge.dst_conn is not None:
                        edge.dst.remove_in_connector(edge.dst_conn)
                for node in [n for n in state.data_nodes() if n.data == name]:
                    state.remove_node(node)
                removed += len(edges)
            nested.remove_data(name, validate=False)
        if names_current_stream:
            sdfg.append_global_code(f"constexpr gpuStream_t {CURRENT_STREAM_NAME} = {DEFAULT_STREAM};\n")
        return removed or None


@transformation.explicit_cf_compatible
class DropExitStreamSync(ppl.Pass):
    """Remove the default-stream waits that end the program, for an entry whose last statement synchronizes the
    device (a CPF unit without boilerplate): that synchronization already waits for the default stream.

    A wait ends the program when it is a sink tasklet of a sink state of the top-level SDFG whose code is
    :func:`stream_sync_code` of :data:`DEFAULT_STREAM` (:class:`BindDefaultStream` wrote it). A state left empty is
    removed when its in-edges carry neither a condition nor an assignment.
    """

    CATEGORY: str = "GPU"

    def modifies(self) -> ppl.Modifies:
        return ppl.Modifies.Tasklets | ppl.Modifies.States

    def should_reapply(self, modified: ppl.Modifies) -> bool:
        return False

    def apply_pass(self, sdfg: SDFG, pipeline_results: dict[str, Any]) -> int | None:
        wait = stream_sync_code([DEFAULT_STREAM])
        removed = 0
        for state in [block for block in sdfg.nodes() if isinstance(block, SDFGState) and sdfg.out_degree(block) == 0]:
            waits = [
                node
                for node in state.sink_nodes()
                if isinstance(node, nodes.Tasklet) and node.code.as_string == wait and not node.in_connectors
            ]
            if not waits:
                continue
            state.remove_nodes_from(waits)
            removed += len(waits)
            plain = all(edge.data.is_unconditional() and not edge.data.assignments for edge in sdfg.in_edges(state))
            if state.is_empty() and state is not sdfg.start_block and plain:
                sdfg.remove_node(state)
        return removed or None
