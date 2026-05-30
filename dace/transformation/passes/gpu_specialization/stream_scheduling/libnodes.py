# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Library nodes representing CUDA stream synchronisation calls.

Three libnodes -- one per CUDA runtime call:

- ``StreamEventRecordLibraryNode``  ↔  ``cudaEventRecord(event, stream)``
- ``StreamWaitEventLibraryNode``    ↔  ``cudaStreamWaitEvent(stream, event)``
- ``StreamSynchronizeLibraryNode``  ↔  ``cudaStreamSynchronize(stream)`` (host)

The scheduler inserts these as first-class nodes in the SDFG; expansion
emits the matching C++ runtime call. Mirrors the ``CopyLibraryNode`` /
``MemsetLibraryNode`` design: one libnode per primitive, single
expansion, no auto-dispatch (the call is fixed).

Stream and event connector names come from
:mod:`gpu_specialization.helpers.gpu_helpers` (``STREAM_CONNECTOR`` and
``EVENT_CONNECTOR`` respectively) so producers, consumers, and the
scheduler share one truth.
"""
import dace
from dace import library, nodes
from dace.codegen import common
from dace.transformation.transformation import ExpandTransformation
from dace.libraries.standard import environments
from dace.transformation.passes.gpu_specialization.helpers.gpu_helpers import (EVENT_CONNECTOR, STREAM_CONNECTOR)


def _backend_call(call_name: str) -> str:
    """Return the backend-prefixed runtime call (``cudaXxx`` or ``hipXxx``)."""
    return f"{common.get_gpu_backend()}{call_name}"


@library.expansion
class ExpandStreamEventRecord(ExpandTransformation):
    environments = [environments.CUDA]

    @staticmethod
    def expansion(node: "StreamEventRecordLibraryNode", parent_state: dace.SDFGState,
                  parent_sdfg: dace.SDFG) -> nodes.Tasklet:
        backend = _backend_call("EventRecord")
        code = (f"DACE_GPU_CHECK({backend}({EVENT_CONNECTOR}, {STREAM_CONNECTOR}));")
        return nodes.Tasklet(node.name,
                             inputs={
                                 STREAM_CONNECTOR: dace.dtypes.gpuStream_t,
                                 EVENT_CONNECTOR: dace.dtypes.gpuEvent_t,
                             },
                             outputs={},
                             code=code,
                             language=dace.Language.CPP,
                             side_effects=True)


@library.expansion
class ExpandStreamWaitEvent(ExpandTransformation):
    environments = [environments.CUDA]

    @staticmethod
    def expansion(node: "StreamWaitEventLibraryNode", parent_state: dace.SDFGState,
                  parent_sdfg: dace.SDFG) -> nodes.Tasklet:
        backend = _backend_call("StreamWaitEvent")
        # Third argument 0 = no flags (CUDA default).
        code = (f"DACE_GPU_CHECK({backend}({STREAM_CONNECTOR}, {EVENT_CONNECTOR}, 0));")
        return nodes.Tasklet(node.name,
                             inputs={
                                 STREAM_CONNECTOR: dace.dtypes.gpuStream_t,
                                 EVENT_CONNECTOR: dace.dtypes.gpuEvent_t,
                             },
                             outputs={},
                             code=code,
                             language=dace.Language.CPP,
                             side_effects=True)


@library.expansion
class ExpandStreamSynchronize(ExpandTransformation):
    environments = [environments.CUDA]

    @staticmethod
    def expansion(node: "StreamSynchronizeLibraryNode", parent_state: dace.SDFGState,
                  parent_sdfg: dace.SDFG) -> nodes.Tasklet:
        backend = _backend_call("StreamSynchronize")
        code = f"DACE_GPU_CHECK({backend}({STREAM_CONNECTOR}));"
        return nodes.Tasklet(node.name,
                             inputs={STREAM_CONNECTOR: dace.dtypes.gpuStream_t},
                             outputs={},
                             code=code,
                             language=dace.Language.CPP,
                             side_effects=True)


@library.node
class StreamEventRecordLibraryNode(nodes.LibraryNode):
    """``cudaEventRecord(event, stream)`` -- mark ``stream``'s current position
    with ``event``.

    Scheduler-inserted node; the codegen emits exactly one
    ``EventRecord`` call per instance. No data flow: the stream and
    event handles are read via the shared ``STREAM_CONNECTOR`` /
    ``EVENT_CONNECTOR`` in-connectors.
    """

    implementations = {"CUDA": ExpandStreamEventRecord}
    default_implementation = "CUDA"

    def __init__(self, name: str = "stream_event_record", *args, **kwargs):
        super().__init__(name, *args, inputs={STREAM_CONNECTOR, EVENT_CONNECTOR}, outputs=set(), **kwargs)


@library.node
class StreamWaitEventLibraryNode(nodes.LibraryNode):
    """``cudaStreamWaitEvent(stream, event, 0)`` -- make ``stream`` block on
    ``event``.

    Scheduler-inserted node mirroring
    :class:`StreamEventRecordLibraryNode` on the consumer side of a
    cross-stream dependency.
    """

    implementations = {"CUDA": ExpandStreamWaitEvent}
    default_implementation = "CUDA"

    def __init__(self, name: str = "stream_wait_event", *args, **kwargs):
        super().__init__(name, *args, inputs={STREAM_CONNECTOR, EVENT_CONNECTOR}, outputs=set(), **kwargs)


@library.node
class StreamSynchronizeLibraryNode(nodes.LibraryNode):
    """``cudaStreamSynchronize(stream)`` -- host blocks until ``stream`` drains.

    Used at the host-visible reads point: an interstate-edge condition
    or assignment that reads a GPU array forces the host to wait on the
    stream that last wrote it.
    """

    implementations = {"CUDA": ExpandStreamSynchronize}
    default_implementation = "CUDA"

    def __init__(self, name: str = "stream_synchronize", *args, **kwargs):
        super().__init__(name, *args, inputs={STREAM_CONNECTOR}, outputs=set(), **kwargs)
