# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""``Sync``: a barrier among the threads of a warp, a thread block or a whole grid of a GPU kernel, or a host wait for
a GPU stream or the whole device."""

import enum

import dace
from dace import dtypes, library, nodes, properties
from dace.codegen.common import get_gpu_backend
from dace.transformation.transformation import ExpandTransformation


class SyncScope(enum.Enum):
    """Which threads a :class:`Sync` waits for."""

    #: The (``mask`` selected) threads of a warp, ``__syncwarp``
    WARP = "warp"
    #: The threads of a thread block (CTA), ``__syncthreads``
    BLOCK = "block"
    #: Every thread of the grid, which a cooperative launch keeps resident
    GRID = "grid"
    #: The host waits for the work of the stream connected to :data:`STREAM_CONNECTOR`, or of the default stream
    #: (``nullptr``) when none is, as a kernel without a stream launches on it
    STREAM = "stream"
    #: The host waits for all work of the device
    DEVICE = "device"


#: The in-connector of a stream barrier, which takes the stream
STREAM_CONNECTOR = "__stream"
#: The type of :data:`STREAM_CONNECTOR`
GPU_STREAM = dtypes.opaque("gpuStream_t")


def sync_code(scope: SyncScope, backend: str, mask: str | None = None, stream: str = "nullptr") -> str:
    """The code of a barrier of ``scope`` for the ``cuda`` or ``hip`` backend.

    :param scope: The threads that wait for each other.
    :param backend: The GPU backend.
    :param mask: The lanes of a warp barrier, or ``None`` for the whole warp.
    :param stream: The stream a stream barrier waits for.
    """
    if scope == SyncScope.STREAM:
        return f"DACE_GPU_CHECK({backend}StreamSynchronize({stream}));"
    if scope == SyncScope.DEVICE:
        return f"DACE_GPU_CHECK({backend}DeviceSynchronize());"
    if scope == SyncScope.BLOCK:
        return "__syncthreads();"
    if scope == SyncScope.GRID:
        return "cooperative_groups::this_grid().sync();"
    if scope == SyncScope.WARP:
        if backend == "hip":
            # A wavefront runs in lockstep, so a barrier of any lanes orders the memory accesses of the wavefront
            return (
                '__builtin_amdgcn_fence(__ATOMIC_RELEASE, "wavefront");\n'
                "__builtin_amdgcn_wave_barrier();\n"
                '__builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "wavefront");'
            )
        else:
            return f"__syncwarp({mask});" if mask is not None else "__syncwarp();"
    raise ValueError(f"Sync {scope.value}: unknown scope for backend {backend}")


@library.node
class Sync(nodes.LibraryNode):
    """A barrier among the threads of a warp, a thread block or the grid of a GPU kernel, or a host wait for a GPU
    stream or the whole device.

    The node orders what runs before and after it in the control flow, so it is the only node of its state. Only a
    stream barrier takes a connector: :data:`STREAM_CONNECTOR`, typed :data:`GPU_STREAM`, which the pass wiring the
    streams adds; without it, the barrier waits for the default stream.
    """

    implementations = {}
    default_implementation = "Auto"

    scope = properties.EnumProperty(dtype=SyncScope, default=SyncScope.BLOCK, desc="Which threads wait for each other.")
    mask = properties.Property(
        dtype=str, allow_none=True, default=None, desc="The lanes of a warp barrier (CUDA), or None for the whole warp."
    )

    def __init__(self, name: str, *, scope: SyncScope = SyncScope.BLOCK, mask: str | None = None, **kwargs):
        super().__init__(name, **kwargs)
        self.scope = scope
        self.mask = mask

    def has_side_effects(self, sdfg) -> bool:
        return True

    def validate(self, sdfg: dace.SDFG, state: dace.SDFGState) -> None:
        if self.mask is not None and self.scope != SyncScope.WARP:
            raise ValueError(f"Sync {self.name}: only a warp barrier takes a mask")
        expected = {STREAM_CONNECTOR} if self.scope == SyncScope.STREAM else set()
        if not set(self.in_connectors) <= expected or self.out_connectors:
            raise ValueError(
                f"Sync {self.name}: a {self.scope.value} barrier takes the in-connectors {expected or '{}'}"
            )


def _expand(node: Sync, state: dace.SDFGState, backend: str) -> nodes.Tasklet:
    wired = any(e.dst_conn == STREAM_CONNECTOR for e in state.in_edges(node))
    stream = STREAM_CONNECTOR if wired else "nullptr"
    tasklet = nodes.Tasklet(
        node.name,
        code=sync_code(node.scope, backend, node.mask, stream),
        language=dace.Language.CPP,
        side_effects=True,
    )
    # Without edges to take over, the expansion would not add the tasklet to the state
    state.add_node(tasklet)
    return tasklet


@library.register_expansion(Sync, "CUDA")
class ExpandCUDA(ExpandTransformation):
    environments = []

    @staticmethod
    def expansion(node: Sync, parent_state: dace.SDFGState, parent_sdfg: dace.SDFG) -> nodes.Tasklet:
        return _expand(node, parent_state, "cuda")


@library.register_expansion(Sync, "HIP")
class ExpandHIP(ExpandTransformation):
    environments = []

    @staticmethod
    def expansion(node: Sync, parent_state: dace.SDFGState, parent_sdfg: dace.SDFG) -> nodes.Tasklet:
        return _expand(node, parent_state, "hip")


@library.register_expansion(Sync, "Auto")
class ExpandAuto(ExpandTransformation):
    """The expansion of the configured GPU backend."""

    environments = []

    @staticmethod
    def expansion(node: Sync, parent_state: dace.SDFGState, parent_sdfg: dace.SDFG) -> nodes.Tasklet:
        return _expand(node, parent_state, get_gpu_backend())
