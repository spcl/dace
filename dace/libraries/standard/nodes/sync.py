# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""``Sync``: a barrier among the threads of a warp, a thread block or a whole grid of a GPU kernel, or of an OpenMP
team, or a host wait for a GPU stream or the whole device."""

import enum

import dace
from dace import dtypes, library, nodes, properties
from dace.codegen.common import get_gpu_backend
from dace.sdfg.scope import is_devicelevel_gpu
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

    The node orders what runs before and after it: through the (empty) edges of its state, or, as the only node of
    its state, through the control flow. Only a
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
    """The OpenMP expansion for a barrier in a CPU team (``CPU_Persistent``), else that of the configured GPU
    backend."""

    environments = []

    @staticmethod
    def expansion(node: Sync, parent_state: dace.SDFGState, parent_sdfg: dace.SDFG) -> nodes.Tasklet:
        in_team = enclosing_cpu_parallel_schedule(parent_sdfg, parent_state, node) == dtypes.ScheduleType.CPU_Persistent
        if in_team and not is_devicelevel_gpu(parent_sdfg, parent_state, node):
            return ExpandOpenMP.expansion(node, parent_state, parent_sdfg)
        return _expand(node, parent_state, get_gpu_backend())


def enclosing_cpu_parallel_schedule(
    sdfg: dace.SDFG, state: dace.SDFGState, node: nodes.Node
) -> dtypes.ScheduleType | None:
    """The schedule of the innermost ``CPU_Multicore`` or ``CPU_Persistent`` map around ``node``, across nested
    SDFGs; ``None`` when one thread runs it."""
    parallel = (dtypes.ScheduleType.CPU_Multicore, dtypes.ScheduleType.CPU_Persistent)
    while True:
        scopes = state.scope_dict()
        scope = scopes[node]
        while scope is not None:
            if scope.schedule in parallel:
                return scope.schedule
            scope = scopes[scope]
        if sdfg.parent_nsdfg_node is None:
            return None
        sdfg, state, node = sdfg.parent_sdfg, sdfg.parent, sdfg.parent_nsdfg_node


@library.register_expansion(Sync, "OpenMP")
class ExpandOpenMP(ExpandTransformation):
    """``#pragma omp barrier`` for the team of the enclosing ``CPU_Persistent`` region (an ``omp parallel``), whether
    the barrier names the block or the grid. A warp is one thread on a CPU and one thread waits for nobody, so both
    need no statement; a worksharing ``omp for`` (``CPU_Multicore``) cannot hold a barrier."""

    environments = []

    @staticmethod
    def expansion(node: Sync, parent_state: dace.SDFGState, parent_sdfg: dace.SDFG) -> nodes.Tasklet:
        if node.scope in (SyncScope.STREAM, SyncScope.DEVICE):
            raise ValueError(f"{node.label}: a {node.scope.value} wait has no OpenMP lowering")
        if is_devicelevel_gpu(parent_sdfg, parent_state, node):
            raise ValueError(f"{node.label}: the OpenMP barrier cannot run in a GPU kernel")
        schedule = enclosing_cpu_parallel_schedule(parent_sdfg, parent_state, node)
        if schedule == dtypes.ScheduleType.CPU_Multicore:
            raise ValueError(f"{node.label}: an OpenMP barrier cannot be inside a worksharing (CPU_Multicore) map")
        code = ""
        if schedule == dtypes.ScheduleType.CPU_Persistent and node.scope != SyncScope.WARP:
            code = "#pragma omp barrier"
        return nodes.Tasklet(node.label, {}, {}, code, language=dace.Language.CPP, side_effects=True)
