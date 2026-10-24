# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""``Sync``: a barrier among the threads of a warp, a thread block or a whole grid of a GPU kernel."""

import enum

import dace
from dace import library, nodes, properties
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


def sync_code(scope: SyncScope, backend: str, mask: str | None = None) -> str:
    """The code of a barrier of ``scope`` for the ``cuda`` or ``hip`` backend.

    :param scope: The threads that wait for each other.
    :param backend: The GPU backend.
    :param mask: The lanes of a warp barrier, or ``None`` for the whole warp.
    """
    if scope == SyncScope.BLOCK:
        return "__syncthreads();"
    if scope == SyncScope.GRID:
        return "cooperative_groups::this_grid().sync();"
    if backend == "hip":
        # A wavefront runs in lockstep, so a barrier of any lanes orders the memory accesses of the wavefront
        return (
            '__builtin_amdgcn_fence(__ATOMIC_RELEASE, "wavefront");\n'
            "__builtin_amdgcn_wave_barrier();\n"
            '__builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "wavefront");'
        )
    return f"__syncwarp({mask});" if mask is not None else "__syncwarp();"


@library.node
class Sync(nodes.LibraryNode):
    """A barrier among the threads of a warp, a thread block or the grid of a GPU kernel.

    The node has no connectors: it orders what runs before and after it in the control flow, so it is the only node
    of its state.
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


def _expand(node: Sync, state: dace.SDFGState, backend: str) -> nodes.Tasklet:
    tasklet = nodes.Tasklet(
        node.name,
        code=sync_code(node.scope, backend, node.mask),
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
