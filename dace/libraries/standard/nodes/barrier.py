# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""``Barrier`` library node: wait until every thread of a warp, a thread block or the grid reaches it."""

import enum

from dace import SDFG, SDFGState, dtypes, library, nodes, properties
from dace.sdfg.scope import is_devicelevel_gpu
from dace.transformation.transformation import ExpandTransformation


class SyncScope(enum.Enum):
    """The threads a barrier waits for."""

    WARP = enum.auto()  #: the threads of a warp (a wavefront on AMD)
    BLOCK = enum.auto()  #: the threads of a thread block
    GRID = enum.auto()  #: every thread of the kernel


#: The statement each GPU backend waits with, per scope. A scope missing here has no lowering yet: a grid barrier needs
#: a cooperative launch.
GPU_BARRIERS: dict[tuple[str, SyncScope], str] = {
    ("cuda", SyncScope.BLOCK): "__syncthreads();",
    ("hip", SyncScope.BLOCK): "__syncthreads();",
    ("cuda", SyncScope.WARP): "__syncwarp();",
    ("hip", SyncScope.WARP): "__builtin_amdgcn_wave_barrier();",
}


@library.expansion
class ExpandBarrierNative(ExpandTransformation):
    """The barrier of the GPU backend in a kernel; on a CPU one core runs the whole group, so nothing waits."""

    environments = []

    @staticmethod
    def expansion(node: "Barrier", parent_state: SDFGState, parent_sdfg: SDFG) -> nodes.Tasklet:
        code = ""
        if is_devicelevel_gpu(parent_sdfg, parent_state, node):
            # Avoid import loop
            from dace.codegen import common

            backend = common.get_gpu_backend()
            if (backend, node.scope) not in GPU_BARRIERS:
                raise NotImplementedError(f"{node.label}: no {node.scope.name} barrier on {backend} yet")
            code = GPU_BARRIERS[(backend, node.scope)]
        return nodes.Tasklet(node.label, {}, {}, code, language=dtypes.Language.CPP, side_effects=True)


@library.node
class Barrier(nodes.LibraryNode):
    """Every thread of ``scope`` waits here until all of them arrive, and sees what they wrote before."""

    implementations = {"native": ExpandBarrierNative}
    default_implementation = "native"

    scope = properties.EnumProperty(dtype=SyncScope, default=SyncScope.BLOCK, desc="The threads the barrier waits for.")

    def __init__(self, name: str, scope: SyncScope = SyncScope.BLOCK, **kwargs):
        super().__init__(name, **kwargs)
        self.scope = scope

    def has_side_effects(self, sdfg: SDFG) -> bool:
        return True
