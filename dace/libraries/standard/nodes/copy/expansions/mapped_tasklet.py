# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Element-wise mapped copy: the general form every other expansion falls back to.
"""
from typing import TYPE_CHECKING

from dace import library
from dace.libraries.standard.nodes.copy.node import CopyLibraryNode
from dace.transformation.transformation import ExpandTransformation
from dace.libraries.standard.nodes.copy.common import (_make_mapped_tasklet_expansion)

if TYPE_CHECKING:
    pass


@library.register_expansion(CopyLibraryNode, 'MappedTasklet')
class ExpandMappedTasklet(ExpandTransformation):
    """Mapped element-wise tasklet ``_cpy_out = _cpy_in`` over the collapsed copy shape. Schedule
    from endpoint storages: ``Sequential`` for Register/Register<->GPU_Shared (thread-level),
    ``GPU_Device`` if any side is GPU, else ``Default``. Raises across the CPU/GPU boundary."""
    environments = []

    @staticmethod
    def expansion(node, parent_state, parent_sdfg):
        return _make_mapped_tasklet_expansion(node, parent_state, allow_cross_storage=True)
