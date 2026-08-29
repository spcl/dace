# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Contiguous device copy through ``gpuMemcpyAsync``.
"""
from typing import TYPE_CHECKING

from dace import library
from dace.libraries.standard import environments
from dace.libraries.standard.nodes.copy.node import CopyLibraryNode
from dace.transformation.transformation import ExpandTransformation
from dace.libraries.standard.nodes.copy.common import (_make_memcpy_tasklet)

if TYPE_CHECKING:
    pass


@library.register_expansion(CopyLibraryNode, 'MemcpyCUDA1D')
class ExpandMemcpyCUDA1D(ExpandTransformation):
    """One ``gpuMemcpyAsync`` for a contiguous copy; direction (H2D/D2H/D2D/H2H) inferred from
    endpoint storages."""
    environments = [environments.CUDA]

    @staticmethod
    def expansion(node, parent_state, parent_sdfg):
        return _make_memcpy_tasklet(node, parent_state, cuda=True)
