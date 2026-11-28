# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Contiguous device copy through ``gpuMemcpyAsync``."""

from typing import TYPE_CHECKING

from dace import library
from dace.libraries.standard import environments
from dace.libraries.standard.helper import auto_dispatch
from dace.libraries.standard.nodes.copy.common import _make_memcpy_tasklet
from dace.libraries.standard.nodes.copy.node import CopyLibraryNode
from dace.libraries.standard.nodes.copy.select import select_copy_implementation
from dace.sdfg.scope import is_devicelevel_gpu
from dace.transformation.transformation import ExpandTransformation

if TYPE_CHECKING:
    pass


@library.register_expansion(CopyLibraryNode, "MemcpyCUDA1D")
class ExpandMemcpyCUDA1D(ExpandTransformation):
    """One ``gpuMemcpyAsync`` for a contiguous copy; direction (H2D/D2H/D2D/H2H) inferred from
    endpoint storages."""

    environments = [environments.CUDA]

    @staticmethod
    def expansion(node, parent_state, parent_sdfg):
        # The memcpy is a host call: inside a kernel the node takes the in-kernel form Auto picks there
        if is_devicelevel_gpu(parent_state.sdfg, parent_state, node):
            return auto_dispatch(node, parent_state, select_copy_implementation, CopyLibraryNode)
        return _make_memcpy_tasklet(node, parent_state, cuda=True)
