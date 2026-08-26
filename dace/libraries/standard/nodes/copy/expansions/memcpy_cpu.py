# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Single ``memcpy`` on the host.
"""
from typing import TYPE_CHECKING

from dace import library
from dace.libraries.standard import environments
from dace.libraries.standard.nodes.copy.node import CopyLibraryNode
from dace.transformation.transformation import ExpandTransformation
from dace.libraries.standard.nodes.copy.common import (_make_memcpy_tasklet)

if TYPE_CHECKING:
    pass


@library.register_expansion(CopyLibraryNode, 'MemcpyCPU')
class ExpandMemcpyCPU(ExpandTransformation):
    """One ``std::memcpy`` for a contiguous CPU<->CPU copy."""
    environments = [environments.CPU]

    @staticmethod
    def expansion(node, parent_state, parent_sdfg):
        return _make_memcpy_tasklet(node, parent_state, cuda=False)
