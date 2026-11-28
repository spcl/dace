# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Block-collective copy inside a GPU kernel: the block's lanes split the elements, adjacent lanes adjacent elements."""

from dace import library
from dace.libraries.standard.nodes.copy.common import _make_mapped_tasklet_expansion
from dace.libraries.standard.nodes.copy.node import CopyLibraryNode
from dace.transformation.transformation import ExpandTransformation


@library.register_expansion(CopyLibraryNode, "CUDA (block)")
class ExpandCUDABlock(ExpandTransformation):
    """The mapped element-wise copy, across the lanes of the kernel's thread block, with a barrier after it."""

    environments = []
    runs_inside_kernel = True

    @staticmethod
    def expansion(node, parent_state, parent_sdfg):
        from dace.libraries.standard.block_reduce import BLOCK_COLLECTIVE_THREADS

        return _make_mapped_tasklet_expansion(
            node, parent_state, allow_cross_storage=True, lanes=BLOCK_COLLECTIVE_THREADS
        )
