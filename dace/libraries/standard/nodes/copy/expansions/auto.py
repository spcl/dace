# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Default expansion: dispatches via :func:`select_copy_implementation`.
"""
from typing import TYPE_CHECKING

from dace import library
from dace.libraries.standard.helper import auto_dispatch
from dace.transformation.transformation import ExpandTransformation
from dace.libraries.standard.nodes.copy.select import select_copy_implementation

if TYPE_CHECKING:
    pass


@library.expansion
class ExpandAuto(ExpandTransformation):
    """Default expansion: dispatches to the implementation chosen by
    :func:`select_copy_implementation` from endpoint storages, subset shapes, and scope."""
    environments = []

    @staticmethod
    def expansion(node, parent_state, parent_sdfg):
        from dace.libraries.standard.nodes.copy.node import CopyLibraryNode  # Avoid import loop
        return auto_dispatch(node, parent_state, select_copy_implementation, CopyLibraryNode)
