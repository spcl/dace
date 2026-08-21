# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Default expansion: dispatches via :func:`select_fill_implementation`."""
from typing import TYPE_CHECKING

import dace
from dace import library
from dace.libraries.standard.helper import auto_dispatch
from dace.libraries.standard.nodes.fill.select import select_fill_implementation
from dace.transformation.transformation import ExpandTransformation

if TYPE_CHECKING:
    from dace.libraries.standard.nodes.fill.node import FillLibraryNode


@library.expansion
class ExpandAuto(ExpandTransformation):
    environments = []

    @staticmethod
    def expansion(node: "FillLibraryNode", parent_state: dace.SDFGState, parent_sdfg: dace.SDFG):
        from dace.libraries.standard.nodes.fill.node import FillLibraryNode  # Avoid import loop
        return auto_dispatch(node, parent_state, select_fill_implementation, FillLibraryNode)
