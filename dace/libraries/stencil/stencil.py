# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
import collections

import dace
import dace.library

from .cpu import ExpandStencilCPU


@dace.library.node
class Stencil(dace.library.LibraryNode):
    """Represents applying a stencil to a full input domain."""

    implementations = {"pure": ExpandStencilCPU}
    default_implementation = "pure"

    # Example:
    # accesses = {
    #   "a": ((True, True, True), [(0, 0, -1), (0, -1, 0), (1, 0, 0)]),
    #   "b": ((True, False, True), [(0, 1), (1, 0), (-1, 0), (0, -1)])
    # }
    accesses = dace.properties.OrderedDictProperty(
        desc=("Dictionary mapping input fields to lists of offsets "
              "and index mapping"),
        default=collections.OrderedDict())
    boundary_conditions = dace.properties.OrderedDictProperty(
        desc="Boundary condition specifications for each accessed field",
        default=collections.OrderedDict())
    code = dace.properties.CodeProperty(
        desc="Stencil code using all inputs to produce all outputs",
        default=dace.properties.CodeBlock(""))

    def __init__(self, label, accesses={}, code="", boundary_conditions={}):
        super().__init__(label)
        self.accesses = accesses
        # Default to only outputting accesses that don't go out of bounds
        if not boundary_conditions:
            for name in accesses:
                boundary_conditions[name] = {"btype": "shrink"}
        self.boundary_conditions = boundary_conditions
        self.code = type(self).code.from_string(code,
                                                dace.dtypes.Language.Python)
