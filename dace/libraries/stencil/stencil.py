# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
import collections
from typing import Dict, Tuple

import dace
import dace.library

from .cpu import ExpandStencilCPU


@dace.library.node
class Stencil(dace.library.LibraryNode):
    """Represents applying a stencil to a full input domain."""

    implementations = {"pure": ExpandStencilCPU}
    default_implementation = "pure"

    code = dace.properties.CodeProperty(
        desc="Stencil code using all inputs to produce all outputs",
        default=dace.properties.CodeBlock(""))
    iterator_mapping = dace.properties.DictProperty(
        str,
        tuple,
        desc=("Dictionary mapping lower-dimensional input fields to a tuple "
              " of booleans indicating which iterators to use for their "
              "accesses, e.g.: {'a': (True, False, True)} uses the first and "
              "last iterator in a 3D iteration space to access a 2D array."),
        default=collections.OrderedDict())
    boundary_conditions = dace.properties.OrderedDictProperty(
        desc="Boundary condition specifications for each accessed field",
        default=collections.OrderedDict())

    def __init__(self,
                 label: str,
                 code: str,
                 iterator_mapping: Dict[str, Tuple[int]] = {},
                 boundary_conditions: Dict[str, Dict] = {}):
        super().__init__(label)
        self.code = type(self).code.from_string(code,
                                                dace.dtypes.Language.Python)
        self.iterator_mapping = iterator_mapping
        self.boundary_conditions = boundary_conditions
