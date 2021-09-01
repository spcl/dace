# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
import collections
from typing import Dict, List, Tuple

import dace
import dace.library

from .cpu import ExpandStencilCPU
from .intel_fpga import ExpandStencilIntelFPGA
# from .xilinx import ExpandStencilXilinx


@dace.library.node
class Stencil(dace.library.LibraryNode):
    """
    Represents applying a stencil that reads at constants offset from one or
    more input connectors, and writes to one or more output connector, using
    the given boundary conditions when accesses are out of bounds.
    The size of the iteration space will be inferred from the largest field
    being accessed, and it is assumed that all other fields accessed have the
    same size in each corresponding dimension.

    For specifying the boundary conditions, the following options are supported:

    boundary_conditions = {
      # When an access into the given input is out of bounds, the code will...
      "a": {"btype": "shrink"},  # ...not output anything
      "b": {"btype": "constant", "value": 0.0},  # ...replace it with a constant
      "c": {"btype": "copy"}  # ...uses the center value, e.g., c[0, 0] in 2D
    }

    When one or more fields accessed are of lower dimensionality than others,
    the `iterator_mapping` argument is used to specify which iterators should
    be used to access it. Consider the following code:

      c[0, 0, 0] = a[0, 0, 0] + b[0, 0]

    This will produce three iterators _i0, _i1, and _i2. Which two of these are
    used to index into b, which is only 2-dimensional, is specified using a
    tuple of booleans choosing the iterators:

    input_mapping = {
      "b": (True, False, True)
    }

    This will use iterators _i0 and _i2 for accessing b.
    """

    implementations = {
        "pure": ExpandStencilCPU,
        "intel_fpga": ExpandStencilIntelFPGA,
        # "xilinx": ExpandStencilXilinx
    }
    default_implementation = "pure"

    code = dace.properties.CodeProperty(
        desc=("Stencil code accessing all the input connector at constant "
              "offsets relative to the center, e.g.: "
              "c[0, 0, 0] = 0.1 * a[-1, 0, 1] + 0.9 * b[0, 0, 1]"),
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
        desc=("Boundary condition specifications for each accessed field, on "
              "the form: {'b': {'btype': 'constant', 'value': 3}}."),
        default=collections.OrderedDict())

    def __init__(self,
                 label: str,
                 code: str = "",
                 iterator_mapping: Dict[str, Tuple[int]] = {},
                 boundary_conditions: Dict[str, Dict] = {},
                 **kwargs):
        super().__init__(label, **kwargs)
        self.code = type(self).code.from_string(code,
                                                dace.dtypes.Language.Python)
        self.iterator_mapping = iterator_mapping
        self.boundary_conditions = boundary_conditions
