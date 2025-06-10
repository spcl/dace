# Copyright 2019-2024 ETH Zurich and the DaCe authors. All rights reserved.
"""
Contains replacements for array metadata (shape, strides, etc.).
"""
from dace.frontend.common import op_repository as oprepo
from dace.frontend.python.replacements.utils import ProgramVisitor, Size
from dace import data, SDFG, SDFGState


@oprepo.replaces('len')
def _len_array(pv: 'ProgramVisitor', sdfg: SDFG, state: SDFGState, a: str):
    # len(numpy_array) is equivalent to numpy_array.shape[0]
    if isinstance(a, str):
        if a in sdfg.arrays:
            return sdfg.arrays[a].shape[0]
        if a in sdfg.constants_prop:
            return len(sdfg.constants[a])
    else:
        return len(a)


@oprepo.replaces_attribute('Array', 'size')
@oprepo.replaces_attribute('Scalar', 'size')
@oprepo.replaces_attribute('View', 'size')
def size(pv: ProgramVisitor, sdfg: SDFG, state: SDFGState, arr: str) -> Size:
    desc = sdfg.arrays[arr]
    totalsize = data._prod(desc.shape)
    return totalsize
