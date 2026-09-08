# Copyright 2019-2025 ETH Zurich and the DaCe authors. All rights reserved.
"""
Contains replacements for array metadata (shape, strides, etc.).
"""
import dace  # noqa
from dace.frontend.common import op_repository as oprepo
from dace.frontend.common.op_repository import infers_attribute_descriptor, infers_descriptor
from dace.frontend.python.replacements.utils import ProgramVisitor, Size
from dace import data, dtypes, SDFG, SDFGState


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


@infers_descriptor('len')
def _infer_len(input_descs, a, **_kw):
    if not isinstance(a, str) or a not in input_descs:
        return None
    return data.Scalar(dtypes.int64, transient=True)


@oprepo.replaces_attribute('Array', 'size')
@oprepo.replaces_attribute('Scalar', 'size')
@oprepo.replaces_attribute('View', 'size')
def size(pv: ProgramVisitor, sdfg: SDFG, state: SDFGState, arr: str) -> Size:
    desc = sdfg.arrays[arr]
    totalsize = data._prod(desc.shape)
    return totalsize


def _infer_size(self_desc, **_kw):
    return data.Scalar(dtypes.int64, transient=True)


for _cls in ('Array', 'Scalar', 'View'):
    infers_attribute_descriptor(_cls, 'size')(_infer_size)
