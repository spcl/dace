# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""
Shared helpers for lightweight descriptor-inference functions used by the
schedule-tree frontend.  The actual ``@infers_descriptor`` (and related)
registrations live next to their SDFG-level replacements in the respective
``replacements/*.py`` modules.
"""

from numbers import Number
from typing import Dict, Optional

import numpy as np

from dace import data, dtypes, symbolic
from dace.frontend.python.replacements.utils import normalize_axes

# -------------------------------------------------------------------- #
#  Helpers                                                               #
# -------------------------------------------------------------------- #


def _get_desc(input_descs: Dict[str, data.Data], arg) -> Optional[data.Data]:
    """Resolve *arg* to a descriptor if it names an input array."""
    if isinstance(arg, str) and arg in input_descs:
        return input_descs[arg]
    if isinstance(arg, data.Data):
        return arg
    return None


def scalar_operand_descriptor(value) -> Optional[data.Data]:
    """
    Descriptor of an operand that is not a data container but a compile-time
    scalar: a Python/NumPy number or a symbolic expression.

    The replacement implementations accept these (``replacements.utils.simple_call``
    materializes a constant into an SDFG constant, and a symbolic expression
    straight into the tasklet's code), producing a scalar of the value's own
    type. Inference functions call this so that ``math.sqrt(20)`` or
    ``dace.float64(3)`` type the same way their implementations lower.

    :return: The descriptor, or None if *value* is neither of those.
    """
    if isinstance(value, (Number, np.number)):
        try:
            return data.create_datadescriptor(value)
        except Exception:
            return None
    if symbolic.issymbolic(value):
        try:
            return data.Scalar(symbolic.symtype(value))
        except Exception:
            return None
    return None


def _to_int(v) -> Optional[int]:
    """Try to convert *v* to a plain Python int (for axis, shape elements)."""
    if isinstance(v, (int, np.integer)):
        return int(v)
    if isinstance(v, Number):
        iv = int(v)
        if iv == v:
            return iv
    return None


def _reduction_descriptor(input_descs: Dict[str, data.Data],
                          arr,
                          axis=None,
                          dtype_override: Optional[dtypes.typeclass] = None) -> Optional[data.Data]:
    """Shared logic for reduction-style operations (sum, max, prod, ...)."""
    desc = _get_desc(input_descs, arr)
    if desc is None:
        return None

    out_dtype = dtype_override or desc.dtype
    shape = list(desc.shape)

    if axis is None:
        return data.Scalar(out_dtype)

    if not isinstance(axis, (tuple, list)):
        axis = (axis, )
    axis = tuple(_to_int(a) for a in axis)
    if any(a is None for a in axis):
        return None
    axis = tuple(normalize_axes(axis, len(shape)))

    if len(axis) == len(shape):
        return data.Scalar(out_dtype)

    out_shape = [s for i, s in enumerate(shape) if i not in axis]
    if not out_shape:
        return data.Scalar(out_dtype)
    return data.Array(out_dtype, out_shape, transient=True)


def _method_reduction_descriptor(self_desc: data.Data,
                                 axis=None,
                                 dtype_override: Optional[dtypes.typeclass] = None) -> Optional[data.Data]:
    """Shared logic for method-style reductions (a.sum(), a.max(), ...)."""
    return _reduction_descriptor({}, self_desc, axis, dtype_override)
