# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Shared pieces of the ``FillLibraryNode`` expansions.

Imported by both the node and its expansions, so it must not import either.
"""

from typing import List, Optional, Tuple, TYPE_CHECKING

import numpy as np

import dace
from dace.libraries.standard.helper import collapse_shape_and_strides

if TYPE_CHECKING:
    from dace.libraries.standard.nodes.fill.node import FillLibraryNode

OUTPUT_CONNECTOR_NAME = "_fill_out"
VALUE_CONNECTOR_NAME = "_fill_val"


def numpy_scalar(value, dtype: dace.dtypes.typeclass) -> np.generic:
    """The fill value as a numpy scalar of the destination type.

    :param value: The Python constant held by the node.
    :param dtype: Destination element type.
    :returns: ``value`` narrowed to ``dtype``.
    """
    return np.array(value, dtype=dtype.as_numpy_dtype())[()]


def byte_pattern(value, dtype: dace.dtypes.typeclass) -> Optional[int]:
    """The single byte a ``memset`` would need, or ``None`` if the value is not byte-splat.

    ``memset`` writes one byte over the whole range, so it can only express values whose object
    representation repeats that byte -- every zero, ``-1`` in two's complement, any 1-byte type.
    ``1.0f`` (``0000803f``) is not one of them.

    :param value: The Python constant held by the node.
    :param dtype: Destination element type.
    :returns: The byte to pass to ``memset``, or ``None``.
    """
    raw = numpy_scalar(value, dtype).tobytes()
    return raw[0] if len(set(raw)) == 1 else None


def cpp_literal(value, dtype: dace.dtypes.typeclass) -> str:
    """Render the fill value as a C++ literal of ``dtype``.

    :param value: The Python constant held by the node.
    :param dtype: Destination element type.
    :returns: A C++ expression of type ``dtype.ctype``.
    """
    narrowed = numpy_scalar(value, dtype).item()
    ctype = dtype.ctype
    if isinstance(narrowed, bool):
        return 'true' if narrowed else 'false'
    if isinstance(narrowed, complex):
        return f"{ctype}({narrowed.real!r}, {narrowed.imag!r})"
    if isinstance(narrowed, float):
        return f"{narrowed!r}f" if ctype == 'float' else repr(narrowed)
    return repr(narrowed)


def python_literal(value, dtype: dace.dtypes.typeclass) -> str:
    """Render the fill value for a Python-language tasklet body.

    :param value: The Python constant held by the node.
    :param dtype: Destination element type.
    :returns: A Python expression.
    """
    return repr(numpy_scalar(value, dtype).item())


def make_fill_skeleton(
    node: "FillLibraryNode", parent_state: dace.SDFGState
) -> Tuple[dace.SDFG, dace.SDFGState, str, dace.data.Data, List]:
    """Build the shared SDFG skeleton for the mapped (``ExpandPure``) fill expansion.

    :param node: The fill library node being expanded.
    :param parent_state: The state containing ``node`` (owning SDFG is ``parent_state.sdfg``).
    :returns: ``(sdfg, state, out_name, out, map_lengths)``.
    """
    out_name, out, out_subset = node.validate(parent_state.sdfg, parent_state)
    out_shape_collapsed, out_strides_collapsed = collapse_shape_and_strides(out_subset, out.strides)

    sdfg = dace.SDFG(f"{node.label}_sdfg")
    sdfg.add_array(out_name, out_shape_collapsed, out.dtype, out.storage, strides=out_strides_collapsed)
    sdfg.schedule = dace.dtypes.ScheduleType.Sequential

    state = sdfg.add_state(f"{node.label}_state")
    # Reuse the array descriptor's collapsed shape as map bounds so extents can't diverge.
    map_lengths = out_shape_collapsed

    return sdfg, state, out_name, out, map_lengths
