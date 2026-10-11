# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""
Contains replacements for the Discrete Fourier Transform numpy package (numpy.fft)
"""

import sympy as sp

from dace import SDFG, Memlet, SDFGState, dtypes, symbolic
from dace.frontend.common import op_repository as oprepo
from dace.frontend.python.common import StringLiteral
from dace.frontend.python.replacements.utils import ProgramVisitor
from dace.utils import prod


def _real_to_complex(real_type: dtypes.typeclass):
    if real_type == dtypes.float32:
        return dtypes.complex64
    elif real_type == dtypes.float64:
        return dtypes.complex128
    else:
        return real_type


def _add_fourier_transform(
    pv: "ProgramVisitor",
    sdfg: SDFG,
    state: SDFGState,
    a: str,
    axis: int | None,
    norm: StringLiteral,
    is_inverse: bool,
) -> str:
    """Transforms ``a`` along ``axis``, or along every axis for ``None``, normalized as ``norm`` asks."""
    from dace.libraries.fft.nodes import FFT, IFFT  # Avoid import loops

    if not isinstance(a, str) or a not in sdfg.arrays:
        raise ValueError("Input must be a valid array")
    desc = sdfg.arrays[a]
    libnode = FFT("fft") if not is_inverse else IFFT("ifft")
    # A rank-1 input is transformed whole
    libnode.axis = axis if len(desc.shape) > 1 else None

    # numpy divides by the number of transformed elements
    size = desc.shape[axis] if axis is not None else prod(desc.shape)
    if norm == "forward":
        factor = (1 / size) if not is_inverse else 1
    elif norm == "backward":
        factor = 1 if not is_inverse else (1 / size)
    elif norm == "ortho":
        factor = sp.sqrt(1 / size)
    else:
        raise ValueError('norm argument can only be one of "forward", "backward", or "ortho".')
    libnode.factor = factor

    # Compute output type from input type
    if is_inverse and desc.dtype not in (dtypes.complex64, dtypes.complex128):
        raise TypeError(f"Inverse FFT only accepts complex inputs, got {desc.dtype}")
    dtype = _real_to_complex(desc.dtype)

    name, odesc = sdfg.add_temp_transient_like(desc, dtype, name=pv.get_target_name())
    r = state.add_read(a)
    w = state.add_write(name)
    state.add_edge(r, None, libnode, "_inp", Memlet.from_array(a, desc))
    state.add_edge(libnode, "_out", w, None, Memlet.from_array(name, odesc))

    return name


def _fft_core(
    pv: "ProgramVisitor",
    sdfg: SDFG,
    state: SDFGState,
    a: str,
    n: symbolic.SymbolicType | None = None,
    axis=-1,
    norm: StringLiteral = StringLiteral("backward"),
    is_inverse: bool = False,
):
    if not isinstance(a, str) or a not in sdfg.arrays:
        raise ValueError("Input must be a valid array")
    # If n is not None, either pad input or slice and add a view
    if n is not None:
        raise NotImplementedError
    rank = len(sdfg.arrays[a].shape)
    if not -rank <= axis < rank:
        raise ValueError(f"axis {axis} is out of range for a rank-{rank} array")
    return _add_fourier_transform(pv, sdfg, state, a, axis % rank, norm, is_inverse)


def _fftn_core(
    pv: "ProgramVisitor",
    sdfg: SDFG,
    state: SDFGState,
    a: str,
    s=None,
    axes=None,
    norm: StringLiteral = StringLiteral("backward"),
    is_inverse: bool = False,
):
    if not isinstance(a, str) or a not in sdfg.arrays:
        raise ValueError("Input must be a valid array")
    if s is not None:
        raise NotImplementedError("numpy.fft.fftn with s (padding or cropping)")
    rank = len(sdfg.arrays[a].shape)
    if axes is not None and sorted(int(x) % rank for x in axes) != list(range(rank)):
        raise NotImplementedError("numpy.fft.fftn along a subset of the axes")
    return _add_fourier_transform(pv, sdfg, state, a, None, norm, is_inverse)


@oprepo.replaces("numpy.fft.fft")
def _fft(
    pv: "ProgramVisitor",
    sdfg: SDFG,
    state: SDFGState,
    a: str,
    n: symbolic.SymbolicType | None = None,
    axis=-1,
    norm: StringLiteral = StringLiteral("backward"),
):
    return _fft_core(pv, sdfg, state, a, n, axis, norm, False)


@oprepo.replaces("numpy.fft.ifft")
def _ifft(
    pv: "ProgramVisitor",
    sdfg: SDFG,
    state: SDFGState,
    a,
    n=None,
    axis=-1,
    norm: StringLiteral = StringLiteral("backward"),
):
    return _fft_core(pv, sdfg, state, a, n, axis, norm, True)


@oprepo.replaces("numpy.fft.fftn")
def _fftn(
    pv: "ProgramVisitor",
    sdfg: SDFG,
    state: SDFGState,
    a: str,
    s=None,
    axes=None,
    norm: StringLiteral = StringLiteral("backward"),
):
    return _fftn_core(pv, sdfg, state, a, s, axes, norm, False)


@oprepo.replaces("numpy.fft.ifftn")
def _ifftn(
    pv: "ProgramVisitor",
    sdfg: SDFG,
    state: SDFGState,
    a: str,
    s=None,
    axes=None,
    norm: StringLiteral = StringLiteral("backward"),
):
    return _fftn_core(pv, sdfg, state, a, s, axes, norm, True)
