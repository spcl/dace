# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""
Contains replacements for the Discrete Fourier Transform numpy package (numpy.fft)
"""

from collections.abc import Sequence

import sympy as sp

from dace import SDFG, Memlet, SDFGState, dtypes, symbolic
from dace.frontend.common import op_repository as oprepo
from dace.frontend.python.common import StringLiteral
from dace.frontend.python.replacements.utils import ProgramVisitor


def _real_to_complex(real_type: dtypes.typeclass):
    if real_type == dtypes.float32:
        return dtypes.complex64
    elif real_type == dtypes.float64:
        return dtypes.complex128
    else:
        return real_type


def _fft_core(
    pv: "ProgramVisitor",
    sdfg: SDFG,
    state: SDFGState,
    a: str,
    s: Sequence[symbolic.SymbolicType] | None = None,
    axes: Sequence[int] | None = None,
    norm: StringLiteral = StringLiteral("backward"),
    is_inverse: bool = False,
):
    from dace.libraries.fft.nodes import FFT, IFFT  # Avoid import loops

    if not isinstance(a, str) or a not in sdfg.arrays:
        raise ValueError("Input must be a valid array")

    libnode = FFT("fft") if not is_inverse else IFFT("ifft")

    desc = sdfg.arrays[a]
    ndim = len(desc.shape)
    if axes is None:
        axes = list(range(ndim))
    else:
        axes = [int(axis) for axis in axes]
        if any(not -ndim <= axis < ndim for axis in axes):
            raise ValueError(f"FFT axes {axes} out of bounds for a {ndim}-dimensional array")
        axes = [axis % ndim for axis in axes]
        if len(set(axes)) != len(axes):
            raise ValueError(f"Repeated FFT axes are not supported, got {axes}")
    libnode.axes = axes
    N = 1
    for axis in axes:
        N *= desc.shape[axis]

    # If s (or n) is not None, either pad input or slice and add a view
    if s is not None:
        raise NotImplementedError

    # Compute factor
    if norm == "forward":
        factor = (1 / N) if not is_inverse else 1
    elif norm == "backward":
        factor = 1 if not is_inverse else (1 / N)
    elif norm == "ortho":
        factor = sp.sqrt(1 / N)
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
    return _fft_core(pv, sdfg, state, a, None if n is None else [n], [axis], norm, False)


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
    return _fft_core(pv, sdfg, state, a, None if n is None else [n], [axis], norm, True)


@oprepo.replaces("numpy.fft.fft2")
def _fft2(
    pv: "ProgramVisitor",
    sdfg: SDFG,
    state: SDFGState,
    a: str,
    s=None,
    axes=(-2, -1),
    norm: StringLiteral = StringLiteral("backward"),
):
    return _fft_core(pv, sdfg, state, a, s, axes, norm, False)


@oprepo.replaces("numpy.fft.ifft2")
def _ifft2(
    pv: "ProgramVisitor",
    sdfg: SDFG,
    state: SDFGState,
    a: str,
    s=None,
    axes=(-2, -1),
    norm: StringLiteral = StringLiteral("backward"),
):
    return _fft_core(pv, sdfg, state, a, s, axes, norm, True)


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
    return _fft_core(pv, sdfg, state, a, s, axes, norm, False)


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
    return _fft_core(pv, sdfg, state, a, s, axes, norm, True)
