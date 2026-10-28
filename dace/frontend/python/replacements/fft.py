# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""
Contains replacements for the Discrete Fourier Transform numpy package (numpy.fft)
"""

from collections.abc import Sequence

import sympy as sp

from dace import SDFG, Memlet, SDFGState, data, dtypes, nodes, symbolic
from dace.codegen.common import sym2cpp
from dace.frontend.common import op_repository as oprepo
from dace.frontend.python.common import StringLiteral
from dace.frontend.python.replacements.array_manipulation import _numpy_roll as numpy_roll
from dace.frontend.python.replacements.utils import ProgramVisitor

BACKWARD = StringLiteral("backward")


def as_list(value) -> list | None:
    """``None``, a sequence, or one value as a one-element list."""
    if value is None:
        return None
    return list(value) if isinstance(value, (list, tuple)) else [value]


def normalization(norm: StringLiteral, size: symbolic.SymbolicType, inverse: bool) -> symbolic.SymbolicType:
    """The factor numpy scales a transform of ``size`` elements by under ``norm``."""
    if norm == "forward":
        return 1 if inverse else 1 / size
    if norm == "backward":
        return 1 / size if inverse else 1
    if norm == "ortho":
        return sp.sqrt(1 / size)
    raise ValueError('norm argument can only be one of "forward", "backward", or "ortho".')


def resized(pv: ProgramVisitor, sdfg: SDFG, state: SDFGState, a: str, shape: list) -> nodes.AccessNode:
    """A copy of ``a`` cropped or zero-padded to ``shape``, as numpy's ``n`` and ``s`` arguments ask; the access
    node the copy writes, which the transform reads so dataflow orders the two."""
    desc = sdfg.arrays[a]
    name = pv.add_temp_transient(shape, desc.dtype, storage=desc.storage)[0]
    index = ", ".join(f"__i{d}" for d in range(len(shape)))
    padded = [d for d, (new, old) in enumerate(zip(shape, desc.shape)) if symbolic.equal(new, old) is not True]
    inside = " and ".join(f"__i{d} < {symbolic.symstr(desc.shape[d])}" for d in padded)
    exit_node = state.add_mapped_tasklet(
        "resize",
        {f"__i{d}": f"0:{symbolic.symstr(extent)}" for d, extent in enumerate(shape)},
        {"i": Memlet.from_array(a, desc)},
        f"o = i[{index}] if {inside} else 0" if inside else f"o = i[{index}]",
        {"o": Memlet(data=name, subset=index)},
        external_edges=True,
    )[2]
    return state.out_edges(exit_node)[0].dst


def add_transform(
    pv: ProgramVisitor,
    sdfg: SDFG,
    state: SDFGState,
    a: str,
    axes: Sequence[int] | None,
    lengths: Sequence | None,
    norm: StringLiteral,
    inverse: bool,
    half_spectrum: bool,
) -> str:
    """``a`` transformed over ``axes`` (every axis for ``None``) at the transform ``lengths`` (``None`` keeps the
    extents), as one FFT or IFFT library node; numpy's ``fft``, ``rfft``, ``fftn`` and their inverses.

    The input is cropped or zero-padded to ``lengths`` first. ``half_spectrum`` keeps only the non-negative
    frequencies of the last transformed axis (``rfft``): a real input to a forward transform, a real output of an
    inverse one, whose last length defaults to ``2 * (m - 1)`` for ``m`` stored frequencies.
    """
    from dace.libraries.fft.algorithms.dft import complex_of, real_of  # Avoid import loops
    from dace.libraries.fft.nodes import FFT, IFFT
    from dace.libraries.fft.nodes.fft import normalize_fft_axes

    if not isinstance(a, str) or a not in sdfg.arrays:
        raise ValueError("Input must be a valid array")
    desc = sdfg.arrays[a]
    rank = len(desc.shape)
    transformed = normalize_fft_axes(rank, axes)
    if not transformed:
        raise NotImplementedError("A Fourier transform over no axes (a copy) is not supported")
    if half_spectrum and not inverse and desc.dtype in (dtypes.complex64, dtypes.complex128):
        raise TypeError(f"numpy.fft.rfft takes a real input, got {desc.dtype}")
    if lengths is None:
        lengths = [desc.shape[axis] for axis in transformed]
        if half_spectrum and inverse:
            lengths[-1] = 2 * (desc.shape[transformed[-1]] - 1)
    elif len(lengths) != len(transformed):
        raise ValueError(f"{len(lengths)} transform lengths for {len(transformed)} axes")
    lengths = [symbolic.pystr_to_symbolic(str(n)) for n in lengths]
    stored = sp.floor(lengths[-1] / 2) + 1

    in_shape, out_shape = list(desc.shape), list(desc.shape)
    for axis, length in zip(transformed, lengths):
        in_shape[axis] = out_shape[axis] = length
    if half_spectrum:
        (in_shape if inverse else out_shape)[transformed[-1]] = stored
    source = None
    if any(symbolic.equal(new, old) is not True for new, old in zip(in_shape, desc.shape)):
        source = resized(pv, sdfg, state, a, in_shape)
        a, desc = source.data, sdfg.arrays[source.data]

    complex_type = complex_of(desc.dtype)
    out_type = real_of(complex_type) if half_spectrum and inverse else complex_type
    every_axis = transformed == list(range(rank)) or (not half_spectrum and sorted(transformed) == list(range(rank)))
    node = (IFFT if inverse else FFT)(
        "ifft" if inverse else "fft", axes=None if every_axis else transformed, half_spectrum=half_spectrum
    )
    size = 1
    for length in lengths:
        size = size * length
    node.factor = normalization(norm, size, inverse)

    # The output keeps the input's layout where the shapes agree, so a vendor plans the two alike. NOT for a
    # View: cloning its descriptor makes the output a View that views nothing.
    if out_shape == list(desc.shape) and not isinstance(desc, data.View):
        name, odesc = sdfg.add_temp_transient_like(desc, out_type, name=pv.get_target_name())
    else:
        name, odesc = sdfg.add_transient(
            pv.get_target_name(), out_shape, out_type, storage=desc.storage, find_new_name=True
        )
    state.add_edge(source or state.add_read(a), None, node, "_inp", Memlet.from_array(a, desc))
    state.add_edge(node, "_out", state.add_write(name), None, Memlet.from_array(name, odesc))
    return name


def transform_axes(a_rank: int, s: Sequence | None, axes: Sequence[int] | None) -> list[int] | None:
    """numpy's ``fftn`` axes: ``axes``, else the last ``len(s)`` axes, else every axis."""
    if axes is None and s is not None:
        return list(range(a_rank - len(s), a_rank))
    return axes


def add_nd_transform(
    pv: ProgramVisitor,
    sdfg: SDFG,
    state: SDFGState,
    a: str,
    s,
    axes,
    norm: StringLiteral,
    inverse: bool,
    half_spectrum: bool,
) -> str:
    if not isinstance(a, str) or a not in sdfg.arrays:
        raise ValueError("Input must be a valid array")
    s, axes = as_list(s), as_list(axes)
    return add_transform(
        pv, sdfg, state, a, transform_axes(len(sdfg.arrays[a].shape), s, axes), s, norm, inverse, half_spectrum
    )


def add_1d_transform(
    pv: ProgramVisitor,
    sdfg: SDFG,
    state: SDFGState,
    a: str,
    n,
    axis: int,
    norm: StringLiteral,
    inverse: bool,
    half_spectrum: bool,
) -> str:
    return add_transform(pv, sdfg, state, a, [axis], None if n is None else [n], norm, inverse, half_spectrum)


@oprepo.replaces("numpy.fft.fft")
def numpy_fft(pv: ProgramVisitor, sdfg: SDFG, state: SDFGState, a: str, n=None, axis=-1, norm=BACKWARD):
    return add_1d_transform(pv, sdfg, state, a, n, axis, norm, inverse=False, half_spectrum=False)


@oprepo.replaces("numpy.fft.ifft")
def numpy_ifft(pv: ProgramVisitor, sdfg: SDFG, state: SDFGState, a: str, n=None, axis=-1, norm=BACKWARD):
    return add_1d_transform(pv, sdfg, state, a, n, axis, norm, inverse=True, half_spectrum=False)


@oprepo.replaces("numpy.fft.rfft")
def numpy_rfft(pv: ProgramVisitor, sdfg: SDFG, state: SDFGState, a: str, n=None, axis=-1, norm=BACKWARD):
    return add_1d_transform(pv, sdfg, state, a, n, axis, norm, inverse=False, half_spectrum=True)


@oprepo.replaces("numpy.fft.irfft")
def numpy_irfft(pv: ProgramVisitor, sdfg: SDFG, state: SDFGState, a: str, n=None, axis=-1, norm=BACKWARD):
    return add_1d_transform(pv, sdfg, state, a, n, axis, norm, inverse=True, half_spectrum=True)


@oprepo.replaces("numpy.fft.fftn")
def numpy_fftn(pv: ProgramVisitor, sdfg: SDFG, state: SDFGState, a: str, s=None, axes=None, norm=BACKWARD):
    return add_nd_transform(pv, sdfg, state, a, s, axes, norm, inverse=False, half_spectrum=False)


@oprepo.replaces("numpy.fft.ifftn")
def numpy_ifftn(pv: ProgramVisitor, sdfg: SDFG, state: SDFGState, a: str, s=None, axes=None, norm=BACKWARD):
    return add_nd_transform(pv, sdfg, state, a, s, axes, norm, inverse=True, half_spectrum=False)


@oprepo.replaces("numpy.fft.rfftn")
def numpy_rfftn(pv: ProgramVisitor, sdfg: SDFG, state: SDFGState, a: str, s=None, axes=None, norm=BACKWARD):
    return add_nd_transform(pv, sdfg, state, a, s, axes, norm, inverse=False, half_spectrum=True)


@oprepo.replaces("numpy.fft.irfftn")
def numpy_irfftn(pv: ProgramVisitor, sdfg: SDFG, state: SDFGState, a: str, s=None, axes=None, norm=BACKWARD):
    return add_nd_transform(pv, sdfg, state, a, s, axes, norm, inverse=True, half_spectrum=True)


@oprepo.replaces("numpy.fft.fft2")
def numpy_fft2(pv: ProgramVisitor, sdfg: SDFG, state: SDFGState, a: str, s=None, axes=(-2, -1), norm=BACKWARD):
    return add_nd_transform(pv, sdfg, state, a, s, axes, norm, inverse=False, half_spectrum=False)


@oprepo.replaces("numpy.fft.ifft2")
def numpy_ifft2(pv: ProgramVisitor, sdfg: SDFG, state: SDFGState, a: str, s=None, axes=(-2, -1), norm=BACKWARD):
    return add_nd_transform(pv, sdfg, state, a, s, axes, norm, inverse=True, half_spectrum=False)


@oprepo.replaces("numpy.fft.rfft2")
def numpy_rfft2(pv: ProgramVisitor, sdfg: SDFG, state: SDFGState, a: str, s=None, axes=(-2, -1), norm=BACKWARD):
    return add_nd_transform(pv, sdfg, state, a, s, axes, norm, inverse=False, half_spectrum=True)


@oprepo.replaces("numpy.fft.irfft2")
def numpy_irfft2(pv: ProgramVisitor, sdfg: SDFG, state: SDFGState, a: str, s=None, axes=(-2, -1), norm=BACKWARD):
    return add_nd_transform(pv, sdfg, state, a, s, axes, norm, inverse=True, half_spectrum=True)


def shift(pv: ProgramVisitor, sdfg: SDFG, state: SDFGState, x: str, axes, inverse: bool) -> str:
    """``numpy.fft.fftshift`` (``ifftshift``): a roll by half of each axis's extent, rounded down (up)."""
    desc = sdfg.arrays[x]
    axes = list(range(len(desc.shape))) if axes is None else as_list(axes)
    halves = [sp.floor(desc.shape[axis] / 2) for axis in axes]
    return numpy_roll(pv, sdfg, state, x, [-half if inverse else half for half in halves], axes)


@oprepo.replaces("numpy.fft.fftshift")
def numpy_fftshift(pv: ProgramVisitor, sdfg: SDFG, state: SDFGState, x: str, axes=None):
    return shift(pv, sdfg, state, x, axes, inverse=False)


@oprepo.replaces("numpy.fft.ifftshift")
def numpy_ifftshift(pv: ProgramVisitor, sdfg: SDFG, state: SDFGState, x: str, axes=None):
    return shift(pv, sdfg, state, x, axes, inverse=True)


def frequencies(pv: ProgramVisitor, sdfg: SDFG, state: SDFGState, n, d, half_spectrum: bool) -> str:
    """``numpy.fft.fftfreq`` (``rfftfreq``): the sample frequencies of an ``n``-point transform at spacing ``d``."""
    if isinstance(d, str):
        raise NotImplementedError("numpy.fft.fftfreq with a spacing computed at runtime")
    n = symbolic.pystr_to_symbolic(str(n))
    count = sp.floor(n / 2) + 1 if half_spectrum else n
    name = pv.add_temp_transient([count], dtypes.float64)[0]
    total = f"({float(d)!r} * {sym2cpp(n)})"
    index = "__i" if half_spectrum else f"(__i < ({sym2cpp(n)} + 1) / 2 ? __i : __i - {sym2cpp(n)})"
    state.add_mapped_tasklet(
        "frequencies",
        {"__i": f"0:{symbolic.symstr(count)}"},
        {},
        f"o = {index} / {total};",
        {"o": Memlet(data=name, subset="__i")},
        language=dtypes.Language.CPP,
        external_edges=True,
    )
    return name


@oprepo.replaces("numpy.fft.fftfreq")
def numpy_fftfreq(pv: ProgramVisitor, sdfg: SDFG, state: SDFGState, n, d=1.0):
    return frequencies(pv, sdfg, state, n, d, half_spectrum=False)


@oprepo.replaces("numpy.fft.rfftfreq")
def numpy_rfftfreq(pv: ProgramVisitor, sdfg: SDFG, state: SDFGState, n, d=1.0):
    return frequencies(pv, sdfg, state, n, d, half_spectrum=True)
