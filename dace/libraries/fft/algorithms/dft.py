# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""
Native Discrete Fourier Transform (DFT) implementations: one-dimensional programs and a separable N-D builder.
"""

import math
from collections.abc import Sequence

import numpy as np
import sympy

import dace
from dace.codegen.common import sym2cpp
from dace.libraries.fft.nodes.fft import normalize_fft_axes


# Native, naive version of the Discrete Fourier Transform
@dace.program
def dft(_inp, _out, N: dace.compiletime, factor: dace.compiletime):
    i = np.arange(N)
    e = np.exp(-2j * np.pi * i * i[:, None] / N)
    _out[:] = factor * (e @ _inp.astype(dace.complex128))


@dace.program
def idft(_inp, _out, N: dace.compiletime, factor: dace.compiletime):
    i = np.arange(N)
    e = np.exp(2j * np.pi * i * i[:, None] / N)
    _out[:] = factor * (e @ _inp.astype(dace.complex128))


# Single-map version of DFT, useful for integrating small Fourier transforms into other operations
@dace.program
def dft_explicit(_inp, _out, N: dace.compiletime, factor: dace.compiletime):
    _out[:] = 0
    for i, n in dace.map[0:N, 0:N]:
        with dace.tasklet:
            inp << _inp[n]
            exponent = 2 * math.pi * i * n / N
            b = decltype(b)(math.cos(exponent), -math.sin(exponent)) * inp * factor
            b >> _out(1, lambda a, b: a + b)[i]


@dace.program
def idft_explicit(_inp, _out, N: dace.compiletime, factor: dace.compiletime):
    _out[:] = 0
    for i, n in dace.map[0:N, 0:N]:
        with dace.tasklet:
            inp << _inp[n]
            exponent = 2 * math.pi * i * n / N
            b = decltype(b)(math.cos(exponent), math.sin(exponent)) * inp * factor
            b >> _out(1, lambda a, b: a + b)[i]


# N-dimensional native DFT (separable, rank-generic)


def in_floating_point(factor: dace.symbolic.SymbolicType) -> sympy.Basic:
    """``factor`` with every symbol cast to ``float64`` and every number a float, so ``1/(M*N)`` and
    ``sqrt(M)`` over integer symbols print as floating-point operations.

    :param factor: The factor.
    :return: The floating-point expression.
    """
    value = dace.symbolic.pystr_to_symbolic(factor)
    return value.xreplace({s: dace.symbolic.float64(s) for s in value.free_symbols}).evalf()


def is_unit(factor: dace.symbolic.SymbolicType) -> bool:
    """Whether ``factor`` is one, as ``1`` or ``1.0``.

    :param factor: The factor.
    :return: True if the factor is one.
    """
    value = dace.symbolic.pystr_to_symbolic(factor)
    return bool(value.is_Number) and float(value) == 1.0


def complex_of(element: dace.typeclass) -> dace.typeclass:
    """The complex type numpy's FFTs compute ``element`` in.

    :param element: The input element type.
    :return: ``complex64`` for half and single precision, the type itself if complex, else ``complex128``.
    """
    if element in (dace.complex64, dace.complex128):
        return element
    return dace.complex64 if element in (dace.float16, dace.float32) else dace.complex128


def real_of(complex_type: dace.typeclass) -> dace.typeclass:
    """The real type of ``complex_type``'s precision; a real type is its own.

    :param complex_type: The complex type.
    :return: The real type.
    """
    return {dace.complex64: dace.float32, dace.complex128: dace.float64}.get(complex_type, complex_type)


def index_ranges(shape: Sequence, prefix: str) -> dict[str, str]:
    return {f"{prefix}{d}": f"0:{dace.symbolic.symstr(extent)}" for d, extent in enumerate(shape)}


def add_elementwise_state(
    sdfg: dace.SDFG,
    after: dace.SDFGState | None,
    label: str,
    src: str,
    dst: str,
    ranges: dict[str, str],
    src_index: str,
    dst_index: str,
    code: str,
    language: dace.Language = dace.Language.Python,
    schedule: dace.ScheduleType = dace.ScheduleType.Default,
) -> dace.SDFGState:
    """Appends a state running ``code`` (``o`` of ``i``) over ``ranges``.

    :param after: The state to append after, ``None`` to start the SDFG.
    :param src: The input array (connector ``i``).
    :param dst: The output array (connector ``o``).
    :param ranges: The map ranges.
    :param src_index: The input subset.
    :param dst_index: The output subset.
    :param code: The tasklet code.
    :return: The new state.
    """
    state = sdfg.add_state(label) if after is None else sdfg.add_state_after(after, label)
    state.add_mapped_tasklet(
        label,
        ranges,
        {"i": dace.Memlet(data=src, subset=src_index)},
        code,
        {"o": dace.Memlet(data=dst, subset=dst_index)},
        language=language,
        schedule=schedule,
        external_edges=True,
    )
    return state


def add_twiddle_state(
    sdfg: dace.SDFG,
    after: dace.SDFGState,
    length,
    axis: int,
    element: dace.typeclass,
    inverse: bool,
    storage: dace.StorageType,
) -> tuple[str, dace.SDFGState]:
    """Appends a state filling the ``length`` twiddle factors ``exp(-+2 pi i m / length)``, so a DFT
    pass evaluates ``length`` sines and cosines, not two per term.

    :param length: The transform length.
    :param axis: The transformed axis, which names the table.
    :param element: The complex element type.
    :param inverse: The sign of the exponent.
    :param storage: The storage of the table.
    :return: The table's name and the new state.
    """

    name = sdfg.add_transient(f"__twiddle{axis}", [length], element, storage=storage, find_new_name=True)[0]
    real = "float" if element == dace.complex64 else "double"
    text = sym2cpp(length)
    state = sdfg.add_state_after(after, f"twiddle_axis{axis}")
    state.add_mapped_tasklet(
        "twiddle",
        {"__m": f"0:{dace.symbolic.symstr(length)}"},
        {},
        f"const double angle = {2.0 * math.pi!r} * double(__m) / double({text});\n"
        f"o = {element.ctype}({real}(cos(angle)), {real}({'' if inverse else '-'}sin(angle)));",
        {"o": dace.Memlet(data=name, subset="__m")},
        language=dace.Language.CPP,
        external_edges=True,
    )
    return name, state


def add_dft_axis_state(
    sdfg: dace.SDFG, after: dace.SDFGState, src: str, dst: str, shape: Sequence, axis: int, inverse: bool, factor
) -> dace.SDFGState:
    """Appends a batched 1-D DFT of ``src`` along ``axis`` into ``dst``, scaled by ``factor``.

    Each output element sums its input line in its own tasklet, so no complex atomics are needed.

    :param shape: The shape of both arrays.
    :param axis: The transformed axis; the others are batch dimensions.
    :param inverse: Whether the transform is the inverse one.
    :param factor: The scaling factor.
    :return: The new state.
    """

    ndim = len(shape)
    length = sym2cpp(shape[axis])
    element = sdfg.arrays[dst].dtype
    real = "float" if element == dace.complex64 else "double"
    line = [f"__i{d}" for d in range(ndim)]
    line[axis] = f"0:{dace.symbolic.symstr(shape[axis])}"
    # Divide by a FLOATING denominator: ``1/(M*N)`` over integer symbols is C integer division, i.e. zero. In the
    # element's own precision: ``complex64 * double`` multiplies to zero.
    scale = "" if is_unit(factor) else f"o *= {real}({sym2cpp(in_floating_point(factor))});"
    twiddles, after = add_twiddle_state(sdfg, after, shape[axis], axis, element, inverse, sdfg.arrays[dst].storage)
    # The exponent is reduced modulo the length, as an FFT's twiddle factors are: one table entry per residue
    code = f"""
{element.ctype} sum({real}(0), {real}(0));
for (long long __n = 0; __n < {length}; ++__n) {{
    sum += tw[(__i{axis} * __n) % {length}] * inp[__n * ({sym2cpp(sdfg.arrays[src].strides[axis])})];
}}
o = sum;
{scale}
"""
    state = sdfg.add_state_after(after, f"dft_axis{axis}")
    state.add_mapped_tasklet(
        "dft",
        index_ranges(shape, "__i"),
        {
            "inp": dace.Memlet(data=src, subset=", ".join(line)),
            "tw": dace.Memlet.from_array(twiddles, sdfg.arrays[twiddles]),
        },
        code,
        {"o": dace.Memlet(data=dst, subset=", ".join(f"__i{d}" for d in range(ndim)))},
        language=dace.Language.CPP,
        external_edges=True,
    )
    return state


def add_dft_passes(
    sdfg: dace.SDFG, state: dace.SDFGState, src: str, dst: str, shape: Sequence, axes: list[int], inverse: bool, factor
) -> dace.SDFGState:
    """Appends one batched 1-D DFT per axis, alternating between ``__ping`` and ``__pong``.

    :param axes: The transformed axes, in order.
    :param factor: The scaling factor, applied by the last pass.
    :return: The last state.
    """
    current = src
    for position, axis in enumerate(axes):
        last = position == len(axes) - 1
        target = dst if last else ("__pong" if current == "__ping" else "__ping")
        if target not in sdfg.arrays:
            sdfg.add_transient(target, shape, sdfg.arrays[current].dtype, storage=sdfg.arrays[current].storage)
        state = add_dft_axis_state(sdfg, state, current, target, shape, axis, inverse, factor if last else 1)
        current = target
    return state


def dft_nd_sdfg(indesc, outdesc, factor, inverse, axes, half_spectrum=False, name="dft_nd"):
    """A library-free N-D DFT: one batched 1-D DFT per transformed axis; the other axes are batch dimensions.

    A real input is cast to complex. With ``half_spectrum`` the forward transform keeps the first half of the spectrum,
    and the inverse one restores the negative frequencies as conjugates and keeps the real part (``irfftn``).

    :param indesc: The input descriptor.
    :param outdesc: The output descriptor.
    :param factor: The scaling factor.
    :param inverse: Whether the transform is the inverse one.
    :param axes: The transformed axes, ``None`` for all.
    :param half_spectrum: Keep only the non-negative frequencies of the last axis.
    :param name: The SDFG name.
    :return: The SDFG.
    """
    ndim = len(indesc.shape)
    axes = normalize_fft_axes(ndim, axes)
    if not axes:
        raise NotImplementedError("A DFT over no axes is not supported")
    complex_type = complex_of(outdesc.dtype)
    sdfg = dace.SDFG(name)
    # The caller's strides: a column-major reshape view steps its axes differently from a C-order buffer.
    sdfg.add_array(
        "_inp", indesc.shape, indesc.dtype, storage=indesc.storage, strides=indesc.strides, offset=indesc.offset
    )
    sdfg.add_array(
        "_out", outdesc.shape, outdesc.dtype, storage=outdesc.storage, strides=outdesc.strides, offset=outdesc.offset
    )
    storage = outdesc.storage
    in_index = ", ".join(f"__c{d}" for d in range(ndim))
    # Copy _inp into the complex scratch (decoupled from any in-place _inp == _out alias). The cast names the
    # type: the write is inlined, so ``decltype(o)`` would become a reference to the target element.
    sdfg.add_transient("__ping", indesc.shape, complex_type, storage=storage)
    state = add_elementwise_state(
        sdfg,
        None,
        "cast_in",
        "_inp",
        "__ping",
        index_ranges(indesc.shape, "__c"),
        in_index,
        in_index,
        f"o = dace.{complex_type.to_string()}(i)",
    )
    if not half_spectrum:
        add_dft_passes(sdfg, state, "__ping", "_out", outdesc.shape, axes, inverse, factor)
        return sdfg
    if not inverse:
        sdfg.add_transient("__spectrum", indesc.shape, complex_type, storage=storage)
        state = add_dft_passes(sdfg, state, "__ping", "__spectrum", indesc.shape, axes, inverse, factor)
        out_index = ", ".join(f"__c{d}" for d in range(ndim))
        add_elementwise_state(
            sdfg,
            state,
            "keep_half",
            "__spectrum",
            "_out",
            index_ranges(outdesc.shape, "__c"),
            out_index,
            out_index,
            "o = i",
        )
        return sdfg
    # Inverse: the axes before the last one are complex transforms over the half spectrum.
    last = axes[-1]
    if len(axes) > 1:
        sdfg.add_transient("__half", indesc.shape, complex_type, storage=storage)
        state = add_dft_passes(sdfg, state, "__ping", "__half", indesc.shape, axes[:-1], inverse, 1)
        half = "__half"
    else:
        half = "__ping"
    sdfg.add_transient("__spectrum", outdesc.shape, complex_type, storage=storage)
    length = outdesc.shape[last]
    kept = indesc.shape[last]
    ranges = index_ranges(outdesc.shape, "__c")
    index = [f"__c{d}" for d in range(ndim)]
    ranges[f"__c{last}"] = f"0:{kept}"
    state = add_elementwise_state(
        sdfg, state, "positive_frequencies", half, "__spectrum", ranges, ", ".join(index), ", ".join(index), "o = i"
    )
    ranges[f"__c{last}"] = f"{kept}:{length}"
    mirrored = list(index)
    mirrored[last] = f"{dace.symbolic.symstr(length)} - __c{last}"
    state = add_elementwise_state(
        sdfg,
        state,
        "negative_frequencies",
        half,
        "__spectrum",
        ranges,
        ", ".join(mirrored),
        ", ".join(index),
        f"o = {complex_type.ctype}(i.real(), -i.imag());",
        dace.Language.CPP,
    )
    sdfg.add_transient("__signal", outdesc.shape, complex_type, storage=storage)
    state = add_dft_passes(sdfg, state, "__spectrum", "__signal", outdesc.shape, [last], inverse, factor)
    add_elementwise_state(
        sdfg,
        state,
        "real_part",
        "__signal",
        "_out",
        index_ranges(outdesc.shape, "__c"),
        ", ".join(index),
        ", ".join(index),
        "o = i.real();",
        dace.Language.CPP,
    )
    return sdfg
