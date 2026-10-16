# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""
Native Discrete Fourier Transform (DFT) implementations: one-dimensional programs and a separable N-D builder.
"""

import math
from collections.abc import Sequence

import numpy as np
import sympy

import dace
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


class FloatingPrinter(sympy.printing.str.StrPrinter):
    """Prints a factor so C evaluates it in floating point: every symbol and every number is a double.

    ``1/(M*N)`` and ``sqrt(M)`` over integer symbols are integer division and an integer root in C.
    """

    def _print_Symbol(self, expr):
        return f"(1.0 * {expr.name})"

    def _print_Rational(self, expr):
        return f"({expr.p}.0 / {expr.q}.0)"

    def _print_Integer(self, expr):
        # The node's factor property keeps ``1/14`` unevaluated, a ``Pow`` of an integer
        return f"{expr.p}.0"

    def _print_Pow(self, expr, rational=False):
        # ``pow``, not ``**``: the text lands in Python and C++ tasklets alike
        return f"pow({self._print(expr.base)}, {self._print(expr.exp)})"


def floating_text(factor: dace.symbolic.SymbolicType) -> str:
    """``factor`` as tasklet text C evaluates in floating point."""
    return FloatingPrinter().doprint(dace.symbolic.pystr_to_symbolic(factor))


def complex_of(real: dace.typeclass) -> dace.typeclass:
    """The complex type with ``real``'s precision; a complex type is its own."""
    return {dace.float32: dace.complex64, dace.float64: dace.complex128}.get(real, real)


def real_of(complex_type: dace.typeclass) -> dace.typeclass:
    """The real type with ``complex_type``'s precision; a real type is its own."""
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
) -> dace.SDFGState:
    """Append (or start with, for ``after=None``) a state running ``o = <code of i>`` over ``ranges``."""
    state = sdfg.add_state(label) if after is None else sdfg.add_state_after(after, label)
    state.add_mapped_tasklet(
        label,
        ranges,
        {"i": dace.Memlet(data=src, subset=src_index)},
        code,
        {"o": dace.Memlet(data=dst, subset=dst_index)},
        language=language,
        external_edges=True,
    )
    return state


def add_dft_axis_state(
    sdfg: dace.SDFG, after: dace.SDFGState, src: str, dst: str, shape: Sequence, axis: int, inverse: bool, factor
) -> dace.SDFGState:
    """Append a state doing a batched 1-D DFT of ``src`` along ``axis`` into ``dst``, scaled by ``factor``.

    Each output element sums its whole input line in its own tasklet: no accumulation across threads, so no complex
    atomics, which the GPU backends do not provide. All other axes are batch dimensions.
    """
    from dace.codegen.targets.cpp import sym2cpp  # avoid import loop

    ndim = len(shape)
    length = sym2cpp(shape[axis])
    element = sdfg.arrays[dst].dtype
    real = "float" if element == dace.complex64 else "double"
    line = [f"__i{d}" for d in range(ndim)]
    line[axis] = f"0:{dace.symbolic.symstr(shape[axis])}"
    # Divide by a FLOATING denominator: ``1/(M*N)`` over integer symbols is C integer division, i.e. zero. In the
    # element's own precision: ``complex64 * double`` multiplies to zero.
    scale = "" if str(factor) == "1" else f"o *= {real}({floating_text(factor)});"
    # The angle's integer part is reduced modulo the length first, as an FFT's twiddle factors are
    code = f"""
{element.ctype} sum({real}(0), {real}(0));
for (long long __n = 0; __n < {length}; ++__n) {{
    const {real} angle = {real}(2.0 * {math.pi!r}) * {real}((__i{axis} * __n) % {length}) / {real}({length});
    sum += {element.ctype}(cos(angle), {"" if inverse else "-"}sin(angle)) * inp[__n * ({sym2cpp(sdfg.arrays[src].strides[axis])})];
}}
o = sum;
{scale}
"""
    state = sdfg.add_state_after(after, f"dft_axis{axis}")
    state.add_mapped_tasklet(
        "dft",
        index_ranges(shape, "__i"),
        {"inp": dace.Memlet(data=src, subset=", ".join(line))},
        code,
        {"o": dace.Memlet(data=dst, subset=", ".join(f"__i{d}" for d in range(ndim)))},
        language=dace.Language.CPP,
        external_edges=True,
    )
    return state


def add_dft_passes(
    sdfg: dace.SDFG, state: dace.SDFGState, src: str, dst: str, shape: Sequence, axes: list[int], inverse: bool, factor
) -> dace.SDFGState:
    """One batched 1-D DFT per entry of ``axes``, from ``src`` into ``dst``; passes between them alternate
    between ``__ping`` and ``__pong``, and ``factor`` scales the last one."""
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
    """Build a native (library-free) N-D / axis-batched DFT as a nested SDFG.

    Separable: a sequence of batched 1-D DFTs, one per entry of ``axes`` (every axis when ``None``); unlisted axes
    are batch dimensions. The input is cast to the complex type on the way in, so real inputs work. With
    ``half_spectrum`` the complex side keeps only the non-negative frequencies of the last transformed axis:
    the forward transform computes the full spectrum and keeps its first half, the inverse one restores the
    negative frequencies as the conjugates of the positive ones (numpy's ``irfftn``) and keeps the real part.
    ``factor`` carries any normalization.
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
