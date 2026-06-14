# Copyright 2019-2024 ETH Zurich and the DaCe authors. All rights reserved.
"""
One-dimensional Discrete Fourier Transform (DFT) native implementations.
"""
import dace
import numpy as np
import math


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


##################################################################################################
# N-dimensional native DFT (separable, rank-generic)
##################################################################################################


def _normalize_axes(ndim, axis):
    """Resolve the FFT node's ``axis`` property to a concrete transform-axis list.

    ``axis is None`` -> every axis (a true N-D ``fftn``); an integer (possibly
    negative) -> a single batched 1-D transform along that axis.
    """
    if axis is None:
        return list(range(ndim))
    a = int(axis)
    if a < 0:
        a += ndim
    if not 0 <= a < ndim:
        raise ValueError(f'FFT axis {axis} out of range for rank-{ndim} input')
    return [a]


def _add_zero_state(sdfg, after, dst, shape):
    """Append a state that zero-fills the whole ``dst`` array (WCR-accumulate seed)."""
    state = sdfg.add_state_after(after, f'zero_{dst}')
    sub = ', '.join(f'__z{d}' for d in range(len(shape)))
    state.add_mapped_tasklet('zero', {f'__z{d}': f'0:{shape[d]}'
                                      for d in range(len(shape))}, {},
                             'o = 0', {'o': dace.Memlet(data=dst, subset=sub)},
                             external_edges=True)
    return state


def _add_dft_axis_state(sdfg, after, src, dst, shape, ax, inverse, factor):
    """Append a state doing a batched 1-D DFT of ``src`` along ``ax`` into ``dst``.

    ``dst`` must be pre-zeroed (the contributions over the summation index
    ``__n`` are accumulated via WCR).  All other axes are pure batch dims.
    ``factor`` scales the result (applied once, on the final pass).
    """
    ndim = len(shape)
    N = shape[ax]
    rng = {f'__i{d}': f'0:{shape[d]}' for d in range(ndim)}
    rng['__n'] = f'0:{N}'
    out_sub = ', '.join(f'__i{d}' for d in range(ndim))
    in_idx = [f'__i{d}' for d in range(ndim)]
    in_idx[ax] = '__n'
    in_sub = ', '.join(in_idx)
    isign = '+' if inverse else '-'  # idft uses exp(+i...), fwd uses exp(-i...)
    fac = '' if str(factor) == '1' else f' * ({factor})'
    code = (f'exponent = (2.0 * {math.pi!r} / {dace.symbolic.symstr(N)}) * __i{ax} * __n\n'
            f'o = decltype(o)(math.cos(exponent), {isign}math.sin(exponent)) * inp{fac}')
    state = sdfg.add_state_after(after, f'dft_ax{ax}')
    state.add_mapped_tasklet('dft', rng, {'inp': dace.Memlet(data=src, subset=in_sub)},
                             code,
                             {'o': dace.Memlet(data=dst, subset=out_sub, wcr='lambda a, b: a + b')},
                             external_edges=True)
    return state


def dft_nd_sdfg(indesc, outdesc, factor, inverse, axis, name='dft_nd'):
    """Build a native (library-free) N-D / axis-batched DFT as a nested SDFG.

    Separable: an ``fftn`` (``axis is None``) is a sequence of batched 1-D
    DFTs, one per axis, ping-ponging through two transients; an ``axis``-set
    node is a single batched 1-D DFT.  The input is cast to the (complex)
    output dtype on the way in, so real->complex inputs work as in the 1-D
    path.  Matches ``np.fft.fftn`` / ``np.fft.fft(x, axis=k)`` (forward) and
    their unnormalised inverses (``factor`` carries any normalisation).
    """
    ndim = len(indesc.shape)
    axes = _normalize_axes(ndim, axis)
    shape = list(outdesc.shape)
    ct = outdesc.dtype  # complex output type
    sdfg = dace.SDFG(name)
    sdfg.add_array('_inp', indesc.shape, indesc.dtype, storage=indesc.storage)
    sdfg.add_array('_out', outdesc.shape, outdesc.dtype, storage=outdesc.storage)
    sdfg.add_transient('__buf0', shape, ct, storage=outdesc.storage)
    if len(axes) >= 2:
        sdfg.add_transient('__buf1', shape, ct, storage=outdesc.storage)

    # Cast/copy _inp -> __buf0 (decouples from any in-place _inp == _out alias).
    s = sdfg.add_state('copy_in')
    sub = ', '.join(f'__c{d}' for d in range(ndim))
    s.add_mapped_tasklet('cast_in', {f'__c{d}': f'0:{shape[d]}'
                                     for d in range(ndim)}, {'i': dace.Memlet(data='_inp', subset=sub)},
                         'o = decltype(o)(i)', {'o': dace.Memlet(data='__buf0', subset=sub)},
                         external_edges=True)

    src = '__buf0'
    for ai, ax in enumerate(axes):
        last = (ai == len(axes) - 1)
        dst = '_out' if last else ('__buf1' if src == '__buf0' else '__buf0')
        s = _add_zero_state(sdfg, s, dst, shape)
        s = _add_dft_axis_state(sdfg, s, src, dst, shape, ax, inverse, factor if last else 1)
        src = dst
    return sdfg
