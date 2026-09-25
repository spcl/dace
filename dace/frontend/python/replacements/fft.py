# Copyright 2019-2025 ETH Zurich and the DaCe authors. All rights reserved.
"""
Contains replacements for the Discrete Fourier Transform numpy package (numpy.fft)
"""
from dace.frontend.common import op_repository as oprepo
from dace.frontend.python.common import StringLiteral
from dace.frontend.python.replacements.utils import ProgramVisitor
from dace import data, dtypes, symbolic, Memlet, SDFG, SDFGState

from typing import Optional

import sympy as sp


def _real_to_complex(real_type: dtypes.typeclass):
    if real_type == dtypes.float32:
        return dtypes.complex64
    elif real_type == dtypes.float64:
        return dtypes.complex128
    else:
        return real_type


def _fft_core(pv: 'ProgramVisitor',
              sdfg: SDFG,
              state: SDFGState,
              a: str,
              n: Optional[symbolic.SymbolicType] = None,
              axis=-1,
              norm: StringLiteral = StringLiteral('backward'),
              is_inverse: bool = False):
    """Replacement for ``numpy.fft.fft`` / ``numpy.fft.ifft``.

    For 1-D input the lib node operates on the whole array.  For
    multi-dim input we set ``axis`` on the lib node so the expansion
    runs a batched 1-D FFT along that axis -- matching numpy's
    "transform along the last axis, batch the rest" semantics.
    """
    from dace.libraries.fft.nodes import FFT, IFFT  # Avoid import loops
    from dace.libraries.fft.nodes.fft import normalize_fft_axes
    if not isinstance(a, str) or a not in sdfg.arrays:
        raise ValueError('Input must be a valid array')

    desc = sdfg.arrays[a]
    ndim = len(desc.shape)
    axis_norm = normalize_fft_axes(ndim, [-1 if axis is None else axis])[0]

    libnode = FFT('fft') if not is_inverse else IFFT('ifft')
    # A 1-D input is a whole-array transform; otherwise the other axes are batch dimensions.
    if ndim > 1:
        libnode.axes = [axis_norm]

    N = desc.shape[axis_norm]

    # If n is not None, either pad input or slice and add a view
    if n is not None:
        raise NotImplementedError

    # Compute factor
    if norm == 'forward':
        factor = (1 / N) if not is_inverse else 1
    elif norm == 'backward':
        factor = 1 if not is_inverse else (1 / N)
    elif norm == 'ortho':
        factor = sp.sqrt(1 / N)
    else:
        raise ValueError('norm argument can only be one of "forward", "backward", or "ortho".')
    libnode.factor = factor

    # Compute output type from input type
    if is_inverse and desc.dtype not in (dtypes.complex64, dtypes.complex128):
        raise TypeError(f'Inverse FFT only accepts complex inputs, got {desc.dtype}')
    dtype = _real_to_complex(desc.dtype)

    # NOT add_temp_transient_like when the input is a View: that clones the descriptor, so an fft
    # of a reshape or a slice produced a View output with nothing viewing it -- an invalid node
    # ("Ambiguous or invalid edge to/from a View access node"). The transform writes a fresh dense
    # buffer whatever it read, so the output is a plain Array with that shape.
    if isinstance(desc, data.View):
        name, odesc = sdfg.add_transient(pv.get_target_name(),
                                         desc.shape,
                                         dtype,
                                         storage=desc.storage,
                                         find_new_name=True)
    else:
        name, odesc = sdfg.add_temp_transient_like(desc, dtype, name=pv.get_target_name())
    r = state.add_read(a)
    w = state.add_write(name)
    state.add_edge(r, None, libnode, '_inp', Memlet.from_array(a, desc))
    state.add_edge(libnode, '_out', w, None, Memlet.from_array(name, odesc))

    return name


@oprepo.replaces('numpy.fft.fft')
def _fft(pv: 'ProgramVisitor',
         sdfg: SDFG,
         state: SDFGState,
         a: str,
         n: Optional[symbolic.SymbolicType] = None,
         axis=-1,
         norm: StringLiteral = StringLiteral('backward')):
    return _fft_core(pv, sdfg, state, a, n, axis, norm, False)


@oprepo.replaces('numpy.fft.ifft')
def _ifft(pv: 'ProgramVisitor',
          sdfg: SDFG,
          state: SDFGState,
          a,
          n=None,
          axis=-1,
          norm: StringLiteral = StringLiteral('backward')):
    return _fft_core(pv, sdfg, state, a, n, axis, norm, True)


def _fftn_core(pv: 'ProgramVisitor',
               sdfg: SDFG,
               state: SDFGState,
               a: str,
               s=None,
               axes=None,
               norm: StringLiteral = StringLiteral('backward'),
               is_inverse: bool = False):
    """N-D FFT over ``axes`` (every axis when ``None``), matching ``numpy.fft.fftn``.

    Unlisted axes are batch dimensions and a repeated axis is transformed once per occurrence. A transform
    over every axis, in any order, leaves the lib node's ``axes`` unset. ``s`` (padding/cropping) and an
    empty ``axes`` are not supported.
    """
    from dace.libraries.fft.nodes import FFT, IFFT  # avoid import loop
    from dace.libraries.fft.nodes.fft import normalize_fft_axes

    if not isinstance(a, str) or a not in sdfg.arrays:
        raise ValueError('Input must be a valid array')
    if s is not None:
        raise NotImplementedError('numpy.fft.fftn ``s`` (padding/cropping) is not yet supported')

    desc = sdfg.arrays[a]
    ndim = len(desc.shape)
    transformed = normalize_fft_axes(ndim, axes)
    if not transformed:
        raise NotImplementedError('numpy.fft.fftn over no axes (a copy) is not supported')

    libnode = FFT('fft') if not is_inverse else IFFT('ifft')
    if sorted(transformed) != list(range(ndim)):
        libnode.axes = transformed

    # numpy normalises by the product of the transformed extents.
    total = 1
    for axis in transformed:
        total = total * desc.shape[axis]
    if norm == 'forward':
        factor = (1 / total) if not is_inverse else 1
    elif norm == 'backward':
        factor = 1 if not is_inverse else (1 / total)
    elif norm == 'ortho':
        factor = sp.sqrt(1 / total)
    else:
        raise ValueError('norm argument can only be one of "forward", "backward", or "ortho".')
    libnode.factor = factor

    if is_inverse and desc.dtype not in (dtypes.complex64, dtypes.complex128):
        raise TypeError(f'Inverse FFT only accepts complex inputs, got {desc.dtype}')
    dtype = _real_to_complex(desc.dtype)

    # See _fft_core: an fft of a View must not clone it, or the output is a View nothing views.
    if isinstance(desc, data.View):
        name, odesc = sdfg.add_transient(pv.get_target_name(),
                                         desc.shape,
                                         dtype,
                                         storage=desc.storage,
                                         find_new_name=True)
    else:
        name, odesc = sdfg.add_temp_transient_like(desc, dtype, name=pv.get_target_name())
    r = state.add_read(a)
    w = state.add_write(name)
    state.add_edge(r, None, libnode, '_inp', Memlet.from_array(a, desc))
    state.add_edge(libnode, '_out', w, None, Memlet.from_array(name, odesc))
    return name


@oprepo.replaces('numpy.fft.fftn')
def _fftn(pv: 'ProgramVisitor',
          sdfg: SDFG,
          state: SDFGState,
          a: str,
          s=None,
          axes=None,
          norm: StringLiteral = StringLiteral('backward')):
    """Full N-D FFT (``numpy.fft.fftn``)."""
    return _fftn_core(pv, sdfg, state, a, s, axes, norm, is_inverse=False)


@oprepo.replaces('numpy.fft.ifftn')
def _ifftn(pv: 'ProgramVisitor',
           sdfg: SDFG,
           state: SDFGState,
           a: str,
           s=None,
           axes=None,
           norm: StringLiteral = StringLiteral('backward')):
    """Full N-D inverse FFT (``numpy.fft.ifftn``)."""
    return _fftn_core(pv, sdfg, state, a, s, axes, norm, is_inverse=True)


@oprepo.replaces('numpy.fft.fft2')
def numpy_fft2(pv: 'ProgramVisitor',
               sdfg: SDFG,
               state: SDFGState,
               a: str,
               s=None,
               axes=(-2, -1),
               norm: StringLiteral = StringLiteral('backward')):
    """2-D FFT (``numpy.fft.fft2``): ``numpy.fft.fftn`` over the last two axes by default."""
    return _fftn_core(pv, sdfg, state, a, s, axes, norm, is_inverse=False)


@oprepo.replaces('numpy.fft.ifft2')
def numpy_ifft2(pv: 'ProgramVisitor',
                sdfg: SDFG,
                state: SDFGState,
                a: str,
                s=None,
                axes=(-2, -1),
                norm: StringLiteral = StringLiteral('backward')):
    """2-D inverse FFT (``numpy.fft.ifft2``): ``numpy.fft.ifftn`` over the last two axes by default."""
    return _fftn_core(pv, sdfg, state, a, s, axes, norm, is_inverse=True)
