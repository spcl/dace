# Copyright 2019-2025 ETH Zurich and the DaCe authors. All rights reserved.
"""
Contains replacements for the Discrete Fourier Transform numpy package (numpy.fft)
"""
from dace.frontend.common import op_repository as oprepo
from dace.frontend.python.common import StringLiteral
from dace.frontend.python.replacements.utils import ProgramVisitor
from dace import dtypes, symbolic, Memlet, SDFG, SDFGState

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
    if not isinstance(a, str) or a not in sdfg.arrays:
        raise ValueError('Input must be a valid array')

    desc = sdfg.arrays[a]
    ndim = len(desc.shape)
    axis = int(axis) if axis is not None else -1
    axis_norm = axis if axis >= 0 else ndim + axis
    if axis_norm < 0 or axis_norm >= ndim:
        raise ValueError(f'axis {axis} out of range for ndim={ndim}')

    libnode = FFT('fft') if not is_inverse else IFFT('ifft')
    # 1-D input: no axis to set; the whole array IS the FFT.  Multi-D
    # input: tag the lib node so the expansion runs a batched 1-D FFT
    # along ``axis_norm`` (matching ``np.fft.fft(x, axis=k)``).
    if ndim > 1:
        libnode.axis = axis_norm

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
    """Full N-D FFT, matching the existing :class:`FFT` / :class:`IFFT` lib
    node semantics (the lib node treats the input shape as the FFT extent).

    ``s`` (per-axis output size override) and ``axes`` (subset of axes) are
    not yet supported -- the node always transforms over all axes of the
    input. They are accepted for signature compatibility with numpy so
    ``@dace.program`` code that passes ``axes=(0, 1)`` of an already-2-D
    array doesn't fail at parse time.
    """
    from dace.libraries.fft.nodes import FFT, IFFT  # avoid import loop

    if not isinstance(a, str) or a not in sdfg.arrays:
        raise ValueError('Input must be a valid array')
    if s is not None:
        raise NotImplementedError('numpy.fft.fftn ``s`` (padding/cropping) is not yet supported')

    desc = sdfg.arrays[a]
    ndim = len(desc.shape)
    if axes is not None:
        # Only the "all axes" identity case (matching numpy's default) is
        # supported -- anything else would need a non-trivial axes-aware
        # lib node, which the existing ``FFT`` connector layout does not
        # express.
        wanted = tuple(range(ndim))
        provided = tuple(int(x) % ndim for x in axes)
        if provided != wanted:
            raise NotImplementedError('numpy.fft.fftn ``axes`` other than '
                                      'the full input is not yet supported '
                                      f'(got {axes}, expected {wanted})')

    libnode = FFT('fft') if not is_inverse else IFFT('ifft')

    # Total transform size for the normalisation factor (numpy normalises by
    # the product of the transformed-axes sizes).
    total = 1
    for d in desc.shape:
        total = total * d
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
