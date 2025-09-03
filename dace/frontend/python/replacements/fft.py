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
    from dace.libraries.fft.nodes import FFT, IFFT  # Avoid import loops
    if axis != 0 and axis != -1:
        raise NotImplementedError('Only one dimensional arrays are supported at the moment')
    if not isinstance(a, str) or a not in sdfg.arrays:
        raise ValueError('Input must be a valid array')

    libnode = FFT('fft') if not is_inverse else IFFT('ifft')

    desc = sdfg.arrays[a]
    N = desc.shape[axis]

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
