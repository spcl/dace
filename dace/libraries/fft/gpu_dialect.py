# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""The vendor vocabulary a GPU FFT expansion needs, in one object per backend.

The cuFFT and hipFFT expansions emit the same plan-cache-execute body and differ only in spelling,
the way :mod:`dace.libraries.blas.gpu_dialect` factors cuBLAS and rocBLAS. hipFFT mirrors cuFFT
name for name behind its own prefix (``cufftMakePlanMany64`` / ``hipfftMakePlanMany64``,
``CUFFT_Z2Z`` / ``HIPFFT_Z2Z``) with one exception: the inverse direction is ``CUFFT_INVERSE``
in cuFFT and ``HIPFFT_BACKWARD`` in hipFFT.
"""
from typing import NamedTuple


class GpuFftDialect(NamedTuple):
    """How one vendor GPU FFT library spells the things an expansion emits."""

    #: Human name, for a fallback warning that has to say which backend refused.
    name: str
    #: Prefix of every function and type (``cufft`` -> ``cufftHandle``, ``cufftXtExec``).
    api: str
    #: Prefix of every enum constant (``CUFFT_`` -> ``CUFFT_Z2Z``, ``CUFFT_SUCCESS``).
    enum: str
    #: The inverse-direction constant, the one spelling that is not a prefix swap.
    inverse: str


CUFFT = GpuFftDialect(name='cuFFT', api='cufft', enum='CUFFT_', inverse='CUFFT_INVERSE')

HIPFFT = GpuFftDialect(name='hipFFT', api='hipfft', enum='HIPFFT_', inverse='HIPFFT_BACKWARD')
