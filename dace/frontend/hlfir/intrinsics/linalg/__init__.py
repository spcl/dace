"""Linear-algebra intrinsics (MATMUL / TRANSPOSE / DOT_PRODUCT / FFT).

Phase 3 — each will emit a DaCe library node directly from the HLFIR op
without going through a post-SDFG pattern match:

    hlfir.matmul       -> dace.libraries.blas.nodes.Matmul
    hlfir.transpose    -> dace.libraries.standard.nodes.Transpose
    hlfir.dot_product  -> dace.libraries.blas.nodes.Dot
    fir.call(@fft_*)   -> dace.libraries.fft.nodes.FFT

Today this package only carries an empty ``LIBNODE_INTRINSICS`` registry
so the top-level ``is_libnode`` helper can stay family-agnostic.
"""
from __future__ import annotations

from dace.frontend.hlfir.intrinsics.base import LibNodeIntrinsic

LIBNODE_INTRINSICS: dict[str, LibNodeIntrinsic] = {}
