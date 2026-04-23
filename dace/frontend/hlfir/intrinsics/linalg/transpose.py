"""``hlfir.transpose`` -> ``dace.libraries.standard.nodes.transpose.Transpose``.

Fortran ``transpose`` is defined on rank-2 arrays only; the library node
already specializes MKL / OpenBLAS / cuBLAS backends internally.
"""
from __future__ import annotations

from dace.frontend.hlfir.intrinsics.base import LibNodeIntrinsic

TRANSPOSE: dict[str, LibNodeIntrinsic] = {
    'transpose': LibNodeIntrinsic('transpose', module='standard', node_cls='Transpose'),
}
