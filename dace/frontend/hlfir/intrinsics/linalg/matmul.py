"""``hlfir.matmul`` -> ``dace.libraries.blas.nodes.matmul.MatMul``.

``MatMul`` is a meta-node: its ``SpecializeMatMul`` expansion dispatches to
``Gemm``, ``BatchedMatMul``, ``Gemv`` or ``Dot`` depending on the operand
ranks, so the same node handles matrix-matrix, matrix-vector, and
vector-matrix Fortran ``matmul`` calls alike.
"""
from __future__ import annotations

from dace.frontend.hlfir.intrinsics.base import LibNodeIntrinsic

MATMUL: dict[str, LibNodeIntrinsic] = {
    'matmul': LibNodeIntrinsic('matmul', module='blas', node_cls='MatMul'),
}
