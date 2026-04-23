"""Linear-algebra intrinsics (MATMUL / TRANSPOSE / DOT_PRODUCT).

Each one lowers to a dedicated DaCe library node:

    hlfir.matmul       -> dace.libraries.blas.nodes.matmul.MatMul
    hlfir.transpose    -> dace.libraries.standard.nodes.transpose.Transpose
    hlfir.dot_product  -> dace.libraries.blas.nodes.dot.Dot

The bridge classifies the op (callee string + operand names) and
``hlfir_to_sdfg._emit_libcall`` does the actual ``add_node`` + connector
wiring.  Connector names per node:

    MatMul    inputs ``_a``, ``_b``  output ``_c``
    Dot       inputs ``_x``, ``_y``  output ``_result``
    Transpose input  ``_inp``        output ``_out``
"""
from __future__ import annotations

from dace.frontend.hlfir.intrinsics.base import LibNodeIntrinsic
from dace.frontend.hlfir.intrinsics.linalg.matmul import MATMUL
from dace.frontend.hlfir.intrinsics.linalg.transpose import TRANSPOSE
from dace.frontend.hlfir.intrinsics.linalg.dot_product import DOT_PRODUCT

LIBNODE_INTRINSICS: dict[str, LibNodeIntrinsic] = {
    **MATMUL,
    **TRANSPOSE,
    **DOT_PRODUCT,
}
