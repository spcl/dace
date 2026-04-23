"""``hlfir.dot_product`` -> ``dace.libraries.blas.nodes.dot.Dot``.

Produces a scalar result from two rank-1 inputs.
"""
from __future__ import annotations

from dace.frontend.hlfir.intrinsics.base import LibNodeIntrinsic

DOT_PRODUCT: dict[str, LibNodeIntrinsic] = {
    'dot_product': LibNodeIntrinsic('dot_product', module='blas', node_cls='Dot'),
}
