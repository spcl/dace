"""Whole-array scalar reductions -> DaCe ``standard.Reduce``.

``sum(a)`` / ``product(a)`` / ``minval(a)`` / ``maxval(a)`` each lower
through Flang into a dedicated HLFIR op (``hlfir.sum``, ``hlfir.product``,
``hlfir.minval``, ``hlfir.maxval``) whose operand is the input array box
and whose result is either a scalar or a reduced-rank array (when a
``dim=`` argument was supplied).

The bridge's extract_ast spots one of these ops as the source of an
``hlfir.assign`` and emits ``kind="reduce"`` ASTNode carrying the
parameters below; hlfir_to_sdfg then calls
``state.add_reduce(wcr, axes, identity)``.
"""

from dace.frontend.hlfir.intrinsics.base import ReductionIntrinsic

REDUCTIONS: dict[str, ReductionIntrinsic] = {
    'sum':
    ReductionIntrinsic(name='sum', wcr='lambda a, b: a + b', identity='0'),
    'product':
    ReductionIntrinsic(name='product', wcr='lambda a, b: a * b', identity='1'),
    'minval':
    ReductionIntrinsic(
        name='minval',
        wcr='lambda a, b: min(a, b)',
        # Start at +inf so the first real element always wins.  DaCe's
        # codegen resolves ``math.inf`` through ``_ALLOWED_MODULES``.
        identity='math.inf'),
    'maxval':
    ReductionIntrinsic(name='maxval', wcr='lambda a, b: max(a, b)', identity='-math.inf'),
}
