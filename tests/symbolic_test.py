# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.

from sympy import Max, Min

import dace
from dace import symbolic
from dace.symbolic import simplify_ext, symbol


def test_simplify_ext_min() -> None:
    N = symbol("N")

    assert simplify_ext(Min(N, 4) + 1) == Min(N + 1, 5)
    assert simplify_ext(Max(N, 4) + 1) == Max(N + 1, 5)

    untouched = Min(N, 4)
    assert simplify_ext(untouched) == untouched


if __name__ == "__main__":
    test_simplify_ext_min()


def test_ifexpr_derivative() -> None:
    # ``IfExpr`` differentiates like ``sympy.Piecewise``: values are differentiated, the
    # condition is kept. Differentiating the (relational) condition used to raise
    # ``AttributeError: 'GreaterThan' object has no attribute 'diff'``.
    x, y = symbol("x"), symbol("y")
    expr = symbolic.IfExpr(x >= y, x * x, y)
    assert expr.diff(x) == symbolic.IfExpr(x >= y, 2 * x, 0)
    assert expr.diff(y) == symbolic.IfExpr(x >= y, 0, 1)


def test_boolean_function_derivative() -> None:
    # Truth values are piecewise constant.
    x, y, z = symbol("x"), symbol("y"), symbol("z")
    assert symbolic.OR(x >= y, y >= z).diff(x) == 0
    assert symbolic.AND(x >= y, y >= z).diff(x) == 0


def test_simplify_comparison_with_ternary() -> None:
    # Simplifying a comparison calls ``(lhs - rhs).equals(0)``, which differentiates; with a
    # ternary on one side this used to raise ``AttributeError``.
    symbolic.simplify(symbolic.pystr_to_symbolic("(x if x >= y else x) >= x"))
    symbolic.simplify(symbolic.pystr_to_symbolic("(x if (x >= y or y >= z) else y) >= x"))


def test_memlet_propagation_with_ternary_map_range() -> None:
    # Memlet propagation simplifies the propagated volume, which contains the map range.
    sdfg = dace.SDFG("ternary_map_range")
    sdfg.add_array("A", ["N"], dace.float64)
    sdfg.add_array("B", ["N"], dace.float64)
    state = sdfg.add_state()
    state.add_mapped_tasklet(
        "copy",
        {"i": "0:(N if (N if N >= M else N) >= N else M)"},
        {"a": dace.Memlet("A[i]")},
        "b = a",
        {"b": dace.Memlet("B[i]")},
        external_edges=True,
    )
    sdfg.validate()
