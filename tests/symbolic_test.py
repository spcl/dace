# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.

from sympy import Max, Min

import dace
from dace.symbolic import simplify_ext, symbol


def test_simplify_ext_min() -> None:
    N = symbol("N")

    assert simplify_ext(Min(N, 4) + 1) == Min(N + 1, 5)
    assert simplify_ext(Max(N, 4) + 1) == Max(N + 1, 5)

    untouched = Min(N, 4)
    assert simplify_ext(untouched) == untouched


def test_conj_is_an_ordinary_symbol_name():
    assert dace.Memlet("A[conj - 1]").subset.free_symbols == {"conj"}
    assert dace.symbolic.symstr(dace.symbolic.pystr_to_symbolic("conj(z)"), cpp_mode=True) == "(conj(z))"


if __name__ == "__main__":
    test_simplify_ext_min()
    test_conj_is_an_ordinary_symbol_name()
