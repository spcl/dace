# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""A guard name bound to a truth test of an array element (``if_cond = clipping[i]``) has no boolean value sympy can
put inside ``And``: the store bound is unknown, not a crash of the whole canonicalization (VT's CFL clipping loop)."""

from collections import Counter

import sympy

from dace import symbolic
from dace.transformation.passes.analysis import array_facts


def test_a_guard_bound_to_an_array_element_truth_test_has_no_bound():
    guard, i = symbolic.symbol("if_cond"), symbolic.symbol("i")
    value = symbolic.pystr_to_symbolic("clipping[i]")
    bound = array_facts.bound_expression(sympy.And(guard, i > 0), {"if_cond": value}, Counter({"if_cond": 1}))
    assert bound is None


if __name__ == "__main__":
    test_a_guard_bound_to_an_array_element_truth_test_has_no_bound()
