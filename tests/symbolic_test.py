# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.

from sympy import Min, Max

from dace import dtypes
from dace.symbolic import simplify_ext, symbol


def test_simplify_ext_min() -> None:
    N = symbol("N")

    assert simplify_ext(Min(N, 4) + 1) == Min(N + 1, 5)
    assert simplify_ext(Max(N, 4) + 1) == Max(N + 1, 5)

    untouched = Min(N, 4)
    assert simplify_ext(untouched) == untouched


def test_int_floor_compares_equal_to_the_same_extent_written_with_python_floordiv():
    """One extent, two spellings, and they have to compare equal.

    ``pystr_to_symbolic`` maps ``//`` to ``int_floor``, so everything DaCe parses from a string --
    a memlet subset, a shape, an interstate assignment -- says ``int_floor``. Python's own ``//``
    says SymPy ``floor``, and a data-descriptor shape built in Python says it that way. SymPy
    relates neither head to the other, so ``inequal_symbols`` answered True (not equal) for two
    spellings of the same number.
    """
    from dace.symbolic import inequal_symbols, pystr_to_symbolic, relax_int_floor, symbol

    h = symbol('h', dtype=dtypes.int64, positive=True)
    assert not inequal_symbols((h - 7) // 2 + 1, pystr_to_symbolic('(h - 7) // 2 + 1'))
    assert not inequal_symbols(h // 2, pystr_to_symbolic('h // 2'))
    # int_ceil too, and the rewrite leaves a DaCe head behind in neither case.
    assert str(relax_int_floor(pystr_to_symbolic('int_ceil(h, 2)'))) == 'ceiling(h/2)'
    assert str(relax_int_floor(pystr_to_symbolic('(h - 7) // 2 + 1'))) == 'floor(h/2 - 7/2) + 1'
    # A genuinely different extent must still compare unequal -- the fix must not make everything
    # equal by relaxing both sides into mush.
    assert inequal_symbols(pystr_to_symbolic('(h - 7) // 2 + 1'), pystr_to_symbolic('(h - 7) // 3 + 1'))


if __name__ == "__main__":
    test_simplify_ext_min()
    test_int_floor_compares_equal_to_the_same_extent_written_with_python_floordiv()
