# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""``Mod`` is SymPy's floored modulo; C++ prints it as ``%`` only where the two agree, which the facts at the code
being generated decide. ``numpy.roll`` wraps ``psi[(i - 1) % N]``: printed as C's ``%`` it reads ``psi[-1]``."""

import dace
from dace import symbolic
from dace.symbolic import Facts, Predicate, Relation, RelationKind, predicate_relation


def iterator_facts() -> Facts:
    """``0 <= i < N`` over integers, with ``N`` positive."""
    i, n = dace.symbol("i"), dace.symbol("N")
    return Facts(
        (
            predicate_relation(Predicate.POSITIVE, n),
            Relation(RelationKind.LE, 0, i),
            Relation(RelationKind.LT, i, n),
        ),
        frozenset({"i", "N"}),
    )


def printed(text: str) -> str:
    return symbolic.symstr(symbolic.pystr_to_symbolic(text), cpp_mode=True)


def test_an_unproven_modulo_prints_the_floored_helper():
    assert "py_mod" in printed("(i - 1) % N")


def test_the_facts_prove_a_nonnegative_modulo_c():
    with symbolic.printing_facts(iterator_facts):
        assert "py_mod" not in printed("(i + 1) % N")
        assert "%" in printed("(i + 1) % N")


def test_the_facts_keep_a_wrapping_modulo_floored():
    with symbolic.printing_facts(iterator_facts):
        assert "py_mod" in printed("(i - 1) % N")


def test_a_modulo_proven_under_facts_is_not_served_where_they_do_not_hold():
    """The printers memoize; the facts at the code being generated have to key that memo, or a ``%`` proven inside
    one map is handed to code outside it."""
    with symbolic.printing_facts(iterator_facts):
        assert "py_mod" not in printed("(i + 2) % N")
    assert "py_mod" in printed("(i + 2) % N")
    with symbolic.printing_facts(iterator_facts):
        assert "py_mod" not in symbolic.symstr(symbolic.pystr_to_symbolic("(i + 3) % N"), cpp_mode=True)
    from dace.codegen.common import sym2cpp

    assert "py_mod" in sym2cpp(symbolic.pystr_to_symbolic("(i + 3) % N"))


if __name__ == "__main__":
    test_an_unproven_modulo_prints_the_floored_helper()
    test_the_facts_prove_a_nonnegative_modulo_c()
    test_the_facts_keep_a_wrapping_modulo_floored()
    test_a_modulo_proven_under_facts_is_not_served_where_they_do_not_hold()
