# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""A program declares its symbols as its globals were declared, not as a same-named object inside an annotation.

A symbol is its name, so SymPy's expression cache can hand an annotation such as ``2 * N`` back holding an ``N``
built elsewhere -- without the ``positive`` this module declared. The argument types' free symbols then replaced the
global, and the SDFG lost the predicate: a callee's ``0 < N`` was no longer provable at its call site.
"""

import dace
from dace.symbolic import Predicate, predicate_relation

# Another module's ``N``, cached into ``2 * N`` before this module's declaration exists
2 * dace.symbol("N")

N = dace.symbol("N", dtype=dace.int64, positive=True)


@dace.program
def doubled(a: dace.float64[2 * N]):
    a[:] = 1.0


def test_an_annotation_does_not_replace_the_declared_global():
    sdfg = doubled.to_sdfg()
    assert predicate_relation(Predicate.POSITIVE, N) in sdfg.facts().relations, sdfg.facts().relations


if __name__ == "__main__":
    test_an_annotation_does_not_replace_the_declared_global()
