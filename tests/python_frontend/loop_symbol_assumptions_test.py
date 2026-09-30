# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
""" A loop iterator whose range has no sign proof is one symbol throughout the loop body. """
from typing import Dict, Set

import dace
from dace import symbolic

M = dace.symbol('M')


@dace.program
def triangular_update(A: dace.float64[M, M], B: dace.float64[M, M]):
    for i in range(M):
        A[i, 0:i + 1] += B[i, 0:i + 1]


def subset_symbols(sdfg: dace.SDFG) -> Dict[str, Set[symbolic.symbol]]:
    found: Dict[str, Set[symbolic.symbol]] = {}
    for state in sdfg.all_states():
        for edge in state.edges():
            if edge.data.is_empty():
                continue
            for rng in edge.data.subset.ndrange():
                for expr in rng:
                    if symbolic.issymbolic(expr):
                        for sym in expr.free_symbols:
                            found.setdefault(sym.name, set()).add(sym)
    return found


def test_loop_iterator_of_a_range_with_no_sign_proof_is_one_symbol():
    sdfg = triangular_update.to_sdfg(simplify=False)
    sdfg.validate()
    assert len(subset_symbols(sdfg)['i']) == 1


if __name__ == '__main__':
    test_loop_iterator_of_a_range_with_no_sign_proof_is_one_symbol()
