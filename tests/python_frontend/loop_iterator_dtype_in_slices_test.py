# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
""" Reproducers: a Python loop over an int64 size symbol whose iterator bounds a slice of an array operation. """
from typing import List

import pytest

import dace

N = dace.symbol('N', dtype=dace.int64)


@dace.program
def lu_column_update(A: dace.float64[N, N]):
    for k in range(N):
        A[k + 1:, k] = A[k + 1:, k] / A[k, k]
        A[k + 1:, k + 1:] = A[k + 1:, k + 1:] - A[k + 1:, k][:, None] * A[k, k + 1:][None, :]


@dace.program
def scaled_row_prefix(C: dace.float64[N, N], beta: dace.float64):
    for i in range(N):
        C[i, :i + 1] = C[i, :i + 1] * beta


@dace.program
def forward_substitution(A: dace.float64[N, N], b: dace.float64[N], y: dace.float64[N]):
    for i in range(N):
        y[i] = b[i] - A[i, :i] @ y[:i]


@dace.program
def backward_row_suffix(A: dace.float64[N, N], x: dace.float64[N]):
    for i in range(N - 1, -1, -1):
        A[i, i + 1:] = A[i, i + 1:] + x[i + 1:]


def iterator_dtypes(sdfg: dace.SDFG, name: str) -> List[dace.typeclass]:
    found: List[dace.typeclass] = []
    for sub in sdfg.all_sdfgs_recursive():
        for state in sub.states():
            for edge in state.edges():
                for subset in (edge.data.subset, edge.data.other_subset):
                    if subset is None:
                        continue
                    for bound in (b for rng in subset.ndrange() for b in rng):
                        found.extend(s.dtype for s in getattr(bound, 'free_symbols', ()) if s.name == name)
    return found


@pytest.mark.parametrize('program, iterator', [(lu_column_update, 'k'), (scaled_row_prefix, 'i'),
                                               (forward_substitution, 'i'), (backward_row_suffix, 'i')])
def test_a_loop_iterator_keeps_one_dtype_through_simplify(program, iterator):
    sdfg = program.to_sdfg(simplify=True)
    sdfg.validate()
    assert set(iterator_dtypes(sdfg, iterator)) <= {dace.int64}


if __name__ == '__main__':
    for case in ((lu_column_update, 'k'), (scaled_row_prefix, 'i'), (forward_substitution, 'i'), (backward_row_suffix,
                                                                                                  'i')):
        test_a_loop_iterator_keeps_one_dtype_through_simplify(*case)
