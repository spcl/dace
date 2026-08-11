"""Repro: linalg.solve validate() assumes B/X are always a matrix; a vector RHS crashes with IndexError.

Exits 0 when every case still behaves as linalg_solve_index_error.md documents, 1 when something moved.
Needs OpenBLAS (or MKL) on PYTHONPATH's dace to reach expansion; no C++ compiler is invoked.
"""
import sys

import numpy as np
import dace as dc

N, K = (dc.symbol(s, dtype=dc.int64) for s in ('N', 'K'))


@dc.program
def solve_vector_rhs(A: dc.float64[N, N], b: dc.float64[N]):
    return np.linalg.solve(A, b)


@dc.program
def solve_matrix_rhs(A: dc.float64[N, N], B: dc.float64[N, K]):
    return np.linalg.solve(A, B)


def run(prog):
    try:
        sdfg = prog.to_sdfg(simplify=False)
        sdfg.expand_library_nodes()
    except Exception as ex:
        return type(ex).__name__ + ': ' + str(ex).splitlines()[-1]
    return None


CASES = (
    ('vector RHS: np.linalg.solve(A, b)', solve_vector_rhs, True),
    ('matrix RHS: np.linalg.solve(A, B)', solve_matrix_rhs, False),
)

MARKER = 'IndexError: list index out of range'

failures = 0
for label, prog, expect_error in CASES:
    err = run(prog)
    ok = (err is not None and err == MARKER) if expect_error else (err is None)
    failures += not ok
    print(('OK      ' if ok else 'CHANGED ') + label)
    print('        ' + (err if err else 'expanded'))

print('\n%d/%d cases as documented' % (len(CASES) - failures, len(CASES)))
sys.exit(1 if failures else 0)
