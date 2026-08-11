#!/usr/bin/env python3
"""Reproducer for 01-symbolic-or.md.

    if sym == a or sym == b:   ->  TypeError: 'Equality' object is not iterable

Run it with no arguments and nothing on the environment but a DaCe checkout:

    PYTHONPATH=/path/to/dace python3 01-symbolic-or.py

Exit status 0 means every case behaved exactly as the issue documents, i.e. the
bug still reproduces. Exit status 1 means something moved -- either the bug was
fixed or this script bit-rotted. The per-case lines say which.

Parse-only: no C++ compiler is invoked.
"""
import os
import sys
import tempfile

# DaCe is only deterministic under a fixed hash seed, and the MPI/UCX probes it
# runs at import time can hang without these. Set them here so that the script
# needs no environment of its own.
os.environ.update({
    'OMP_NUM_THREADS': '1',
    'OMPI_MCA_pml': 'ob1',
    'OMPI_MCA_btl': 'self,vader,tcp',
    'PMIX_MCA_gds': 'hash',
    'UCX_VFS_ENABLE': 'n',
    'HWLOC_COMPONENTS': '-gl',
    'MPI4PY_RC_INITIALIZE': '0',
})
if os.environ.get('PYTHONHASHSEED') != '0':
    # The hash seed is fixed at interpreter startup, so it needs one re-exec.
    os.environ['PYTHONHASHSEED'] = '0'
    os.execv(sys.executable, [sys.executable, os.path.abspath(__file__)] + sys.argv[1:])
os.environ.setdefault('DACE_default_build_folder', tempfile.mkdtemp(prefix='dace-issue-repro-'))

import dace  # noqa: E402

N = dace.symbol('N', dtype=dace.int64)


def bug_or():
    """The reported case: a disjunction of two symbolic equalities."""

    @dace.program
    def symbolic_or(x: dace.float64[10]):
        if N == 0 or N == 1:
            x[:] = 1.0
        else:
            x[:] = 2.0

    symbolic_or.to_sdfg(simplify=False)


def bug_and():
    """`and` fails the same way -- both are ast.BoolOp."""

    @dace.program
    def symbolic_and(x: dace.float64[10]):
        if N == 0 and N == 1:
            x[:] = 1.0
        else:
            x[:] = 2.0

    symbolic_and.to_sdfg(simplify=False)


def bug_ifexp():
    """Second symptom: on a single-AST-valued field the sympy object is planted
    in the tree instead of raising, and the unparser trips over it later."""

    @dace.program
    def symbolic_ifexp(x: dace.float64[10]):
        if (N == 0) if N > 5 else (N == 1):
            x[:] = 1.0
        else:
            x[:] = 2.0

    symbolic_ifexp.to_sdfg(simplify=False)


def ok_single():
    """Control: a single Compare is the transformer's root return value and is
    consumed directly by visit_If, so it never reaches generic_visit."""

    @dace.program
    def f(x: dace.float64[10]):
        if N == 0:
            x[:] = 1.0
        else:
            x[:] = 2.0

    f.to_sdfg(simplify=False)


def ok_inequality():
    """Control: visit_Compare only short-circuits for Eq/NotEq, so a BoolOp over
    inequalities is untouched."""

    @dace.program
    def f(x: dace.float64[10]):
        if N < 1 or N > 5:
            x[:] = 1.0
        else:
            x[:] = 2.0

    f.to_sdfg(simplify=False)


def ok_not():
    """Control: `not` over a single equality survives."""

    @dace.program
    def f(x: dace.float64[10]):
        if not (N == 0):
            x[:] = 1.0
        else:
            x[:] = 2.0

    f.to_sdfg(simplify=False)


def ok_workaround():
    """The workaround: an elif chain keeps every `if` test a single Compare."""

    @dace.program
    def f(x: dace.float64[10]):
        if N == 0:
            x[:] = 1.0
        elif N == 1:
            x[:] = 1.0
        else:
            x[:] = 2.0

    f.to_sdfg(simplify=False)


CASES = [
    ('bug   if N == 0 or N == 1', bug_or, TypeError, "'Equality' object is not iterable"),
    ('bug   if N == 0 and N == 1', bug_and, TypeError, "'Equality' object is not iterable"),
    ('bug   if (N==0) if N>5 else (N==1)', bug_ifexp, AttributeError, "has no attribute '_Equality'"),
    ('ctrl  if N == 0', ok_single, None, ''),
    ('ctrl  if N < 1 or N > 5', ok_inequality, None, ''),
    ('ctrl  if not (N == 0)', ok_not, None, ''),
    ('ctrl  elif chain (workaround)', ok_workaround, None, ''),
]


def check(label, fn, exc, message):
    """Run one parse-only case and report whether it matched the issue text."""
    try:
        fn()
    except BaseException as e:  # noqa: BLE001 - any escaping exception is a result
        text = str(e) or '<empty message>'
        if exc is None:
            print(f'CHANGED  {label}: expected a clean parse, got {type(e).__name__}: {text}')
            return False
        if not isinstance(e, exc) or message not in str(e):
            print(f'CHANGED  {label}: expected {exc.__name__} containing {message!r}, '
                  f'got {type(e).__name__}: {text}')
            return False
        print(f'REPRO    {label}: {type(e).__name__}: {text}')
        return True
    if exc is not None:
        print(f'CHANGED  {label}: expected {exc.__name__}, but it parsed cleanly')
        return False
    print(f'ok       {label}: parses')
    return True


def main():
    print(f'dace {dace.__version__} from {os.path.dirname(os.path.abspath(dace.__file__))}')
    results = [check(*case) for case in CASES]
    changed = results.count(False)
    print()
    if changed:
        print(f'{changed} of {len(results)} cases no longer match 01-symbolic-or.md.')
        return 1
    print(f'All {len(results)} cases match 01-symbolic-or.md; the bug still reproduces.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
