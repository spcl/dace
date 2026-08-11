#!/usr/bin/env python3
"""Reproducer for 04-arrayview-copy.md.

    out[:] = path[:, 1].copy()   ->  NotImplementedError   (with an empty message)

Run it with no arguments and nothing on the environment but a DaCe checkout:

    PYTHONPATH=/path/to/dace python3 04-arrayview-copy.py

Exit status 0 means every case behaved exactly as the issue documents, i.e. the
bug still reproduces. Exit status 1 means something moved -- either the bug was
fixed or this script bit-rotted. The per-case lines say which.

The script also prints the descriptor probe the issue quotes: the type actually
handed to _add_transient_data, its MRO, and the keys the dispatch dict holds.

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

import numpy as np  # noqa: E402
import dace  # noqa: E402


def bug_column_copy():
    """The reported case: .copy() of a strided column slice."""

    @dace.program
    def f(path: dace.float64[10, 10], out: dace.float64[10]):
        out[:] = path[:, 1].copy()

    f.to_sdfg(simplify=False)


def bug_row_copy():
    """A contiguous row slice fails identically, so it is not about strides."""

    @dace.program
    def f(path: dace.float64[10, 10], out: dace.float64[10]):
        out[:] = path[1, :].copy()

    f.to_sdfg(simplify=False)


def bug_free_function_copy():
    """np.copy() of the same slice hits the same _add_transient_data path."""

    @dace.program
    def f(path: dace.float64[10, 10], out: dace.float64[10]):
        tmp = np.copy(path[:, 1])
        out[:] = tmp

    f.to_sdfg(simplify=False)


def ok_whole_array_copy():
    """Control: copying the whole array works, because the descriptor is an
    Array, which is one of the four registered keys."""

    @dace.program
    def f(path: dace.float64[10, 10], out: dace.float64[10, 10]):
        out[:] = path.copy()

    f.to_sdfg(simplify=False)


def ok_explicit_temp():
    """The only workaround we found: allocate the temporary and assign into it."""

    @dace.program
    def f(path: dace.float64[10, 10], out: dace.float64[10]):
        tmp = np.zeros(10, dtype=np.float64)
        tmp[:] = path[:, 1]
        out[:] = tmp

    f.to_sdfg(simplify=False)


CASES = [
    ('bug   path[:, 1].copy()', bug_column_copy, NotImplementedError, ''),
    ('bug   path[1, :].copy()', bug_row_copy, NotImplementedError, ''),
    ('bug   np.copy(path[:, 1])', bug_free_function_copy, NotImplementedError, ''),
    ('ctrl  path.copy()', ok_whole_array_copy, None, ''),
    ('ctrl  explicit temp (workaround)', ok_explicit_temp, None, ''),
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


def probe():
    """Recover the descriptor that _add_transient_data refused, straight off the
    traceback, and show it against the keys the dispatch dict actually holds."""
    from dace.frontend.python.newast import AddTransientMethods

    try:
        bug_column_copy()
    except NotImplementedError:
        tb = sys.exc_info()[2]
        while tb.tb_next:
            tb = tb.tb_next
        sample = tb.tb_frame.f_locals.get('sample_data')
        print(f'descriptor handed to _add_transient_data: {type(sample).__name__}')
        print(f'its MRO:                                  {[c.__name__ for c in type(sample).__mro__]}')
        print(f'AddTransientMethods keys:                 {[k.__name__ for k in AddTransientMethods._methods]}')
        return True
    print('descriptor probe: path[:, 1].copy() no longer raises, nothing to probe')
    return False


def main():
    print(f'dace {dace.__version__} from {os.path.dirname(os.path.abspath(dace.__file__))}')
    results = [check(*case) for case in CASES]
    print()
    probe()
    changed = results.count(False)
    print()
    if changed:
        print(f'{changed} of {len(results)} cases no longer match 04-arrayview-copy.md.')
        return 1
    print(f'All {len(results)} cases match 04-arrayview-copy.md; the bug still reproduces.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
