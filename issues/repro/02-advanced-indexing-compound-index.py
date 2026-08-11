#!/usr/bin/env python3
"""Reproducer for 02-advanced-indexing-compound-index.md.

    out[i] = p[int(cols[i])]   ->  AttributeError: 'str' object has no attribute '_fields'

Run it with no arguments and nothing on the environment but a DaCe checkout:

    PYTHONPATH=/path/to/dace python3 02-advanced-indexing-compound-index.py

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

import numpy as np  # noqa: E402
import dace  # noqa: E402

N = dace.symbol('N', dtype=dace.int64)


def bug_cast_gather():
    """(a) The reported case: a gather whose index is a cast of an array read.
    The replacement returns ['int_cols_slice'], which _fill_missing_slices then
    mistakes for a Python list literal, i.e. for advanced indexing."""

    @dace.program
    def f(p: dace.float64[10], cols: dace.int64[5], out: dace.float64[5]):
        for i in range(5):
            out[i] = p[int(cols[i])]

    f.to_sdfg(simplify=False)


def bug_reduction_index():
    """(b) The same crash from an index produced by a reduction -- pivot
    selection, the shape this takes in real code."""

    @dace.program
    def g(U: dace.float64[10, 10], absU: dace.float64[10, 10], out: dace.float64[10]):
        for j in range(10):
            out[j] = U[np.argmax(absU[:, j]), j]

    g.to_sdfg(simplify=False)


def bug_symbolic_list_element():
    """(c) The same line reached from a genuine list literal holding a symbol.
    visit_List yields [0, symbol('N')], and _parse_value chokes on the symbol
    rather than on a str."""

    @dace.program
    def h(x: dace.float64[10], out: dace.float64[2]):
        out[:] = x[[0, N]]

    h.to_sdfg(simplify=False)


def ok_number_list():
    """Control and discriminator: a list-literal index whose elements are all
    plain Numbers parses. The crash needs a non-Number component."""

    @dace.program
    def f(x: dace.float64[10], out: dace.float64[3]):
        out[:] = x[[0, 2, 4]]

    f.to_sdfg(simplify=False)


def ok_hoisted_index():
    """Workaround for (a): hoist the index so the subscript sees a plain name."""

    @dace.program
    def f(p: dace.float64[10], cols: dace.int64[5], out: dace.float64[5]):
        for i in range(5):
            k = int(cols[i])
            out[i] = p[k]

    f.to_sdfg(simplify=False)


def hoist_not_enough_for_reduction():
    """The plain hoist is NOT enough for (b): it trades the AttributeError for a
    separate failure in _views_to_data."""

    @dace.program
    def g(U: dace.float64[10, 10], absU: dace.float64[10, 10], out: dace.float64[10]):
        for j in range(10):
            k = np.argmax(absU[:, j])
            out[j] = U[k, j]

    g.to_sdfg(simplify=False)


def ok_explicit_temp():
    """Workaround for (b): materialise the slice into an explicit temporary.

    This holds on `main` @ d7efcef0c. It does NOT hold on the downstream branch as
    of 9feb23929 ("Reduce argmax/argmin through two scalars instead of a struct"),
    where it fails with `ValueError: View "tmp_0" already has both incoming and
    outgoing edges`. This case is calibrated against `main`, so a run on that
    branch reports it as CHANGED. See 02-advanced-indexing-compound-index.md.
    """

    @dace.program
    def g(U: dace.float64[10, 10], absU: dace.float64[10, 10], out: dace.float64[10]):
        for j in range(10):
            tmp = np.zeros(10, dtype=np.float64)
            tmp[:] = absU[:, j]
            k = np.argmax(tmp)
            out[j] = U[k, j]

    g.to_sdfg(simplify=False)


CASES = [
    ('bug   (a) p[int(cols[i])]', bug_cast_gather, AttributeError, "'str' object has no attribute '_fields'"),
    ('bug   (b) U[np.argmax(absU[:, j]), j]', bug_reduction_index, AttributeError,
     "'str' object has no attribute '_fields'"),
    ('bug   (c) x[[0, N]]', bug_symbolic_list_element, AttributeError, "'symbol' object has no attribute '_fields'"),
    ('ctrl  x[[0, 2, 4]]', ok_number_list, None, ''),
    ('ctrl  hoisted index (workaround a)', ok_hoisted_index, None, ''),
    ('ctrl  hoisted argmax is not enough', hoist_not_enough_for_reduction, ValueError,
     'already has both incoming and outgoing edges'),
    ('ctrl  explicit temp (workaround b)', ok_explicit_temp, None, ''),
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
        print(f'{changed} of {len(results)} cases no longer match 02-advanced-indexing-compound-index.md.')
        return 1
    print(f'All {len(results)} cases match 02-advanced-indexing-compound-index.md; the bug still reproduces.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
