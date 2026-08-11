#!/usr/bin/env python3
"""Reproducer for 03-numpy-empty-dtype.md.

    y = np.empty(10)   ->  TypeError: _numpy_empty() missing 1 required positional argument: 'dtype'

Run it with no arguments and nothing on the environment but a DaCe checkout:

    PYTHONPATH=/path/to/dace python3 03-numpy-empty-dtype.py

Exit status 0 means every case behaved exactly as the issue documents, i.e. the
bug still reproduces. Exit status 1 means something moved -- either the bug was
fixed or this script bit-rotted. The per-case lines say which.

Once the one-line patch in 03-numpy-empty-dtype.md is applied, the first case
flips to a clean parse and this script exits 1, reporting that. The last case
then also checks that the resulting transient is float64, matching numpy.

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


def bug_empty_no_dtype():
    """The reported case: numpy.empty's replacement declares dtype as a required
    positional parameter, so a call that is valid NumPy raises an arity error."""

    @dace.program
    def f(x: dace.float64[10]):
        y = np.empty(10)
        y[:] = 1.0
        x[:] = y

    f.to_sdfg(simplify=False)


def ok_zeros_no_dtype():
    """Control: the sibling numpy.zeros defaults dtype to float64."""

    @dace.program
    def f(x: dace.float64[10]):
        y = np.zeros(10)
        y[:] = 1.0
        x[:] = y

    f.to_sdfg(simplify=False)


def ok_ones_no_dtype():
    """Control: so does numpy.ones."""

    @dace.program
    def f(x: dace.float64[10]):
        y = np.ones(10)
        x[:] = y

    f.to_sdfg(simplify=False)


def ok_empty_like_no_dtype():
    """Control: even numpy.empty_like, the closest relative, defaults dtype."""

    @dace.program
    def f(x: dace.float64[10]):
        y = np.empty_like(x)
        y[:] = 1.0
        x[:] = y

    f.to_sdfg(simplify=False)


def ok_explicit_dtype():
    """The workaround: pass the dtype explicitly."""

    @dace.program
    def f(x: dace.float64[10]):
        y = np.empty(10, dtype=np.float64)
        y[:] = 1.0
        x[:] = y

    f.to_sdfg(simplify=False)


CASES = [
    ('bug   np.empty(10)', bug_empty_no_dtype, TypeError, "missing 1 required positional argument: 'dtype'"),
    ('ctrl  np.zeros(10)', ok_zeros_no_dtype, None, ''),
    ('ctrl  np.ones(10)', ok_ones_no_dtype, None, ''),
    ('ctrl  np.empty_like(x)', ok_empty_like_no_dtype, None, ''),
    ('ctrl  np.empty(10, dtype=np.float64)', ok_explicit_dtype, None, ''),
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


def check_patched_dtype():
    """Only meaningful once the patch is applied: np.empty(10) must yield a
    float64 transient, and an explicit dtype must still win."""

    @dace.program
    def f(x: dace.float64[10]):
        y = np.empty(10)
        y[:] = 1.0
        x[:] = y

    @dace.program
    def g(x: dace.int32[10]):
        y = np.empty(10, dtype=np.int32)
        y[:] = 1
        x[:] = y

    default = {n: d.dtype for n, d in f.to_sdfg(simplify=False).arrays.items() if d.transient}
    explicit = {n: d.dtype for n, d in g.to_sdfg(simplify=False).arrays.items() if d.transient}
    if not any(dt == dace.float64 for dt in default.values()):
        print(f'         patched np.empty(10) is not float64: {default}')
        return False
    if not any(dt == dace.int32 for dt in explicit.values()):
        print(f'         patched np.empty(10, dtype=np.int32) lost its dtype: {explicit}')
        return False
    print('         patched np.empty defaults to float64 and still honours an explicit dtype')
    return True


def main():
    print(f'dace {dace.__version__} from {os.path.dirname(os.path.abspath(dace.__file__))}')
    results = [check(*case) for case in CASES]
    changed = results.count(False)
    print()
    if changed:
        print(f'{changed} of {len(results)} cases no longer match 03-numpy-empty-dtype.md.')
        if not results[0]:
            print('np.empty(10) parses, so the patch looks applied. Checking the resulting dtype:')
            return 0 if check_patched_dtype() else 1
        return 1
    print(f'All {len(results)} cases match 03-numpy-empty-dtype.md; the bug still reproduces.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
