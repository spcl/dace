# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
""" Map iteration must be ascending.

    A ``Map`` with a negative step fails SDFG validation. An empty
    ``end < begin`` positive-step range is valid (it iterates zero times).
    A symbolic step is checked at runtime by an ``assert(step > 0)`` the
    CPU codegen emits, active only in Debug builds.

    The symbolic-step assertion runs in a subprocess: a failing C
    ``assert`` raises ``SIGABRT`` rather than a Python exception, so the
    Debug build type is set in the child's environment only.
"""
import os
import subprocess
import sys
import tempfile
import textwrap

import pytest

import dace

# SDFG validation: negative-step maps are invalid; empty positive-step maps are valid.


def test_concrete_negative_step_map_is_invalid():
    """A map with a concrete negative step fails SDFG validation."""

    @dace.program
    def neg_step(a: dace.float64[8], b: dace.float64[8]):
        for i in dace.map[5:0:-1]:
            b[i] = a[i] + 1.0

    with pytest.raises(Exception) as ei:
        neg_step.to_sdfg(simplify=True).validate()
    assert 'negative step' in str(ei.value)


def test_empty_positive_step_map_is_valid():
    """An empty ``end < begin`` positive-step map validates (a warning is
    acceptable)."""

    @dace.program
    def empty(a: dace.float64[16], b: dace.float64[16]):
        for i in dace.map[10:3:1]:
            b[i] = a[i] + 1.0

    empty.to_sdfg(simplify=True).validate()  # must not raise


def test_positive_step_map_is_valid():
    """An ascending map (unit or non-unit step) validates."""

    @dace.program
    def ok(a: dace.float64[32], b: dace.float64[32]):
        for i in dace.map[2:30:2]:
            b[i] = a[i] * 2.0

    ok.to_sdfg(simplify=True).validate()  # must not raise


# CPU codegen: a symbolic step is checked by a Debug-build runtime assertion.

_CHILD = textwrap.dedent('''
    import numpy as np, dace
    N = dace.symbol("N")
    @dace.program
    def symstep(a: dace.float64[N], b: dace.float64[N], S: dace.int64):
        for i in dace.map[0:N:S]:
            b[i] = a[i] + 1.0
    sdfg = symstep.to_sdfg(simplify=True)
    a = np.random.rand(8); b = np.zeros(8)
    sdfg(a=a, b=b, N=8, S={step})
    print("NO_ABORT")
''')


def _run_child(step: int):
    """Run the symbolic-step kernel in a child Debug build (build type and
    cache set only in the child env, never in the test process)."""
    env = dict(os.environ)
    env['DACE_compiler_build_type'] = 'Debug'
    env['DACE_cache'] = 'unique'
    # Must be a real file: the dace Python frontend cannot obtain source for
    # a program defined via ``python -c``.
    with tempfile.NamedTemporaryFile('w', suffix='.py', delete=False) as fh:
        fh.write(_CHILD.format(step=step))
        path = fh.name
    try:
        return subprocess.run([sys.executable, path], env=env, capture_output=True, timeout=600)
    finally:
        os.unlink(path)


def test_symbolic_negative_step_aborts_in_debug_build():
    """A symbolic step that is negative at runtime trips the debug-only
    ``assert(step > 0)`` -> the process aborts (SIGABRT)."""
    proc = _run_child(-1)
    assert proc.returncode != 0 and b'NO_ABORT' not in proc.stdout, \
        f"expected abort, rc={proc.returncode} out={proc.stdout!r}"
    assert (proc.returncode == -6 or b'requires a positive step' in proc.stderr
            or b'Assertion' in proc.stderr), \
        f"expected assertion failure, stderr={proc.stderr[-400:]!r}"


def test_symbolic_positive_step_runs_in_debug_build():
    """A positive symbolic step passes the assertion and runs normally in a
    Debug build."""
    proc = _run_child(2)
    assert proc.returncode == 0 and b'NO_ABORT' in proc.stdout, \
        f"expected clean run, rc={proc.returncode} stderr={proc.stderr[-400:]!r}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
