# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Unit tests for :class:`AssumeSymbolsNonnegative`.

Canonicalization treats every free symbol as nonnegative (offset-sign reasoning);
this pass makes that contract runtime-checked by prepending a side-effecting
``__builtin_trap`` start state that aborts when a signed-integer free symbol is
negative. The guard must be the first state, be marked side-effecting so simplify
keeps it, be a no-op when there is nothing signed to guard, and survive the full
canonicalize pipeline.
"""
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MPI4PY_RC_INITIALIZE", "0")
os.environ.setdefault("OMPI_MCA_pml", "ob1")
os.environ.setdefault("OMPI_MCA_btl", "self,vader")
os.environ.setdefault("UCX_VFS_ENABLE", "n")

import subprocess
import sys
import textwrap

import numpy as np
import pytest

import dace
from dace.sdfg import nodes
from dace.transformation.passes.canonicalize.pipeline import canonicalize
from dace.transformation.passes.canonicalize.assume_symbols_nonnegative import (
    AssumeSymbolConstraints, AssumeSymbolsNonnegative, insert_assumption_guards, insert_symbol_nonnegative_guard,
    _GUARD_STATE_LABEL)
from dace.transformation.passes.canonicalize.tracked_assumptions import record_assumption, tracked_assumptions

N = dace.symbol('N', dtype=dace.int64)
K = dace.symbol('K', dtype=dace.int64)


def _axpy_sdfg():

    @dace.program
    def axpy(a: dace.float64, x: dace.float64[N], y: dace.float64[N]):
        for i in dace.map[0:N]:
            y[i] = a * x[i] + y[i]

    return axpy.to_sdfg(simplify=True)


def _trap_tasklets(sdfg):
    return [
        n for st in sdfg.all_states() for n in st.nodes()
        if isinstance(n, nodes.Tasklet) and '__builtin_trap' in n.code.as_string
    ]


def test_emits_guard_as_first_state():
    sdfg = _axpy_sdfg()
    assert insert_symbol_nonnegative_guard(sdfg) == 1
    assert sdfg.start_block.label == _GUARD_STATE_LABEL
    traps = _trap_tasklets(sdfg)
    assert len(traps) == 1
    assert 'N < 0' in traps[0].code.as_string
    # Must be side-effecting so DeadDataflowElimination does not prune the
    # output-less trap (and, with it, the whole guard).
    assert traps[0].side_effects is True
    sdfg.validate()


def test_idempotent():
    sdfg = _axpy_sdfg()
    assert insert_symbol_nonnegative_guard(sdfg) == 1
    assert insert_symbol_nonnegative_guard(sdfg) is None
    assert len(_trap_tasklets(sdfg)) == 1


def test_noop_without_signed_int_symbols():

    @dace.program
    def noop(x: dace.float64[8]):
        x[:] = x + 1.0

    sdfg = noop.to_sdfg(simplify=True)
    assert sdfg.free_symbols == set()
    assert insert_symbol_nonnegative_guard(sdfg) is None
    assert _trap_tasklets(sdfg) == []


def test_pass_wrapper_matches_helper():
    sdfg = _axpy_sdfg()
    assert AssumeSymbolsNonnegative().apply_pass(sdfg, {}) == 1
    assert AssumeSymbolsNonnegative().apply_pass(sdfg, {}) is None


def test_survives_full_canonicalize_and_runs():
    sdfg = _axpy_sdfg()
    canonicalize(sdfg)
    traps = _trap_tasklets(sdfg)
    assert len(traps) == 1
    assert traps[0].side_effects is True
    # The trap must still sit in the start block after all structural passes.
    start = sdfg.start_block
    assert any(t in start.nodes() for t in traps)
    sdfg.validate()

    a = 2.0
    x = np.random.rand(16)
    y = np.random.rand(16)
    ref = a * x + y
    sdfg(a=a, x=x, y=y, N=16)
    assert np.allclose(y, ref)


def test_guard_aborts_on_negative_symbol():
    """A negative symbol must abort the compiled program (SIGTRAP/SIGILL)."""
    script = textwrap.dedent(f"""
        import os
        for k, v in dict(OMP_NUM_THREADS='1', MPI4PY_RC_INITIALIZE='0', OMPI_MCA_pml='ob1',
                         OMPI_MCA_btl='self,vader', UCX_VFS_ENABLE='n').items():
            os.environ.setdefault(k, v)
        import numpy as np
        import dace
        from dace.transformation.passes.canonicalize.pipeline import canonicalize
        N = dace.symbol('N', dtype=dace.int64)
        @dace.program
        def axpy(a: dace.float64, x: dace.float64[N], y: dace.float64[N]):
            for i in dace.map[0:N]:
                y[i] = a * x[i] + y[i]
        sdfg = axpy.to_sdfg(simplify=True)
        canonicalize(sdfg)
        csdfg = sdfg.compile()
        # Allocate a real buffer so only the guard (not an OOB) can fault; pass N < 0.
        x = np.ones(4); y = np.ones(4)
        csdfg(a=2.0, x=x, y=y, N=-1)
        print("NO_TRAP")
    """)
    proc = subprocess.run([sys.executable, '-c', script],
                          env={**os.environ, 'PYTHONPATH': os.path.dirname(dace.__path__[0])},
                          capture_output=True,
                          text=True)
    # __builtin_trap terminates via a signal -> negative returncode, and "NO_TRAP"
    # must not have been reached.
    assert 'NO_TRAP' not in proc.stdout
    assert proc.returncode != 0


def _kn_sdfg():
    """An SDFG whose free symbols are ``K`` and ``N`` (both signed ints), so a
    recorded ``K < N`` relation is in scope at the entry state."""

    @dace.program
    def kn(x: dace.float64[N]):
        for i in dace.map[0:N]:
            x[i] = x[i] + K

    return kn.to_sdfg(simplify=True)


def test_tracked_assumption_emitted_as_own_tasklet():
    """A recorded relation becomes its own trap tasklet (guarded on its negation),
    in the single guard state alongside the one-per-symbol nonnegativity tasklets."""
    sdfg = _kn_sdfg()
    record_assumption(sdfg, K < N)
    assert insert_assumption_guards(sdfg) == 1
    traps = _trap_tasklets(sdfg)
    # One tasklet per assumption: nonneg K, nonneg N, and the tracked K < N.
    assert len(traps) == 3
    assert all(t.side_effects is True for t in traps)
    # All three tasklets live in the single guard start state.
    assert all(t in sdfg.start_block.nodes() for t in traps)
    conds = [t.code.as_string for t in traps]
    assert any('K < 0' in c for c in conds) and any('N < 0' in c for c in conds)  # nonnegativity
    assert any('K >= N' in c for c in conds)  # the tracked K < N, guarded on its negation
    sdfg.validate()


def test_tracked_assumption_deduped_and_true_dropped():
    sdfg = _kn_sdfg()
    record_assumption(sdfg, K < N)
    record_assumption(sdfg, K < N)          # duplicate -> single entry
    record_assumption(sdfg, N < N + 1)      # simplifies to True -> dropped
    assert [str(r) for r in tracked_assumptions(sdfg)] == ['K < N']


def test_tracked_assumption_out_of_scope_skipped():
    """A relation over a symbol that is not an SDFG free symbol cannot be checked
    at the entry state, so it is skipped (only the nonneg tasklets remain)."""
    sdfg = _kn_sdfg()
    record_assumption(sdfg, dace.symbol('Q', dtype=dace.int64) < N)  # Q is not in the SDFG
    assert insert_assumption_guards(sdfg) == 1
    conds = [t.code.as_string for t in _trap_tasklets(sdfg)]
    assert all('Q' not in c for c in conds)
    assert any('K < 0' in c for c in conds) and any('N < 0' in c for c in conds)


def test_back_compat_aliases():
    assert AssumeSymbolsNonnegative is AssumeSymbolConstraints
    assert insert_symbol_nonnegative_guard is insert_assumption_guards


if __name__ == '__main__':
    test_emits_guard_as_first_state()
    test_idempotent()
    test_noop_without_signed_int_symbols()
    test_pass_wrapper_matches_helper()
    test_survives_full_canonicalize_and_runs()
    test_guard_aborts_on_negative_symbol()
    test_tracked_assumption_emitted_as_own_tasklet()
    test_tracked_assumption_deduped_and_true_dropped()
    test_tracked_assumption_out_of_scope_skipped()
    test_back_compat_aliases()
    print("OK")
