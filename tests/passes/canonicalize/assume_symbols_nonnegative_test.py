# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Unit tests for :class:`AssumeSymbolsNonnegative`.

Canonicalization treats every free symbol as nonnegative (offset-sign reasoning);
this pass makes that contract runtime-checked by prepending a side-effecting
``std::abort`` start state that aborts when a signed-integer free symbol is
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
import sympy

import dace
from dace import subsets
from dace.sdfg import nodes
from dace.transformation.passes.canonicalize.pipeline import canonicalize
from dace.transformation.passes.canonicalize.assume_symbols_nonnegative import (
    AssumeSymbolConstraints, AssumeSymbolsNonnegative, insert_assumption_guards, insert_symbol_nonnegative_guard,
    set_symbol_nonnegative_assumptions, _GUARD_STATE_LABEL)
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
        if isinstance(n, nodes.Tasklet) and 'std::abort' in n.code.as_string
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
    """The wrapper reports BOTH halves of the contract it applies: the symbols it re-declared
    nonnegative plus the guard state it emitted. ``axpy`` has one signed-integer free symbol
    (``N``), so a first application counts 1 + 1; a second changes nothing."""
    sdfg = _axpy_sdfg()
    assert AssumeSymbolsNonnegative().apply_pass(sdfg, {}) == 2
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


def test_second_canonicalize_keeps_the_guard_its_own_block():
    """The trap has to run BEFORE the computation it guards, and only its own block keeps it
    there. A re-canonicalize (the vectorizer canonicalizes at its own entry) must therefore not
    let state fusion merge the guard into the compute state."""
    sdfg = _axpy_sdfg()
    canonicalize(sdfg)
    canonicalize(sdfg)
    traps = _trap_tasklets(sdfg)
    assert len(traps) == 1
    assert sdfg.start_block.nodes() == traps
    sdfg.validate()


def _plain_map_bound_symbols(sdfg):
    """Symbol names a map range still spells WITHOUT the nonnegativity assumption."""
    return {
        str(s)
        for sd in sdfg.all_sdfgs_recursive()
        for st in sd.states()
        for n in st.nodes() if isinstance(n, nodes.MapEntry) for rng in n.map.range.ndrange() for bound in rng
        if isinstance(bound, sympy.Basic) for s in bound.free_symbols if not s.is_nonnegative
    }


def test_map_bounds_keep_the_assumption_across_a_second_canonicalize():
    """The assumption lives on the symbol OBJECTS threaded through the graph, and a stage that
    rebuilds a map bound by PARSING mints a plain one. The terminal pass therefore has to decide
    "already assumed" over everything it rewrites, not over the descriptors alone: run 1 marks
    the descriptors, so a descriptor-only look makes run 2 skip and leaves the map bounds spelled
    ``N`` where run 1 spelled ``symbol(N, nonnegative=True)`` -- the same bound, two digests."""
    sdfg = _axpy_sdfg()
    canonicalize(sdfg)
    assert not _plain_map_bound_symbols(sdfg)
    canonicalize(sdfg)
    assert not _plain_map_bound_symbols(sdfg)


def test_reassumes_a_map_bound_that_was_rebuilt_plain():
    """A bound re-parsed after the pass ran is exactly what a later stage leaves behind. The pass
    must see it and re-mark -- and report that it did, since it rewrote the graph."""
    sdfg = _axpy_sdfg()
    canonicalize(sdfg)
    entries = [n for st in sdfg.states() for n in st.nodes() if isinstance(n, nodes.MapEntry)]
    assert entries
    entries[0].map.range = subsets.Range.from_string(str(entries[0].map.range))
    assert _plain_map_bound_symbols(sdfg) == {'N'}
    assert set_symbol_nonnegative_assumptions(sdfg) == 1
    assert not _plain_map_bound_symbols(sdfg)
    assert set_symbol_nonnegative_assumptions(sdfg) is None


def test_guard_leads_the_block_list_on_every_canonicalize():
    """The guard is the start block, and it must also be block 0 of the list ``nodes()``
    reports. That list is insertion order, so the guard -- emitted by the terminal stage --
    is APPENDED on a first canonicalize while a second run inherits it already in place and
    fuses away the blocks that preceded it. Same graph, permuted node indices, different
    content digest: the position has to be pinned to the role, on both runs."""
    sdfg = _axpy_sdfg()
    canonicalize(sdfg)
    first = sdfg.nodes()
    assert first[0].label == _GUARD_STATE_LABEL and first[0] is sdfg.start_block
    canonicalize(sdfg)
    second = sdfg.nodes()
    assert second[0].label == _GUARD_STATE_LABEL and second[0] is sdfg.start_block
    assert len(first) == len(second)
    sdfg.validate()


def test_repositioning_alone_is_reported_as_a_change():
    """Moving the guard IS a rewrite, so the pass must report it -- a pass that mutates while
    returning ``None`` stalls a fixed-point pipeline. Re-applied with the guard already at the
    head it must then report nothing."""
    sdfg = _axpy_sdfg()
    canonicalize(sdfg)
    guard = sdfg.start_block
    sdfg.reorder_nodes([b for b in sdfg.nodes() if b is not guard] + [guard])
    assert sdfg.nodes()[0] is not guard
    assert insert_assumption_guards(sdfg) == 1
    assert sdfg.nodes()[0] is guard
    assert insert_assumption_guards(sdfg) is None


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
                          env={
                              **os.environ, 'PYTHONPATH': os.path.dirname(dace.__path__[0])
                          },
                          capture_output=True,
                          text=True)
    # std::abort terminates via a signal -> negative returncode, and "NO_TRAP"
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
    record_assumption(sdfg, K < N)  # duplicate -> single entry
    record_assumption(sdfg, N < N + 1)  # simplifies to True -> dropped
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
    test_second_canonicalize_keeps_the_guard_its_own_block()
    test_map_bounds_keep_the_assumption_across_a_second_canonicalize()
    test_reassumes_a_map_bound_that_was_rebuilt_plain()
    test_guard_leads_the_block_list_on_every_canonicalize()
    test_repositioning_alone_is_reported_as_a_change()
    test_guard_aborts_on_negative_symbol()
    test_tracked_assumption_emitted_as_own_tasklet()
    test_tracked_assumption_deduped_and_true_dropped()
    test_tracked_assumption_out_of_scope_skipped()
    test_back_compat_aliases()
    print("OK")
