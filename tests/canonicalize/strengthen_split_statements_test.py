# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Soundness strengthening for ``SplitStatements`` -- the forward-read snapshot.

``SplitStatements._snapshot_forward_reads`` snapshots a global array before a loop
and redirects a read-ahead access ``arr[i + off]`` to the pre-loop snapshot so the
loop can fission. For a symbolic offset it plants a runtime positive-check guard on
``off`` and, under the nonnegative-symbol assumption, admits every provably
nonnegative ``off``.

The crux: in the MIXED split shape ``arr[i] = ..; d[i] = arr[i] + arr[i + off]`` a
SIBLING statement writes ``arr[i]`` earlier in the SAME iteration. A read
``arr[i + off]`` with ``off == 0`` is therefore a same-iteration read of the
just-written value and must keep reading the LIVE array. Redirecting it to the
pre-loop snapshot returns the stale original -> wrong numbers. A numeric offset of 0
is classified ``'none'`` (correctly kept live); but a SYMBOLIC offset ``K`` was
routed to ``WAR_symbolic`` with only a ``>= 0`` guard, so at runtime ``K == 0``
passed the guard and the read was silently miscompiled.

The fix emits a STRICT (``off >= 1``) guard for the split's mixed shape, so ``K == 0``
faults loudly instead of corrupting the result, while every valid ``K >= 1`` still
snapshots-and-parallelizes bit-exactly.
"""
import os
import numpy as np
import pytest

import dace
from dace.sdfg import nodes
from dace.transformation.passes.canonicalize.pipeline import canonicalize

N = dace.symbol('N')
K = dace.symbol('K')


@dace.program
def _split_sym_forward(A: dace.float64[N], B: dace.float64[N], D: dace.float64[N]):
    for i in range(N - K):
        A[i] = B[i] + 1.0
        D[i] = A[i] + A[i + K]  # A[i+K] is the forward read snapshotted by SplitStatements


def _nmaps(sdfg):
    return sum(1 for n, _ in sdfg.all_nodes_recursive() if isinstance(n, nodes.MapEntry))


def test_split_symbolic_forward_offset_valid_k_is_bit_exact_and_parallel():
    """The valid case (strictly positive symbolic offset K=3): the read A[i+3] is
    genuinely read-ahead, so SplitStatements snapshots A and the cone parallelizes.
    Bit-exact against the raw (un-canonicalized) SDFG -- guards the valid path so the
    strict-guard fix does not over-refuse."""
    n, k = 50, 3
    A0 = np.random.default_rng(21).random(n)
    B0 = np.random.default_rng(22).random(n)

    raw = _split_sym_forward.to_sdfg(simplify=True)
    Ar, Br, Dr = A0.copy(), B0.copy(), np.zeros(n)
    raw.compile()(A=Ar, B=Br, D=Dr, N=n, K=k)

    cand = _split_sym_forward.to_sdfg(simplify=True)
    canonicalize(cand, validate=True, peel_limit=4, break_anti_dependence=True)
    assert any('snap' in nm for nm in cand.arrays), 'the forward-read cone should snapshot A'
    assert _nmaps(cand) >= 1, 'the symbolic-offset anti-dependence cone should parallelize'
    Ac, Bc, Dc = A0.copy(), B0.copy(), np.zeros(n)
    cand.compile()(A=Ac, B=Bc, D=Dc, N=n, K=k)
    assert np.allclose(Dr, Dc, equal_nan=True), 'K=3 snapshot must be value-preserving'


def test_split_symbolic_forward_offset_zero_never_silently_miscompiles():
    """The unsound case: K=0 makes ``A[i+K] == A[i]`` a same-iteration read of the
    value the sibling ``A[i] = B[i]+1`` just wrote. It must NOT be redirected to the
    pre-loop snapshot.

    Runs the canonicalized kernel in a forked child (a trap must not kill pytest).
    Pre-fix the child computes wrong D and exits 7 (silent miscompile). Post-fix the
    strict ``>0`` guard traps, so the child dies by signal -- no corruption. The
    assertion is exactly 'the child never returns a clean-but-wrong result'."""
    n, k = 40, 0
    A0 = np.random.default_rng(21).random(n)
    B0 = np.random.default_rng(22).random(n)

    raw = _split_sym_forward.to_sdfg(simplify=True)
    Ar, Br, Dr = A0.copy(), B0.copy(), np.zeros(n)
    raw.compile()(A=Ar, B=Br, D=Dr, N=n, K=k)

    pid = os.fork()
    if pid == 0:  # child: compile+run the canonicalized kernel; compare to the reference
        try:
            cand = _split_sym_forward.to_sdfg(simplify=True)
            canonicalize(cand, validate=True, peel_limit=4, break_anti_dependence=True)
            Ac, Bc, Dc = A0.copy(), B0.copy(), np.zeros(n)
            cand.compile()(A=Ac, B=Bc, D=Dc, N=n, K=k)
            os._exit(0 if np.allclose(Dr, Dc, equal_nan=True) else 7)
        except BaseException:
            os._exit(5)  # a clean Python-level refusal is also acceptable (no wrong numbers)
    _, status = os.waitpid(pid, 0)

    if os.WIFSIGNALED(status):
        return  # trapped by the strict positive-check guard -> sound (loud, not silent)
    assert os.WIFEXITED(status), 'child neither exited nor signalled'
    code = os.WEXITSTATUS(status)
    assert code != 7, ('SplitStatements silently miscompiled the K=0 same-iteration read: '
                       'A[i+0] was redirected to the stale pre-loop snapshot instead of the '
                       'just-written live value')
    assert code in (0, 5), f'unexpected child exit code {code}'


@dace.program
def _split_mixed_numeric_none(A: dace.float64[N], B: dace.float64[N], E: dace.float64[N], D: dace.float64[N]):
    for i in range(1, N - 2):
        A[i] = B[i]
        A[i + 1] = E[i]
        D[i] = A[i + 1] * 2.0  # reads the value the SIBLING ``A[i+1]=E[i]`` just wrote this iteration


def test_split_mixed_read_kept_live_when_none_coexists_with_war():
    """K0a (compile-time, no symbolic guard involved): the read ``A[i+1]`` is read-ahead (``WAR``,
    offset +1) versus the write ``A[i]`` but offset-0 (``'none'``) versus the write ``A[i+1]=E[i]`` in
    the SAME iteration. It must keep the just-written live value (``D == 2*E``), never be redirected to
    the pre-loop snapshot. Pre-fix the gate skipped only on ``RAW``/``complex`` and fired on ANY
    ``WAR``, so this ``{'WAR','none'}`` mix slipped through and D read the stale original A -- a silent
    miscompile with a purely numeric offset (the strict symbolic guard cannot catch it)."""
    n = 32
    A0 = np.random.default_rng(1).random(n)
    B0 = np.random.default_rng(2).random(n)
    E0 = np.random.default_rng(3).random(n)

    raw = _split_mixed_numeric_none.to_sdfg(simplify=True)
    Ar, Br, Er, Dr = A0.copy(), B0.copy(), E0.copy(), np.zeros(n)
    raw.compile()(A=Ar, B=Br, E=Er, D=Dr, N=n)

    cand = _split_mixed_numeric_none.to_sdfg(simplify=True)
    canonicalize(cand, validate=True, peel_limit=4, break_anti_dependence=True)
    Ac, Bc, Ec, Dc = A0.copy(), B0.copy(), E0.copy(), np.zeros(n)
    cand.compile()(A=Ac, B=Bc, E=Ec, D=Dc, N=n)

    assert np.allclose(Dr, Dc, equal_nan=True), ('the same-iteration read A[i+1] was redirected to the '
                                                 'stale snapshot instead of the just-written live value')
    assert np.allclose(Dc[1:n - 2], 2.0 * E0[1:n - 2]), 'D[i] must equal 2*E[i]'


if __name__ == '__main__':
    pytest.main([__file__, '-q'])
