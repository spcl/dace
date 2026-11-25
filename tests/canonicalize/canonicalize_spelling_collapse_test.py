# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Regression tests for multi-spelling canonical collapse (CPF prime examples).

Each test exercises distinct source spellings of the same computation -- reverse loops,
prime-factor hand unrolls with ``while`` + remainder loops, multi-accumulator reductions
with a separate ``tail`` accumulator, and dimension-swapped / multi-level tiled stencils --
and verifies that :func:`canonicalize` collapses them to the same high-level parallel IR
while preserving exact numerical semantics (and refusing unsound accumulator merges).
"""

import numpy as np
import pytest

import dace
from dace.sdfg import nodes
from dace.sdfg.state import LoopRegion
from dace.transformation.passes.canonicalize import canonicalize
from dace.transformation.passes.canonicalize.assume_symbols_nonnegative import declare_nonnegative_symbols

N = dace.symbol("N")
NBLK = dace.symbol("NBLK")
TSTEPS = dace.symbol("TSTEPS")


def map_entries_of(sdfg: dace.SDFG) -> list[nodes.MapEntry]:
    return [n for n, unused in sdfg.all_nodes_recursive() if isinstance(n, nodes.MapEntry)]


def loops_of(sdfg: dace.SDFG) -> list[LoopRegion]:
    return [b for b in sdfg.all_control_flow_blocks(recursive=True) if isinstance(b, LoopRegion)]


def reduces_of(sdfg: dace.SDFG) -> list[nodes.LibraryNode]:
    return [
        n
        for n, unused in sdfg.all_nodes_recursive()
        if isinstance(n, nodes.LibraryNode) and "Reduce" in type(n).__name__
    ]


# --- Family 1: sum (4 distinct spellings -> single Reduce over a[0:N]) ---


@dace.program
def sum_plain(a: dace.float64[N], out: dace.float64[1]):
    s = 0.0
    for i in range(N):
        s += a[i]
    out[0] = s


@dace.program
def sum_unrolled_counter(a: dace.float64[4 * NBLK], out: dace.float64[1]):
    s = 0.0
    idx = 0
    for unused in range(NBLK):
        s += a[idx]
        idx += 1
        s += a[idx]
        idx += 1
        s += a[idx]
        idx += 1
        s += a[idx]
        idx += 1
    out[0] = s


@dace.program
def sum_11_accs_while_tail(a: dace.float64[N], out: dace.float64[1]):
    """Benchmark ``unroll_reduction_11_accs`` shape: while-loop 11x unroll + separate tail accumulator."""
    s0 = 0.0
    s1 = 0.0
    s2 = 0.0
    s3 = 0.0
    s4 = 0.0
    s5 = 0.0
    s6 = 0.0
    s7 = 0.0
    s8 = 0.0
    s9 = 0.0
    s10 = 0.0
    i = 0
    while i + 11 <= N:
        s0 += a[i + 0]
        s1 += a[i + 1]
        s2 += a[i + 2]
        s3 += a[i + 3]
        s4 += a[i + 4]
        s5 += a[i + 5]
        s6 += a[i + 6]
        s7 += a[i + 7]
        s8 += a[i + 8]
        s9 += a[i + 9]
        s10 += a[i + 10]
        i += 11
    tail = 0.0
    while i < N:
        tail += a[i]
        i += 1
    out[0] = s0 + s1 + s2 + s3 + s4 + s5 + s6 + s7 + s8 + s9 + s10 + tail


# Reproducer: CPF canonicalize: collapse prime-unrolled and split-accumulator spellings (E5) with regres... (74a4d75479)
def test_sum_spellings_collapse_to_single_reduce():
    """Plain loop, 4x carried counter, and 11-accumulator while+tail all lift to ONE Reduce over [0, N)."""
    rng = np.random.default_rng(101)
    for prog, sym, sizes in (
        (sum_plain, "N", [1, 11, 25, 100]),
        (sum_11_accs_while_tail, "N", [1, 5, 10, 11, 12, 25, 100]),
        (sum_unrolled_counter, "NBLK", [1, 4, 25]),
    ):
        sdfg = prog.to_sdfg(simplify=False)
        declare_nonnegative_symbols(sdfg, [sym])
        canonicalize(sdfg, target="cpu", validate=True)
        assert len(reduces_of(sdfg)) == 1, f"{prog.name}: expected 1 Reduce, got {len(reduces_of(sdfg))}"
        assert len(loops_of(sdfg)) == 0, f"{prog.name}: expected 0 residual loops"
        assert len(map_entries_of(sdfg)) == 0, f"{prog.name}: expected 0 residual maps"
        for sz in sizes:
            arr_len = 4 * sz if sym == "NBLK" else sz
            a = rng.standard_normal(arr_len)
            out = np.zeros(1)
            sdfg(a=a.copy(), out=out, **{sym: sz})
            assert np.isclose(out[0], np.sum(a), rtol=1e-12, atol=1e-12), f"{prog.name} ({sym}={sz})"


# --- Family 2: add_one (forward, reverse, prime-17 while+remainder -> single 0:N Map) ---


@dace.program
def add_one_forward(a: dace.float64[N], b: dace.float64[N]):
    for i in range(N):
        b[i] = a[i] + 1.0


@dace.program
def add_one_reverse(a: dace.float64[N], b: dace.float64[N]):
    for i in range(N - 1, -1, -1):
        b[i] = a[i] + 1.0


@dace.program
def add_one_prime17_while(a: dace.float64[N], b: dace.float64[N]):
    i = 0
    while i + 17 <= N:
        b[i + 0] = a[i + 0] + 1.0
        b[i + 1] = a[i + 1] + 1.0
        b[i + 2] = a[i + 2] + 1.0
        b[i + 3] = a[i + 3] + 1.0
        b[i + 4] = a[i + 4] + 1.0
        b[i + 5] = a[i + 5] + 1.0
        b[i + 6] = a[i + 6] + 1.0
        b[i + 7] = a[i + 7] + 1.0
        b[i + 8] = a[i + 8] + 1.0
        b[i + 9] = a[i + 9] + 1.0
        b[i + 10] = a[i + 10] + 1.0
        b[i + 11] = a[i + 11] + 1.0
        b[i + 12] = a[i + 12] + 1.0
        b[i + 13] = a[i + 13] + 1.0
        b[i + 14] = a[i + 14] + 1.0
        b[i + 15] = a[i + 15] + 1.0
        b[i + 16] = a[i + 16] + 1.0
        i += 17
    while i < N:
        b[i] = a[i] + 1.0
        i += 1


# Reproducer: CPF canonicalize: collapse prime-unrolled and split-accumulator spellings (E5) with regres... (74a4d75479)
def test_add_one_spellings_collapse_to_single_unit_map():
    """Forward, reverse, and prime-17 while+remainder all collapse to ONE Map over 0:N."""
    rng = np.random.default_rng(102)
    for prog in (add_one_forward, add_one_reverse, add_one_prime17_while):
        sdfg = prog.to_sdfg(simplify=False)
        declare_nonnegative_symbols(sdfg, ["N"])
        canonicalize(sdfg, target="cpu", validate=True)
        maps = map_entries_of(sdfg)
        assert len(maps) == 1, f"{prog.name}: expected 1 Map, got {len(maps)} ({[m.map.range for m in maps]})"
        assert str(maps[0].map.range) == "0:N", f"{prog.name}: expected 0:N, got {maps[0].map.range}"
        assert len(loops_of(sdfg)) == 0
        for n in (1, 16, 17, 18, 37, 68):
            a = rng.standard_normal(n)
            b = np.zeros(n)
            sdfg(a=a.copy(), b=b, N=n)
            assert np.allclose(b, a + 1.0, rtol=1e-12, atol=1e-12), f"{prog.name} (N={n})"


# --- Family 3: saxpy (reroll_saxpy7 simplifies exact 7*NBLK bound without int_floor) ---


@dace.program
def saxpy_unrolled_7(a: dace.float64[7 * NBLK], b: dace.float64[7 * NBLK]):
    for i in range(0, 7 * NBLK, 7):
        a[i] = a[i] + b[i] * 2.0
        a[i + 1] = a[i + 1] + b[i + 1] * 2.0
        a[i + 2] = a[i + 2] + b[i + 2] * 2.0
        a[i + 3] = a[i + 3] + b[i + 3] * 2.0
        a[i + 4] = a[i + 4] + b[i + 4] * 2.0
        a[i + 5] = a[i + 5] + b[i + 5] * 2.0
        a[i + 6] = a[i + 6] + b[i + 6] * 2.0


# Reproducer: CPF canonicalize: collapse prime-unrolled and split-accumulator spellings (E5) with regres... (74a4d75479)
def test_saxpy_reroll_simplifies_exact_multiple_bound():
    """Exact-multiple unrolled loops simplify ``7 * int_floor(7*NBLK - 1, 7) + 7`` to ``7*NBLK``."""
    sdfg = saxpy_unrolled_7.to_sdfg(simplify=False)
    declare_nonnegative_symbols(sdfg, ["NBLK"])
    canonicalize(sdfg, target="cpu", validate=True)
    maps = map_entries_of(sdfg)
    assert len(maps) == 1
    assert str(maps[0].map.range) == "0:7*NBLK", f"expected clean 0:7*NBLK, got {maps[0].map.range}"
    rng = np.random.default_rng(103)
    for nblk in (1, 4, 20):
        a = rng.standard_normal(7 * nblk)
        b = rng.standard_normal(7 * nblk)
        ref = a + 2.0 * b
        sdfg(a=a, b=b.copy(), NBLK=nblk)
        assert np.allclose(a, ref, rtol=1e-12, atol=1e-12)


# --- Soundness refusal tests for split-accumulator & partial-fold rewrites ---


@dace.program
def split_acc_escaped_prefix_must_not_fuse(a: dace.float64[N], out: dace.float64[2]):
    """``s`` is read both in ``s + tail`` AND directly in ``out[1] = s``: must NOT merge ``tail`` into ``s``."""
    s = 0.0
    i = 0
    while i + 4 <= N:
        s += a[i + 0]
        s += a[i + 1]
        s += a[i + 2]
        s += a[i + 3]
        i += 4
    tail = 0.0
    while i < N:
        tail += a[i]
        i += 1
    out[0] = s + tail
    out[1] = s


@dace.program
def multi_acc_split_outputs_must_not_reroll(a: dace.float64[N], out: dace.float64[2]):
    """Two accumulators written to distinct outputs share an AccessNode sink but never combine."""
    s0 = 0.0
    s1 = 0.0
    for i in range(0, N - 1, 2):
        s0 += a[i]
        s1 += a[i + 1]
    out[0] = s0 + 1.0
    out[1] = s1 + 1.0


@pytest.mark.parametrize(
    "prog,check_fn",
    [
        (split_acc_escaped_prefix_must_not_fuse, lambda a: np.array([np.sum(a), np.sum(a[: (len(a) // 4) * 4])])),
        (multi_acc_split_outputs_must_not_reroll, lambda a: np.array([np.sum(a[0::2]) + 1.0, np.sum(a[1::2]) + 1.0])),
    ],
)
# Reproducer: CPF canonicalize: collapse prime-unrolled and split-accumulator spellings (E5) with regres... (74a4d75479)
def test_unsound_accumulator_merges_refused(prog, check_fn):
    sdfg = prog.to_sdfg(simplify=False)
    declare_nonnegative_symbols(sdfg, ["N"])
    canonicalize(sdfg, target="cpu", validate=True)
    rng = np.random.default_rng(104)
    for n in (6, 14, 22):
        a = rng.standard_normal(n)
        out = np.zeros(2)
        sdfg(a=a.copy(), out=out, N=n)
        assert np.allclose(out, check_fn(a), rtol=1e-12, atol=1e-12), f"{prog.name} (N={n}): got {out}"
