# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Strengthening probe for LoopToScan: reverse-iteration residue-class (stride>1) scan.

A backward loop whose carry links elements TWO apart:

    for jm in range(N, 1, -1):
        acc[jm - 2] = acc[jm] + delta[jm]

is a REVERSE stride-2 residue-class prefix sum: even array positions form one
independent scan, odd positions another, each seeded by its own pre-loop head
value (``acc[N]`` for one class, ``acc[N-1]`` for the other). ``_emit_seed_add``
fans the ``S`` pre-loop seeds out by ``_i mod S`` in BOTH directions -- forward
class ``k`` seeds from ``iter_start + k_r + k``, reverse class ``k`` seeds from
``k_r - k`` (the head at the high end of the range) -- so the shape lifts.
"""
import numpy as np

import dace
from dace.libraries.standard.nodes.scan import Scan
from dace.transformation.passes.loop_to_scan import LoopToScan

N = dace.symbol('N')


def _num_scan_nodes(sdfg):
    return sum(1 for n, _ in sdfg.all_nodes_recursive() if isinstance(n, Scan))


def test_reverse_stride2_residue_class_scan():
    @dace.program
    def backward_stride2(acc: dace.float64[N + 1], delta: dace.float64[N + 1]):
        for jm in range(N, 1, -1):
            acc[jm - 2] = acc[jm] + delta[jm]

    sdfg = backward_stride2.to_sdfg(simplify=True)
    res = LoopToScan().apply_pass(sdfg, {})
    sdfg.validate()

    # The reverse residue-class (stride>1) shape LIFTS: one Scan libnode running the
    # S independent class scans, plus the per-class seed fan-out in the apply Map.
    assert res, 'reverse stride-2 scan must lift'
    assert _num_scan_nodes(sdfg) == 1

    n = 12
    rng = np.random.default_rng(2079)
    acc = rng.standard_normal(n + 1)
    delta = rng.standard_normal(n + 1)
    expected = acc.copy()
    for jm in range(n, 1, -1):
        expected[jm - 2] = expected[jm] + delta[jm]

    got = acc.copy()
    sdfg(acc=got, delta=delta, N=n)
    # Bit-exact: each residue class is summed in the SAME order as the sequential
    # loop (the strided scan walks the class in iteration order), so no reassociation.
    assert np.array_equal(got, expected), (f'reverse stride-2 scan mismatch (lifted={res}, '
                                           f'scan_nodes={_num_scan_nodes(sdfg)}):\n got={got}\n exp={expected}\n'
                                           f' diff={np.abs(got - expected)}')


def test_reverse_stride3_residue_class_scan_odd_trip():
    """Stride 3 with a trip count that is NOT a multiple of the stride -- the class
    scans have unequal lengths and the last class's tail is short. Exercises the
    ``_i mod S`` fan-out at a non-aligned boundary."""

    @dace.program
    def backward_stride3(acc: dace.float64[N + 1], delta: dace.float64[N + 1]):
        for jm in range(N, 2, -1):
            acc[jm - 3] = acc[jm] + delta[jm]

    sdfg = backward_stride3.to_sdfg(simplify=True)
    res = LoopToScan().apply_pass(sdfg, {})
    sdfg.validate()
    assert res, 'reverse stride-3 scan must lift'
    assert _num_scan_nodes(sdfg) == 1

    n = 14
    rng = np.random.default_rng(311)
    acc = rng.standard_normal(n + 1)
    delta = rng.standard_normal(n + 1)
    expected = acc.copy()
    for jm in range(n, 2, -1):
        expected[jm - 3] = expected[jm] + delta[jm]

    got = acc.copy()
    sdfg(acc=got, delta=delta, N=n)
    assert np.array_equal(got, expected), f'reverse stride-3 scan mismatch:\n got={got}\n exp={expected}'
