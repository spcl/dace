# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""A constant-stride residue-class scan is a CPU lift, not a GPU one.

The decomposition of ``b[i] = b[i - S] + a[i]`` into ``S`` independent prefix scans exposes exactly
``S`` units of parallelism. ``S = 4`` saturates a four-thread team and leaves a device serial, and
the lift has displaced a form the GPU pipeline handled far better -- measured at XL, canonicalize
against the same graph without it: ``s1221`` 1.30e4 vs 1.02e3 ms, ``fission_dep_const_offset``
8.89e3 vs 2.14e2 ms. The same lift is a CPU win, so the gate is a target policy: it must fire on
``target='gpu'`` and must leave the CPU pipeline alone.
"""
import numpy as np
import pytest

import dace
from dace.sdfg import nodes
from dace.sdfg.state import LoopRegion
from dace.transformation.passes.canonicalize import pipeline as canon
from dace.transformation.passes.loop_to_scan import residue_classes_too_narrow_for_gpu

LEN_1D = dace.symbol('LEN_1D', dtype=dace.int64)


@dace.program
def s1221(a: dace.float64[LEN_1D], b: dace.float64[LEN_1D]):
    for i in range(4, LEN_1D):
        b[i] = b[i - 4] + a[i]


@dace.program
def contiguous_scan(a: dace.float64[LEN_1D], b: dace.float64[LEN_1D]):
    for i in range(1, LEN_1D):
        b[i] = b[i - 1] + a[i]


def scans_of(sdfg):
    return [
        n for g in sdfg.all_sdfgs_recursive() for st in g.all_states() for n in st.nodes()
        if isinstance(n, nodes.LibraryNode) and type(n).__name__ == 'Scan'
    ]


def test_the_predicate_only_claims_a_constant_stride_above_one():
    """``S == 1`` is the contiguous scan and says nothing about width; a symbolic S is unknown."""

    class Info:

        def __init__(self, stride):
            self.scan_stride = stride

    assert residue_classes_too_narrow_for_gpu([Info(4)])
    assert residue_classes_too_narrow_for_gpu([Info(2), Info(4)])
    assert not residue_classes_too_narrow_for_gpu([Info(1)])
    assert not residue_classes_too_narrow_for_gpu([Info(4), Info(1)])
    assert not residue_classes_too_narrow_for_gpu([Info(dace.symbol('K'))])
    assert not residue_classes_too_narrow_for_gpu([])


def test_cpu_still_lifts_the_residue_class_scan():
    sdfg = s1221.to_sdfg(simplify=False)
    canon.canonicalize(sdfg, target='cpu')
    assert scans_of(sdfg), 'the CPU lift is a measured win and must not regress'


def test_gpu_refuses_it_and_leaves_a_loop():
    sdfg = s1221.to_sdfg(simplify=False)
    canon.canonicalize(sdfg, target='gpu')
    assert not scans_of(sdfg), 'four residue classes is not GPU parallelism'
    assert any(
        isinstance(b, LoopRegion) for g in sdfg.all_sdfgs_recursive()
        for b in g.all_control_flow_regions(recursive=True)), 'the recurrence must survive as a loop'


@pytest.mark.parametrize('target', ['cpu', 'gpu'])
def test_a_contiguous_scan_is_never_gated(target):
    """Stride 1 is not a residue-class decomposition -- the gate must not touch it on any target."""
    sdfg = contiguous_scan.to_sdfg(simplify=False)
    canon.canonicalize(sdfg, target=target)
    assert scans_of(sdfg), f'the contiguous scan was gated on {target}'


def test_the_gated_graph_still_computes():
    sdfg = s1221.to_sdfg(simplify=False)
    canon.canonicalize(sdfg, target='gpu')
    n = 37
    rng = np.random.default_rng(5)
    a = rng.random(n)
    b = rng.random(n)
    want = b.copy()
    for i in range(4, n):
        want[i] = want[i - 4] + a[i]
    got = b.copy()
    sdfg(a=a, b=got, LEN_1D=n)
    assert np.allclose(got, want), f'{got} != {want}'


if __name__ == '__main__':
    test_the_predicate_only_claims_a_constant_stride_above_one()
    test_cpu_still_lifts_the_residue_class_scan()
    test_gpu_refuses_it_and_leaves_a_loop()
    test_a_contiguous_scan_is_never_gated('cpu')
    test_a_contiguous_scan_is_never_gated('gpu')
    test_the_gated_graph_still_computes()
