# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Tests for the CPU specialization RecomputeOversizedIntermediates (gated producer -> consumer recompute).

Every "declines" case asserts that the UNGATED ``OTFMapFusion`` would have applied, so a shape change that makes the
pattern disappear fails the test instead of passing it vacuously.
"""

import copy

import numpy as np

import dace
from dace import data, symbolic
from dace.sdfg import nodes
from dace.transformation.dataflow import OTFMapFusion
from dace.transformation.passes.cpu_specialization import RecomputeOversizedIntermediates
from dace.transformation.passes.cpu_specialization.machine import topology
from dace.transformation.passes.cpu_specialization.recompute_oversized_intermediates import intermediate_outgrows_cache
from tests.sdfg.cfg_list_in_place_test import assert_tree_consistent


def assert_cfg_list_matches_reset(sdfg: dace.SDFG) -> None:
    """The kept CFG list and ids equal a fresh copy's after ``reset_cfg_list``, and every parent pointer holds."""
    fresh = copy.deepcopy(sdfg)
    fresh.reset_cfg_list()
    kept = [(type(r).__name__, r.label, r.cfg_id) for r in sdfg.cfg_list]
    assert kept == [(type(r).__name__, r.label, r.cfg_id) for r in fresh.cfg_list]
    assert_tree_consistent(sdfg)


N = dace.symbol("N")
M = dace.symbol("M")

#: Small enough to fit any last-level cache this could run on: 16 doubles is 128 bytes.
SMALL = 16


@dace.program
def chain_symbolic(a: dace.float64[N], out: dace.float64[N]):
    t = dace.define_local([N], dace.float64)
    for i in dace.map[0:N]:
        t[i] = a[i] * 2.0
    for i in dace.map[0:N]:
        out[i] = t[i] + 1.0


@dace.program
def chain_small(a: dace.float64[SMALL], out: dace.float64[SMALL]):
    t = dace.define_local([SMALL], dace.float64)
    for i in dace.map[0:SMALL]:
        t[i] = a[i] * 2.0
    for i in dace.map[0:SMALL]:
        out[i] = t[i] + 1.0


@dace.program
def stencil_symbolic(a: dace.float64[N + 2], out: dace.float64[N]):
    t = dace.define_local([N + 2], dace.float64)
    for i in dace.map[0 : N + 2]:
        t[i] = a[i] * 2.0 + 1.0
    for i in dace.map[0:N]:
        out[i] = t[i] + t[i + 1] + t[i + 2]


@dace.program
def stencil_small(a: dace.float64[SMALL + 2], out: dace.float64[SMALL]):
    t = dace.define_local([SMALL + 2], dace.float64)
    for i in dace.map[0 : SMALL + 2]:
        t[i] = a[i] * 2.0 + 1.0
    for i in dace.map[0:SMALL]:
        out[i] = t[i] + t[i + 1] + t[i + 2]


@dace.program
def stencil_small_symbol(a: dace.float64[M + 2], out: dace.float64[M]):
    t = dace.define_local([M + 2], dace.float64)
    for i in dace.map[0 : M + 2]:
        t[i] = a[i] * 2.0 + 1.0
    for i in dace.map[0:M]:
        out[i] = t[i] + t[i + 1] + t[i + 2]


@dace.program
def chain_two_consumers(a: dace.float64[N], out: dace.float64[N], other: dace.float64[N]):
    t = dace.define_local([N], dace.float64)
    for i in dace.map[0:N]:
        t[i] = a[i] * 2.0
    for i in dace.map[0:N]:
        out[i] = t[i] + 1.0
    for i in dace.map[0:N]:
        other[i] = t[i] * 3.0


@dace.program
def chain_consumer_in_a_loop(a: dace.float64[N], out: dace.float64[N]):
    t = dace.define_local([N], dace.float64)
    for i in dace.map[0:N]:
        t[i] = a[i] * 2.0
    for _ in range(3):
        for i in dace.map[0:N]:
            out[i] = out[i] + t[i]


def transients(sdfg):
    return [n for n, d in sdfg.arrays.items() if d.transient and isinstance(d, data.Array)]


def top_level_maps(sdfg):
    return [n for st in sdfg.states() for n in st.nodes() if isinstance(n, nodes.MapEntry) and st.entry_node(n) is None]


def assert_ungated_fusion_applies(prog):
    """The pattern is there: only the gate is what declines it below."""
    probe = prog.to_sdfg(simplify=True)
    assert probe.apply_transformations_repeated(OTFMapFusion, validate_all=False) > 0


def assert_matches_unfused(fused: dace.SDFG, unfused: dace.SDFG, n: int, halo: int = 0, **symbols) -> None:
    """The fused SDFG computes the unfused SDFG's output on random input."""
    unfused.name = fused.name + "_unfused"
    a = np.random.default_rng(11).random(n + halo)
    expected, got = np.zeros(n), np.zeros(n)
    unfused(a=a.copy(), out=expected, **symbols)
    fused(a=a.copy(), out=got, **symbols)
    assert np.allclose(got, expected)


def test_oversized_intermediate_is_recomputed():
    """A symbolic extent is assumed big, so the intermediate goes and the two maps become one."""
    sdfg = chain_symbolic.to_sdfg(simplify=True)
    before = transients(sdfg)
    assert len(before) == 1 and len(top_level_maps(sdfg)) == 2

    assert RecomputeOversizedIntermediates().apply_pass(sdfg, {}) == 1
    sdfg.validate()
    assert_cfg_list_matches_reset(sdfg)
    assert before[0] not in sdfg.arrays
    assert len(top_level_maps(sdfg)) == 1

    n = 64
    rng = np.random.default_rng(5)
    a = rng.random(n)
    out = np.zeros(n)
    sdfg(a=a, out=out, N=n)
    assert np.allclose(out, a * 2.0 + 1.0)


def test_small_read_once_intermediate_is_fused():
    """Each element read once recomputes nothing, so even a cache-resident intermediate goes."""
    sdfg = chain_small.to_sdfg(simplify=True)
    unfused = copy.deepcopy(sdfg)
    intermediate = transients(sdfg)
    assert RecomputeOversizedIntermediates().apply_pass(sdfg, {}) == 1
    sdfg.validate()
    assert intermediate[0] not in sdfg.arrays and len(top_level_maps(sdfg)) == 1
    assert_matches_unfused(sdfg, unfused, SMALL)


def test_large_multiply_read_pointwise_intermediate_is_recomputed():
    """A symbolic (not SMALL) intermediate read at three offsets is recomputed from its pointwise producer."""
    sdfg = stencil_symbolic.to_sdfg(simplify=True)
    unfused = copy.deepcopy(sdfg)
    intermediate = transients(sdfg)
    assert RecomputeOversizedIntermediates().apply_pass(sdfg, {}) == 1
    sdfg.validate()
    assert_cfg_list_matches_reset(sdfg)
    assert intermediate[0] not in sdfg.arrays and len(top_level_maps(sdfg)) == 1
    assert_matches_unfused(sdfg, unfused, 64, halo=2, N=64)


def test_small_multiply_read_intermediate_stays_materialized():
    """A 144-byte intermediate read at three offsets is below L2: reading it back beats recomputing."""
    assert_ungated_fusion_applies(stencil_small)

    sdfg = stencil_small.to_sdfg(simplify=True)
    unfused = copy.deepcopy(sdfg)
    assert RecomputeOversizedIntermediates().apply_pass(sdfg, {}) is None
    assert len(transients(sdfg)) == 1 and len(top_level_maps(sdfg)) == 2
    assert_matches_unfused(sdfg, unfused, SMALL, halo=2)


def test_small_symbol_multiply_read_intermediate_stays_materialized():
    """An intermediate sized by a SMALL symbol only is cache-resident by the size facts."""
    assert_ungated_fusion_applies(stencil_small_symbol)

    sdfg = stencil_small_symbol.to_sdfg(simplify=True)
    sdfg.set_symbol_scale("M", symbolic.Scale.SMALL)
    assert RecomputeOversizedIntermediates().apply_pass(sdfg, {}) is None
    assert len(transients(sdfg)) == 1 and len(top_level_maps(sdfg)) == 2


def test_two_consumers_stay_materialized():
    """Sharing one buffer across consumers is what the cache is for; fusing would re-read the
    producer's inputs once per consumer, which this pass has no model for."""
    assert_ungated_fusion_applies(chain_two_consumers)

    sdfg = chain_two_consumers.to_sdfg(simplify=True)
    assert RecomputeOversizedIntermediates().apply_pass(sdfg, {}) is None
    assert len(transients(sdfg)) == 1
    assert len(top_level_maps(sdfg)) == 3


def test_consumers_in_separate_states_stay_materialized():
    """The consumer count is a property of the SDFG, not of the state the match was found in.

    stencil_3d reads its ``padded`` intermediate at six offsets per tap inside the radius loop, so
    every individual state sees exactly one consumer. Counting per state fused the producer into
    one of them and left the rest reading a buffer nothing writes -- the surviving access node came
    out ``in=0, out=6``, the SDFG still validated, and the kernel returned uninitialized memory.
    """
    sdfg = chain_consumer_in_a_loop.to_sdfg(simplify=True)
    intermediate = transients(sdfg)
    assert len(intermediate) == 1

    assert RecomputeOversizedIntermediates().apply_pass(sdfg, {}) is None
    sdfg.validate()
    assert intermediate[0] in sdfg.arrays

    writers = [
        (st, n)
        for st in sdfg.states()
        for n in st.nodes()
        if isinstance(n, nodes.AccessNode) and n.data == intermediate[0] and st.in_degree(n) > 0
    ]
    assert writers, "the intermediate is read but nothing writes it"

    n = 64
    rng = np.random.default_rng(7)
    a = rng.random(n)
    out = np.zeros(n)
    sdfg(a=a, out=out, N=n)
    assert np.allclose(out, 3.0 * (a * 2.0))


def test_the_gate_reads_the_host_l2_cache_and_size_facts():
    """The threshold is the host's own per-core L2; a size over SMALL symbols only is below it."""
    l2 = topology().l2_bytes
    sdfg = dace.SDFG("gate")
    sdfg.add_symbol("M", dace.int64)
    sdfg.set_symbol_scale("M", symbolic.Scale.SMALL)
    sdfg.add_array("over", [l2 // 8 + 1], dace.float64, transient=True)
    sdfg.add_array("under", [l2 // 8 - 1], dace.float64, transient=True)
    sdfg.add_array("symbolic", [N], dace.float64, transient=True)
    sdfg.add_array("small_symbol", [M, M], dace.float64, transient=True)
    sdfg.add_array("mixed", [M, N], dace.float64, transient=True)
    sdfg.add_scalar("scalar", dace.float64, transient=True)
    facts = sdfg.facts()

    assert intermediate_outgrows_cache(sdfg, "over", facts)
    assert not intermediate_outgrows_cache(sdfg, "under", facts)
    assert intermediate_outgrows_cache(sdfg, "symbolic", facts)
    assert not intermediate_outgrows_cache(sdfg, "small_symbol", facts)
    assert intermediate_outgrows_cache(sdfg, "mixed", facts)
    assert not intermediate_outgrows_cache(sdfg, "scalar", facts)


if __name__ == "__main__":
    test_oversized_intermediate_is_recomputed()
    test_small_read_once_intermediate_is_fused()
    test_large_multiply_read_pointwise_intermediate_is_recomputed()
    test_small_multiply_read_intermediate_stays_materialized()
    test_small_symbol_multiply_read_intermediate_stays_materialized()
    test_two_consumers_stay_materialized()
    test_consumers_in_separate_states_stay_materialized()
    test_the_gate_reads_the_host_l2_cache_and_size_facts()
