# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Tests for the CPU specialization RecomputeOversizedIntermediates (producer -> consumer recompute).

The trade is cache-resident vs not: an intermediate that fits in the last-level cache is cheaper to
materialize once and read back, one that does not costs a full DRAM write plus a full DRAM read
that recomputing never pays. Every "declines" case below asserts that the UNGATED ``OTFMapFusion``
would have applied, so a shape change that makes the pattern disappear fails the test instead of
passing it vacuously.
"""
import numpy as np

import dace
from dace import data
from dace.sdfg import nodes
from dace.transformation.dataflow import OTFMapFusion
from dace.transformation.passes.cpu_specialization import RecomputeOversizedIntermediates
from dace.transformation.passes.cpu_specialization.machine import topology
from dace.transformation.passes.cpu_specialization.recompute_oversized_intermediates import (intermediate_outgrows_cache
                                                                                             )

N = dace.symbol('N')

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
    return [
        n for st in sdfg.all_states() for n in st.nodes() if isinstance(n, nodes.MapEntry) and st.entry_node(n) is None
    ]


def assert_ungated_fusion_applies(prog):
    """The pattern is there: only the gate is what declines it below."""
    probe = prog.to_sdfg(simplify=True)
    assert probe.apply_transformations_repeated(OTFMapFusion, validate_all=False) > 0


def test_oversized_intermediate_is_recomputed():
    """A symbolic extent is assumed big, so the intermediate goes and the two maps become one."""
    sdfg = chain_symbolic.to_sdfg(simplify=True)
    before = transients(sdfg)
    assert len(before) == 1 and len(top_level_maps(sdfg)) == 2

    assert RecomputeOversizedIntermediates().apply_pass(sdfg, {}) == 1
    sdfg.validate()
    assert before[0] not in sdfg.arrays
    assert len(top_level_maps(sdfg)) == 1

    n = 64
    rng = np.random.default_rng(5)
    a = rng.random(n)
    out = np.zeros(n)
    sdfg(a=a, out=out, N=n)
    assert np.allclose(out, a * 2.0 + 1.0)


def test_cache_resident_intermediate_stays_materialized():
    """128 bytes fits any LLC, so reading it back beats recomputing and the pass declines."""
    assert_ungated_fusion_applies(chain_small)

    sdfg = chain_small.to_sdfg(simplify=True)
    assert RecomputeOversizedIntermediates().apply_pass(sdfg, {}) is None
    assert len(transients(sdfg)) == 1
    assert len(top_level_maps(sdfg)) == 2


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

    writers = [(st, n) for st in sdfg.all_states() for n in st.nodes()
               if isinstance(n, nodes.AccessNode) and n.data == intermediate[0] and st.in_degree(n) > 0]
    assert writers, 'the intermediate is read but nothing writes it'

    n = 64
    rng = np.random.default_rng(7)
    a = rng.random(n)
    out = np.zeros(n)
    sdfg(a=a, out=out, N=n)
    assert np.allclose(out, 3.0 * (a * 2.0))


def test_the_gate_reads_the_host_last_level_cache():
    """The threshold is the host's own LLC, not a constant measured on the development box."""
    llc = topology().llc_bytes
    sdfg = dace.SDFG('gate')
    sdfg.add_array('over', [llc // 8 + 1], dace.float64, transient=True)
    sdfg.add_array('under', [llc // 8 - 1], dace.float64, transient=True)
    sdfg.add_array('symbolic', [N], dace.float64, transient=True)
    sdfg.add_scalar('scalar', dace.float64, transient=True)

    assert intermediate_outgrows_cache(sdfg, 'over')
    assert not intermediate_outgrows_cache(sdfg, 'under')
    assert intermediate_outgrows_cache(sdfg, 'symbolic')
    assert not intermediate_outgrows_cache(sdfg, 'scalar')


if __name__ == '__main__':
    test_oversized_intermediate_is_recomputed()
    test_cache_resident_intermediate_stays_materialized()
    test_two_consumers_stay_materialized()
    test_the_gate_reads_the_host_last_level_cache()
