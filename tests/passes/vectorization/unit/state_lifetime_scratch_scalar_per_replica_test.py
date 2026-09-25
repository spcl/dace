# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""A ``State``-lifetime scratch scalar written in a map gets its own name in every scope replica.

``replicate_scope`` renamed only ``Scope``-lifetime transients. The tile splitter replicates a map into
``__tile_main`` and ``__masked_tail``, so a ``State``-lifetime per-iteration scalar (CloudSC's
``_wcr_priv__Sub____out_4``) stayed shared by both. Nesting each body then saw a container used outside
it, passed it in as a non-transient ``Scalar``, and the widener took it for loop-invariant: W lanes were
copied into one element, surfacing only at codegen as an out-of-bounds copy.
"""
import numpy as np
import pytest

import dace
from dace.libraries.tileops._dispatch import detect_host_isa
from dace.transformation.helpers import replicate_scope
from dace.transformation.passes.vectorization.config import VectorizeConfig
from dace.transformation.passes.vectorization.vectorize_cpu_multi_dim import VectorizeCPUMultiDim

N = dace.symbol('N')
WIDTH = 8


def copy_through_scratch_scalar(lifetime: dace.AllocationLifetime) -> dace.SDFG:
    """``for i: t = b[i]; c[i] = t`` with ``t`` a one-element transient of the given lifetime."""
    sdfg = dace.SDFG(f'copy_through_{lifetime.name.lower()}_scalar')
    sdfg.add_array('b', [N], dace.float64)
    sdfg.add_array('c', [N], dace.float64)
    sdfg.add_scalar('t', dace.float64, transient=True, lifetime=lifetime)
    state = sdfg.add_state()
    entry, exit_ = state.add_map('m', {'i': '0:N'})
    tasklet = state.add_tasklet('assign', {'__in'}, {'__out'}, '__out = __in')
    scratch = state.add_access('t')
    state.add_memlet_path(state.add_read('b'), entry, tasklet, dst_conn='__in', memlet=dace.Memlet('b[i]'))
    state.add_edge(tasklet, '__out', scratch, None, dace.Memlet('t[0]'))
    state.add_memlet_path(scratch, exit_, state.add_write('c'), memlet=dace.Memlet('c[i]'))
    sdfg.validate()
    return sdfg


def replica_scratch_names(lifetime: dace.AllocationLifetime) -> set:
    sdfg = copy_through_scratch_scalar(lifetime)
    state = sdfg.start_state
    entry = next(n for n in state.nodes() if isinstance(n, dace.nodes.MapEntry))
    replica = replicate_scope(sdfg, state, state.scope_subgraph(entry))
    return {n.data for n in replica.nodes() if isinstance(n, dace.nodes.AccessNode)}


@pytest.mark.parametrize('lifetime, renamed', [
    (dace.AllocationLifetime.Scope, True),
    (dace.AllocationLifetime.State, True),
    (dace.AllocationLifetime.Persistent, False),
])
def test_a_replica_renames_a_scratch_scalar_only_when_its_value_ends_with_the_state(lifetime, renamed):
    names = replica_scratch_names(lifetime)
    assert ('t' not in names) == renamed, names


def test_a_state_lifetime_scratch_scalar_vectorizes_per_lane():
    sdfg = copy_through_scratch_scalar(dace.AllocationLifetime.State)
    VectorizeCPUMultiDim(VectorizeConfig(widths=(WIDTH, ), target_isa=detect_host_isa(),
                                         validate=True)).apply_pass(sdfg, {})
    size = 3 * WIDTH + 5
    b = np.random.default_rng(0).standard_normal(size)
    c = np.zeros(size)
    sdfg(b=b, c=c, N=size)
    np.testing.assert_array_equal(c, b)
