# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""RV-1b-alpha: the ``"vectorized"`` Reduce implementation + dispatcher.

The vectorization pipeline lifts recognised accumulators into a
standard ``Reduce`` node with ``implementation='vectorized'``.
``ExpandReduceVectorized`` is a schedule-aware dispatcher: Sequential /
Default fold via the (currently delegated) sequential path,
CPU_Multicore via the OpenMP reduction, and any other schedule or a
non-associative / custom reduction operator must raise (loud failure,
no silent fallback). These tests pin the registration, the
numerically-correct expand+compile for the two supported schedules,
and the loud raises.
"""
import numpy as np
import pytest

import dace
from dace import dtypes
# Importing the vectorization package registers the implementation.
import dace.transformation.passes.vectorization  # noqa: F401
from dace.libraries.standard.nodes.reduce import Reduce
from dace.transformation.passes.vectorization.reduce_expansion import (
    ExpandReduceVectorized,
)

N = dace.symbol("N")


def test_vectorized_implementation_is_registered():
    assert Reduce.implementations.get("vectorized") is ExpandReduceVectorized
    assert ExpandReduceVectorized._match_node is Reduce


def _sum_sdfg(schedule):
    """A[N] -> scalar sum via a Reduce node with implementation='vectorized'."""
    sdfg = dace.SDFG(f"vecred_{schedule.name}")
    sdfg.add_array("A", [N], dace.float64)
    sdfg.add_array("out", [1], dace.float64)
    st = sdfg.add_state()
    rnode = Reduce("reduce_sum", wcr="lambda a, b: a + b", axes=[0], identity=0.0)
    rnode.implementation = "vectorized"
    rnode.schedule = schedule
    st.add_node(rnode)
    st.add_edge(st.add_read("A"), None, rnode, None, dace.Memlet("A[0:N]"))
    st.add_edge(rnode, None, st.add_write("out"), None, dace.Memlet("out[0]"))
    return sdfg


@pytest.mark.parametrize("schedule", [
    dtypes.ScheduleType.Sequential,
    dtypes.ScheduleType.Default,
    dtypes.ScheduleType.CPU_Multicore,
])
def test_vectorized_reduce_expands_and_is_numerically_correct(schedule):
    sdfg = _sum_sdfg(schedule)
    sdfg.expand_library_nodes()
    assert not any(isinstance(n, Reduce) for n, _ in sdfg.all_nodes_recursive())
    a = np.random.rand(64).astype(np.float64)
    out = np.zeros(1, dtype=np.float64)
    sdfg(A=a, out=out, N=64)
    assert np.allclose(out[0], a.sum(), atol=1e-12), f"{out[0]} vs {a.sum()}"


def test_unsupported_schedule_raises():
    sdfg = _sum_sdfg(dtypes.ScheduleType.GPU_Device)
    rnode = next(n for n, _ in sdfg.all_nodes_recursive() if isinstance(n, Reduce))
    state = next(s for s in sdfg.states() if rnode in s.nodes())
    with pytest.raises(NotImplementedError, match="not supported in the CPU"):
        ExpandReduceVectorized.expansion(rnode, state, sdfg)


def test_unsupported_reduction_operator_raises():
    sdfg = dace.SDFG("vecred_custom")
    sdfg.add_array("A", [N], dace.float64)
    sdfg.add_array("out", [1], dace.float64)
    st = sdfg.add_state()
    # A non-associative / custom reduction (no horizontal-reduce identity).
    rnode = Reduce("reduce_custom", wcr="lambda a, b: a - b", axes=[0])
    rnode.implementation = "vectorized"
    rnode.schedule = dtypes.ScheduleType.Sequential
    st.add_node(rnode)
    st.add_edge(st.add_read("A"), None, rnode, None, dace.Memlet("A[0:N]"))
    st.add_edge(rnode, None, st.add_write("out"), None, dace.Memlet("out[0]"))
    with pytest.raises(NotImplementedError, match="no associative"):
        ExpandReduceVectorized.expansion(rnode, st, sdfg)


if __name__ == "__main__":
    pytest.main([__file__, "-q"])
