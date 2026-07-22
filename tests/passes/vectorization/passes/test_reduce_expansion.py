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
    ExpandReduceVectorized, )

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


def _reduce_sdfg(wcr, identity, dtype, schedule=dtypes.ScheduleType.Sequential, tag=""):
    # A unique SDFG name per parametrization: parallel (xdist) workers
    # compiling different cases must not collide in .dacecache/<name>/.
    # Hash the raw wcr (a regex sanitiser would collapse + * & | ^ all to
    # '_', aliasing distinct operators into the same .dacecache dir).
    import hashlib as _hl
    h = _hl.md5(f"{tag}|{wcr}|{dtype.ctype}|{schedule.name}".encode()).hexdigest()[:10]
    sdfg = dace.SDFG(f"vr_{tag}_{h}")
    sdfg.add_array("A", [N], dtype)
    sdfg.add_array("out", [1], dtype)
    st = sdfg.add_state()
    rnode = Reduce("red", wcr=wcr, axes=[0], identity=identity)
    rnode.implementation = "vectorized"
    rnode.schedule = schedule
    st.add_node(rnode)
    st.add_edge(st.add_read("A"), None, rnode, None, dace.Memlet("A[0:N]"))
    st.add_edge(rnode, None, st.add_write("out"), None, dace.Memlet("out[0]"))
    return sdfg


_FP_OPS = [
    ("lambda a, b: a + b", 0.0, lambda x: np.add.reduce(x)),
    ("lambda a, b: a * b", 1.0, lambda x: np.multiply.reduce(x)),
    ("lambda a, b: max(a, b)", None, lambda x: np.maximum.reduce(x)),
    ("lambda a, b: min(a, b)", None, lambda x: np.minimum.reduce(x)),
]
_INT_OPS = [
    ("lambda a, b: a + b", 0, lambda x: np.add.reduce(x)),
    ("lambda a, b: a & b", -1, lambda x: np.bitwise_and.reduce(x)),
    ("lambda a, b: a | b", 0, lambda x: np.bitwise_or.reduce(x)),
    ("lambda a, b: a ^ b", 0, lambda x: np.bitwise_xor.reduce(x)),
]
_SIZES = [7, 8, 9, 16, 17, 64]


@pytest.mark.parametrize("size", _SIZES)
@pytest.mark.parametrize("wcr,identity,ref", _FP_OPS)
def test_vectorized_fp_reduction_matrix(wcr, identity, ref, size):
    sdfg = _reduce_sdfg(wcr, identity, dace.float64, tag=f"fp{size}")
    sdfg.expand_library_nodes()
    a = (np.random.rand(size) + 0.5).astype(np.float64)
    out = np.zeros(1, dtype=np.float64)
    sdfg(A=a, out=out, N=size)
    assert np.allclose(out[0], ref(a), atol=1e-10), f"size={size} {wcr}: {out[0]} vs {ref(a)}"


@pytest.mark.parametrize("size", _SIZES)
@pytest.mark.parametrize("wcr,identity,ref", _INT_OPS)
def test_vectorized_int_reduction_matrix(wcr, identity, ref, size):
    sdfg = _reduce_sdfg(wcr, identity, dace.int64, tag=f"int{size}")
    sdfg.expand_library_nodes()
    a = np.random.randint(1, 9999, size=size, dtype=np.int64)
    out = np.zeros(1, dtype=np.int64)
    sdfg(A=a, out=out, N=size)
    assert out[0] == ref(a), f"size={size} {wcr}: {out[0]} vs {ref(a)}"


def test_symbolic_length_vectorized_reduction():
    sdfg = _reduce_sdfg("lambda a, b: a + b", 0.0, dace.float64, tag="symlen")
    sdfg.expand_library_nodes()
    for n in (13, 100, 257):
        a = np.random.rand(n).astype(np.float64)
        out = np.zeros(1, dtype=np.float64)
        sdfg(A=a, out=out, N=n)
        assert np.allclose(out[0], a.sum(), atol=1e-10), f"N={n}"


def test_1d_full_reduction_takes_vectorized_path():
    sdfg = _reduce_sdfg("lambda a, b: a + b", 0.0, dace.float64, tag="path1d")
    sdfg.expand_library_nodes()
    codes = [n.code.as_string for n, _ in sdfg.all_nodes_recursive() if isinstance(n, dace.nodes.Tasklet)]
    assert any("horizontal_reduce_add" in c for c in codes), \
        "1-D full reduction must use the vectorized horizontal_reduce kernel"


def test_2d_partial_reduction_falls_back_to_pure():
    M = dace.symbol("M")
    sdfg = dace.SDFG("vecred_2d")
    sdfg.add_array("A", [M, N], dace.float64)
    sdfg.add_array("out", [M], dace.float64)
    st = sdfg.add_state()
    rnode = Reduce("red2d", wcr="lambda a, b: a + b", axes=[1], identity=0.0)
    rnode.implementation = "vectorized"
    rnode.schedule = dtypes.ScheduleType.Sequential
    st.add_node(rnode)
    st.add_edge(st.add_read("A"), None, rnode, None, dace.Memlet("A[0:M, 0:N]"))
    st.add_edge(rnode, None, st.add_write("out"), None, dace.Memlet("out[0:M]"))
    sdfg.expand_library_nodes()
    codes = [n.code.as_string for n, _ in sdfg.all_nodes_recursive() if isinstance(n, dace.nodes.Tasklet)]
    assert not any("horizontal_reduce" in c for c in codes), \
        "partial 2-D reduction must fall back to ExpandReducePure (no horizontal_reduce)"
    a = np.random.rand(5, 9).astype(np.float64)
    out = np.zeros(5, dtype=np.float64)
    sdfg(A=a, out=out, M=5, N=9)
    assert np.allclose(out, a.sum(axis=1), atol=1e-12)


if __name__ == "__main__":
    pytest.main([__file__, "-q"])
