# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""``DistributeTaskletIntoMap`` must only absorb a free Tasklet that is what blocks Map fusion."""
import copy
from typing import Any, Dict

import numpy as np
import pytest

import dace
from dace import nodes
from dace.transformation.dataflow import DistributeTaskletIntoMap, MapFusionVertical
from tests.helpers.isolation import call_in_child

N = 8


def build(ordering_edge: bool = True,
          tasklet_reads_data: bool = False,
          tasklet_has_side_effects: bool = False,
          scalar_used_later: bool = False,
          second_map_end: int = N,
          tasklet_reads_symbol: bool = False,
          extra_consumer: bool = False) -> dace.SDFG:
    """``map1{T[i]=A[i]*2}`` and ``map2{B[i]=T[i]+S}``, with a free ``S = 3.0`` Tasklet beside them.

    Without ``ordering_edge`` the two Maps already fuse; with it the Tasklet becomes an
    intermediate of ``map1``'s exit and fusion is refused. The remaining flags each break one
    precondition of the absorption, except:

    :param tasklet_reads_symbol: The free Tasklet computes ``S = dt`` from a genuine symbol rather
        than a literal, so absorbing it moves a symbol reference inside a Map scope.
    :param extra_consumer: ``S`` also feeds an output ``C`` in the SAME state (``scalar_used_later``
        puts that second consumer in a later state instead).
    """
    sdfg = dace.SDFG("distribute_tasklet")
    for name in ("A", "B"):
        sdfg.add_array(name, [N], dace.float64)
    sdfg.add_array("T", [N], dace.float64, transient=True)
    sdfg.add_scalar("S", dace.float64, transient=True)
    if tasklet_reads_symbol:
        sdfg.add_symbol("dt", dace.float64)
    if extra_consumer:
        sdfg.add_array("C", [1], dace.float64)
    state = sdfg.add_state("main", is_start_block=True)

    a, t, b, s = (state.add_access(name) for name in ("A", "T", "B", "S"))

    entry1, exit1 = state.add_map("map1", {"i": f"0:{N}"})
    body1 = state.add_tasklet("t1", {"a": None}, {"o": None}, "o = a * 2.0")
    state.add_memlet_path(a, entry1, body1, dst_conn="a", memlet=dace.Memlet("A[i]"))
    state.add_memlet_path(body1, exit1, t, src_conn="o", memlet=dace.Memlet("T[i]"))

    if tasklet_reads_data:
        free = state.add_tasklet("free", {"a": None}, {"o": None}, "o = a + 3.0")
        state.add_edge(state.add_access("A"), None, free, "a", dace.Memlet("A[0]"))
    else:
        code = "o = dt" if tasklet_reads_symbol else "o = 3.0"
        free = state.add_tasklet("free", {}, {"o": None}, code, side_effects=tasklet_has_side_effects or None)
    state.add_edge(free, "o", s, None, dace.Memlet("S[0]"))
    if extra_consumer:
        state.add_edge(s, None, state.add_access("C"), None, dace.Memlet("S[0] -> [0]"))

    entry2, exit2 = state.add_map("map2", {"i": f"0:{second_map_end}"})
    body2 = state.add_tasklet("t2", {"t": None, "s": None}, {"o": None}, "o = t + s")
    state.add_memlet_path(t, entry2, body2, dst_conn="t", memlet=dace.Memlet("T[i]"))
    state.add_memlet_path(s, entry2, body2, dst_conn="s", memlet=dace.Memlet("S[0]"))
    state.add_memlet_path(body2, exit2, b, src_conn="o", memlet=dace.Memlet("B[i]"))

    if ordering_edge:
        state.add_edge(exit1, None, free, None, dace.Memlet())
    if scalar_used_later:
        later = sdfg.add_state_after(state, "later")
        later.add_edge(later.add_access("S"), None, later.add_access("B"), None, dace.Memlet("S[0] -> [0]"))

    sdfg.validate()
    return sdfg


def num_maps(sdfg: dace.SDFG) -> int:
    return sum(1 for state in sdfg.states() for node in state.nodes() if isinstance(node, nodes.MapEntry))


def shape_of(sdfg: dace.SDFG):
    return [(state.label, state.number_of_nodes(), state.number_of_edges()) for state in sdfg.states()]


def run_in_child(sdfg: dace.SDFG, extra: Dict[str, Any], wants_c: bool) -> np.ndarray:
    """Child body for :func:`run`: call ``sdfg`` on fresh buffers and return the outputs.

    Allocating here rather than shipping buffers in keeps every array owning its own memory -- DaCe
    rejects an argument whose ``.base`` is the pickle buffer an unpickled array carries.
    """
    a = np.arange(1, N + 1, dtype=np.float64)
    b = np.zeros(N, dtype=np.float64)
    c = np.zeros(1, dtype=np.float64) if wants_c else None
    sdfg(A=a, B=b, **({} if c is None else {"C": c}), **extra)
    return b if c is None else np.concatenate((b, c))


def run(sdfg: dace.SDFG, **extra) -> np.ndarray:
    """Execute in a spawned child so a compiled kernel cannot take the test process down.

    ``extra`` carries the optional fixture arguments (``dt``, ``C``); ``C`` is echoed back after
    ``B`` so a caller that asked for it can check the second consumer was still written. Spawned
    rather than forked: this process has already run compiled kernels, and a fork child deadlocks
    on the OpenMP team it inherits but does not own.
    """
    wants_c = extra.pop("C", None) is not None
    return call_in_child(run_in_child, sdfg, extra, wants_c)


def test_absorbing_the_free_tasklet_unblocks_fusion():
    """The ordering edge is the only obstacle, and removing the Tasklet from it restores fusion."""
    sdfg = build()
    oracle = np.arange(1, N + 1, dtype=np.float64) * 2.0 + 3.0
    assert np.array_equal(run(copy.deepcopy(sdfg)), oracle)
    assert sdfg.apply_transformations_repeated(MapFusionVertical, validate_all=True) == 0, "expected fusion to block"

    assert sdfg.apply_transformations_repeated(DistributeTaskletIntoMap, validate_all=True) == 1
    sdfg.validate()
    assert np.array_equal(run(copy.deepcopy(sdfg)), oracle), "absorbing the Tasklet changed B"

    assert sdfg.apply_transformations_repeated(MapFusionVertical, validate_all=True) == 1
    assert num_maps(sdfg) == 1
    assert np.array_equal(run(sdfg), oracle), "fusing after the absorption changed B"


def test_a_shape_that_already_fuses_is_left_alone():
    """Without the ordering edge the Tasklet blocks nothing, so moving it would be pure churn."""
    sdfg = build(ordering_edge=False)
    before = shape_of(sdfg)

    assert sdfg.apply_transformations_repeated(DistributeTaskletIntoMap) == 0
    assert shape_of(sdfg) == before
    assert sdfg.apply_transformations_repeated(MapFusionVertical, validate_all=True) == 1


def test_a_free_tasklet_reading_a_symbol_is_absorbed_and_keeps_the_symbol():
    """A literal-valued Tasklet says nothing about scoping. This one computes ``S = dt`` from a real
    symbol, so absorbing it moves that reference inside a Map scope -- the value has to survive."""
    dt = 0.25
    sdfg = build(tasklet_reads_symbol=True)
    oracle = np.arange(1, N + 1, dtype=np.float64) * 2.0 + dt
    assert np.array_equal(run(copy.deepcopy(sdfg), dt=dt), oracle)

    assert sdfg.apply_transformations_repeated(DistributeTaskletIntoMap, validate_all=True) == 1
    sdfg.validate()
    assert np.array_equal(run(copy.deepcopy(sdfg), dt=dt), oracle), "absorbing the Tasklet changed B"

    assert sdfg.apply_transformations_repeated(MapFusionVertical, validate_all=True) == 1
    assert num_maps(sdfg) == 1
    assert np.array_equal(run(sdfg, dt=dt), oracle), "fusing after the absorption changed B"


def test_a_second_same_state_consumer_of_the_scalar_still_gets_its_value():
    """``scalar_used_later`` covers a consumer in a LATER state. This one reads ``S`` in the same
    state, where the buffer being Map-scoped after the move would silently leave ``C`` unwritten --
    so whether or not the transformation fires, both outputs must come out right."""
    sdfg = build(extra_consumer=True)
    b_oracle = np.arange(1, N + 1, dtype=np.float64) * 2.0 + 3.0
    got = run(copy.deepcopy(sdfg), C=np.zeros(1, dtype=np.float64))
    assert np.array_equal(got[:N], b_oracle) and got[N] == 3.0

    sdfg.apply_transformations_repeated(DistributeTaskletIntoMap, validate_all=True)
    sdfg.validate()
    got = run(sdfg, C=np.zeros(1, dtype=np.float64))
    assert np.array_equal(got[:N], b_oracle), "absorbing the Tasklet changed B"
    assert got[N] == 3.0, "the second consumer of S lost its value"


@pytest.mark.parametrize("flags,reason", [
    ({
        "tasklet_reads_data": True
    }, "a replica would read whatever the input holds at that iteration"),
    ({
        "tasklet_has_side_effects": True
    }, "the side effect would happen once per iteration instead of once"),
    ({
        "scalar_used_later": True
    }, "a Map-scoped buffer does not survive to the next state"),
    ({
        "second_map_end": N - 1
    }, "the Maps cannot fuse anyway"),
])
def test_unsound_or_pointless_absorptions_are_refused(flags: dict, reason: str):
    sdfg = build(**flags)
    before = shape_of(sdfg)

    assert sdfg.apply_transformations_repeated(DistributeTaskletIntoMap) == 0, reason
    assert shape_of(sdfg) == before, "a refused match must not mutate the SDFG"


if __name__ == "__main__":
    pytest.main([__file__])
