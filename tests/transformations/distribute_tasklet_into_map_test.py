# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""``DistributeTaskletIntoMap`` must only absorb a free Tasklet that is what blocks Map fusion."""
import copy
import os

import numpy as np
import pytest

import dace
from dace import nodes
from dace.transformation.dataflow import DistributeTaskletIntoMap, MapFusionVertical

N = 8


def build(ordering_edge: bool = True,
          tasklet_reads_data: bool = False,
          tasklet_has_side_effects: bool = False,
          scalar_used_later: bool = False,
          second_map_end: int = N) -> dace.SDFG:
    """``map1{T[i]=A[i]*2}`` and ``map2{B[i]=T[i]+S}``, with a free ``S = 3.0`` Tasklet beside them.

    Without ``ordering_edge`` the two Maps already fuse; with it the Tasklet becomes an
    intermediate of ``map1``'s exit and fusion is refused. The remaining flags each break one
    precondition of the absorption.
    """
    sdfg = dace.SDFG("distribute_tasklet")
    for name in ("A", "B"):
        sdfg.add_array(name, [N], dace.float64)
    sdfg.add_array("T", [N], dace.float64, transient=True)
    sdfg.add_scalar("S", dace.float64, transient=True)
    state = sdfg.add_state("main", is_start_block=True)

    a, t, b, s = (state.add_access(name) for name in ("A", "T", "B", "S"))

    entry1, exit1 = state.add_map("map1", {"i": f"0:{N}"})
    body1 = state.add_tasklet("t1", {"a"}, {"o"}, "o = a * 2.0")
    state.add_memlet_path(a, entry1, body1, dst_conn="a", memlet=dace.Memlet("A[i]"))
    state.add_memlet_path(body1, exit1, t, src_conn="o", memlet=dace.Memlet("T[i]"))

    if tasklet_reads_data:
        free = state.add_tasklet("free", {"a"}, {"o"}, "o = a + 3.0")
        state.add_edge(state.add_access("A"), None, free, "a", dace.Memlet("A[0]"))
    else:
        free = state.add_tasklet("free", {}, {"o"}, "o = 3.0", side_effects=tasklet_has_side_effects or None)
    state.add_edge(free, "o", s, None, dace.Memlet("S[0]"))

    entry2, exit2 = state.add_map("map2", {"i": f"0:{second_map_end}"})
    body2 = state.add_tasklet("t2", {"t", "s"}, {"o"}, "o = t + s")
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


def run(sdfg: dace.SDFG) -> np.ndarray:
    """Execute in a forked child so a compiled kernel cannot take the test process down."""
    a = np.arange(1, N + 1, dtype=np.float64)
    b = np.zeros(N, dtype=np.float64)
    read_fd, write_fd = os.pipe()
    if os.fork() == 0:
        os.close(read_fd)
        sdfg(A=a, B=b)
        os.write(write_fd, b.tobytes())
        os._exit(0)
    os.close(write_fd)
    raw = b""
    while len(raw) < b.nbytes:
        chunk = os.read(read_fd, b.nbytes - len(raw))
        if not chunk:
            break
        raw += chunk
    os.close(read_fd)
    os.wait()
    assert len(raw) == b.nbytes, "child died before writing its result"
    return np.frombuffer(raw, dtype=np.float64)


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
