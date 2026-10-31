# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Tests for :class:`SplitMapLocalTransients`: a map-local scratch shared by name across states is split per scope."""

import numpy as np

import dace
from dace import dtypes
from dace.sdfg import nodes
from dace.transformation.passes.canonicalize.split_map_local_transients import SplitMapLocalTransients

N = 1000


def _two_branch_sdfg(read_before_write: bool = False) -> dace.SDFG:
    """``if flag: B = 2 * A + 1 else: B = 3 * A + 1``, each branch a parallel map through the scratch ``t``."""
    sdfg = dace.SDFG("shared_scratch" + ("_rbw" if read_before_write else ""))
    sdfg.add_symbol("flag", dace.int64)
    sdfg.add_array("A", [N], dace.float64)
    sdfg.add_array("B", [N], dace.float64)
    sdfg.add_scalar("t", dace.float64, transient=True)
    init = sdfg.add_state("init", is_start_block=True)
    for k, (cond, factor) in enumerate((("flag != 0", 2.0), ("flag == 0", 3.0))):
        st = sdfg.add_state(f"branch{k}")
        sdfg.add_edge(init, st, dace.InterstateEdge(cond))
        me, mx = st.add_map(f"m{k}", {"i": f"0:{N}"}, schedule=dtypes.ScheduleType.CPU_Multicore)
        t = st.add_access("t")
        first = st.add_tasklet(f"scale{k}", {"x"}, {"y"}, f"y = {factor} * x")
        second = st.add_tasklet(f"shift{k}", {"y"}, {"z"}, "z = y + 1.0")
        st.add_memlet_path(st.add_read("A"), me, first, dst_conn="x", memlet=dace.Memlet("A[i]"))
        if read_before_write:
            # ``t`` is read (the value an earlier map left) before this scope writes it
            st.add_edge(me, None, t, None, dace.Memlet())
            st.add_edge(t, None, second, "y", dace.Memlet("t[0]"))
            sink = st.add_access("t")
            st.add_edge(first, "y", sink, None, dace.Memlet("t[0]"))
            st.add_edge(sink, None, mx, None, dace.Memlet())
        else:
            st.add_edge(first, "y", t, None, dace.Memlet("t[0]"))
            st.add_edge(t, None, second, "y", dace.Memlet("t[0]"))
        st.add_memlet_path(second, mx, st.add_write("B"), src_conn="z", memlet=dace.Memlet("B[i]"))
    return sdfg


def _scratch_names(sdfg: dace.SDFG) -> set[str]:
    return {n.data for n, _ in sdfg.all_nodes_recursive() if isinstance(n, nodes.AccessNode) and n.data.startswith("t")}


def test_shared_scratch_is_split_per_map_and_allocated_inside_it():
    sdfg = _two_branch_sdfg()
    sdfg.validate()
    assert SplitMapLocalTransients().apply_pass(sdfg, {}) == 1
    sdfg.validate()
    assert len(_scratch_names(sdfg)) == 2
    code = sdfg.generate_code()[0].clean_code
    # Each instance is declared inside its parallel loop, not once for the function
    body = code[code.index("#pragma omp parallel for") :]
    for name in _scratch_names(sdfg):
        assert f"double {name};" in body or f"double {name}[1]" in body
    A = np.random.default_rng(3).random(N)
    for flag, factor in ((1, 2.0), (0, 3.0)):
        B = np.zeros(N)
        sdfg(A=A, B=B, flag=flag)
        np.testing.assert_allclose(B, factor * A + 1.0)


def test_a_scratch_read_before_its_write_in_a_scope_is_kept():
    sdfg = _two_branch_sdfg(read_before_write=True)
    assert SplitMapLocalTransients().apply_pass(sdfg, {}) is None
    assert _scratch_names(sdfg) == {"t"}


if __name__ == "__main__":
    test_shared_scratch_is_split_per_map_and_allocated_inside_it()
    test_a_scratch_read_before_its_write_in_a_scope_is_kept()
