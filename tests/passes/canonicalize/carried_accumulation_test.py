# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""A loop whose only cross-iteration writes are ``X[s] = X[s] + r`` at a data-dependent slice maps once the updates
are WCR, and the result is the same whichever slices overlap (lavamd's box loop stayed sequential)."""

import numpy as np

import dace
from dace import memlet as mm
from dace.sdfg.state import LoopRegion
from dace.transformation.passes.canonicalize.carried_accumulation import CarriedAccumulationToWCR
from dace.transformation.passes.canonicalize.pipeline import canonicalize

NB, P, Q, NT = dace.symbol("NB"), dace.symbol("P"), dace.symbol("Q"), dace.symbol("NT")


@dace.program
def boxes(offsets: dace.int64[NB], vals: dace.float64[NB, P, Q], out: dace.float64[NT]):
    for b in range(NB):
        o = offsets[b]
        row = np.sum(vals[b, :, :], axis=1)
        out[o : o + P] = out[o : o + P] + row


def test_a_box_loop_accumulating_at_data_offsets_maps_and_matches():
    sdfg = boxes.to_sdfg(simplify=True)
    canonicalize(sdfg, target="cpu")
    assert not [r for r in sdfg.all_control_flow_regions() if isinstance(r, LoopRegion)]
    rng = np.random.default_rng(0)
    nb, p = 9, 5
    offsets = np.array([0, 3, 3, 10, 12, 20, 21, 30, 35], dtype=np.int64)  # boxes 1 and 2 overlap
    vals = rng.random((nb, p, 4))
    out = rng.random(40)
    want = out.copy()
    for b in range(nb):
        want[offsets[b] : offsets[b] + p] += vals[b].sum(axis=1)
    sdfg(offsets=offsets, vals=vals, out=out, NB=nb, P=p, Q=4, NT=40)
    assert np.allclose(out, want)


@dace.program
def scatter(idx: dace.int64[NB], vals: dace.float64[NB, P, Q], out: dace.float64[NT]):
    for b in range(NB):
        s = np.sum(vals[b, :, :])
        out[idx[b]] = out[idx[b]] + s


def test_a_scatter_of_one_element_per_iteration_maps_and_matches():
    """A free tasklet's ``out[idx[b]] = out[idx[b]] + s`` (gromacs's pair forces): repeated indices accumulate."""
    sdfg = scatter.to_sdfg(simplify=True)
    canonicalize(sdfg, target="cpu")
    assert not [r for r in sdfg.all_control_flow_regions() if isinstance(r, LoopRegion)]
    rng = np.random.default_rng(1)
    idx = np.array([3, 1, 3, 0, 7, 1], dtype=np.int64)
    vals = rng.random((6, 2, 3))
    out = rng.random(9)
    want = out.copy()
    for b in range(6):
        want[idx[b]] += vals[b].sum()
    sdfg(idx=idx, vals=vals, out=out, NB=6, P=2, Q=3, NT=9)
    assert np.allclose(out, want)


def update_loop(recurrence: bool) -> dace.SDFG:
    """``out[b // 2] = out[b // 2] + v[b]`` over ``b``, then, if ``recurrence``, ``c[b + 1] = 0.5 * c[b]``."""
    sdfg = dace.SDFG("update_loop")
    sdfg.add_array("out", [4], dace.float64)
    sdfg.add_array("v", [8], dace.float64)
    sdfg.add_array("c", [9], dace.float64)
    loop = LoopRegion("over_b", "b < 8", "b", "b = 0", "b = b + 1")
    sdfg.add_node(loop, is_start_block=True)
    update = loop.add_state("update", is_start_block=True)
    add = update.add_tasklet("add", {"__in1": None, "__in2": None}, {"__out": None}, "__out = __in1 + __in2")
    update.add_edge(update.add_read("out"), None, add, "__in1", mm.Memlet("out[b // 2]"))
    update.add_edge(update.add_read("v"), None, add, "__in2", mm.Memlet("v[b]"))
    update.add_edge(add, "__out", update.add_write("out"), None, mm.Memlet("out[b // 2]"))
    if recurrence:
        recur = loop.add_state_after(update, "recur")
        half = recur.add_tasklet("half", {"__in": None}, {"__out": None}, "__out = 0.5 * __in")
        recur.add_edge(recur.add_read("c"), None, half, "__in", mm.Memlet("c[b]"))
        recur.add_edge(half, "__out", recur.add_write("c"), None, mm.Memlet("c[b + 1]"))
    sdfg.reset_cfg_list()
    return sdfg


def test_an_update_of_a_loop_a_recurrence_keeps_sequential_stays_plain():
    """With ``c[b + 1] = 0.5 * c[b]`` beside it the loop stays a loop whatever the update is, so it stays plain
    (vexx_k's ``result`` update; RevertNonReductionWCR undid it); alone it becomes WCR."""
    sdfg = update_loop(recurrence=True)
    assert CarriedAccumulationToWCR().apply_pass(sdfg, {}) is None
    assert not [e for st in sdfg.states() for e in st.edges() if e.data.wcr]
    assert CarriedAccumulationToWCR().apply_pass(update_loop(recurrence=False), {}) == 1


if __name__ == "__main__":
    test_a_box_loop_accumulating_at_data_offsets_maps_and_matches()
    test_a_scatter_of_one_element_per_iteration_maps_and_matches()
    test_an_update_of_a_loop_a_recurrence_keeps_sequential_stays_plain()
