# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Sub-cases of ``MapFusionVertical`` refusals that are provably safe: an intermediate read again past the fused Map,
and two Maps whose ranges spell one extent as ``int_floor(x, 2)`` and ``floor(x/2)``."""

import numpy as np
import sympy

import dace
from dace import subsets
from dace.memlet import Memlet
from dace.sdfg import nodes
from dace.transformation.dataflow import map_fusion_helper as mfhelper
from dace.transformation.dataflow.map_fusion_vertical import MapFusionVertical
from dace.transformation.passes.fuse_maps import FuseMaps, iteration_space

H = dace.symbol("H")


def downstream_sdfg(name: str, third_reads_intermediate: bool) -> tuple[dace.SDFG, nodes.MapExit, nodes.MapEntry]:
    """``T = A + 1; O = 2 T; T' = (T +) O; R = T'``: the third Map writes ``T`` again, reading the first ``T`` too
    when ``third_reads_intermediate``, else only ``O``."""
    sdfg = dace.SDFG(name)
    state = sdfg.add_state(is_start_block=True)
    for arr in ("A", "O", "R"):
        sdfg.add_array(arr, (10,), dace.float64)
    sdfg.add_array("T", (10,), dace.float64, transient=True)
    a, t, o, t_again = (state.add_access(n) for n in ("A", "T", "O", "T"))
    first_tasklet, first_entry, first_exit = state.add_mapped_tasklet(
        "producer",
        {"i": "0:10"},
        inputs={"__a": Memlet("A[i]")},
        code="__t = __a + 1.0",
        outputs={"__t": Memlet("T[i]")},
        input_nodes={"A": a},
        output_nodes={"T": t},
        external_edges=True,
    )
    second_tasklet, second_entry, second_exit = state.add_mapped_tasklet(
        "consumer",
        {"j": "0:10"},
        inputs={"__t": Memlet("T[j]")},
        code="__o = 2.0 * __t",
        outputs={"__o": Memlet("O[j]")},
        input_nodes={"T": t},
        output_nodes={"O": o},
        external_edges=True,
    )
    inputs = {"__o": Memlet("O[m]")}
    input_nodes = {"O": o}
    if third_reads_intermediate:
        inputs["__t"] = Memlet("T[m]")
        input_nodes["T"] = t
    state.add_mapped_tasklet(
        "rewrite",
        {"m": "0:10"},
        inputs=inputs,
        code="__u = __t + __o" if third_reads_intermediate else "__u = __o + 10.0",
        outputs={"__u": Memlet("T[m]")},
        input_nodes=input_nodes,
        output_nodes={"T": t_again},
        external_edges=True,
    )
    after = sdfg.add_state_after(state)
    after.add_nedge(after.add_access("T"), after.add_access("R"), sdfg.make_array_memlet("T"))
    sdfg.validate()
    return sdfg, first_exit, second_entry


def fusable(sdfg: dace.SDFG, first_exit: nodes.MapExit, second_entry: nodes.MapEntry, relaxed: bool) -> bool:
    state = sdfg.start_block
    (t,) = [e.dst for e in state.out_edges(first_exit) if isinstance(e.dst, nodes.AccessNode)]
    return MapFusionVertical.can_be_applied_to(
        sdfg,
        options={"relaxed_dataflow": relaxed},
        first_map_exit=first_exit,
        array=t,
        second_map_entry=second_entry,
    )


def map_count(sdfg: dace.SDFG) -> int:
    return sum(isinstance(n, nodes.MapEntry) for state in sdfg.states() for n in state.nodes())


def test_an_intermediate_written_again_past_another_reader_fuses_under_relaxed_dataflow():
    """The third Map reads ``T`` itself, so ``T``'s rewrite stays ordered after the rebuilt ``T`` of the fused Map."""
    sdfg, first_exit, second_entry = downstream_sdfg("mfv_downstream_past_reader", third_reads_intermediate=True)
    assert not fusable(sdfg, first_exit, second_entry, relaxed=False)
    assert fusable(sdfg, first_exit, second_entry, relaxed=True)
    (t,) = [e.dst for e in sdfg.start_block.out_edges(first_exit) if isinstance(e.dst, nodes.AccessNode)]
    MapFusionVertical.apply_to(
        sdfg, options={"relaxed_dataflow": True}, first_map_exit=first_exit, array=t, second_map_entry=second_entry
    )
    sdfg.validate()
    assert map_count(sdfg) == 2
    a = np.random.default_rng(0).random(10)
    o, r = np.zeros(10), np.zeros(10)
    sdfg(A=a, O=o, R=r)
    assert np.allclose(o, 2.0 * (a + 1.0))
    assert np.allclose(r, 3.0 * (a + 1.0))


def test_an_intermediate_written_again_only_past_the_second_map_stays_refused():
    """Reached only through the second Map, the rewrite would stand beside the rebuilt ``T``, unordered with it."""
    sdfg, first_exit, second_entry = downstream_sdfg("mfv_downstream_through_second", third_reads_intermediate=False)
    assert not fusable(sdfg, first_exit, second_entry, relaxed=True)


def test_floor_and_int_floor_spell_one_range():
    parsed = subsets.Range.from_string("0:H//2")[0]
    python = (0, sympy.floor(H / 2) - 1, 1)
    assert parsed != python
    assert mfhelper.same_range(parsed, python)
    assert not mfhelper.same_range(parsed, (0, sympy.floor(H / 3) - 1, 1))


def spelled_sdfg(name: str, divisor: int) -> dace.SDFG:
    """``T[i] = A[i] + 1`` over ``0:H//2`` (``int_floor``), then ``B[j] = 2 T[j]`` over ``floor(H/divisor)``."""
    sdfg = dace.SDFG(name)
    state = sdfg.add_state(is_start_block=True)
    sdfg.add_array("A", (H,), dace.float64)
    sdfg.add_array("B", (H,), dace.float64)
    sdfg.add_array("T", (H,), dace.float64, transient=True)
    t = state.add_access("T")
    state.add_mapped_tasklet(
        "first",
        {"i": "0:H//2"},
        inputs={"__a": Memlet("A[i]")},
        code="__t = __a + 1.0",
        outputs={"__t": Memlet("T[i]")},
        output_nodes={"T": t},
        external_edges=True,
    )
    tasklet, entry, exit_node = state.add_mapped_tasklet(
        "second",
        {"j": "0:H//2"},
        inputs={"__t": Memlet("T[j]")},
        code="__b = 2.0 * __t",
        outputs={"__b": Memlet("B[j]")},
        input_nodes={"T": t},
        external_edges=True,
    )
    entry.map.range = subsets.Range([(0, sympy.floor(H / divisor) - 1, 1)])
    return sdfg


def fuse_maps(sdfg: dace.SDFG) -> None:
    FuseMaps(
        allow_intermediates_with_symbolic_size=True, relaxed_dataflow=True, perform_horizontal_map_fusion=False
    ).apply_pass(sdfg, {})
    sdfg.validate()


def test_maps_over_one_extent_in_two_spellings_fuse():
    sdfg = spelled_sdfg("mfv_floor_spellings", divisor=2)
    first, second = (n.map for n in sdfg.start_block.nodes() if isinstance(n, nodes.MapEntry))
    assert iteration_space(first) == iteration_space(second)
    fuse_maps(sdfg)
    assert map_count(sdfg) == 1
    a = np.random.default_rng(1).random(9)
    b = np.zeros(9)
    sdfg(A=a, B=b, H=9)
    assert np.allclose(b[:4], 2.0 * (a[:4] + 1.0))


def test_maps_over_different_floors_stay_apart():
    sdfg = spelled_sdfg("mfv_floor_other_divisor", divisor=3)
    first, second = (n.map for n in sdfg.start_block.nodes() if isinstance(n, nodes.MapEntry))
    assert iteration_space(first) != iteration_space(second)
    assert mfhelper.find_parameter_remapping(first, second) is None
    fuse_maps(sdfg)
    assert map_count(sdfg) == 2


if __name__ == "__main__":
    test_an_intermediate_written_again_past_another_reader_fuses_under_relaxed_dataflow()
    test_an_intermediate_written_again_only_past_the_second_map_stays_refused()
    test_floor_and_int_floor_spell_one_range()
    test_maps_over_one_extent_in_two_spellings_fuse()
    test_maps_over_different_floors_stay_apart()
