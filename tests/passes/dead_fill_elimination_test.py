# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""A memset whose array the next map overwrites element for element, unread, is dropped."""

import numpy as np

import dace
from dace.libraries.standard.nodes.fill import FillLibraryNode
from dace.transformation.passes.dead_fill_elimination import DeadFillElimination

N = dace.symbol("N")


def zeroed_then_written(index: str = "i", map_range: str = "0:N", read_back: bool = False) -> dace.SDFG:
    """``tmp = 0; tmp[index] = b[i] (+ tmp[i]) for i in map_range; out = tmp`` in one state."""
    sdfg = dace.SDFG(f"zeroed_then_written_{abs(hash((index, map_range, read_back)))}")
    sdfg.add_array("b", [N], dace.float64)
    sdfg.add_array("idx", [N], dace.int64)
    sdfg.add_array("out", [N], dace.float64)
    sdfg.add_transient("tmp", [N], dace.float64)
    state = sdfg.add_state()
    zeroed = state.add_access("tmp")
    fill = FillLibraryNode("memset_tmp", value=0)
    state.add_node(fill)
    state.add_edge(fill, FillLibraryNode.OUTPUT_CONNECTOR_NAME, zeroed, None, dace.Memlet("tmp[0:N]"))
    b = state.add_read("b")
    state.add_nedge(zeroed, b, dace.Memlet())
    written = state.add_access("tmp")
    inputs = {"__b": dace.Memlet("b[i]")}
    code = "__t = __b"
    if read_back:
        inputs["__old"] = dace.Memlet("tmp[i]")
        code = "__t = __b + __old"
    if index != "i":
        inputs["__k"] = dace.Memlet("idx[i]")
    state.add_mapped_tasklet(
        "write",
        {"i": map_range},
        inputs,
        code,
        {"__t": dace.Memlet(f"tmp[{index}]")},
        input_nodes={"b": b, **({"tmp": zeroed} if read_back else {})},
        output_nodes={"tmp": written},
        external_edges=True,
    )
    state.add_nedge(written, state.add_write("out"), dace.Memlet("tmp[0:N]"))
    return sdfg


def fills(sdfg: dace.SDFG) -> int:
    return sum(isinstance(n, FillLibraryNode) for n, _ in sdfg.all_nodes_recursive())


def test_overwritten_memset_is_dropped():
    sdfg = zeroed_then_written()
    assert DeadFillElimination().apply_pass(sdfg, {}) == 1
    assert fills(sdfg) == 0
    sdfg.validate()
    b = np.random.rand(17)
    out = np.zeros(17)
    sdfg(b=b, idx=np.arange(17), out=out, N=17)
    assert np.allclose(out, b)


def test_memset_kept_when_the_map_misses_elements():
    sdfg = zeroed_then_written(map_range="1:N")
    assert DeadFillElimination().apply_pass(sdfg, {}) is None
    assert fills(sdfg) == 1


def test_memset_kept_under_an_indirect_write():
    sdfg = zeroed_then_written(index="__k")
    assert DeadFillElimination().apply_pass(sdfg, {}) is None
    assert fills(sdfg) == 1


def test_memset_kept_when_the_map_reads_it():
    sdfg = zeroed_then_written(read_back=True)
    assert DeadFillElimination().apply_pass(sdfg, {}) is None
    assert fills(sdfg) == 1


if __name__ == "__main__":
    test_overwritten_memset_is_dropped()
    test_memset_kept_when_the_map_misses_elements()
    test_memset_kept_under_an_indirect_write()
    test_memset_kept_when_the_map_reads_it()
