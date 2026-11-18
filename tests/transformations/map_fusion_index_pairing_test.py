# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""``MapFusionVertical`` pairs the parameters of two Maps over one square range by how both index the intermediate:
the order of the parameters alone pairs ``T[i, j]`` written with ``T[j, i]`` read as two different elements."""

import numpy as np

import dace
from dace.sdfg import nodes
from dace.transformation.dataflow import map_fusion_helper as mfhelper
from dace.transformation.passes.fuse_maps import FuseMaps

N = dace.symbol("N")


@dace.program
def transposed_read_same_names(A: dace.float64[N, N], B: dace.float64[N, N]):
    T = np.empty((N, N), dtype=np.float64)
    for i, j in dace.map[0:N, 0:N]:
        T[i, j] = A[i, j] + 1.0
    for i, j in dace.map[0:N, 0:N]:
        B[i, j] = T[j, i] * 2.0


@dace.program
def transposed_read_other_names(A: dace.float64[N, N], B: dace.float64[N, N]):
    T = np.empty((N, N), dtype=np.float64)
    for i, j in dace.map[0:N, 0:N]:
        T[i, j] = A[i, j] + 1.0
    for k, m in dace.map[0:N, 0:N]:
        B[k, m] = T[m, k] * 2.0


@dace.program
def both_orientations_read(A: dace.float64[N, N], B: dace.float64[N, N]):
    T = np.empty((N, N), dtype=np.float64)
    for i, j in dace.map[0:N, 0:N]:
        T[i, j] = A[i, j] + 1.0
    for i, j in dace.map[0:N, 0:N]:
        B[i, j] = T[i, j] - T[j, i]


def top_level_maps(sdfg: dace.SDFG) -> list[nodes.Map]:
    return [
        n.map
        for state in sdfg.states()
        for n in state.nodes()
        if isinstance(n, nodes.MapEntry) and state.entry_node(n) is None
    ]


def intermediate_nodes(sdfg: dace.SDFG) -> list[nodes.AccessNode]:
    return [n for state in sdfg.states() for n in state.data_nodes() if n.data == "T"]


def fuse(sdfg: dace.SDFG) -> None:
    FuseMaps(
        allow_intermediates_with_symbolic_size=True, relaxed_dataflow=True, perform_horizontal_map_fusion=False
    ).apply_pass(sdfg, {})
    sdfg.validate()


def test_the_remapping_takes_the_pair_both_maps_index_the_intermediate_by():
    first = dace.nodes.Map("first", ["i", "j"], dace.subsets.Range.from_string("0:N, 0:N"))
    second = dace.nodes.Map("second", ["i", "j"], dace.subsets.Range.from_string("0:N, 0:N"))
    assert mfhelper.find_parameter_remapping(first, second) == {}
    assert mfhelper.find_parameter_remapping(first, second, preferred={"i": "j", "j": "i"}) == {"i": "j", "j": "i"}


def test_a_preferred_pair_over_different_ranges_is_ignored():
    first = dace.nodes.Map("first", ["i", "j"], dace.subsets.Range.from_string("0:N, 0:8"))
    second = dace.nodes.Map("second", ["k", "m"], dace.subsets.Range.from_string("0:N, 0:8"))
    assert mfhelper.find_parameter_remapping(first, second, preferred={"k": "j"}) == {"k": "i", "m": "j"}


def test_a_transposed_read_of_a_square_intermediate_fuses_under_one_name_set():
    sdfg = transposed_read_same_names.to_sdfg(simplify=True)
    assert len(top_level_maps(sdfg)) == 2
    fuse(sdfg)
    assert len(top_level_maps(sdfg)) == 1
    assert intermediate_nodes(sdfg) == []
    a = np.random.default_rng(0).random((7, 7))
    b = np.zeros((7, 7))
    sdfg(A=a, B=b, N=7)
    assert np.allclose(b, 2.0 * (a + 1.0).T)


def test_a_transposed_read_of_a_square_intermediate_fuses_under_other_names():
    sdfg = transposed_read_other_names.to_sdfg(simplify=True)
    assert len(top_level_maps(sdfg)) == 2
    fuse(sdfg)
    assert len(top_level_maps(sdfg)) == 1
    assert intermediate_nodes(sdfg) == []
    a = np.random.default_rng(1).random((7, 7))
    b = np.zeros((7, 7))
    sdfg(A=a, B=b, N=7)
    assert np.allclose(b, 2.0 * (a + 1.0).T)


def test_a_consumer_reading_both_orientations_keeps_the_intermediate():
    """``T[j, i]`` is written by another iteration under every pairing that makes ``T[i, j]`` one element."""
    sdfg = both_orientations_read.to_sdfg(simplify=True)
    fuse(sdfg)
    assert len(top_level_maps(sdfg)) == 2
    a = np.random.default_rng(2).random((7, 7))
    b = np.zeros((7, 7))
    sdfg(A=a, B=b, N=7)
    assert np.allclose(b, (a + 1.0) - (a + 1.0).T)


if __name__ == "__main__":
    test_the_remapping_takes_the_pair_both_maps_index_the_intermediate_by()
    test_a_preferred_pair_over_different_ranges_is_ignored()
    test_a_transposed_read_of_a_square_intermediate_fuses_under_one_name_set()
    test_a_transposed_read_of_a_square_intermediate_fuses_under_other_names()
    test_a_consumer_reading_both_orientations_keeps_the_intermediate()
